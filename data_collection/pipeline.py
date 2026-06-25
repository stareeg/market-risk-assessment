"""
Этап 1. Сбор данных из первоисточников.

Качаем всё из пункта 1 задания (кривая ЦБ, ОФЗ, акции, индексы, Brent, курсы,
срочный рынок) и складываем в data/ как parquet. Сами запросы лежат в
sources.py, тут только оркестрация: что за чем тянем и в каком виде сохраняем.

Запуск: python main.py --stage data
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from . import sources as S

pd.options.io.parquet.engine = "fastparquet"


def _make_saver(data_dir: Path):
    """Возвращает функцию сохранения в parquet с короткой сводкой в консоль."""
    def save_parquet(df: pd.DataFrame, name: str) -> Path:
        path = data_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  сохранено {name}.parquet: строк {len(df)}, колонок {df.shape[1]}")
        return path
    return save_parquet


def run(data_dir: str | Path | None = None) -> None:
    """
    Качает все данные пункта 1 и сохраняет в data_dir.
    По умолчанию пишем в config.DATA_DIR. Можно передать свою папку, чтобы
    проверить загрузку, не трогая уже сохранённые данные.
    """
    data_dir = Path(data_dir) if data_dir is not None else C.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    save_parquet = _make_saver(data_dir)

    np.random.seed(C.RANDOM_SEED)   # фиксируем seed (воспроизводимость)
    session = S._session()          # один HTTP-сеанс на всю загрузку
    print(f"Период загрузки: с {C.START_DATE} по {C.END_DATE}")
    print(f"Каталог данных: {data_dir}")

    # Кривая бескупонной доходности ЦБ РФ (спот-доходности по срокам)
    print("\nКривая бескупонной доходности ЦБ РФ")
    zcyc = S.get_cbr_zcyc(session, C.START_DATE, C.END_DATE)
    print(f"  период кривой: с {zcyc['DATE'].min().date()} по {zcyc['DATE'].max().date()}")
    save_parquet(zcyc, "zcyc_cbr")

    # Описания ОФЗ и расписания купонов. Берём 10 выпусков с запасом.
    # По каждому находим полный SECID, тянем карточку и расписание выплат.
    print(f"\nОблигации ОФЗ: описания и купоны ({len(C.OFZ_PD_NUMBERS)} выпусков)")
    desc_rows, coupon_frames, secids = [], [], {}
    for num in C.OFZ_PD_NUMBERS:
        secid = S.resolve_ofz_secid(session, num)
        secids[num] = secid
        d = S.get_bond_description(session, secid)
        desc_rows.append({
            "NUMBER": num, "SECID": secid, "ISIN": d.get("ISIN"),
            "NAME": d.get("NAME"), "MATDATE": d.get("MATDATE"),
            "COUPONPERCENT": d.get("COUPONPERCENT"), "COUPONPERIOD": d.get("COUPONPERIOD"),
            "FACEVALUE": d.get("FACEVALUE"), "TYPE": d.get("TYPE"),
            "OFFERDATE": d.get("OFFERDATE"),   # пусто значит без оферты
        })
        cp, am, of = S.get_bond_schedule(session, secid)
        cp = cp.copy()
        cp["NUMBER"] = num
        coupon_frames.append(cp)

    bonds_desc = pd.DataFrame(desc_rows)
    bonds_desc["MATDATE"] = pd.to_datetime(bonds_desc["MATDATE"])
    # Проверяем критерии задания: погашение после 2026 и без оферты.
    bonds_desc["after_2026"] = bonds_desc["MATDATE"] > pd.Timestamp("2026-01-01")
    bonds_desc["no_offer"] = bonds_desc["OFFERDATE"].isna()
    assert bonds_desc["after_2026"].all(), "Есть выпуск с погашением до 2026"
    assert bonds_desc["no_offer"].all(), "Есть выпуск с офертой"
    print("  все выпуски без оферты и с погашением после 2026, критерии задания сходятся")
    save_parquet(bonds_desc, "bonds_descriptions")

    # Расписания купонов всех выбранных ОФЗ
    bonds_coupons = pd.concat(coupon_frames, ignore_index=True)
    bonds_coupons["coupondate"] = pd.to_datetime(bonds_coupons["coupondate"])
    save_parquet(bonds_coupons, "bonds_coupons")

    # Котировки выбранных ОФЗ (доска TQOB)
    print("\nКотировки ОФЗ (доска TQOB)")
    bh_frames = []
    for num, secid in secids.items():
        h = S.get_bond_history(session, secid, C.START_DATE, C.END_DATE)
        h["NUMBER"] = num
        bh_frames.append(h)
    bonds_history = pd.concat(bh_frames, ignore_index=True)
    bonds_history["TRADEDATE"] = pd.to_datetime(bonds_history["TRADEDATE"])
    for c in ["CLOSE", "LEGALCLOSEPRICE", "ACCINT", "WAPRICE", "YIELDCLOSE",
              "DURATION", "VOLUME", "VALUE"]:
        bonds_history[c] = pd.to_numeric(bonds_history[c], errors="coerce")
    save_parquet(bonds_history, "bonds_history")

    # Котировки акций (доска TQBR). Берём 12 тикеров с запасом.
    print(f"\nКотировки акций (доска TQBR, {len(C.STOCK_TICKERS)} тикеров)")
    sh_frames = []
    for t in C.STOCK_TICKERS:
        h = S.get_share_history(session, t, C.START_DATE, C.END_DATE)
        sh_frames.append(h)
    stocks_history = pd.concat(sh_frames, ignore_index=True)
    stocks_history["TRADEDATE"] = pd.to_datetime(stocks_history["TRADEDATE"])
    for c in ["CLOSE", "LEGALCLOSEPRICE", "WAPRICE", "OPEN", "HIGH", "LOW",
              "VOLUME", "VALUE"]:
        stocks_history[c] = pd.to_numeric(stocks_history[c], errors="coerce")
    save_parquet(stocks_history, "stocks_history")

    # Индексы МосБиржи (IMOEX) и РТС (RTSI)
    print("\nИндексы МосБиржи (IMOEX) и РТС (RTSI)")
    idx_frames = []
    for t in C.INDEX_TICKERS:
        h = S.get_index_history(session, t, C.START_DATE, C.END_DATE)
        idx_frames.append(h)
    indices_history = pd.concat(idx_frames, ignore_index=True)
    indices_history["TRADEDATE"] = pd.to_datetime(indices_history["TRADEDATE"])
    for c in ["CLOSE", "OPEN", "HIGH", "LOW", "VALUE"]:
        indices_history[c] = pd.to_numeric(indices_history[c], errors="coerce")
    save_parquet(indices_history, "indices_history")

    # Нефть Brent. Спота нет, строим непрерывный фронт-месяц из фьючерсов BR.
    # Это самый долгий шаг, перебираем все месячные контракты.
    print("\nНефть Brent, склеиваем фронт-месяц из фьючерсов BR (шаг долгий)")
    brent = S.get_brent_front_month(session, C.START_DATE, C.END_DATE, C.BRENT_ASSETCODE)
    brent["TRADEDATE"] = pd.to_datetime(brent["TRADEDATE"])
    print(f"  собрано {len(brent)} торговых дней")
    save_parquet(brent, "brent_history")

    # Официальные курсы USD и EUR (ЦБ РФ, непрерывный ряд)
    print("\nОфициальные курсы USD и EUR (ЦБ РФ)")
    fx_frames = []
    for code, val in C.CBR_CURRENCIES.items():
        f = S.get_cbr_fx(session, val, C.START_DATE, C.END_DATE)
        f["CCY"] = code
        fx_frames.append(f)
    fx = pd.concat(fx_frames, ignore_index=True)
    save_parquet(fx, "fx_cbr")

    # Фьючерс и опционы на Si за выбранный день. Берём ближайший фьючерс
    # с экспирацией больше месяца и опционы той же серии.
    print(f"\nСрочный рынок Si за {C.FORTS_TRADE_DAY}")
    fut_day = S.get_forts_futures_on_date(session, C.FORTS_ASSETCODE, C.FORTS_TRADE_DAY)
    fut_specs = S.get_futures_specs(session, fut_day["SECID"].tolist())
    front = S.pick_front_future(fut_specs, C.FORTS_TRADE_DAY, C.FORTS_MIN_DAYS_TO_EXPIRY)
    fut_day = fut_day.merge(fut_specs[["SECID", "LSTDELDATE"]], on="SECID", how="left")
    fut_day["IS_CHOSEN_FRONT"] = fut_day["SECID"] == front
    print(f"  выбранный фьючерс: {front}")
    save_parquet(fut_day, f"forts_futures_{C.FORTS_TRADE_DAY}")

    # Опционная цепочка на выбранный фьючерс (Call и Put, все страйки)
    opt_day = S.get_forts_options_on_date(session, C.FORTS_ASSETCODE, C.FORTS_TRADE_DAY)
    # Предфильтр по суффиксу SECID (декабрьская квартальная серия) ускоряет выгрузку
    cand = [x for x in opt_day["SECID"] if re.search(r"(L5|X5)$", x)]
    specs = S.get_option_specs(session, cand)
    specs["LSTDELDATE"] = pd.to_datetime(specs["LSTDELDATE"])
    chosen_exp = fut_specs.set_index("SECID").loc[front, "LSTDELDATE"]
    keep = specs[(specs["UNDERLYINGASSET"] == front) & (specs["LSTDELDATE"] == chosen_exp)]
    chain = opt_day.merge(
        keep[["SECID", "STRIKE", "OPTIONTYPE", "UNDERLYINGASSET", "LSTDELDATE"]],
        on="SECID", how="inner")
    chain["STRIKE"] = pd.to_numeric(chain["STRIKE"], errors="coerce")
    chain = chain.sort_values(["OPTIONTYPE", "STRIKE"]).reset_index(drop=True)
    print(f"  опционов в цепочке: {len(chain)}")
    save_parquet(chain, f"forts_options_chain_{C.FORTS_TRADE_DAY}")

    # Короткий итог по сохранённым файлам
    print("\nИтог, сохранённые файлы:")
    for f in sorted(data_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        print(f"  {f.name:42} строк {len(df):>6}, колонок {df.shape[1]}")


if __name__ == "__main__":
    run()
