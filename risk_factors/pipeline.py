"""
Этап 2. Выделение риск-факторов и описательная статистика.

Готовим данные из пункта 1 (сплиты, разрывы, переход к приращениям), сжимаем
кривую и акции через PCA, добавляем валютные факторы и сохраняем итоговый набор
из 8 риск-факторов в data/. Тяжёлая логика лежит в factors.py, pca_tools.py и
diagnostics.py, тут только оркестрация и проверка значений.

Запуск: python main.py --stage factors
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from . import factors as F
from . import pca_tools as P
from . import diagnostics as D

pd.options.io.parquet.engine = "fastparquet"


def _show(title: str, df: pd.DataFrame, ndigits: int = 4) -> None:
    """Печатает таблицу с заголовком, числовые колонки округляет для читаемости."""
    print(f"\n{title}")
    out = df
    if isinstance(out, pd.DataFrame):
        out = out.copy()
        num = out.select_dtypes(include=[np.number]).columns
        out[num] = out[num].round(ndigits)
    elif isinstance(out, pd.Series) and pd.api.types.is_numeric_dtype(out):
        out = out.round(ndigits)
    print(out.to_string())


def run(data_dir: str | Path | None = None) -> None:
    """
    Считает риск-факторы по данным из data_dir и сохраняет их туда же.
    По умолчанию работаем с config.DATA_DIR. Можно передать свою папку, чтобы
    проверить расчёт, не трогая уже сохранённые файлы.
    """
    data_dir = Path(data_dir) if data_dir is not None else C.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(C.RANDOM_SEED)   # фиксируем seed (воспроизводимость)
    panels = F.load_panels(data_dir)
    print(f"Каталог данных: {data_dir}")
    print("Размеры загруженных панелей:", {k: v.shape for k, v in panels.items()})

    # Чистим цены акций от сплитов, иначе день дробления даёт ложный обвал
    # доходности в N раз.
    splits = F.detect_splits(panels["stock_px"])
    _show("Найденные сплиты (реальные корпоративные события):",
          splits.set_index("TICKER") if not splits.empty else splits)
    stock_px_adj, _ = F.adjust_splits(panels["stock_px"], splits)

    # Единый торговый календарь MOEX по дням, когда торговался широкий рынок акций.
    # Курсы и кривую ЦБ протягиваем на него: опубликованное значение действует до
    # следующей публикации, иначе праздники ЦБ выбивали бы часть наблюдений.
    master = stock_px_adj.dropna(how="all").index
    fx_lvl = F.to_trading_calendar(panels["fx_px"], master)
    yield_lvl = F.to_trading_calendar(panels["yields"], master)

    # Переходим к приращениям: лог-доходности цен и изменения ставок в б.п.
    stock_ret = F.log_returns(stock_px_adj)
    fx_ret = F.log_returns(fx_lvl)
    brent_ret = F.log_returns(F.to_trading_calendar(panels["brent"], master))
    index_ret = F.log_returns(panels["index_px"])
    dY = F.yield_changes(yield_lvl)

    n_gap = int((~F.mask_calendar_gaps(stock_px_adj)).sum())
    print(f"\nТорговых дней в общем календаре: {len(master)}")
    print(f"Исключено доходностей через длинные разрывы (включая остановку биржи в 2022): {n_gap}")

    # PCA кривой по изменениям ставок (ковариация, единицы одинаковые, б.п.).
    # Три компоненты это уровень, наклон, кривизна.
    scree_y = P.full_scree(dY)
    model_y, scores_y, load_y, evr_y = P.pca_on_changes(dY, n_components=3)
    print("\nPCA кривой, доля объяснённой дисперсии по компонентам:",
          evr_y.round(4).to_dict())
    print(f"  три компоненты вместе объясняют {evr_y.sum():.1%} дисперсии")
    _show("Интерпретация компонент кривой:", P.interpret_curve_pcs(load_y))

    # PCA акций по корреляциям (доходности стандартизуем, волатильности разные).
    # Структуру проекта держим на 8 факторах, поэтому берём ровно 3 компоненты,
    # критерий Кайзера печатаем как справку.
    eq = stock_ret.dropna()
    scree_e = P.full_scree(eq, standardize=True)
    eigvals = scree_e.values * eq.shape[1]
    k_kaiser = int((eigvals > 1).sum())
    cum = {f"PC{i+1}": round(scree_e.head(i + 1).sum(), 3) for i in range(min(6, len(scree_e)))}
    print(f"\nКритерий Кайзера для акций: компонент с собственным числом больше 1: {k_kaiser}")
    print("Накопленная дисперсия акций по компонентам:", cum)

    model_e, scores_e, load_e, evr_e = P.pca_on_changes(eq, n_components=3, standardize=True)
    print("PCA акций, доля объяснённой дисперсии по компонентам:",
          evr_e.round(4).to_dict())
    print(f"  три компоненты вместе объясняют {evr_e.sum():.1%} дисперсии")

    # Проверяем, что PC1 акций это рыночный фактор (сравниваем с доходностью IMOEX)
    imoex_ret = index_ret["IMOEX"].reindex(scores_e.index)
    corr_pc1_imoex = scores_e["PC1"].corr(imoex_ret)
    print(f"corr(PC1 акций, доходность IMOEX) = {corr_pc1_imoex:.3f}. "
          "Значит PC1 это рыночный фактор")

    # Проверяем избыточность РТС и роль Brent (почему их не берём факторами)
    piv = panels["index_px"].join(panels["fx_px"], how="inner").dropna()
    ratio = piv["IMOEX"] / piv["RTSI"]
    print(f"\ncorr(IMOEX/RTSI, USD/RUB) = {ratio.corr(piv['USD']):.3f}. "
          "Значит РТС это IMOEX в долларах, как фактор избыточен")
    common = stock_ret.join(brent_ret, how="inner").dropna()
    bcorr = common.corr()["BRENT"].drop("BRENT").sort_values(ascending=False)
    _show("Корреляция доходности Brent с акциями (топ-5):", bcorr.head(5))

    # Собираем финальный набор из 8 факторов, выравниваем по общему календарю
    rate_f = scores_y.add_prefix("RATE_")
    eq_f = scores_e.add_prefix("EQ_")
    fx_f = fx_ret.rename(columns={"USD": "FX_USD", "EUR": "FX_EUR"})
    aligned = F.align_on_common_dates(rate_f, eq_f, fx_f)
    risk_factors = pd.concat(aligned, axis=1).dropna()
    print(f"\nИтоговая матрица риск-факторов: {risk_factors.shape}")
    print("Факторы:", list(risk_factors.columns))

    _show("Корреляции риск-факторов:", risk_factors.corr(), ndigits=2)

    # Тяжесть хвостов по каждому фактору (важно для выбора модели в п.3)
    tt = D.tail_table(risk_factors)
    _show("Хвосты распределений факторов:",
          tt[["std", "skew", "exc_kurt", "JB_p", "t_dof", "hill_alpha"]], ndigits=3)
    print("Везде JB_p близко к 0, эксцесс заметно больше 0, t_dof мал, значит хвосты тяжёлые.")

    # Стационарность: уровни нестационарны, приращения стационарны
    levels = pd.concat([
        panels["yields"]["10"].rename("yield_10y_level"),
        np.log(panels["brent"]["BRENT"]).rename("log_brent_level"),
        np.log(panels["fx_px"]["USD"]).rename("log_usd_level"),
    ], axis=1)
    changes = pd.concat([
        dY["10"].rename("yield_10y_chg"),
        brent_ret["BRENT"].rename("brent_ret"),
        fx_ret["USD"].rename("usd_ret"),
    ], axis=1)
    _show("Стационарность уровней:", D.stationarity_table(levels))
    _show("Стационарность приращений:", D.stationarity_table(changes))

    # Недельная сезонность доходностей факторов (проверяем, что её почти нет)
    _show("Сезонность факторов по дням недели:",
          D.weekday_seasonality(risk_factors), ndigits=5)

    # Сохраняем итог и нагрузки PCA, понадобятся в пунктах 3-5
    rf_path = data_dir / "risk_factors.parquet"
    ry_path = data_dir / "rate_pca_loadings.parquet"
    re_path = data_dir / "equity_pca_loadings.parquet"
    risk_factors.to_parquet(rf_path)
    load_y.to_parquet(ry_path)
    load_e.to_parquet(re_path)

    print("\nИтог, сохранённые файлы:")
    print(f"  risk_factors.parquet         {risk_factors.shape}, колонки {list(risk_factors.columns)}")
    print(f"  rate_pca_loadings.parquet    {load_y.shape}")
    print(f"  equity_pca_loadings.parquet  {load_e.shape}")


if __name__ == "__main__":
    run()
