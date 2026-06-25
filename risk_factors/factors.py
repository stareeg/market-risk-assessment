"""
Подготовка данных и построение риск-факторов для пункта 2 задания.

Логика:
  1) загрузка сохранённых parquet-панелей (load_panels);
  2) очистка данных:
       - корректировка цен акций на сплиты (adjust_splits), иначе день сплита
         даёт ложный «обвал» доходности в N раз;
       - пометка длинных календарных разрывов (например, остановка торгов на MOEX
         в феврале-марте 2022), доходность через такой разрыв не является
         однодневной и исключается из выборки (mask_calendar_gaps);
  3) переход от уровней к приращениям: лог-доходности цен (log_returns) и дневные
     изменения доходностей кривой в б.п. (yield_changes);
  4) выравнивание всех рядов по общему торговому календарю (align_on_common_dates).

Сами риск-факторы (через PCA) считаются в pca_tools.py и в pipeline.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
pd.options.io.parquet.engine = 'fastparquet'

# Известные сроки кривой (в годах) для упорядочивания и интерпретации PCA.
TENOR_LABELS = ["0,25", "0,5", "0,75", "1", "2", "3", "5", "7", "10", "15", "20", "30"]
TENOR_YEARS = [0.25, 0.5, 0.75, 1, 2, 3, 5, 7, 10, 15, 20, 30]


# Загрузка панелей из data/
def load_panels(data_dir=None) -> dict[str, pd.DataFrame]:
    """Читает parquet-файлы и приводит их к «широким» панелям (дата × инструмент)."""
    d = data_dir or C.DATA_DIR

    stocks = pd.read_parquet(d / "stocks_history.parquet")
    stock_px = (stocks.pivot(index="TRADEDATE", columns="SECID", values="CLOSE")
                .sort_index())

    bonds = pd.read_parquet(d / "bonds_history.parquet")
    bond_px = (bonds.pivot(index="TRADEDATE", columns="NUMBER", values="CLOSE")
               .sort_index())

    zcyc = pd.read_parquet(d / "zcyc_cbr.parquet").set_index("DATE").sort_index()
    yields = zcyc[TENOR_LABELS].copy()

    fx = pd.read_parquet(d / "fx_cbr.parquet")
    fx_px = fx.pivot(index="DATE", columns="CCY", values="RATE").sort_index()

    brent = (pd.read_parquet(d / "brent_history.parquet")
             .set_index("TRADEDATE")["BRENT_USD"].sort_index().rename("BRENT"))

    idx = pd.read_parquet(d / "indices_history.parquet")
    index_px = idx.pivot(index="TRADEDATE", columns="SECID", values="CLOSE").sort_index()

    return {
        "stock_px": stock_px,
        "bond_px": bond_px,
        "yields": yields,
        "fx_px": fx_px,
        "brent": brent.to_frame(),
        "index_px": index_px,
    }


# Очистка: сплиты и календарные разрывы
def detect_splits(prices: pd.DataFrame, ratio_threshold: float = 3.0) -> pd.DataFrame:
    """
    Находит дни, где цена инструмента изменилась более чем в ratio_threshold раз.
    Это почти наверняка дробление или консолидация акций (сплит), а не рыночное движение.
    Возвращает таблицу найденных событий с оценкой кратности.
    """
    rows = []
    for col in prices.columns:
        s = prices[col].dropna()
        rel = s / s.shift(1)
        flag = rel[(rel > ratio_threshold) | (rel < 1 / ratio_threshold)]
        for dt, r in flag.items():
            factor = 1 / r if r < 1 else r          # во сколько раз изменилась цена
            rows.append({"TICKER": col, "DATE": dt, "price_ratio": round(r, 4),
                         "approx_split": int(round(factor))})
    return pd.DataFrame(rows)


def adjust_splits(prices: pd.DataFrame, splits: pd.DataFrame | None = None,
                  ratio_threshold: float = 3.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Делает ценовой ряд непрерывным относительно сплитов: цены ДО даты сплита
    масштабируются так, чтобы стыковаться с ценами ПОСЛЕ. Это устраняет ложный
    однодневный «скачок» доходности и при этом сохраняет настоящие движения.
    """
    if splits is None:
        splits = detect_splits(prices, ratio_threshold)
    adj = prices.copy()
    for _, ev in splits.iterrows():
        col, dt, r = ev["TICKER"], ev["DATE"], ev["price_ratio"]
        # все котировки строго ДО дня сплита умножаем на коэффициент r (цена_после/цена_до)
        mask = adj.index < dt
        adj.loc[mask, col] = adj.loc[mask, col] * r
    return adj, splits


def mask_calendar_gaps(prices: pd.DataFrame, max_gap_days: int = 7) -> pd.DataFrame:
    """
    Возвращает булеву маску (по индексу дат): True там, где доходность считается
    через «нормальный» промежуток (<= max_gap_days). Длинные разрывы (остановка
    биржи в 2022 г., длинные праздники) помечаются False, соответствующая
    доходность не является однодневной и будет исключена из выборки.
    """
    dates = prices.dropna(how="all").index
    gap = pd.Series(dates, index=dates).diff().dt.days
    return gap.le(max_gap_days)


# Переход к приращениям
def log_returns(prices: pd.DataFrame, max_gap_days: int = 7) -> pd.DataFrame:
    """
    Лог-доходности цен. Доходности через длинные календарные разрывы зануляются
    в NaN (см. mask_calendar_gaps), чтобы не смешивать многонедельный скачок с
    однодневными движениями.
    """
    r = np.log(prices).diff()
    good = mask_calendar_gaps(prices, max_gap_days)
    r = r.loc[good.index]
    r[~good.values] = np.nan
    return r


def yield_changes(yields: pd.DataFrame, max_gap_days: int = 7) -> pd.DataFrame:
    """Дневные изменения доходностей кривой в базисных пунктах (1 б.п. = 0.01%)."""
    dy = yields.diff() * 100.0  # из % годовых в б.п.
    good = mask_calendar_gaps(yields, max_gap_days)
    dy = dy.loc[good.index]
    dy[~good.values] = np.nan
    return dy


# Выравнивание по общему календарю
def align_on_common_dates(*frames: pd.DataFrame) -> list[pd.DataFrame]:
    """Оставляет только даты, присутствующие во всех переданных панелях."""
    common = None
    for f in frames:
        idx = f.dropna(how="all").index
        common = idx if common is None else common.intersection(idx)
    return [f.loc[common].sort_index() for f in frames]


def to_trading_calendar(df: pd.DataFrame, calendar: pd.Index) -> pd.DataFrame:
    """
    Приводит панель уровней к торговому календарю MOEX с протяжкой последнего
    значения (forward-fill). Корректно для курсов/ставок ЦБ: опубликованный курс
    «действует» до следующей публикации, поэтому в дни, когда ЦБ не выставлял новый
    курс (праздники), берётся последний известный. Это устраняет потерю ~20%
    наблюдений из-за несовпадения календарей ЦБ и биржи.
    """
    return df.reindex(calendar).ffill()
