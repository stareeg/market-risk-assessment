"""
Переоценка валютных позиций (пункт 4).

Тут всё просто. Курс это и есть цена валюты в рублях. Факторы FX_USD и FX_EUR это
лог-доходности курсов ЦБ, поэтому новый курс это
    курс_новый = курс_базовый * exp(сдвиг фактора).
Стоимость позиции в рублях двигается ровно во столько же раз.

Отдельной модели цены тут нет: сам курс и есть риск-фактор, поэтому проверка
точности как у облигаций и акций не нужна, оценка точная по построению.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

FX_FACTOR = {"USD": "FX_USD", "EUR": "FX_EUR"}


def base_fx(data_dir, eval_date: pd.Timestamp) -> pd.Series:
    """Курсы USD и EUR на дату оценки (последние известные, если в этот день нет)."""
    fx = pd.read_parquet(data_dir / "fx_cbr.parquet")
    wide = fx.pivot(index="DATE", columns="CCY", values="RATE").sort_index()
    if eval_date in wide.index:
        row = wide.loc[eval_date]
    else:
        row = wide.loc[:eval_date].iloc[-1]
    return row[["USD", "EUR"]].astype(float)


def revalue(base_rate, logret) -> np.ndarray:
    """Новый курс по сдвигу фактора (лог-доходности)."""
    return np.asarray(base_rate, dtype=float) * np.exp(np.asarray(logret, dtype=float))


def positions(base_rate: pd.Series) -> pd.Series:
    """Количество валюты в позиции: рублёвый объём поделить на курс."""
    return pd.Series({ccy: C.FX_NOTIONAL_RUB[ccy] / base_rate[ccy]
                      for ccy in ["USD", "EUR"]})
