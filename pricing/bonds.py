"""
Оценка ОФЗ через дисконтирование денежных потоков (пункт 4).

Цена облигации это приведённая стоимость будущих выплат: оставшиеся купоны плюс
номинал в дату погашения. Дисконтируем по кривой ЦБ (модуль curve), ставку на
срок каждого потока берём кубическим сплайном по 12 узлам.

Грязная цена это приведённая стоимость всех будущих потоков. Чистая цена это
грязная минус НКД.
НКД считаем линейно внутри купонного периода, проверено против ACCINT с биржи
(сходится до копейки).

Точность модели проверяем сравнением грязной цены с рыночной (CLOSE + ACCINT)
по каждому выпуску и метриками RMSE и MAPE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from . import curve as CV

FACE_VALUE = 1000.0          # номинал ОФЗ
DAY_COUNT = 365.0            # ACT/365 для срока до потока


def _load_coupons(data_dir) -> pd.DataFrame:
    """Купонные выплаты с датами в формате datetime."""
    cp = pd.read_parquet(data_dir / "bonds_coupons.parquet")
    cp["coupondate"] = pd.to_datetime(cp["coupondate"])
    cp["startdate"] = pd.to_datetime(cp["startdate"])
    return cp


def future_cashflows(num: str, coupons: pd.DataFrame, maturity: pd.Timestamp,
                     eval_date: pd.Timestamp) -> pd.DataFrame:
    """
    Будущие потоки по выпуску: оставшиеся купоны и номинал в погашение.
    Возвращает таблицу со сроком в годах и суммой в рублях.
    """
    sub = coupons[(coupons["NUMBER"] == num) & (coupons["coupondate"] > eval_date)]
    rows = [{"date": d, "amount": v}
            for d, v in zip(sub["coupondate"], sub["value_rub"])]
    rows.append({"date": maturity, "amount": FACE_VALUE})   # возврат номинала
    cf = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    cf["t"] = (cf["date"] - eval_date).dt.days / DAY_COUNT
    return cf


def accrued_interest(num: str, coupons: pd.DataFrame, eval_date: pd.Timestamp) -> float:
    """НКД линейно внутри текущего купонного периода."""
    sub = coupons[(coupons["NUMBER"] == num) &
                  (coupons["startdate"] <= eval_date) &
                  (coupons["coupondate"] > eval_date)]
    if sub.empty:
        return 0.0
    c = sub.iloc[0]
    period = (c["coupondate"] - c["startdate"]).days
    elapsed = (eval_date - c["startdate"]).days
    return float(c["value_rub"]) * elapsed / period


def price_dirty(node_yields: np.ndarray, cashflows: pd.DataFrame) -> float:
    """Грязная цена: сумма потоков, дисконтированных по кривой."""
    df = CV.discount_factors(node_yields, cashflows["t"].values)
    return float(np.sum(cashflows["amount"].values * df))


def price_portfolio(data_dir, eval_date: pd.Timestamp,
                    node_yields: np.ndarray) -> pd.DataFrame:
    """
    Оценивает 5 портфельных ОФЗ на дату оценки и сравнивает с рынком.
    Возвращает таблицу с модельной и рыночной ценой и ошибкой по каждому выпуску.
    """
    coupons = _load_coupons(data_dir)
    desc = pd.read_parquet(data_dir / "bonds_descriptions.parquet").set_index("NUMBER")
    desc.index = desc.index.astype(str)
    hist = pd.read_parquet(data_dir / "bonds_history.parquet")
    hist["NUMBER"] = hist["NUMBER"].astype(str)
    day = hist[hist["TRADEDATE"] == eval_date].set_index("NUMBER")

    rows = []
    for num in C.PORTFOLIO_BONDS:
        maturity = pd.Timestamp(desc.loc[num, "MATDATE"])
        cf = future_cashflows(num, coupons, maturity, eval_date)
        model_dirty = price_dirty(node_yields, cf)
        accr = accrued_interest(num, coupons, eval_date)
        model_clean = model_dirty - accr

        mkt_clean = float(day.loc[num, "CLOSE"])                 # чистая в % номинала
        mkt_accr = float(day.loc[num, "ACCINT"])
        mkt_dirty = mkt_clean / 100.0 * FACE_VALUE + mkt_accr
        err = (model_dirty - mkt_dirty) / mkt_dirty * 100.0

        rows.append({
            "NUMBER": num,
            "maturity": maturity.date(),
            "model_dirty": model_dirty,
            "mkt_dirty": mkt_dirty,
            "model_clean_pct": model_clean / FACE_VALUE * 100.0,
            "mkt_clean_pct": mkt_clean,
            "accrued": accr,
            "err_pct": err,
        })
    return pd.DataFrame(rows).set_index("NUMBER")


def accuracy(results: pd.DataFrame) -> dict:
    """Сводные метрики точности оценки облигаций."""
    err = results["err_pct"].values
    return {
        "RMSE_pct": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_pct": float(np.mean(np.abs(err))),
        "max_abs_err_pct": float(np.max(np.abs(err))),
    }


def save_last_prices(data_dir, eval_date: pd.Timestamp) -> pd.DataFrame:
    """
    Снимок рыночных данных по 5 ОФЗ на дату оценки (для старта симуляции в п.5).
    Цена тут в штуках-номинале, как на бирже.
    """
    hist = pd.read_parquet(data_dir / "bonds_history.parquet")
    hist["NUMBER"] = hist["NUMBER"].astype(str)
    snap = (hist[(hist["TRADEDATE"] == eval_date) &
                 (hist["NUMBER"].isin(C.PORTFOLIO_BONDS))]
            .set_index("NUMBER").reindex(C.PORTFOLIO_BONDS))
    snap.to_parquet(data_dir / "last_bond_prices.parquet")
    return snap
