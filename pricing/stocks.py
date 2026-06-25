"""
Оценка акций через факторную модель на EQ-факторах (пункт 4).

Доходность акции раскладываем на общий рыночный кусок (три EQ-фактора из PCA) и
собственный остаток:
    r = alpha + b1*EQ_PC1 + b2*EQ_PC2 + b3*EQ_PC3 + e
Коэффициенты оцениваем МНК по истории. Цена в симуляции это
    цена = последняя_цена * exp(r).

Тут же закрываем проблему заниженного риска акций. Три фактора ловят около двух
третей дисперсии, оставшаяся треть это идиосинкразия отдельных бумаг. Поэтому по
каждой бумаге считаем волатильность остатка и тяжесть его хвоста (число степеней
свободы t). В симуляции (п.5) к факторной доходности добавим этот собственный
шок, иначе риск подпортфеля акций будет занижен.

Раньше коэффициентов и волатильности остатка в коде не было, файлы появлялись
вручную. Тут всё считается из данных, воспроизводимо.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import config as C
from risk_factors import factors as F

EQ_COLS = ["EQ_PC1", "EQ_PC2", "EQ_PC3"]


def portfolio_returns(data_dir, rf_index: pd.Index) -> pd.DataFrame:
    """
    Лог-доходности 10 портфельных акций на тех же датах, что и риск-факторы.
    Цены чистим от сплитов и считаем доходности так же, как в пункте 2, чтобы
    регрессия была согласована с факторами.
    """
    panels = F.load_panels(data_dir)
    px_adj, _ = F.adjust_splits(panels["stock_px"])
    ret = F.log_returns(px_adj)[C.PORTFOLIO_STOCKS]
    return ret.reindex(rf_index).dropna()


def fit_factor_model(returns: pd.DataFrame, eq_factors: pd.DataFrame):
    """
    МНК доходности каждой акции на три EQ-фактора.
    Возвращает таблицу коэффициентов (alpha, beta_EQ1, EQ2, EQ3, R2) и таблицу
    параметров остатка (волатильность и число степеней свободы t).
    """
    F_ = eq_factors.loc[returns.index, EQ_COLS].values
    X = np.column_stack([np.ones(len(F_)), F_])     # константа плюс три фактора
    n, k = X.shape

    coeff_rows, idio_rows, resid_cols = {}, {}, {}
    for tic in returns.columns:
        y = returns[tic].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        ss_res = float(resid @ resid)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot
        idio_vol = np.sqrt(ss_res / (n - k))        # стандартная ошибка регрессии
        # хвост остатка: подгоняем t с фиксированным центром в нуле
        dof = float(stats.t.fit(resid, floc=0.0)[0])

        coeff_rows[tic] = {"alpha": beta[0], "beta_EQ1": beta[1],
                           "beta_EQ2": beta[2], "beta_EQ3": beta[3], "R2": r2}
        idio_rows[tic] = {"idio_vol": idio_vol, "idio_dof": dof}
        resid_cols[tic] = resid

    coeff = pd.DataFrame(coeff_rows).T.loc[returns.columns]
    idio = pd.DataFrame(idio_rows).T.loc[returns.columns]
    resid_df = pd.DataFrame(resid_cols, index=returns.index)
    return coeff, idio, resid_df


def last_prices(data_dir, eval_date: pd.Timestamp) -> pd.DataFrame:
    """
    Цены 10 акций на дату оценки, как торговались (без поправки на сплиты, это
    реальная цена для расчёта количества бумаг в портфеле). Одна строка, колонки
    это тикеры.
    """
    stocks = pd.read_parquet(data_dir / "stocks_history.parquet")
    px = stocks.pivot(index="TRADEDATE", columns="SECID", values="CLOSE").sort_index()
    if eval_date in px.index:
        row = px.loc[[eval_date]]
    else:
        row = px.loc[:eval_date].iloc[[-1]]
    return row[C.PORTFOLIO_STOCKS]


def price_stock(last_price, coeff_row, factor_shock, idio_shock=0.0):
    """
    Цена акции при сдвиге факторов factor_shock (приращения EQ_PC1, EQ_PC2, EQ_PC3).
    idio_shock это собственный шок бумаги (в симуляции, тут по умолчанию 0).
    """
    betas = np.array([coeff_row["beta_EQ1"], coeff_row["beta_EQ2"], coeff_row["beta_EQ3"]])
    r = coeff_row["alpha"] + betas @ np.asarray(factor_shock) + idio_shock
    return last_price * np.exp(r)
