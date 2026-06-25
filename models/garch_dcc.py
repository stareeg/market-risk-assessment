"""
Стохастические модели динамики риск-факторов (пункт 3).

Единый подход для всех 8 PCA-факторов: одномерная GARCH(1,1) с распределением
Стьюдента для условной волатильности плюс DCC для совместной динамики корреляций.
Так держится связь с пунктом 2: моделируем те же самые PCA-факторы.

Почему так:
- приращения факторов стационарны, поэтому моделируем именно приращения;
- хвосты тяжёлые, поэтому берём t, а не нормальное распределение;
- волатильность кластеризуется, это и ловит GARCH;
- корреляции непостоянны, для них ставим DCC.

GARCH оцениваем через библиотеку arch (это MLE). DCC оцениваем своей функцией
правдоподобия, тоже MLE, двухшаговой схемой: сначала одномерные GARCH, потом по
их стандартизованным остаткам подбираем theta1 и theta2.

Старый подход с CIR и GBM в обход PCA сюда сознательно не переносим, он
противоречит пункту 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize
from statsmodels.stats.diagnostic import acorr_ljungbox

# Сколько лагов берём в тесте Льюнга-Бокса и порог значимости.
LJUNG_LAGS = 10
ALPHA_LB = 0.05

# Числовые колонки таблицы параметров (нужно, чтобы привести типы к float).
_PARAM_COLS = ["const", "ar1", "omega", "alpha", "beta", "nu"]


def _fit(series: pd.Series, dist: str, mean: str):
    """
    Подгоняет одну GARCH(1,1). mean это 'zero' или 'AR' (тогда лаг 1).

    rescale=True нужен из-за разного масштаба факторов: ставки в б.п. имеют размах
    порядка 50, а валютные лог-доходности порядка 0.01. На таком мелком масштабе
    оптимизатор расходится, поэтому arch временно домножает ряд на степень десятки.
    Масштаб запоминаем в res.scale и потом возвращаем параметры к исходным единицам.
    """
    if mean == "AR":
        model = arch_model(series, mean="AR", lags=1, vol="Garch", p=1, q=1,
                           dist=dist, rescale=True)
    else:
        model = arch_model(series, mean="zero", vol="Garch", p=1, q=1,
                           dist=dist, rescale=True)
    return model.fit(disp="off")


def ljung_box_p(resid, lags: int = LJUNG_LAGS) -> float:
    """p-value теста Льюнга-Бокса. Маленькое p значит автокорреляция осталась."""
    r = pd.Series(resid).dropna()
    lb = acorr_ljungbox(r, lags=lags, return_df=True)
    return float(lb["lb_pvalue"].iloc[-1])


def _params_row(res, name: str, mean: str) -> dict:
    """
    Достаёт параметры выбранной модели в единый плоский вид и возвращает их к
    исходному масштабу фактора.

    arch отдаёт omega и const на домноженном ряде (см. res.scale). Дисперсия растёт
    как масштаб в квадрате, поэтому omega делим на scale^2, а среднее линейно,
    поэтому const делим на scale. Коэффициенты alpha, beta, ar1 и число степеней
    свободы nu безразмерны, их не трогаем.
    """
    p = res.params
    s = res.scale
    # У AR-модели лаговый коэффициент назван по имени ряда, например RATE_PC1[1].
    ar1 = float(p[f"{name}[1]"]) if f"{name}[1]" in p.index else 0.0
    const = float(p["Const"]) / s if "Const" in p.index else 0.0
    return {
        "mean": "AR1" if mean == "AR" else "zero",
        "const": const,
        "ar1": ar1,
        "omega": float(p["omega"]) / s ** 2,
        "alpha": float(p["alpha[1]"]),
        "beta": float(p["beta[1]"]),
        "nu": float(p["nu"]) if "nu" in p.index else np.nan,
    }


def fit_factors(risk_factors: pd.DataFrame, lags: int = LJUNG_LAGS,
                alpha_lb: float = ALPHA_LB):
    """
    Подбирает GARCH(1,1) по каждому фактору.

    Сначала сравниваем t и нормальное распределение при нулевом среднем (по
    AIC/BIC). Дальше берём t. Если в остатках t-модели осталась автокорреляция
    (Льюнг-Бокс), уточняем уравнение среднего через AR(1) и проверяем заново.

    Возвращает: параметры, стандартизованные остатки, условные волатильности,
    таблицу сравнения t и нормального, таблицу диагностики остатков.
    """
    params = {}
    std_resid = {}
    cond_vol = {}
    compare = []
    diag = []

    for col in risk_factors.columns:
        s = risk_factors[col]
        res_t = _fit(s, "t", "zero")
        res_n = _fit(s, "normal", "zero")
        compare.append({
            "factor": col,
            "AIC_t": res_t.aic, "BIC_t": res_t.bic,
            "AIC_norm": res_n.aic, "BIC_norm": res_n.bic,
        })

        # Если у t-модели в остатках есть автокорреляция, добавляем AR(1).
        lb_zero = ljung_box_p(res_t.std_resid, lags)
        if lb_zero < alpha_lb:
            res = _fit(s, "t", "AR")
            mean = "AR"
        else:
            res = res_t
            mean = "zero"

        lb_final = ljung_box_p(res.std_resid, lags)
        lb_sq = ljung_box_p(pd.Series(res.std_resid).dropna() ** 2, lags)

        row = _params_row(res, col, mean)
        params[col] = row
        std_resid[col] = res.std_resid                       # безразмерны
        cond_vol[col] = res.conditional_volatility / res.scale  # к исходному масштабу
        diag.append({
            "factor": col,
            "mean": row["mean"],
            "lb_p_zero": lb_zero,
            "lb_p_final": lb_final,
            "lb_p_sq": lb_sq,
            "alpha+beta": row["alpha"] + row["beta"],
        })

    order = list(risk_factors.columns)
    params_df = (pd.DataFrame([{"factor": c, **params[c]} for c in order])
                 .set_index("factor"))
    params_df[_PARAM_COLS] = params_df[_PARAM_COLS].astype(float)

    std_resid_df = pd.DataFrame(std_resid)[order]
    cond_vol_df = pd.DataFrame(cond_vol)[order]
    compare_df = pd.DataFrame(compare).set_index("factor")
    diag_df = pd.DataFrame(diag).set_index("factor")
    return params_df, std_resid_df, cond_vol_df, compare_df, diag_df


def order_search(risk_factors: pd.DataFrame,
                 orders=((1, 1), (2, 1), (1, 2), (2, 2))) -> pd.DataFrame:
    """
    Средние AIC и BIC по факторам для разных порядков GARCH-t.
    Нужно, чтобы обосновать выбор простого (1,1).
    """
    rows = []
    for col in risk_factors.columns:
        for p, q in orders:
            res = arch_model(risk_factors[col], mean="zero", vol="Garch",
                             p=p, q=q, dist="t", rescale=True).fit(disp="off")
            rows.append({"p": p, "q": q, "AIC": res.aic, "BIC": res.bic})
    return pd.DataFrame(rows).groupby(["p", "q"])[["AIC", "BIC"]].mean()


def dcc_negloglik(theta, Z: np.ndarray, Q_bar: np.ndarray) -> float:
    """
    Отрицательное лог-правдоподобие DCC (гауссова квазиформа для шага корреляции).
    Минимизация по theta1, theta2 это и есть MLE второго шага.
    """
    theta1, theta2 = theta
    # Условие стационарности: оба неотрицательны и в сумме меньше 1.
    if theta1 < 0 or theta2 < 0 or theta1 + theta2 >= 1:
        return 1e10
    T = Z.shape[0]
    Q = Q_bar.copy()
    total = 0.0
    for t in range(T):
        zt = Z[t].reshape(-1, 1)
        Q = (1 - theta1 - theta2) * Q_bar + theta1 * (zt @ zt.T) + theta2 * Q
        d = np.sqrt(np.diag(Q))
        R = Q / np.outer(d, d)
        sign, logdet = np.linalg.slogdet(R)
        if sign <= 0:
            return 1e10
        total += logdet + (zt.T @ np.linalg.solve(R, zt))[0, 0]
    return total


def fit_dcc(std_resid_df: pd.DataFrame, init=(0.05, 0.9)):
    """
    Оценивает параметры DCC по стандартизованным остаткам одномерных GARCH.
    Возвращает theta1, theta2 и безусловную корреляцию Q_bar (как DataFrame).
    """
    Z = std_resid_df.dropna()
    Q_bar = Z.corr().values
    res = minimize(
        dcc_negloglik, np.array(init, dtype=float),
        args=(Z.values, Q_bar),
        bounds=[(0, 1), (0, 1)],
        constraints={"type": "ineq", "fun": lambda x: 1 - x[0] - x[1]},
        method="SLSQP",
    )
    theta1, theta2 = res.x
    q_bar_df = pd.DataFrame(Q_bar, index=Z.columns, columns=Z.columns)
    return float(theta1), float(theta2), q_bar_df
