"""
Описательная статистика и тесты для пункта 2:
  - стационарность (ADF + KPSS);
  - «тяжесть хвостов» (эксцесс, Jarque-Bera, число степеней свободы t-Стьюдента,
    оценка хвостового индекса Хилла);
  - помощник для сезонности (день недели).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss


# Стационарность
def stationarity_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Для каждого столбца: ADF (нулевая гипотеза H0 это единичный корень, то есть
    нестационарность) и KPSS (H0 это стационарность). Два теста сразу надёжнее
    одного: вывод делаем, когда они согласованы.
    """
    rows = []
    for col in df.columns:
        x = df[col].dropna().values
        if len(x) < 30:
            continue
        adf_p = adfuller(x, autolag="AIC")[1]
        try:
            # KPSS предупреждает, когда p-value за пределами таблицы. Это не
            # ошибка (значимость просто ещё выше), глушим, чтобы не сорить в консоль.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_p = kpss(x, regression="c", nlags="auto")[1]
        except Exception:
            kpss_p = np.nan
        # Считаем ряд стационарным, если ADF отвергает H0 (p<0.05), а KPSS не отвергает (p>0.05)
        verdict = "стационарен" if (adf_p < 0.05 and (np.isnan(kpss_p) or kpss_p > 0.05)) \
            else ("нестационарен" if adf_p > 0.05 else "спорно")
        rows.append({"series": col, "ADF_p": round(adf_p, 4),
                     "KPSS_p": round(kpss_p, 4) if not np.isnan(kpss_p) else np.nan,
                     "verdict": verdict})
    return pd.DataFrame(rows).set_index("series")


# Тяжесть хвостов
def hill_estimator(x: np.ndarray, tail_frac: float = 0.05) -> float:
    """
    Оценка хвостового индекса Хилла по объединённым модулям доходностей.
    Малое alpha это тяжёлые хвосты. У нормального распределения alpha уходит в
    бесконечность, у финансовых рядов обычно 2-5.
    """
    a = np.sort(np.abs(x[~np.isnan(x)]))[::-1]
    k = max(10, int(len(a) * tail_frac))
    k = min(k, len(a) - 1)
    xk = a[k]
    return 1.0 / np.mean(np.log(a[:k] / xk))


def tail_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Описательная статистика хвостов по каждому фактору-доходности:
    среднее, ст.отклонение, асимметрия, эксцесс (избыточный), p-value Jarque-Bera
    (H0: нормальность), число степеней свободы t-Стьюдента (df: чем меньше, тем
    тяжелее хвост) и хвостовой индекс Хилла.
    """
    rows = []
    for col in df.columns:
        x = df[col].dropna().values
        if len(x) < 30:
            continue
        jb_p = stats.jarque_bera(x)[1]
        # подгоняем t-Стьюдента, берём число степеней свободы
        try:
            dof = stats.t.fit(x)[0]
        except Exception:
            dof = np.nan
        rows.append({
            "series": col,
            "mean": x.mean(),
            "std": x.std(ddof=1),
            "skew": stats.skew(x),
            "exc_kurt": stats.kurtosis(x, fisher=True),   # 0 у нормального
            "JB_p": jb_p,
            "t_dof": dof,
            "hill_alpha": hill_estimator(x),
            "min": x.min(),
            "max": x.max(),
        })
    out = pd.DataFrame(rows).set_index("series")
    return out


# Сезонность (день недели)
def weekday_seasonality(returns: pd.DataFrame) -> pd.DataFrame:
    """Средняя доходность и волатильность по дням недели, простой тест сезонности."""
    r = returns.copy()
    r["weekday"] = r.index.dayofweek
    g = r.groupby("weekday")
    res = pd.concat({"mean": g.mean().mean(axis=1),
                     "std": g.std().mean(axis=1)}, axis=1)
    res.index = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"][:len(res)]
    return res
