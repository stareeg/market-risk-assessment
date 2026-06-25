"""
Меры риска VaR и ES с ежедневной ребалансировкой (пункт 5).

Ежедневная ребалансировка значит, что каждый день портфель возвращают к целевым
пропорциям. Тогда дневная доходность портфеля это взвешенная по пропорциям сумма
дневных доходностей инструментов, а стоимость портфеля за горизонт растёт
произведением дневных множителей. Поэтому P&L за горизонт это
    V0 * (произведение по дням (1 + дневная доходность портфеля) - 1).
Однодневный P&L это просто V0 на дневную доходность первого дня.

Меры риска как в задании, на разных уровнях:
  VaR 99%   убыток на уровне первого процентиля распределения P&L,
  ES 97.5%  средний убыток по худшим 2.5% сценариев (хвост за 2.5-м процентилем).
Это разные уровни, ES не равен VaR.
"""
from __future__ import annotations

import numpy as np


def rebalanced_pnl(ret: np.ndarray, weights: np.ndarray, value0: float,
                   horizon: int) -> np.ndarray:
    """
    P&L подпортфеля за горизонт при ежедневной ребалансировке.

    ret это дневные доходности инструментов (n_sim, дни, инструменты), weights это
    целевые пропорции (в сумме 1) по этим инструментам, value0 это стоимость на
    дату оценки. Берём первые horizon дней.
    """
    daily = ret[:, :horizon, :] @ weights          # дневная доходность портфеля
    growth = np.prod(1.0 + daily, axis=1)           # произведение дневных множителей
    return value0 * (growth - 1.0)


def var_es(pnl: np.ndarray, var_level: float = 0.99,
           es_level: float = 0.975) -> dict:
    """
    VaR на уровне var_level и ES на уровне es_level по выборке P&L.
    Оба возвращаем как положительный убыток.
    """
    var = -np.percentile(pnl, (1.0 - var_level) * 100.0)
    thr = np.percentile(pnl, (1.0 - es_level) * 100.0)
    tail = pnl[pnl <= thr]
    es = -tail.mean() if tail.size else float("nan")
    return {"VaR": float(var), "ES": float(es)}


def measure_all(ret: np.ndarray, notional: np.ndarray, groups: dict,
                horizons=(1, 10)) -> "pd.DataFrame":
    """
    Считает VaR 99% и ES 97.5% для всего портфеля и подпортфелей на заданных
    горизонтах. Для каждого подпортфеля пропорции нормируем внутри него, а
    стоимость берём суммой его рублёвых объёмов.

    Возвращает таблицу (строки это пара портфель и горизонт) и словарь с выборками
    P&L по всему портфелю для гистограмм.
    """
    import pandas as pd

    rows = []
    pnl_full = {}
    for name, idx in groups.items():
        idx = np.array(idx)
        sub_notional = notional[idx]
        value0 = float(sub_notional.sum())
        weights = sub_notional / value0
        sub_ret = ret[:, :, idx]
        for h in horizons:
            pnl = rebalanced_pnl(sub_ret, weights, value0, h)
            m = var_es(pnl)
            rows.append({
                "Портфель": name,
                "Горизонт": h,
                "Стоимость": value0,
                "VaR99": m["VaR"],
                "ES975": m["ES"],
                "VaR99_pct": m["VaR"] / value0 * 100.0,
                "ES975_pct": m["ES"] / value0 * 100.0,
            })
            if name == "Портфель":
                pnl_full[h] = pnl

    table = pd.DataFrame(rows)
    return table, pnl_full
