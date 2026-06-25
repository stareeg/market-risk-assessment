"""
Кривая бескупонной доходности для оценки облигаций (часть пункта 4).

Тут две задачи.
1) По 12 узлам кривой (ставки ЦБ на сроки от 0.25 до 30 лет) восстановить ставку
   на любой срок. Берём кубический сплайн по всем узлам, а не интерполяцию по трём
   точкам как в черновике, потому что на длинных ОФЗ три точки давали ошибку до
   1.5%.
2) Сдвигать кривую в симуляции через PCA-нагрузки: изменение ставок в б.п. это
   нагрузки, умноженные на приращения первых трёх компонент (уровень, наклон,
   кривизна). Это те же факторы RATE_PC, что и в пункте 2.

Дисконтируем по эффективной годовой ставке: DF(t) = (1 + y)^(-t), y в долях.
Проверка на 5 ОФЗ показала, что эта формула (а не непрерывная exp(-y t)) даёт
ошибку около процента, то есть совпадает с соглашением кривой ЦБ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# Сроки кривой: подписи как в данных (десятичная запятая) и те же в годах.
TENOR_LABELS = ["0,25", "0,5", "0,75", "1", "2", "3", "5", "7", "10", "15", "20", "30"]
TENOR_YEARS = np.array([0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])


def load_base_curve(data_dir, eval_date: pd.Timestamp) -> pd.Series:
    """
    Кривая ЦБ на дату оценки, 12 ставок в процентах годовых.
    Если в этот день публикации не было, берём последнюю известную (как и в
    остальном пайплайне, опубликованная ставка действует до следующей).
    """
    z = pd.read_parquet(data_dir / "zcyc_cbr.parquet").set_index("DATE").sort_index()
    y = z[TENOR_LABELS]
    if eval_date in y.index:
        row = y.loc[eval_date]
    else:
        row = y.loc[:eval_date].iloc[-1]
    return row.astype(float)


def curve_spline(node_yields: np.ndarray) -> CubicSpline:
    """Кубический сплайн ставки по сроку. На вход 12 ставок в порядке TENOR_YEARS."""
    return CubicSpline(TENOR_YEARS, np.asarray(node_yields, dtype=float))


def yield_at(node_yields: np.ndarray, t) -> np.ndarray:
    """
    Ставка (в процентах) на срок t лет. Сроки за пределами узлов прижимаем к
    границам, чтобы сплайн не уходил в дикую экстраполяцию на хвостах.
    """
    spline = curve_spline(node_yields)
    tt = np.clip(np.asarray(t, dtype=float), TENOR_YEARS[0], TENOR_YEARS[-1])
    return spline(tt)


def discount_factors(node_yields: np.ndarray, times) -> np.ndarray:
    """
    Дисконт-факторы для сроков times (в годах) по эффективной годовой ставке:
    DF = (1 + y/100)^(-t).
    """
    times = np.asarray(times, dtype=float)
    y = yield_at(node_yields, times) / 100.0
    return (1.0 + y) ** (-times)


def reconstruct_curve(base_node_yields: np.ndarray, d_pc: np.ndarray,
                      loadings: pd.DataFrame) -> np.ndarray:
    """
    Двигаем кривую через PCA-нагрузки. d_pc это приращения RATE_PC1, PC2, PC3.
    Нагрузки построены по изменениям ставок в б.п., поэтому новое изменение
    ставок (в б.п.) это нагрузки на d_pc, а в проценты переводим делением на 100.
    Возвращает новые 12 ставок в процентах.
    """
    load = loadings.reindex(TENOR_LABELS).values  # 12 x 3, порядок узлов как у base
    d_bp = load @ np.asarray(d_pc, dtype=float)
    return np.asarray(base_node_yields, dtype=float) + d_bp / 100.0
