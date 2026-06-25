"""
Этап 3. Оценка стохастических моделей для риск-факторов.

Читает risk_factors.parquet из пункта 2, подбирает по каждому фактору GARCH(1,1)-t,
при необходимости уточняет среднее через AR(1), оценивает DCC для совместной
динамики корреляций и сохраняет параметры в data/. Тяжёлая логика в garch_dcc.py,
тут только оркестрация и проверка значений.

Запуск: python main.py --stage models
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from . import garch_dcc as G

pd.options.io.parquet.engine = "fastparquet"


def _show(title: str, df, ndigits: int = 4) -> None:
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
    Оценивает модели по риск-факторам из data_dir и сохраняет параметры туда же.
    По умолчанию работаем с config.DATA_DIR. Можно передать свою папку, чтобы
    проверить расчёт, не трогая сохранённые файлы.
    """
    data_dir = Path(data_dir) if data_dir is not None else C.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(C.RANDOM_SEED)   # фиксируем seed (воспроизводимость)
    rf = pd.read_parquet(data_dir / "risk_factors.parquet")
    print(f"Каталог данных: {data_dir}")
    print(f"Факторов: {rf.shape[1]}, наблюдений: {rf.shape[0]}")
    print("Факторы:", list(rf.columns))

    # Подбираем GARCH по каждому фактору и сравниваем t с нормальным.
    params_df, std_resid, cond_vol, compare, diag = G.fit_factors(rf)
    _show("Сравнение GARCH(1,1): t против нормального (AIC/BIC):", compare, 1)
    aic_wins = int((compare["AIC_t"] < compare["AIC_norm"]).sum())
    bic_wins = int((compare["BIC_t"] < compare["BIC_norm"]).sum())
    print(f"t лучше нормального: по AIC у {aic_wins} из {len(compare)}, "
          f"по BIC у {bic_wins} из {len(compare)} факторов")

    # Перебор порядков, чтобы обосновать простой (1,1).
    orders = G.order_search(rf)
    _show("Средние AIC/BIC по факторам для разных порядков GARCH-t:", orders, 1)

    # Параметры выбранных моделей и проверка устойчивости.
    _show("Параметры выбранных моделей GARCH(1,1)-t:", params_df, 5)
    persist = params_df["alpha"] + params_df["beta"]
    # Взрывной волатильности (a+b>1) быть не должно.
    print("alpha+beta <= 1 у всех факторов (нет взрывной волатильности):",
          bool((persist <= 1 + 1e-6).all()))
    # Часть факторов выходит ровно на a+b=1. Это интегрированный GARCH (IGARCH),
    # волатильность как экспоненциальное сглаживание, стандартная модель RiskMetrics.
    igarch = list(persist.index[persist > 1 - 1e-4])
    print(f"На границе a+b=1 (IGARCH, память волатильности бесконечна): {igarch}")
    nu_min_factor = params_df["nu"].idxmin()
    print(f"Среднее nu = {params_df['nu'].mean():.2f}, минимум {params_df['nu'].min():.2f} "
          f"у {nu_min_factor} (самые тяжёлые хвосты, скачки рубля 2022-2023)")

    # Диагностика остатков. Хотим, чтобы ARCH-эффекты ушли (lb_p_sq > 0.05),
    # а автокорреляция (lb_p_final) была не значима.
    _show("Диагностика остатков (p-value Льюнга-Бокса):", diag, 4)
    ar_used = list(params_df.index[params_df["mean"] == "AR1"])
    print("AR(1) в среднем добавлен для:", ar_used if ar_used else "никого")
    bad_ac = list(diag.index[diag["lb_p_final"] < G.ALPHA_LB])
    print("Осталась значимая автокорреляция в остатках у:",
          bad_ac if bad_ac else "никого")
    arch_ok = bool((diag["lb_p_sq"] > G.ALPHA_LB).all())
    print("ARCH-эффекты убраны у всех факторов (lb_p_sq > 0.05):", arch_ok)

    # DCC на стандартизованных остатках выбранных моделей.
    theta1, theta2, q_bar = G.fit_dcc(std_resid)
    print(f"\nDCC: theta1={theta1:.4f}, theta2={theta2:.4f}, "
          f"сумма={theta1 + theta2:.4f}. Меньше 1, значит корреляции устойчивы")

    # Сохраняем параметры для пунктов 4 и 5.
    gp_path = data_dir / "garch_params.parquet"
    dcc_path = data_dir / "dcc_params.parquet"
    qbar_path = data_dir / "q_bar.parquet"
    diag_path = data_dir / "garch_diagnostics.parquet"

    params_df.to_parquet(gp_path)
    dcc = pd.DataFrame({"value": [theta1, theta2]}, index=["theta1", "theta2"])
    dcc.index.name = "param"
    dcc.to_parquet(dcc_path)
    q_bar.to_parquet(qbar_path)
    diag.to_parquet(diag_path)

    print("\nИтог, сохранённые файлы:")
    print(f"  garch_params.parquet         {params_df.shape}, колонки {list(params_df.columns)}")
    print("  dcc_params.parquet           theta1, theta2")
    print(f"  q_bar.parquet                {q_bar.shape}")
    print(f"  garch_diagnostics.parquet    {diag.shape}")


if __name__ == "__main__":
    run()
