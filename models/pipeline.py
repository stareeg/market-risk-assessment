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


# Представители трёх типов факторов для графика волатильности.
_VOL_FACTORS = ["RATE_PC1", "EQ_PC1", "FX_USD"]


def _plot_garch_vol(cond_vol: pd.DataFrame, out_dir: Path) -> Path:
    """
    Подогнанная условная волатильность GARCH во времени.
    У факторов разные единицы (ставки в б.п., валюта в долях), поэтому каждую
    волатильность делим на её среднее. Так видно общее: волатильность скачет
    кластерами, ярче всего весной 2022. Это и ловит GARCH.
    """
    import matplotlib.pyplot as plt
    from viz.style import set_slide_style, COLORS, save_slide

    set_slide_style()
    colors = [COLORS["main"], COLORS["second"], COLORS["accent"]]
    names = {"RATE_PC1": "ставки (RATE_PC1)", "EQ_PC1": "акции (EQ_PC1)",
             "FX_USD": "валюта (FX_USD)"}

    fig, ax = plt.subplots()
    for col, color in zip(_VOL_FACTORS, colors):
        v = cond_vol[col]
        ax.plot(v.index, v / v.mean(), color=color, linewidth=1.4,
                label=names.get(col, col))
    ax.set_title("Условная волатильность GARCH относительно своей средней")
    ax.set_xlabel("Год")
    ax.set_ylabel("Волатильность к средней, разы")
    ax.legend(loc="upper right")
    return save_slide(fig, "models_garch_vol", out_dir)


def _plot_t_vs_normal(std_resid: pd.DataFrame, params_df: pd.DataFrame,
                      out_dir: Path) -> Path:
    """
    Сравнение распределения t с нормальным на стандартизованных остатках.
    Берём самый тяжелохвостый фактор. Слева QQ против нормального, точки на хвостах
    уходят от прямой. Справа QQ против t с подогнанным числом степеней свободы,
    точки ложатся на прямую. Значит t описывает хвосты, а нормальное нет.
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    from viz.style import set_slide_style, COLORS, save_slide

    set_slide_style()
    name = params_df["nu"].idxmin()           # минимальное nu это самые тяжёлые хвосты
    nu = float(params_df.loc[name, "nu"])
    r = std_resid[name].dropna()
    z = (r - r.mean()) / r.std()
    samp = np.sort(z.values)

    n = len(samp)
    p = (np.arange(1, n + 1) - 0.5) / n
    q_norm = stats.norm.ppf(p)
    # Квантили t приводим к единичной дисперсии, чтобы прямая была y=x.
    q_t = stats.t.ppf(p, nu) * np.sqrt((nu - 2) / nu)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, theo, title in (
        (axes[0], q_norm, "против нормального"),
        (axes[1], q_t, f"против t (nu = {nu:.1f})"),
    ):
        lim = max(abs(theo[0]), abs(theo[-1]), abs(samp[0]), abs(samp[-1]))
        ax.plot([-lim, lim], [-lim, lim], color=COLORS["grey"], linestyle="--",
                linewidth=1.5)
        ax.scatter(theo, samp, s=12, color=COLORS["main"])
        ax.set_title(title)
        ax.set_xlabel("Теоретические квантили")
    axes[0].set_ylabel("Квантили остатков")
    fig.suptitle(f"Хвосты остатков фактора {name}: t против нормального", y=1.02)
    return save_slide(fig, "models_t_vs_normal", out_dir)


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

    # Графики для слайдов: волатильность GARCH и сравнение t с нормальным.
    fig_dir = C.PROJECT_DIR / "docs" / "figures"
    p1 = _plot_garch_vol(cond_vol, fig_dir)
    p2 = _plot_t_vs_normal(std_resid, params_df, fig_dir)
    print(f"\nГрафик волатильности GARCH: {p1}")
    print(f"График сравнения t с нормальным: {p2}")

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
