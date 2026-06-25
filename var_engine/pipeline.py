"""
Этап 5. Monte Carlo, VaR и ES.

Симулирует риск-факторы (модуль simulation), переоценивает портфель по сценариям
(модуль revaluation) и считает меры риска с ежедневной ребалансировкой
(модуль risk_measures). Оценка на 2 декабря 2025 г., горизонты 1 и 10 дней,
VaR 99% и ES 97.5%. Итог это четыре числа по всему портфелю плюс разбивка по
подпортфелям, гистограмма P&L и таблица в data/.

Запуск: python main.py --stage var
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from . import simulation as SIM
from . import revaluation as REV
from . import risk_measures as RM

pd.options.io.parquet.engine = "fastparquet"


def _format_rub(x: float) -> str:
    """Рубли с разделением разрядов, для читаемого вывода."""
    return f"{x:,.0f}".replace(",", " ")


def _days(h: int) -> str:
    """Правильное склонение слова день для подписей."""
    return f"{h} день" if h == 1 else f"{h} дней"


def _plot_pnl(pnl: np.ndarray, var: float, es: float, horizon: int,
              out_dir: Path) -> Path:
    """
    Гистограмма P&L по всему портфелю с линиями VaR и ES.
    Ось обрезаем по процентилям: из-за тяжёлых хвостов валюты единичные сценарии
    уходят далеко и без обрезки растягивают картинку до нечитаемой. Обрезка только
    для отображения, на сами меры риска не влияет.
    """
    import matplotlib.pyplot as plt
    from viz.style import set_slide_style, COLORS, save_slide

    set_slide_style()
    fig, ax = plt.subplots()
    pnl_mln = pnl / 1e6
    lo, hi = np.percentile(pnl_mln, [0.3, 99.7])
    ax.hist(pnl_mln, bins=80, range=(lo, hi), color=COLORS["main"], alpha=0.8)
    ax.axvline(-var / 1e6, color=COLORS["accent"], linestyle="--", linewidth=2,
               label=f"VaR 99% = {_format_rub(var)} руб")
    ax.axvline(-es / 1e6, color=COLORS["grey"], linestyle=":", linewidth=2,
               label=f"ES 97.5% = {_format_rub(es)} руб")
    ax.set_xlim(lo, hi)
    ax.set_title(f"Распределение P&L портфеля, горизонт {_days(horizon)}")
    ax.set_xlabel("Прибыль и убыток, млн руб")
    ax.set_ylabel("Число сценариев")
    ax.legend()
    return save_slide(fig, f"pnl_hist_{horizon}d", out_dir)


def run(data_dir: str | Path | None = None) -> None:
    """
    Считает VaR и ES на дату оценки и сохраняет результаты в data_dir.
    По умолчанию работаем с config.DATA_DIR.
    """
    data_dir = Path(data_dir) if data_dir is not None else C.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    eval_date = pd.Timestamp(C.EVAL_DATE)
    print(f"Каталог данных: {data_dir}")
    print(f"Дата оценки: {eval_date.date()}, горизонты 1 и 10 дней")
    print(f"Сценариев Monte Carlo: {SIM.N_SIM}, seed {C.RANDOM_SEED}")

    # Симуляция факторов и собственных шоков акций (seed фиксирован внутри).
    incr, idio, info = SIM.simulate(data_dir)
    print(f"Симулировано факторов: {incr.shape[2]}, дней: {incr.shape[1]}")

    # Переоценка инструментов по сценариям.
    ret, notional, groups, labels = REV.instrument_returns(incr, idio, info, data_dir)
    value0 = float(notional.sum())
    print(f"\nСтоимость портфеля на дату оценки: {_format_rub(value0)} руб")
    print("  облигации " + _format_rub(notional[:len(C.PORTFOLIO_BONDS)].sum())
          + ", акции " + _format_rub(
              notional[len(C.PORTFOLIO_BONDS):len(C.PORTFOLIO_BONDS) + len(C.PORTFOLIO_STOCKS)].sum())
          + ", валюта " + _format_rub(notional[-2:].sum()) + " руб")

    # Меры риска по всему портфелю и подпортфелям.
    table, pnl_full = RM.measure_all(ret, notional, groups)

    # Главный результат: четыре числа по всему портфелю.
    full = table[table["Портфель"] == "Портфель"].set_index("Горизонт")
    print("\nГлавный результат, весь портфель:")
    for h in (1, 10):
        r = full.loc[h]
        print(f"  горизонт {_days(h)}: VaR 99% = {_format_rub(r['VaR99'])} руб "
              f"({r['VaR99_pct']:.2f}%), ES 97.5% = {_format_rub(r['ES975'])} руб "
              f"({r['ES975_pct']:.2f}%)")

    # Прикидка масштаба: при близкой к независимой динамике 10-дневный VaR
    # примерно в корень из 10 больше однодневного.
    ratio = full.loc[10, "VaR99"] / full.loc[1, "VaR99"]
    print(f"\nОтношение VaR 10 к 1 дню: {ratio:.2f} (корень из 10 это {np.sqrt(10):.2f})")

    # Разбивка по подпортфелям.
    show = table.copy()
    for col in ["Стоимость", "VaR99", "ES975"]:
        show[col] = show[col].map(_format_rub)
    show["VaR99_pct"] = show["VaR99_pct"].round(2)
    show["ES975_pct"] = show["ES975_pct"].round(2)
    print("\nРазбивка по портфелям и горизонтам:")
    print(show.to_string(index=False))

    # Гистограммы P&L для слайдов.
    fig_dir = C.PROJECT_DIR / "docs" / "figures"
    for h in (1, 10):
        r = full.loc[h]
        path = _plot_pnl(pnl_full[h], r["VaR99"], r["ES975"], h, fig_dir)
        print(f"График P&L (горизонт {h}): {path}")

    # Сохраняем таблицу результатов.
    res_path = data_dir / "var_es_results.parquet"
    table.to_parquet(res_path)
    print("\nИтог, сохранённые файлы:")
    print(f"  var_es_results.parquet       {table.shape} (VaR99 и ES97.5 по портфелям и горизонтам)")


if __name__ == "__main__":
    run()
