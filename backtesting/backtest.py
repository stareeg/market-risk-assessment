"""
Этап 6. Бэктестинг VaR на каждый торговый день 2025 года.

Идея простого бэктеста: каждый день заново считаем VaR по той же модели, что в
пункте 5, и смотрим, как часто фактический убыток следующего дня оказывается
больше предсказанного VaR. Такое событие называем пробоем. Для корректной модели
доля пробоев должна быть близка к уровню значимости (для VaR 99% это около 1%, то
есть примерно 2-3 пробоя за 250 торговых дней).

Считаем по всему портфелю и по трём подпортфелям (облигации, акции, валюта).

Горизонт берём один день. Это стандартный бэктест: один прогноз и один факт на
торговый день, пробои не пересекаются во времени. На горизонте 10 дней окна
накладывались бы и ломали проверку независимости пробоев в этапе 7, поэтому
десятидневный VaR оставляем для самой оценки риска в пункте 5, а проверяем дневной.

Как устроен один день. Берём торговый день d из 2025 года и предыдущий торговый
день d_prev. На конец дня d_prev прогнозируем VaR на один день вперёд (вызываем ту
же симуляцию, что в пункте 5, с датой прогноза d_prev). Затем сравниваем прогноз с
фактической доходностью за день d_prev в d. Так каждый торговый день 2025 года
становится днём проверки, всего их 250.

Что переоцениваем каждый день (решение по пункту 6). Параметры моделей GARCH-DCC и
факторной модели оставляем оценёнными один раз на всей истории. Заново каждый день
прогоняем только фильтр волатильности: условную дисперсию каждого фактора, матрицу
Q для DCC и затравку AR восстанавливаем по истории строго до даты прогноза. Так
дисперсия (главный двигатель дневного VaR) всегда отражает только доступную на эту
дату информацию. Полная переоценка параметров каждый день это 250 умножить на 9
подгонок MLE, дорого и неустойчиво на коротких окнах, выигрыш в точности на горизонте
один день мал. Поэтому фиксируем параметры и обновляем фильтр. Это стандартная схема
с замороженными параметрами и скользящим фильтром.

Фактический P&L считаем на реализованных риск-факторах через тот же прайсер, что и
в симуляции. Облигации переоцениваем по реализованному сдвигу кривой (те же три
RATE_PC и те же нагрузки). Валюта это реализованная доходность курса. Акции берём по
фактической доходности рынка (с поправкой на сплиты), она включает и факторную, и
идиосинкразическую часть, ровно то, что в симуляции мы моделируем шоком по бумаге.
Так тестируем именно риск-модель, а не базис цены облигаций из пункта 4.

Запуск: python main.py --stage backtest
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from var_engine import simulation as SIM
from var_engine import revaluation as REV
from var_engine import risk_measures as RM
from pricing import curve as CV
from risk_factors import factors as F

pd.options.io.parquet.engine = "fastparquet"

# Год проверки и порядок подпортфелей в выводе.
BACKTEST_YEAR = 2025
GROUP_ORDER = ["Портфель", "Облигации", "Акции", "Валюта"]

# Сценариев Monte Carlo на каждый день. Берём столько же, сколько в пункте 5, чтобы
# VaR считался той же моделью. Один процентный хвост это 500 сценариев, оценка
# устойчива. Seed фиксируем общим для всех дней: случайные инновации одинаковы,
# меняются только дисперсия, корреляция и кривая, поэтому путь VaR гладкий и зависит
# от динамики модели, а не от шума Monte Carlo.
N_SIM = 50_000
SEED = C.RANDOM_SEED

VAR_LEVEL = 0.99
ES_LEVEL = 0.975


def _format_rub(x: float) -> str:
    """Рубли с разделением разрядов."""
    return f"{x:,.0f}".replace(",", " ")


def load_inputs(data_dir: Path):
    """
    Готовит постоянные входные данные для бэктеста.

    Возвращает:
      rf       приращения 8 факторов по дням (как в пунктах 2-3),
      adj_px   цены акций, скорректированные на сплиты (для фактической доходности),
      notional рублёвый объём по инструментам,
      groups   индексы инструментов по подпортфелям.
    """
    rf = pd.read_parquet(data_dir / "risk_factors.parquet")

    panels = F.load_panels(data_dir)
    adj_px, _ = F.adjust_splits(panels["stock_px"])
    adj_px = adj_px[C.PORTFOLIO_STOCKS]

    notional, groups, _ = REV.portfolio_layout()
    return rf, adj_px, notional, groups


def estimate_var(as_of: pd.Timestamp, data_dir: Path,
                 n_sim: int = N_SIM, seed: int = SEED) -> pd.DataFrame:
    """
    Прогноз VaR 99% и ES 97.5% на один день вперёд по состоянию на дату as_of.

    Возвращает таблицу мер риска по всему портфелю и подпортфелям (та же функция
    measure_all, что в пункте 5, только на горизонте один день).
    """
    incr, idio, info = SIM.simulate(data_dir, n_sim=n_sim, horizon=1,
                                    seed=seed, as_of=as_of)
    ret, notional, groups, _ = REV.instrument_returns(incr, idio, info, data_dir)
    table, _ = RM.measure_all(ret, notional, groups, horizons=(1,))
    return table.set_index("Портфель")


def realized_instrument_returns(d_prev: pd.Timestamp, d: pd.Timestamp,
                                rf: pd.DataFrame, adj_px: pd.DataFrame,
                                data_dir: Path) -> np.ndarray:
    """
    Фактические дневные доходности всех инструментов за день d_prev в d.
    Порядок инструментов как в revaluation: 5 ОФЗ, 10 акций, USD, EUR.
    """
    inc = rf.loc[d]

    # Облигации: реализованный сдвиг кривой через три RATE_PC и те же нагрузки.
    rate_incr = inc[["RATE_PC1", "RATE_PC2", "RATE_PC3"]].values.reshape(1, 1, 3)
    base_yields = CV.load_base_curve(data_dir, d_prev).values
    r_bonds = REV.bond_returns(rate_incr, data_dir, pd.Timestamp(d_prev),
                               base_yields)[0, 0, :]

    # Акции: фактическая рыночная доходность с поправкой на сплиты.
    r_stocks = (adj_px.loc[d] / adj_px.loc[d_prev] - 1.0).reindex(C.PORTFOLIO_STOCKS).values

    # Валюта: факторы это лог-доходности курса, простая доходность это exp минус 1.
    r_usd = np.exp(inc["FX_USD"]) - 1.0
    r_eur = np.exp(inc["FX_EUR"]) - 1.0

    return np.concatenate([r_bonds, r_stocks, [r_usd, r_eur]])


def realized_pnl(realized_ret: np.ndarray, notional: np.ndarray,
                 groups: dict) -> dict:
    """
    Фактический P&L (рубли) по каждому подпортфелю за один день.
    При ежедневной ребалансировке однодневный P&L это взвешенная по пропорциям
    доходность, умноженная на стоимость подпортфеля.
    """
    out = {}
    for name, idx in groups.items():
        idx = np.array(idx)
        sub_notional = notional[idx]
        value0 = float(sub_notional.sum())
        weights = sub_notional / value0
        out[name] = value0 * float(realized_ret[idx] @ weights)
    return out


def run_backtest(data_dir: Path, n_sim: int = N_SIM, seed: int = SEED) -> pd.DataFrame:
    """
    Прогоняет бэктест по всем торговым дням 2025 года.
    Возвращает длинную таблицу: по строке на пару день и портфель с прогнозом VaR,
    фактическим P&L и отметкой пробоя.
    """
    rf, adj_px, notional, groups = load_inputs(data_dir)

    idx = rf.index
    days = idx[idx.year == BACKTEST_YEAR]
    pos = idx.get_indexer(days)

    rows = []
    n = len(days)
    print(f"Торговых дней для проверки: {n} "
          f"({days[0].date()} по {days[-1].date()})")
    print(f"Сценариев на каждый день: {n_sim}, seed {seed}")

    for i, (d, p) in enumerate(zip(days, pos)):
        d_prev = idx[p - 1]                       # день, на конец которого даём прогноз

        table = estimate_var(d_prev, data_dir, n_sim=n_sim, seed=seed)
        realized_ret = realized_instrument_returns(d_prev, d, rf, adj_px, data_dir)
        pnl = realized_pnl(realized_ret, notional, groups)

        for name in GROUP_ORDER:
            var99 = float(table.loc[name, "VaR99"])
            es975 = float(table.loc[name, "ES975"])
            value0 = float(table.loc[name, "Стоимость"])
            p_pnl = pnl[name]
            rows.append({
                "Дата": d,
                "Портфель": name,
                "Стоимость": value0,
                "VaR99": var99,
                "ES975": es975,
                "PnL": p_pnl,
                "Пробой": int(p_pnl < -var99),
            })

        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"  обработано дней: {i + 1} из {n}")

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """
    Сводка по портфелям: число дней, число и доля пробоев, ожидаемая доля.
    Ожидаемая доля это 1 минус уровень VaR (для VaR 99% это 1%).
    """
    expected = (1.0 - VAR_LEVEL) * 100.0
    rows = []
    for name in GROUP_ORDER:
        sub = results[results["Портфель"] == name]
        n_days = len(sub)
        n_breach = int(sub["Пробой"].sum())
        rows.append({
            "Портфель": name,
            "Дней": n_days,
            "Пробоев": n_breach,
            "Ожидалось": round(n_days * expected / 100.0, 1),
            "Доля пробоев, %": round(n_breach / n_days * 100.0, 2),
            "Ожидаемая доля, %": expected,
        })
    return pd.DataFrame(rows)


def _plot_one(ax, sub: pd.DataFrame, title: str) -> None:
    """Один график бэктеста: фактический P&L, линия минус VaR, пробои точками."""
    from viz.style import COLORS

    dates = sub["Дата"].values
    pnl = sub["PnL"].values / 1e6
    var_line = -sub["VaR99"].values / 1e6
    breach = sub["Пробой"].values.astype(bool)

    ax.plot(dates, pnl, color=COLORS["main"], linewidth=1.0,
            label="Фактический P&L")
    ax.plot(dates, var_line, color=COLORS["accent"], linewidth=1.8,
            label="Минус VaR 99%")
    ax.scatter(dates[breach], pnl[breach], color=COLORS["accent"], s=45,
               zorder=5, label=f"Пробои: {int(breach.sum())}")
    ax.axhline(0.0, color=COLORS["grey"], linewidth=0.8, alpha=0.6)
    ax.set_title(title)
    ax.set_ylabel("млн руб")
    ax.legend(loc="lower left", ncol=3)


def plot_backtest(results: pd.DataFrame, out_dir: Path):
    """Графики бэктеста для слайдов: весь портфель и три подпортфеля."""
    import matplotlib.pyplot as plt
    from viz.style import set_slide_style, save_slide

    set_slide_style()

    # Весь портфель отдельной картинкой.
    fig, ax = plt.subplots()
    full = results[results["Портфель"] == "Портфель"].sort_values("Дата")
    _plot_one(ax, full, "Бэктест VaR 99%, весь портфель, 2025 год")
    ax.set_xlabel("Дата")
    path_full = save_slide(fig, "backtest_full", out_dir)

    # Три подпортфеля в столбик.
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, name in zip(axes, ["Облигации", "Акции", "Валюта"]):
        sub = results[results["Портфель"] == name].sort_values("Дата")
        _plot_one(ax, sub, name)
    axes[-1].set_xlabel("Дата")
    path_sub = save_slide(fig, "backtest_subportfolios", out_dir)

    return path_full, path_sub


def run(data_dir: str | Path | None = None) -> None:
    """
    Запускает бэктест: считает пробои по дням 2025 года, печатает сводку,
    сохраняет результаты и графики.
    """
    data_dir = Path(data_dir) if data_dir is not None else C.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Каталог данных: {data_dir}")

    results = run_backtest(data_dir)

    summary = summarize(results)
    print("\nСводка по пробоям VaR 99%:")
    print(summary.to_string(index=False))
    print("\nДля корректной модели доля пробоев близка к 1%. "
          "Формальные тесты (Купиец, Кристофферсен) считаем в этапе 7.")

    # Графики для слайдов.
    fig_dir = C.PROJECT_DIR / "docs" / "figures"
    path_full, path_sub = plot_backtest(results, fig_dir)
    print(f"\nГрафик бэктеста, весь портфель: {path_full}")
    print(f"График бэктеста, подпортфели: {path_sub}")

    # Сохраняем результаты для этапа 7.
    res_path = data_dir / "backtest_results.parquet"
    results.to_parquet(res_path)
    sum_path = data_dir / "backtest_summary.parquet"
    summary.to_parquet(sum_path)
    print("\nИтог, сохранённые файлы:")
    print(f"  backtest_results.parquet     {results.shape} (VaR, P&L и пробои по дням и портфелям)")
    print(f"  backtest_summary.parquet     {summary.shape} (число и доля пробоев по портфелям)")


if __name__ == "__main__":
    run()
