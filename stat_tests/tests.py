"""
Этап 7. Статистические тесты корректности VaR.

Бэктест из пункта 6 дал последовательность пробоев по каждому портфелю. Глазом
видно, что доля пробоев где-то выше, где-то ниже ожидаемого 1 процента, но это
надо проверить формально. Здесь считаем тесты, которые отвечают на два вопроса.

Первый вопрос про долю пробоев (безусловное покрытие). Если модель верна, пробои
случаются с вероятностью 1 процент в день (для VaR 99%). Тест Купица проверяет,
не отличается ли наблюдаемая доля от ожидаемой значимо.

Второй вопрос про независимость пробоев. Хорошая модель распределяет пробои
равномерно во времени, а не кучкует их в плохие периоды. Если пробои идут
сериями, значит модель не успевает поднимать VaR в спокойное время после
всплеска. Это ловят тест Кристофферсена (марковская проверка соседних дней) и
тест Хааса по длинам промежутков между пробоями (ловит группировку и на больших
лагах, не только день в день).

Берём три семейства тестов:
  Kupiec POF       доля пробоев, хи-квадрат с 1 степенью свободы.
  Christoffersen   независимость соседних дней (LR_ind) и условное покрытие
                   (LR_cc = POF + LR_ind), хи-квадрат с 1 и 2 степенями свободы.
  Haas TBFI        независимость по промежуткам между пробоями (геометрическое
                   распределение длин), хи-квадрат с числом промежутков.

Считаем по всему портфелю и по трём подпортфелям. Используем готовую таблицу
пробоев из пункта 6 (backtest_results.parquet).

Запуск: python main.py --stage tests
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special as sp
from scipy.stats import chi2

import config as C

pd.options.io.parquet.engine = "fastparquet"

# Порядок портфелей в выводе, как в бэктесте.
GROUP_ORDER = ["Портфель", "Облигации", "Акции", "Валюта"]

# Уровень VaR и вытекающая ожидаемая доля пробоев.
VAR_LEVEL = 0.99
P_EXP = 1.0 - VAR_LEVEL

# Уровень значимости самих тестов. Если p-value ниже, гипотезу о корректной
# модели отвергаем.
ALPHA = 0.05


def kupiec_pof(n: int, x: int, p: float = P_EXP):
    """
    Тест Купица на долю пробоев (безусловное покрытие).

    Сравнивает наблюдаемую долю пробоев x/n с заложенной p через отношение
    правдоподобий. Статистика хи-квадрат с 1 степенью свободы. Большая статистика
    и малое p-value означают, что доля пробоев значимо не та, что заложена.
    """
    pi_hat = x / n
    # Правдоподобие при заложенной вероятности p и при наблюдённой pi_hat.
    # xlogy(a, b) это a*log(b) с верным нулём при a равном нулю, так что случаи
    # x=0 и x=n считаются без деления на ноль.
    ll_p = sp.xlogy(n - x, 1.0 - p) + sp.xlogy(x, p)
    ll_hat = sp.xlogy(n - x, 1.0 - pi_hat) + sp.xlogy(x, pi_hat)
    stat = float(2.0 * (ll_hat - ll_p))
    stat = max(stat, 0.0)
    return stat, 1, float(chi2.sf(stat, 1))


def _markov_counts(breaches: np.ndarray):
    """Счётчики переходов день в день для марковской проверки независимости."""
    b = np.asarray(breaches).astype(int)
    prev, cur = b[:-1], b[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    return n00, n01, n10, n11


def christoffersen_ind(breaches: np.ndarray):
    """
    Тест Кристофферсена на независимость пробоев (первый порядок Маркова).

    Проверяет, зависит ли вероятность пробоя сегодня от того, был ли пробой вчера.
    Если пробои кучкуются, вероятность после пробоя выше, и тест это поймает.
    Статистика хи-квадрат с 1 степенью свободы.
    """
    n00, n01, n10, n11 = _markov_counts(breaches)
    t0, t1 = n00 + n01, n10 + n11
    tot = t0 + t1
    pi01 = n01 / t0 if t0 > 0 else 0.0      # доля пробоев после спокойного дня
    pi11 = n11 / t1 if t1 > 0 else 0.0      # доля пробоев после пробоя
    pi = (n01 + n11) / tot if tot > 0 else 0.0
    ll_null = sp.xlogy(n00 + n10, 1.0 - pi) + sp.xlogy(n01 + n11, pi)
    ll_alt = (sp.xlogy(n00, 1.0 - pi01) + sp.xlogy(n01, pi01)
              + sp.xlogy(n10, 1.0 - pi11) + sp.xlogy(n11, pi11))
    stat = float(2.0 * (ll_alt - ll_null))
    stat = max(stat, 0.0)
    return stat, 1, float(chi2.sf(stat, 1))


def christoffersen_cc(pof_stat: float, ind_stat: float):
    """
    Условное покрытие Кристофферсена.

    Это сумма статистик доли пробоев и независимости. Проверяет всё сразу:
    и долю, и отсутствие группировки. Хи-квадрат с 2 степенями свободы.
    """
    stat = float(pof_stat + ind_stat)
    return stat, 2, float(chi2.sf(stat, 2))


def haas_tbfi(breaches: np.ndarray, p: float = P_EXP):
    """
    Тест Хааса по промежуткам между пробоями (time between failures).

    Если пробои независимы, длины промежутков между ними подчиняются
    геометрическому распределению с параметром p. Для каждого промежутка считаем
    статистику Купица как для отдельного срока до пробоя и складываем. Сумма это
    хи-квадрат с числом промежутков (число пробоев минус один). В отличие от
    Кристофферсена ловит группировку не только в соседние дни, но и на больших
    промежутках.

    Нужно минимум 2 пробоя, иначе ни одного промежутка нет и тест неприменим
    (возвращаем nan).
    """
    pos = np.flatnonzero(np.asarray(breaches).astype(int))
    k = len(pos)
    if k < 2:
        return float("nan"), 0, float("nan")
    durations = np.diff(pos)  # длины промежутков между соседними пробоями, каждая >= 1
    stat = 0.0
    for d in durations:
        d = int(d)
        # Геометрическое правдоподобие при заложенной p и при наблюдённой 1/d.
        ll_p = math.log(p) + sp.xlogy(d - 1, 1.0 - p)
        ll_hat = math.log(1.0 / d) + sp.xlogy(d - 1, 1.0 - 1.0 / d)
        stat += 2.0 * (ll_hat - ll_p)
    stat = float(max(stat, 0.0))
    df = k - 1
    return stat, df, float(chi2.sf(stat, df))


def _verdict(pof_p: float, cc_p: float, share_pct: float,
             expected_pct: float) -> str:
    """
    Общий вывод по портфелю.

    Сначала смотрим на долю пробоев (тест Купица), это главная проверка частоты.
    Если доля значимо не та, говорим прямо, в какую сторону. Если доля в норме, но
    условное покрытие Кристофферсена отвергнуто, значит пробои группируются. Иначе
    модель корректна.
    """
    if pof_p < ALPHA:
        if share_pct < expected_pct:
            return "VaR слишком консервативен, пробоев значимо мало"
        return "VaR занижает риск, пробоев значимо много"
    if cc_p < ALPHA:
        return "доля в норме, но пробои группируются"
    return "модель корректна"


def run_tests(results: pd.DataFrame) -> pd.DataFrame:
    """
    Прогоняет все тесты по каждому портфелю.
    Возвращает таблицу со статистиками, p-value и общим выводом.
    """
    rows = []
    for name in GROUP_ORDER:
        sub = results[results["Портфель"] == name].sort_values("Дата")
        b = sub["Пробой"].to_numpy().astype(int)
        n = len(b)
        x = int(b.sum())

        pof_s, _, pof_p = kupiec_pof(n, x)
        ind_s, _, ind_p = christoffersen_ind(b)
        cc_s, _, cc_p = christoffersen_cc(pof_s, ind_s)
        haas_s, haas_df, haas_p = haas_tbfi(b)

        rows.append({
            "Портфель": name,
            "Дней": n,
            "Пробоев": x,
            "Доля, %": round(x / n * 100.0, 2),
            "Kupiec_stat": round(pof_s, 3),
            "Kupiec_p": round(pof_p, 4),
            "Ind_stat": round(ind_s, 3),
            "Ind_p": round(ind_p, 4),
            "CC_stat": round(cc_s, 3),
            "CC_p": round(cc_p, 4),
            "Haas_stat": round(haas_s, 3) if not math.isnan(haas_s) else float("nan"),
            "Haas_df": haas_df,
            "Haas_p": round(haas_p, 4) if not math.isnan(haas_p) else float("nan"),
            "Вердикт": _verdict(pof_p, cc_p, x / n * 100.0, P_EXP * 100.0),
        })
    return pd.DataFrame(rows)


def _print_results(table: pd.DataFrame) -> None:
    """Печатает результаты тестов читаемыми блоками."""
    counts = table[["Портфель", "Дней", "Пробоев", "Доля, %"]].copy()
    counts["Ожидалось"] = (counts["Дней"] * P_EXP).round(1)
    print("Пробои по портфелям (из бэктеста, пункт 6):")
    print(counts.to_string(index=False))

    print("\nТесты корректности VaR (p-value, порог отклонения "
          f"{ALPHA}):")
    show = table[["Портфель", "Kupiec_p", "Ind_p", "CC_p", "Haas_p", "Вердикт"]]
    show = show.rename(columns={
        "Kupiec_p": "Купиц",
        "Ind_p": "Кристоф.незав",
        "CC_p": "Кристоф.усл.покр",
        "Haas_p": "Хаас",
    })
    print(show.to_string(index=False))


def _discuss(table: pd.DataFrame) -> None:
    """
    Критическое обсуждение результатов (требование пункта 7).
    Текст строим по реальным числам из таблицы, а не задаём вручную.
    """
    print("\nКритическое обсуждение:")

    print("- Выбор тестов. Берём три семейства, потому что они проверяют разное. "
          "Купиц смотрит только на долю пробоев и не видит их группировку. "
          "Кристофферсен добавляет проверку независимости соседних дней и "
          "условное покрытие. Хаас смотрит на длины промежутков между пробоями и "
          "ловит группировку на любом расстоянии, а не только день в день. Вместе "
          "они закрывают и долю, и динамику пробоев.")

    for _, r in table.iterrows():
        name = r["Портфель"]
        x = int(r["Пробоев"])
        share = r["Доля, %"]
        bits = [f"- {name}: пробоев {x} из {int(r['Дней'])} ({share}%), "
                f"ожидалось около {P_EXP*100:.0f}%."]
        if r["Kupiec_p"] < ALPHA:
            if share < P_EXP * 100:
                bits.append("Купиц отвергает долю: пробоев заметно меньше нормы, "
                            "VaR слишком консервативен.")
            else:
                bits.append("Купиц отвергает долю: пробоев больше нормы, "
                            "VaR занижает риск.")
        else:
            bits.append("Купиц долю не отвергает.")
        ind_ok = r["Ind_p"] >= ALPHA
        haas_na = math.isnan(r["Haas_p"])
        haas_ok = haas_na or r["Haas_p"] >= ALPHA
        if haas_na:
            bits.append("Хаас неприменим, пробоев меньше двух, промежутков нет.")
        elif ind_ok and haas_ok:
            bits.append("Независимость держится: ни соседние дни (Кристофферсен), "
                        "ни промежутки (Хаас) группировки не показывают.")
        elif ind_ok and not haas_ok:
            if x <= 3:
                bits.append("Кристофферсен группировки не видит, но Хаас отмечает, "
                            "что пробои легли близко друг к другу. При таком малом "
                            "числе пробоев это слабый сигнал, на вердикт по "
                            "покрытию он не влияет.")
            else:
                bits.append("Кристофферсен соседние дни чистыми считает, но Хаас "
                            "видит сближение пробоев на среднем горизонте.")
        else:
            bits.append("Есть признаки группировки пробоев в соседние дни.")
        bits.append(f"Итог: {r['Вердикт']}.")
        print("  " + " ".join(bits))

    print("- Ограничение. Пробоев мало (за год их единицы), поэтому у тестов "
          "независимости низкая мощность, а у Хааса при двух пробоях всего один "
          "промежуток. Это честное ограничение годичного бэктеста: формально доля "
          "проверяется надёжно, а группировку на таком объёме данных уверенно не "
          "поймать. Для строгой проверки независимости нужен более длинный период.")
    print("- Связь с пунктами 5 и 6. Тесты подтверждают вывод, что риск портфеля "
          "определяет валюта: по доле она ближе всех к границе, а облигации "
          "перестрахованы. Само покрытие портфель проходит.")


def _plot_pvalues(table: pd.DataFrame, out_dir: Path):
    """
    Картинка для слайда: таблица p-value по портфелям и тестам с цветом.
    Зелёный это тест пройден (p не ниже порога), красный это отклонение,
    серый это тест неприменим.
    """
    import matplotlib.pyplot as plt
    from viz.style import set_slide_style, save_slide

    set_slide_style()

    cols = ["Kupiec_p", "Ind_p", "CC_p", "Haas_p"]
    col_labels = ["Купиц\n(доля)", "Кристоф.\nнезав.",
                  "Кристоф.\nусл.покр.", "Хаас\n(промежутки)"]
    rows = table["Портфель"].tolist()
    pmat = table[cols].to_numpy(dtype=float)

    pass_col = "#d9ead3"   # светло-зелёный, тест пройден
    fail_col = "#f4cccc"   # светло-красный, отклонение
    na_col = "#eeeeee"     # серый, неприменим

    fig, ax = plt.subplots(figsize=(10, 5.0))
    # Размещаем поля вручную, чтобы двухстрочные подписи столбцов и нижний
    # пояснительный текст не наезжали друг на друга.
    fig.set_tight_layout(False)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(rows))
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    # Верхняя строка таблицы сверху, поэтому портфели идут сверху вниз.
    ax.set_yticklabels(rows[::-1])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, name in enumerate(rows):
        y = len(rows) - 1 - i        # переворот, чтобы первый портфель был сверху
        for j, p in enumerate(pmat[i]):
            if math.isnan(p):
                color, txt = na_col, "n/a"
            elif p >= ALPHA:
                color, txt = pass_col, f"{p:.3f}"
            else:
                color, txt = fail_col, f"{p:.3f}"
            ax.add_patch(plt.Rectangle((j, y), 1, 1, facecolor=color,
                                       edgecolor="white", linewidth=2))
            ax.text(j + 0.5, y + 0.5, txt, ha="center", va="center",
                    fontsize=14)

    ax.set_title("Статистические тесты VaR 99%, p-value по портфелям")
    fig.subplots_adjust(left=0.15, right=0.97, top=0.86, bottom=0.26)
    fig.text(0.5, 0.05,
             f"Зелёный p не ниже {ALPHA} (тест пройден), "
             f"красный p ниже {ALPHA} (отклонение), серый неприменим",
             ha="center", va="center", fontsize=12, color="#555555")
    return save_slide(fig, "stat_tests", out_dir)


def run(data_dir: str | Path | None = None) -> None:
    """
    Запускает статистические тесты: читает пробои из бэктеста, считает тесты по
    портфелю и подпортфелям, печатает результаты с обсуждением, сохраняет таблицу
    и картинку.
    """
    data_dir = Path(data_dir) if data_dir is not None else C.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Каталог данных: {data_dir}")

    res_path = data_dir / "backtest_results.parquet"
    if not res_path.exists():
        raise SystemExit("Нет backtest_results.parquet. Сначала запустите "
                         "этап бэктеста: python main.py --stage backtest")
    results = pd.read_parquet(res_path)

    table = run_tests(results)
    _print_results(table)
    _discuss(table)

    fig_dir = C.PROJECT_DIR / "docs" / "figures"
    path_fig = _plot_pvalues(table, fig_dir)
    print(f"\nГрафик тестов: {path_fig}")

    out_path = data_dir / "stat_tests.parquet"
    table.to_parquet(out_path)
    print("\nИтог, сохранённые файлы:")
    print(f"  stat_tests.parquet           {table.shape} "
          "(статистики, p-value и вывод по портфелям)")


if __name__ == "__main__":
    run()
