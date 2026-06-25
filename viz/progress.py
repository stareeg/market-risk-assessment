"""
Блок-схема прогресса проекта для вводного слайда (шаг 13).

Рисует восемь пунктов задания сквозным пайплайном сверху вниз: на каждый пункт
прямоугольник, между ними стрелки, цвет по статусу (зелёный готово, жёлтый чиним,
серый не начато). Картинку кладём в docs/figures/progress.png.

Запуск: python -m viz.progress
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

from viz.style import set_slide_style, save_slide


# Восемь пунктов задания: номер, название, пакет, метод, статус.
# Статус один из: готово, чиним, не начато.
STAGES = [
    ("п.1", "Данные",       "data_collection", "MOEX ISS и ЦБ РФ, 2021-2026",             "готово"),
    ("п.2", "Риск-факторы", "risk_factors",    "PCA кривой и акций, 8 факторов",          "готово"),
    ("п.3", "Модели",       "models",          "GARCH(1,1)-t и DCC, оценка MLE",          "готово"),
    ("п.4", "Прайсинг",     "pricing",         "кривая по 12 узлам, факторная модель",    "готово"),
    ("п.5", "VaR и ES",     "var_engine",      "Monte Carlo, горизонты 1 и 10, ребаланс", "готово"),
    ("п.6", "Бэктестинг",   "backtesting",     "VaR по дням 2025, пробои",                "готово"),
    ("п.7", "Стат-тесты",   "stat_tests",      "Kupiec, Christoffersen, Haas",            "готово"),
    ("п.8", "Бонус",        "bonus",           "опционы Black-76, встроенные опционы",    "готово"),
]

# Цвета статусов: светлая заливка и насыщенная рамка, чтобы тёмный текст читался.
STATUS_FILL = {"готово": "#dbe9cc", "чиним": "#fdebc8", "не начато": "#e6e6e6"}
STATUS_EDGE = {"готово": "#548235", "чиним": "#dd8b00", "не начато": "#7f7f7f"}
TEXT_COLOR = "#1f4e79"


def build_progress_figure():
    """Собирает фигуру с блок-схемой и возвращает её."""
    set_slide_style()
    n = len(STAGES)
    fig, ax = plt.subplots(figsize=(13.5, 8.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n + 1.3)
    ax.axis("off")

    box_h, gap = 0.78, 0.22
    x0, x1 = 0.4, 9.6
    top = n + 0.9   # верхняя кромка под заголовок

    for i, (point, name, pkg, method, status) in enumerate(STAGES):
        # первый пункт выше всех, дальше вниз
        yc = top - 0.9 - i * (box_h + gap) - box_h / 2
        box = FancyBboxPatch((x0, yc - box_h / 2), x1 - x0, box_h,
                             boxstyle="round,pad=0.02,rounding_size=0.08",
                             linewidth=2.2, edgecolor=STATUS_EDGE[status],
                             facecolor=STATUS_FILL[status])
        ax.add_patch(box)
        # пункт и название слева жирным
        ax.text(x0 + 0.25, yc, f"{point}  {name}", va="center", ha="left",
                fontsize=15, fontweight="bold", color=TEXT_COLOR)
        # метод посередине
        ax.text(x0 + 3.1, yc, method, va="center", ha="left",
                fontsize=12.5, color="#333333")
        # пакет справа моноширинным
        ax.text(x1 - 0.25, yc, pkg, va="center", ha="right",
                fontsize=11, family="monospace", color=STATUS_EDGE[status])
        # стрелка вниз к следующему пункту
        if i < n - 1:
            xm = (x0 + x1) / 2
            y_from = yc - box_h / 2
            ax.annotate("", xy=(xm, y_from - gap), xytext=(xm, y_from),
                        arrowprops=dict(arrowstyle="-|>", color="#7f7f7f", lw=1.6))

    ax.text(5, n + 0.7, "Прогресс проекта: восемь пунктов задания",
            ha="center", va="center", fontsize=19, fontweight="bold",
            color=TEXT_COLOR)
    ax.text(5, n + 0.28, "сквозной пайплайн, каждый этап читает результат предыдущего",
            ha="center", va="center", fontsize=12.5, color="#555555")

    handles = [Patch(facecolor=STATUS_FILL[s], edgecolor=STATUS_EDGE[s],
                     linewidth=2, label=s) for s in ["готово", "чиним", "не начато"]]
    ax.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
              fontsize=13, bbox_to_anchor=(0.5, 0.04))
    return fig


def run(out_dir: str | Path = "docs/figures") -> Path:
    """Строит блок-схему и сохраняет в png под слайд."""
    fig = build_progress_figure()
    return save_slide(fig, "progress", out_dir)


if __name__ == "__main__":
    print(f"Блок-схема прогресса: {run()}")
