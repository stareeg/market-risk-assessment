"""
Единый стиль графиков под слайды.

Цель - чтобы надпись было видно с задних рядов и подписи не наезжали друг на
друга. Поэтому крупные шрифты, светлый фон, тонкая сетка, без рамок сверху и
справа. Никаких смайликов и спецсимволов в подписях.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


# Палитра: спокойные цвета, хорошо различимые на проекторе
COLORS = {
    "main": "#1f4e79",    # тёмно-синий, основной
    "accent": "#c00000",  # красный, для линий VaR и пробоев
    "second": "#548235",  # зелёный, для сравнения
    "grey": "#7f7f7f",    # серый, для вспомогательного
}


def set_slide_style() -> None:
    """Включает крупный читаемый стиль. Вызывать один раз перед построением."""
    mpl.rcParams.update({
        "figure.figsize": (10, 5.5),   # под широкий слайд 16:9
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",       # чтобы подписи не обрезались
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,      # убираем лишние рамки
        "axes.spines.right": False,
        "axes.titlepad": 12,
        "figure.autolayout": True,     # авторазмещение, чтобы не наезжало
    })


def save_slide(fig, name: str, out_dir: str | Path = "docs/figures") -> Path:
    """Сохраняет график в png под слайд и возвращает путь к файлу."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path
