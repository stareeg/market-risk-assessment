"""
Бонусный портфель 1. Опционы на фьючерс Si, оценка по модели Блэка-76 (пункт 8).

Почему Блэк-76, а не Блэк-Шоулз. Это опционы на фьючерс, а не на спот, поэтому
базовый актив в формуле это фьючерсная цена F, а не цена спота. Это прямая
оговорка задания.

Опционы на срочном рынке MOEX маржируемые (futures-style): премию не платят
сразу, она ходит вариационной маржой как у фьючерса. Поэтому дисконтирования в
формуле нет, дисконт-фактор равен единице. Это видно и в данных: у глубоко
денежного колла расчётная цена (16279) выше внутренней стоимости (16176), то есть
премия не дисконтируется ниже внутренней стоимости.

Подразумеваемую волатильность калибруем по цепочке опционов: разворачиваем цену
каждого страйка обратно в волатильность и получаем улыбку. Для оценки берём
волатильность на деньгах (страйк у фьючерсной цены).

Дата оценки тут 2025-10-15, потому что именно на этот день собрана цепочка
опционов и фьючерс (выбран так, чтобы до экспирации было больше месяца).

Проверка (пункт 8b): сравниваем расчётную цену по Блэку-76 с наблюдаемой
расчётной ценой биржи, расхождение в процентах.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

import config as C

pd.options.io.parquet.engine = "fastparquet"

# Два опциона в деньгах с разными страйками. Колл со страйком ниже фьючерсной цены,
# пут со страйком выше. Оба заметно в деньгах, но не глубоко (там улыбку считать
# неустойчиво из-за малой временной стоимости).
CALL_STRIKE = 75000
PUT_STRIKE = 85000

# Размер позиции: стоимость базового актива по страйку около 1 млн руб.
NOTIONAL_RUB = 1_000_000


def black76_price(F: float, K: float, T: float, sigma: float, kind: str) -> float:
    """
    Цена опциона на фьючерс по Блэку-76 без дисконтирования (маржируемый опцион).
    kind это 'C' для колла и 'P' для пута.
    """
    F, K, T, sigma = float(F), float(K), float(T), float(sigma)
    if T <= 0 or sigma <= 0:
        return max(F - K, 0.0) if kind == "C" else max(K - F, 0.0)
    srt = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / srt
    d2 = d1 - srt
    if kind == "C":
        return F * norm.cdf(d1) - K * norm.cdf(d2)
    return K * norm.cdf(-d2) - F * norm.cdf(-d1)


def implied_vol(price: float, F: float, K: float, T: float, kind: str) -> float:
    """
    Подразумеваемая волатильность: разворачиваем цену опциона обратно через Блэка-76.
    Если цена не выше внутренней стоимости, временной стоимости нет и вернуть
    волатильность нельзя (возвращаем nan).
    """
    intrinsic = max(F - K, 0.0) if kind == "C" else max(K - F, 0.0)
    if price <= intrinsic + 1e-6:
        return float("nan")
    try:
        return float(brentq(lambda s: black76_price(F, K, T, s, kind) - price,
                            1e-4, 5.0, maxiter=200))
    except ValueError:
        return float("nan")


def load_chain(data_dir: Path):
    """
    Загружает цепочку опционов и фьючерс на дату сбора данных.
    Возвращает таблицу опционов с подразумеваемой волатильностью, фьючерсную цену
    F, срок до экспирации T (в годах) и сами даты.
    """
    fut = pd.read_parquet(data_dir / f"forts_futures_{C.FORTS_TRADE_DAY}.parquet")
    front = fut[fut["IS_CHOSEN_FRONT"]].iloc[0]
    F = float(front["SETTLEPRICE"])
    expiry = pd.Timestamp(front["LSTDELDATE"])
    trade_day = pd.Timestamp(C.FORTS_TRADE_DAY)
    T = (expiry - trade_day).days / 365.0

    opt = pd.read_parquet(data_dir / f"forts_options_chain_{C.FORTS_TRADE_DAY}.parquet")
    opt = opt[["SECID", "STRIKE", "OPTIONTYPE", "SETTLEPRICE"]].copy()
    opt["IV"] = [implied_vol(p, F, k, T, cp)
                 for p, k, cp in zip(opt["SETTLEPRICE"], opt["STRIKE"], opt["OPTIONTYPE"])]
    return opt, F, T, trade_day, expiry


def atm_vol(chain: pd.DataFrame, F: float) -> float:
    """
    Волатильность на деньгах: линейно интерполируем улыбку коллов в точке страйк
    равен фьючерсной цене. Колл и пут на одном страйке дают одну волатильность
    (паритет при единичном дисконте), поэтому достаточно коллов.
    """
    calls = chain[chain["OPTIONTYPE"] == "C"].dropna(subset=["IV"]).sort_values("STRIKE")
    return float(np.interp(F, calls["STRIKE"].values, calls["IV"].values))


def price_selected(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    """
    Оценивает выбранные колл и пут в деньгах и сравнивает с наблюдаемой ценой.

    Считаем два варианта расчётной цены:
      по своей волатильности страйка (проверка, что реализация Блэка-76 верна,
      ошибка около нуля по построению),
      по единой волатильности на деньгах (так видно вклад улыбки: на крыльях
      волатильность выше, поэтому единая ставка на деньгах слегка занижает цену).
    """
    chain, F, T, trade_day, expiry = load_chain(data_dir)
    sigma_atm = atm_vol(chain, F)

    rows = []
    for K, kind, label in [(CALL_STRIKE, "C", "Колл ITM"), (PUT_STRIKE, "P", "Пут ITM")]:
        obs = float(chain[(chain["STRIKE"] == K) &
                          (chain["OPTIONTYPE"] == kind)]["SETTLEPRICE"].iloc[0])
        own_iv = implied_vol(obs, F, K, T, kind)
        price_own = black76_price(F, K, T, own_iv, kind)
        price_atm = black76_price(F, K, T, sigma_atm, kind)
        n_contracts = int(round(NOTIONAL_RUB / K))
        rows.append({
            "Опцион": label,
            "Страйк": K,
            "Тип": kind,
            "Контрактов": n_contracts,
            "Наблюдаемая": round(obs, 1),
            "Своя волат.": round(own_iv, 4),
            "Цена по своей": round(price_own, 1),
            "Волат. на деньгах": round(sigma_atm, 4),
            "Цена на деньгах": round(price_atm, 1),
            "Расхождение, %": round((price_atm - obs) / obs * 100.0, 2),
        })
    table = pd.DataFrame(rows)
    info = {"F": F, "T": T, "sigma_atm": sigma_atm,
            "trade_day": trade_day, "expiry": expiry}
    return table, info, chain


def plot_options(table: pd.DataFrame, info: dict, chain: pd.DataFrame,
                 out_dir: Path):
    """
    График для слайда: расчётная цена по Блэку-76 против наблюдаемой по обоим
    опционам. Маленькой вставкой показываем улыбку волатильности с отметками
    выбранных страйков.
    """
    import matplotlib.pyplot as plt
    from viz.style import set_slide_style, save_slide, COLORS

    set_slide_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))
    fig.set_tight_layout(False)   # поля задаём вручную, чтобы подписи не наезжали

    # Левая панель: цена по Блэку-76 против наблюдаемой.
    labels = table["Опцион"].tolist()
    x = np.arange(len(labels))
    w = 0.36
    obs = table["Наблюдаемая"].values
    model = table["Цена на деньгах"].values

    ax1.bar(x - w / 2, obs, w, label="Наблюдаемая (биржа)", color=COLORS["main"])
    ax1.bar(x + w / 2, model, w, label="Блэк-76 на деньгах", color=COLORS["second"])
    for xi, m, d in zip(x, model, table["Расхождение, %"].values):
        ax1.text(xi + w / 2, m, f"{d:+.1f}%", ha="center", va="bottom", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{l}\nстрайк {k}" for l, k in zip(labels, table["Страйк"])])
    ax1.set_ylabel("рублей за контракт")
    ax1.set_ylim(0, max(obs.max(), model.max()) * 1.28)
    ax1.set_title("Цена против наблюдаемой")
    ax1.legend(loc="upper center", ncol=1, fontsize=12)

    # Правая панель: улыбка волатильности с отметками выбранных страйков.
    calls = chain[chain["OPTIONTYPE"] == "C"].dropna(subset=["IV"]).sort_values("STRIKE")
    sub = calls[(calls["STRIKE"] >= 70000) & (calls["STRIKE"] <= 92000)]
    ax2.plot(sub["STRIKE"].values, sub["IV"].values * 100, color=COLORS["grey"],
             linewidth=2.0)
    ax2.axvline(info["F"], color=COLORS["main"], linewidth=1.2, linestyle="--")
    ax2.text(info["F"], ax2.get_ylim()[0], "  на деньгах", color=COLORS["main"],
             fontsize=11, va="bottom")
    for K in (CALL_STRIKE, PUT_STRIKE):
        iv = float(np.interp(K, sub["STRIKE"], sub["IV"])) * 100
        ax2.scatter([K], [iv], color=COLORS["accent"], s=60, zorder=5)
        ax2.annotate(f"{K}", (K, iv), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=11, color=COLORS["accent"])
    ax2.set_xlabel("страйк")
    ax2.set_ylabel("волатильность, %")
    ax2.set_title("Улыбка волатильности")

    fig.suptitle("Опционы Si, модель Блэка-76", fontsize=18)
    fig.subplots_adjust(top=0.86, bottom=0.2, wspace=0.28)
    fig.text(0.5, 0.04,
             f"Фьючерс F={info['F']:.0f}, до экспирации {info['T']*365:.0f} дней, "
             f"волатильность на деньгах {info['sigma_atm']*100:.1f}%",
             ha="center", fontsize=12, color="#555555")

    return save_slide(fig, "bonus_options", out_dir)
