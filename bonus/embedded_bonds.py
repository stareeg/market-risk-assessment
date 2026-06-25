"""
Бонусный портфель 2. Облигации со встроенными опционами (пункт 8).

Берём одну портфельную ОФЗ (длинную 26230) и строим два гипотетических варианта,
одинаковых по купонам и погашению, но с правом досрочного выкупа 01.01.2026 по
цене 100% номинала:
  с офертой put - право инвестора продать облигацию эмитенту по номиналу,
  отзывную call - право эмитента выкупить облигацию у инвестора по номиналу.

Цена облигации с опционом это цена обычной облигации плюс или минус стоимость
встроенного опциона. Опцион оцениваем короткой моделью ставок на биномиальной
решётке Блэка-Дермана-Тоя (BDT). Решётку калибруем на текущую кривую ЦБ, чтобы
обычная облигация на решётке совпадала с оценкой по кривой из пункта 4. Это и есть
проверка, что решётка построена верно.

Логика выкупа на дату оферты:
  put - инвестор выберет максимум из стоимости удержания и номинала, поэтому цена с
        офертой put не ниже обычной,
  call - эмитент оставит инвестору минимум из стоимости удержания и номинала,
         поэтому отзывная не дороже обычной.

При нынешних ставках длинная ОФЗ стоит заметно ниже номинала, поэтому put глубоко в
деньгах (инвестор почти наверняка предъявит к выкупу), а call вне денег (эмитенту
невыгодно выкупать дороже рынка). Это видно в результатах.

Проверка (пункт 8b): сравниваем цену с опционом с ценой обычной ОФЗ, показываем
сдвиг.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from pricing import bonds as B
from pricing import curve as CV

pd.options.io.parquet.engine = "fastparquet"

# Какая ОФЗ и параметры встроенного опциона.
BOND_NUMBER = "26230"
OPTION_DATE = pd.Timestamp("2026-01-01")
STRIKE_PCT = 100.0          # страйк в процентах номинала
FACE = B.FACE_VALUE


def estimate_short_rate_vol(data_dir: Path, start: str = "2023-01-01") -> float:
    """
    Волатильность короткой ставки для решётки (логнормальная, годовая).
    Берём ставку ЦБ на 0.25 года, считаем стандартное отклонение дневных
    логарифмических изменений и приводим к году. Считаем по недавнему периоду
    (с 2023 года), чтобы не тащить разовый скачок ставок 2022 года.
    """
    z = pd.read_parquet(data_dir / "zcyc_cbr.parquet").set_index("DATE").sort_index()
    r = z["0,25"].astype(float) / 100.0
    dl = np.log(r).diff().dropna()
    dl = dl[dl.index >= start]
    return float(dl.std() * np.sqrt(252))


def build_grid(num: str, eval_date: pd.Timestamp, data_dir: Path):
    """
    Строит временную сетку решётки и денежные потоки на её узлах.

    Узлы это дата оценки, дата оферты и все даты будущих выплат (купоны и
    погашение). Потоки ложатся ровно на свои узлы, поэтому решётка точна по
    срокам. Возвращает сетку в годах, потоки по узлам и номер узла оферты.
    """
    coupons = pd.read_parquet(data_dir / "bonds_coupons.parquet")
    coupons["coupondate"] = pd.to_datetime(coupons["coupondate"])
    coupons["startdate"] = pd.to_datetime(coupons["startdate"])
    desc = pd.read_parquet(data_dir / "bonds_descriptions.parquet")
    desc["NUMBER"] = desc["NUMBER"].astype(str)
    maturity = pd.Timestamp(desc.set_index("NUMBER").loc[num, "MATDATE"])

    cf = B.future_cashflows(num, coupons, maturity, eval_date)

    # Уникальные даты узлов: оценка, оферта, все потоки.
    dates = sorted(set([eval_date, OPTION_DATE] + list(cf["date"])))
    grid_t = np.array([(d - eval_date).days / B.DAY_COUNT for d in dates])

    cf_nodes = np.zeros(len(dates))
    idx_of = {d: i for i, d in enumerate(dates)}
    for d, a in zip(cf["date"], cf["amount"]):
        cf_nodes[idx_of[d]] += a
    opt_step = idx_of[OPTION_DATE]
    return grid_t, cf_nodes, opt_step, coupons


def calibrate_bdt(grid_t: np.ndarray, zero_prices: np.ndarray, sigma: float):
    """
    Калибрует решётку BDT на кривую методом прямой прогонки.

    На каждом шаге подбираем базовую (нижнюю) ставку так, чтобы решётка точно
    воспроизвела цену бескупонной облигации на следующий срок из кривой. Ставка в
    узле растёт от базовой вверх множителем по волатильности. Возвращает
    пошаговые дисконт-факторы по узлам.
    """
    n = len(grid_t) - 1
    Q = np.array([1.0])             # цены Эрроу-Дебре на текущем шаге
    disc_steps = []
    for i in range(n):
        dt = grid_t[i + 1] - grid_t[i]
        spread = np.exp(2.0 * sigma * np.sqrt(dt) * np.arange(i + 1))  # узлы j=0..i
        target = zero_prices[i + 1]

        def pv_gap(u):
            disc = (1.0 + u * spread) ** (-dt)
            return float(np.sum(Q * disc)) - target

        u = brentq(pv_gap, 1e-9, 1e6, maxiter=200)
        disc = (1.0 + u * spread) ** (-dt)
        disc_steps.append(disc)

        # Прогоняем цены Эрроу-Дебре на следующий шаг (вероятности 0.5/0.5).
        q_next = np.zeros(i + 2)
        q_next[:i + 1] += 0.5 * Q * disc      # ход вниз, состояние j
        q_next[1:i + 2] += 0.5 * Q * disc     # ход вверх, состояние j+1
        Q = q_next
    return disc_steps


def value_on_lattice(grid_t, disc_steps, cf_nodes, opt_step, redemption, mode):
    """
    Грязная цена облигации обратной прогонкой по решётке.
    mode: 'straight' обычная, 'put' с офертой инвестора, 'call' отзывная эмитентом.
    """
    n = len(grid_t) - 1
    V = np.full(n + 1, cf_nodes[n])           # узлы погашения, выплата номинала и купона
    for i in range(n - 1, -1, -1):
        disc = disc_steps[i]                  # длина i+1
        cont = 0.5 * disc * (V[:i + 1] + V[1:i + 2])
        node = cf_nodes[i] + cont
        if i == opt_step:
            if mode == "put":
                node = np.maximum(node, redemption)
            elif mode == "call":
                node = np.minimum(node, redemption)
        V = node
    return float(V[0])


def evaluate(data_dir: Path, eval_date: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """
    Оценивает обычную ОФЗ и оба варианта со встроенным опционом, сравнивает.
    """
    sigma = estimate_short_rate_vol(data_dir)
    node_yields = CV.load_base_curve(data_dir, eval_date).values

    grid_t, cf_nodes, opt_step, coupons = build_grid(BOND_NUMBER, eval_date, data_dir)
    zero_prices = np.array([1.0] + [float(CV.discount_factors(node_yields, t))
                                    for t in grid_t[1:]])
    disc_steps = calibrate_bdt(grid_t, zero_prices, sigma)

    accr_today = B.accrued_interest(BOND_NUMBER, coupons, eval_date)
    accr_opt = B.accrued_interest(BOND_NUMBER, coupons, OPTION_DATE)
    redemption = STRIKE_PCT / 100.0 * FACE + accr_opt   # выкуп по номиналу плюс НКД

    dirty = {m: value_on_lattice(grid_t, disc_steps, cf_nodes, opt_step,
                                 redemption, m)
             for m in ("straight", "put", "call")}

    # Чистая цена в процентах номинала.
    clean_pct = {m: (dirty[m] - accr_today) / FACE * 100.0 for m in dirty}

    # Сверка обычной облигации с оценкой по кривой (пункт 4) и с рынком.
    desc = pd.read_parquet(data_dir / "bonds_descriptions.parquet")
    desc["NUMBER"] = desc["NUMBER"].astype(str)
    maturity = pd.Timestamp(desc.set_index("NUMBER").loc[BOND_NUMBER, "MATDATE"])
    cf = B.future_cashflows(BOND_NUMBER, coupons, maturity, eval_date)
    curve_dirty = B.price_dirty(node_yields, cf)
    hist = pd.read_parquet(data_dir / "bonds_history.parquet")
    hist["NUMBER"] = hist["NUMBER"].astype(str)
    day = hist[(hist["TRADEDATE"] == eval_date) &
               (hist["NUMBER"] == BOND_NUMBER)].iloc[0]
    mkt_clean = float(day["CLOSE"])

    rows = [
        {"Вариант": "Обычная ОФЗ", "Чистая цена, %": round(clean_pct["straight"], 2),
         "Сдвиг, п.п.": 0.0},
        {"Вариант": "С офертой put", "Чистая цена, %": round(clean_pct["put"], 2),
         "Сдвиг, п.п.": round(clean_pct["put"] - clean_pct["straight"], 2)},
        {"Вариант": "Отзывная call", "Чистая цена, %": round(clean_pct["call"], 2),
         "Сдвиг, п.п.": round(clean_pct["call"] - clean_pct["straight"], 2)},
    ]
    table = pd.DataFrame(rows)

    info = {
        "bond": BOND_NUMBER,
        "sigma": sigma,
        "n_steps": len(grid_t) - 1,
        "lattice_straight_pct": clean_pct["straight"],
        "curve_straight_pct": (curve_dirty - accr_today) / FACE * 100.0,
        "market_pct": mkt_clean,
        "put_value_pct": clean_pct["put"] - clean_pct["straight"],
        "call_value_pct": clean_pct["call"] - clean_pct["straight"],
        "option_date": OPTION_DATE,
        "eval_date": eval_date,
    }
    return table, info


def plot_embedded(table: pd.DataFrame, info: dict, out_dir: Path):
    """
    График для слайда: чистая цена обычной ОФЗ против цены с офертой put и
    отзывной call, чтобы видеть, в какую сторону и насколько двигает цену опцион.
    """
    import matplotlib.pyplot as plt
    from viz.style import set_slide_style, save_slide, COLORS

    set_slide_style()
    fig, ax = plt.subplots()

    labels = table["Вариант"].tolist()
    vals = table["Чистая цена, %"].values
    colors = [COLORS["grey"], COLORS["second"], COLORS["accent"]]
    bars = ax.bar(labels, vals, color=colors, width=0.6)

    base = float(table.loc[table["Вариант"] == "Обычная ОФЗ", "Чистая цена, %"].iloc[0])
    ax.axhline(base, color=COLORS["grey"], linewidth=1.0, linestyle="--", alpha=0.7)
    for b, v, sh in zip(bars, vals, table["Сдвиг, п.п."].values):
        txt = f"{v:.1f}%" + (f"\n({sh:+.1f} п.п.)" if sh != 0 else "")
        ax.text(b.get_x() + b.get_width() / 2, v, txt, ha="center", va="bottom",
                fontsize=13)

    ax.set_ylabel("чистая цена, % номинала")
    ax.set_title(f"ОФЗ {info['bond']} со встроенным опционом на {info['option_date'].date()}")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.text(0.0, -0.16,
            f"страйк 100%, волатильность короткой ставки {info['sigma']*100:.0f}%, "
            f"обычная на решётке {info['lattice_straight_pct']:.1f}% против "
            f"кривой {info['curve_straight_pct']:.1f}%",
            transform=ax.transAxes, fontsize=11, color="#555555")

    return save_slide(fig, "bonus_embedded_bonds", out_dir)
