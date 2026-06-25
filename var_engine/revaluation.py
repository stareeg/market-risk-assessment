"""
Переоценка портфеля по сценариям факторов (пункт 5).

Из приращений факторов получаем дневную доходность каждого инструмента:
  облигации  полная переоценка по сдвинутой кривой (модуль pricing.curve и bonds),
  акции      факторная модель на EQ-факторах плюс собственный шок,
  валюта     курс мультипликативно по своему фактору.

Считаем именно дневные простые доходности, потому что в пункте 5 нужна ежедневная
ребалансировка: портфель каждый день возвращают к целевым пропорциям, поэтому
важна доходность за каждый день, а не один скачок на весь горизонт.

Кривую двигаем строго по требованию задания: сдвиг ставок это PCA-нагрузки на
приращения RATE_PC. Облигации переоцениваем по всей кривой. Чтобы не гонять
сплайн в цикле, один раз строим линейную карту (ставки в узлах в ставки на сроки
потоков): кубический сплайн линеен по значениям в узлах, поэтому такая карта
точная. Дальше переоценка облигаций это умножение матриц, векторно по сценариям.

Дату оценки в потоках держим фиксированной (на 2 декабря 2025): на горизонте
1-10 дней набегающий купонный доход мал и детерминирован, а VaR измеряет риск от
движения рынка, то есть от сдвигов кривой. Это стандартное упрощение.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from pricing import bonds as B
from pricing import curve as CV

pd.options.io.parquet.engine = "fastparquet"

# Индексы RATE и EQ факторов внутри вектора факторов задаём по именам, чтобы не
# зависеть от их позиции.
RATE_NAMES = ["RATE_PC1", "RATE_PC2", "RATE_PC3"]
EQ_NAMES = ["EQ_PC1", "EQ_PC2", "EQ_PC3"]


def _bond_interp_matrix(times: np.ndarray) -> np.ndarray:
    """
    Линейная карта: ставки в 12 узлах кривой в ставки на сроки потоков times.
    Кубический сплайн линеен по значениям в узлах, поэтому карту строим прогоном
    базисных векторов (в одном узле единица, в остальных ноль).
    """
    n_nodes = len(CV.TENOR_YEARS)
    M = np.empty((len(times), n_nodes))
    for j in range(n_nodes):
        e = np.zeros(n_nodes)
        e[j] = 1.0
        M[:, j] = CV.yield_at(e, times)
    return M


def build_bond_pricers(data_dir: Path, eval_date: pd.Timestamp,
                       base_node_yields: np.ndarray):
    """
    Готовит данные для быстрой переоценки облигаций.
    По каждому выпуску: сроки потоков, суммы, карта сплайна и базовая грязная цена.
    """
    coupons = B._load_coupons(data_dir)
    desc = pd.read_parquet(data_dir / "bonds_descriptions.parquet").set_index("NUMBER")
    desc.index = desc.index.astype(str)

    pricers = []
    for num in C.PORTFOLIO_BONDS:
        maturity = pd.Timestamp(desc.loc[num, "MATDATE"])
        cf = B.future_cashflows(num, coupons, maturity, eval_date)
        times = cf["t"].values
        amounts = cf["amount"].values
        M = _bond_interp_matrix(times)
        base_price = B.price_dirty(base_node_yields, cf)
        pricers.append({"num": num, "times": times, "amounts": amounts,
                        "M": M, "base_price": base_price})
    return pricers


def _bond_prices(curves: np.ndarray, pricer: dict) -> np.ndarray:
    """
    Грязная цена выпуска по набору кривых.
    curves имеет форму (..., 12) в процентах годовых, на выходе цена той же формы
    без последней оси.
    """
    y = curves @ pricer["M"].T                      # ставки на сроки потоков, в процентах
    df = (1.0 + y / 100.0) ** (-pricer["times"])    # дисконт-факторы
    return df @ pricer["amounts"]


def bond_returns(rate_incr: np.ndarray, data_dir: Path, eval_date: pd.Timestamp,
                 base_node_yields: np.ndarray) -> np.ndarray:
    """
    Дневные простые доходности 5 ОФЗ.

    rate_incr это приращения RATE_PC1/2/3, форма (n_sim, horizon, 3). Кривую на
    день d получаем из накопленных приращений: сдвиг ставок (в б.п.) это нагрузки
    на накопленную сумму PC, переводим в проценты делением на 100.
    """
    loadings = pd.read_parquet(data_dir / "rate_pca_loadings.parquet")
    load = loadings.reindex(CV.TENOR_LABELS).values  # 12 x 3

    n_sim, horizon, _ = rate_incr.shape
    # Накопленные приращения PC по дням, с нулём в день 0 (старт).
    cum = np.zeros((n_sim, horizon + 1, 3))
    cum[:, 1:, :] = np.cumsum(rate_incr, axis=1)
    # Кривые на каждый день: базовая плюс сдвиг через нагрузки.
    curves = base_node_yields[None, None, :] + np.einsum("sdk,nk->sdn", cum, load) / 100.0

    pricers = build_bond_pricers(data_dir, eval_date, base_node_yields)
    ret = np.empty((n_sim, horizon, len(pricers)))
    for i, pr in enumerate(pricers):
        price = _bond_prices(curves, pr)            # (n_sim, horizon+1)
        ret[:, :, i] = price[:, 1:] / price[:, :-1] - 1.0
    return ret


def stock_returns(eq_incr: np.ndarray, idio: np.ndarray, data_dir: Path) -> np.ndarray:
    """
    Дневные простые доходности 10 акций.

    Лог-доходность бумаги это факторная модель на трёх EQ-факторах плюс
    собственный шок: r = alpha + beta @ EQ + e. Простая доходность это exp(r) - 1.
    """
    coeff = pd.read_parquet(data_dir / "stock_factors_coeff.parquet").reindex(C.PORTFOLIO_STOCKS)
    alpha = coeff["alpha"].values
    betas = coeff[["beta_EQ1", "beta_EQ2", "beta_EQ3"]].values   # 10 x 3

    factor_part = np.einsum("sdk,jk->sdj", eq_incr, betas)       # (n_sim, horizon, 10)
    logret = alpha[None, None, :] + factor_part + idio
    return np.exp(logret) - 1.0


def fx_returns(fx_usd_incr: np.ndarray, fx_eur_incr: np.ndarray) -> np.ndarray:
    """
    Дневные простые доходности валют в порядке USD, EUR.
    Факторы это лог-доходности курса, поэтому доходность за день это exp(сдвиг) - 1.
    """
    r_usd = np.exp(fx_usd_incr) - 1.0
    r_eur = np.exp(fx_eur_incr) - 1.0
    return np.stack([r_usd, r_eur], axis=-1)


def portfolio_layout():
    """
    Постоянная раскладка портфеля: рублёвый объём по каждому инструменту, индексы
    подпортфелей и подписи. От даты не зависит, поэтому вынесено отдельно (этим же
    пользуется бэктестинг в пункте 6).
    """
    notional = np.array(
        [C.BOND_NOTIONAL_RUB] * len(C.PORTFOLIO_BONDS)
        + [C.STOCK_NOTIONAL_RUB] * len(C.PORTFOLIO_STOCKS)
        + [C.FX_NOTIONAL_RUB["USD"], C.FX_NOTIONAL_RUB["EUR"]],
        dtype=float)

    n_b = len(C.PORTFOLIO_BONDS)
    n_s = len(C.PORTFOLIO_STOCKS)
    groups = {
        "Облигации": list(range(0, n_b)),
        "Акции": list(range(n_b, n_b + n_s)),
        "Валюта": list(range(n_b + n_s, n_b + n_s + 2)),
        "Портфель": list(range(0, n_b + n_s + 2)),
    }
    labels = (list(C.PORTFOLIO_BONDS)
              + list(C.PORTFOLIO_STOCKS)
              + ["USD", "EUR"])
    return notional, groups, labels


def instrument_returns(incr: np.ndarray, idio: np.ndarray, info: dict,
                       data_dir: str | Path | None = None):
    """
    Собирает дневные доходности всех инструментов портфеля.

    Возвращает:
      ret        (n_sim, horizon, 17): 5 ОФЗ, 10 акций, USD, EUR,
      notional   рублёвый объём по каждому инструменту (17,),
      groups     индексы инструментов по подпортфелям,
      labels     подписи инструментов.
    """
    data_dir = Path(data_dir) if data_dir is not None else C.DATA_DIR
    # Дату прогноза берём из симуляции (по умолчанию дата оценки из config).
    # В бэктестинге сюда приходит произвольный торговый день 2025 года.
    eval_date = pd.Timestamp(info.get("as_of", C.EVAL_DATE))
    order = info["factor_order"]

    rate_idx = [order.index(n) for n in RATE_NAMES]
    eq_idx = [order.index(n) for n in EQ_NAMES]
    usd_idx = order.index("FX_USD")
    eur_idx = order.index("FX_EUR")

    # Кривая ЦБ на дату прогноза. На дату оценки совпадает с base_curve.parquet,
    # для остальных дней берёт исторический срез (нужно для бэктестинга).
    base_node_yields = CV.load_base_curve(data_dir, eval_date).values

    r_bonds = bond_returns(incr[:, :, rate_idx], data_dir, eval_date, base_node_yields)
    r_stocks = stock_returns(incr[:, :, eq_idx], idio, data_dir)
    r_fx = fx_returns(incr[:, :, usd_idx], incr[:, :, eur_idx])

    ret = np.concatenate([r_bonds, r_stocks, r_fx], axis=-1)

    notional, groups, labels = portfolio_layout()
    return ret, notional, groups, labels
