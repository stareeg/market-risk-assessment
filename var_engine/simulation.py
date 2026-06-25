"""
Симуляция риск-факторов методом Monte Carlo (пункт 5).

Гоняем те же 8 PCA-факторов, что и в пунктах 2-3, по модели GARCH(1,1)-t плюс DCC.
На каждый день горизонта:
  обновляем условную дисперсию каждого фактора (GARCH),
  обновляем матрицу Q и из неё корреляцию (DCC),
  тянем коррелированные инновации с t-хвостами и считаем приращения факторов.
Всё векторно по сценариям, цикл идёт только по дням горизонта (их мало).

Начальное состояние берём по истории строго до даты оценки (2 декабря 2025),
чтобы не заглядывать в будущее. Условную дисперсию и матрицу Q восстанавливаем
прогоном фильтра по этой истории. Для факторов на границе a+b=1 (IGARCH)
безусловная дисперсия бесконечна, поэтому стартовую дисперсию берём именно из
фильтра, последнюю внутривыборочную.

Коррелированные инновации строим гауссовой копулой: тянем нормальные величины с
корреляцией из DCC, прогоняем через нормальную функцию распределения и обратную
функцию Стьюдента с числом степеней свободы своим у каждого фактора. Так держим
и динамическую корреляцию из DCC, и тяжёлые хвосты с разной тяжестью по факторам.

Дополнительно тянем независимые собственные шоки по каждой акции (та часть риска,
которую три EQ-фактора не ловят, около трети дисперсии). Их масштаб и тяжесть
хвоста берём из пункта 4.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist

import config as C

pd.options.io.parquet.engine = "fastparquet"

# Сколько сценариев гоняем и на какой максимальный горизонт.
# Однодневный результат берём из первого шага этой же симуляции.
N_SIM = 50_000
HORIZON = 10

# Тикеры акций берём в порядке портфеля из config.
STOCK_ORDER = C.PORTFOLIO_STOCKS


def load_models(data_dir: Path):
    """
    Читает параметры моделей из пунктов 2-3 и историю факторов.
    Всё приводим к единому порядку факторов (как в garch_params).
    """
    rf = pd.read_parquet(data_dir / "risk_factors.parquet")
    gp = pd.read_parquet(data_dir / "garch_params.parquet")
    factor_order = list(gp.index)
    rf = rf[factor_order]

    dcc = pd.read_parquet(data_dir / "dcc_params.parquet")["value"]
    theta1 = float(dcc.loc["theta1"])
    theta2 = float(dcc.loc["theta2"])
    q_bar = pd.read_parquet(data_dir / "q_bar.parquet").reindex(
        index=factor_order, columns=factor_order).values

    return rf, gp, theta1, theta2, q_bar, factor_order


def _garch_filter(x: np.ndarray, const: float, ar1: float,
                  omega: float, alpha: float, beta: float):
    """
    Прогон одномерного фильтра GARCH(1,1) по ряду фактора.

    Среднее: mean_t = const + ar1 * x_{t-1}, остаток eps_t = x_t - mean_t.
    Дисперсия: h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}.

    Возвращает последний остаток, последнюю дисперсию, стандартизованные остатки
    и прогноз дисперсии на первый день вне выборки. Первую дисперсию берём
    выборочной (бэккаст), после тысячи с лишним шагов её влияние пренебрежимо.
    """
    n = len(x)
    eps = np.empty(n)
    eps[0] = x[0] - const
    eps[1:] = x[1:] - const - ar1 * x[:-1]

    h = np.empty(n)
    h[0] = float(np.var(x))
    for t in range(1, n):
        h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]

    h_next = omega + alpha * eps[-1] ** 2 + beta * h[-1]
    z = eps / np.sqrt(h)
    return eps, h, z, h_next


def filter_state(rf: pd.DataFrame, gp: pd.DataFrame,
                 theta1: float, theta2: float, q_bar: np.ndarray):
    """
    Восстанавливает состояние моделей на первый день прогноза.

    Возвращает:
      h1   прогноз условной дисперсии каждого фактора на первый день,
      Q1   прогноз матрицы Q на первый день (для DCC),
      f_last значения факторов в последний день (затравка для AR в среднем),
      Z    стандартизованные остатки по всей истории (понадобятся внутри).
    """
    cols = list(gp.index)
    k = len(cols)
    Z = np.empty((len(rf), k))
    h1 = np.empty(k)
    for j, col in enumerate(cols):
        p = gp.loc[col]
        _, _, z, h_next = _garch_filter(
            rf[col].values, p["const"], p["ar1"], p["omega"], p["alpha"], p["beta"])
        Z[:, j] = z
        h1[j] = h_next

    # Прогон DCC по стандартизованным остаткам: Q_1 = Qbar,
    # Q_t = (1-t1-t2) Qbar + t1 z_{t-1} z_{t-1}' + t2 Q_{t-1}.
    Q = q_bar.copy()
    for t in range(1, len(Z)):
        z = Z[t - 1]
        Q = (1 - theta1 - theta2) * q_bar + theta1 * np.outer(z, z) + theta2 * Q
    z_last = Z[-1]
    Q1 = (1 - theta1 - theta2) * q_bar + theta1 * np.outer(z_last, z_last) + theta2 * Q

    f_last = rf.iloc[-1].values
    return h1, Q1, f_last, Z


def _correlation_from_q(Q: np.ndarray) -> np.ndarray:
    """Из матриц Q (n_sim x k x k) делает корреляционные матрицы той же формы."""
    d = np.sqrt(np.einsum("sii->si", Q))
    return Q / (d[:, :, None] * d[:, None, :])


def _draw_correlated_t(R: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """
    Тянет стандартизованные инновации Стьюдента (единичная дисперсия) с заданной
    по сценариям корреляцией R через гауссову копулу. nu это число степеней
    свободы по каждому фактору.
    """
    n_sim, k = R.shape[0], R.shape[1]
    L = np.linalg.cholesky(R)
    w = np.random.standard_normal((n_sim, k))
    g = np.einsum("sij,sj->si", L, w)              # нормальные с корреляцией R
    u = norm.cdf(g)
    u = np.clip(u, 1e-12, 1 - 1e-12)               # чтобы обратная Стьюдента не ушла в бесконечность
    scale = np.sqrt((nu - 2.0) / nu)               # приводим дисперсию Стьюдента к единице
    return t_dist.ppf(u, nu) * scale


def simulate(data_dir: str | Path | None = None,
             n_sim: int = N_SIM, horizon: int = HORIZON,
             seed: int = C.RANDOM_SEED, as_of: str | pd.Timestamp | None = None):
    """
    Симулирует приращения факторов и собственные шоки акций.

    as_of это дата, на которую делаем прогноз (по умолчанию дата оценки из config).
    Начальное состояние моделей восстанавливаем по истории строго до неё. В
    бэктестинге (пункт 6) сюда подставляют любой торговый день, поэтому параметр
    вынесен наружу.

    Возвращает:
      incr   приращения 8 факторов, массив (n_sim, horizon, 8),
      idio   собственные шоки 10 акций (лог-доходности), (n_sim, horizon, 10),
      info   словарь со служебными данными (порядок факторов, q_bar и т.п.).
    """
    data_dir = Path(data_dir) if data_dir is not None else C.DATA_DIR
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(C.EVAL_DATE)

    rf_full, gp, theta1, theta2, q_bar, factor_order = load_models(data_dir)
    # История строго до даты прогноза, иначе заглянем в будущее.
    rf = rf_full.loc[:as_of]

    h1, Q1, f_last, _ = filter_state(rf, gp, theta1, theta2, q_bar)

    const = gp["const"].values
    ar1 = gp["ar1"].values
    omega = gp["omega"].values
    alpha = gp["alpha"].values
    beta = gp["beta"].values
    nu = gp["nu"].values

    np.random.seed(seed)
    k = len(factor_order)

    # Стартовое состояние, размноженное по сценариям.
    h = np.tile(h1, (n_sim, 1))                    # условная дисперсия на текущий день
    Q = np.tile(Q1, (n_sim, 1, 1))
    f_prev = np.tile(f_last, (n_sim, 1))
    eps_prev = None
    z_prev = None

    incr = np.empty((n_sim, horizon, k))
    for d in range(horizon):
        if d > 0:
            # Обновляем дисперсию и Q по вчерашним остаткам.
            h = omega + alpha * eps_prev ** 2 + beta * h
            zz = z_prev[:, :, None] * z_prev[:, None, :]
            Q = (1 - theta1 - theta2) * q_bar + theta1 * zz + theta2 * Q

        R = _correlation_from_q(Q)
        z = _draw_correlated_t(R, nu)
        eps = np.sqrt(h) * z
        f = const + ar1 * f_prev + eps             # приращение фактора за день
        incr[:, d, :] = f

        f_prev = f
        eps_prev = eps
        z_prev = z

    idio = _draw_idiosyncratic(data_dir, n_sim, horizon)

    info = {"factor_order": factor_order, "q_bar": q_bar,
            "theta1": theta1, "theta2": theta2, "n_sim": n_sim, "horizon": horizon,
            "as_of": as_of}
    return incr, idio, info


def _draw_idiosyncratic(data_dir: Path, n_sim: int, horizon: int) -> np.ndarray:
    """
    Собственные шоки по каждой акции: распределение Стьюдента со своим числом
    степеней свободы, масштаб это волатильность остатка из пункта 4. Независимы
    по бумагам и по дням. Возвращает (n_sim, horizon, число_акций).
    """
    idio = pd.read_parquet(data_dir / "stock_idio_vol.parquet").reindex(STOCK_ORDER)
    vol = idio["idio_vol"].values
    dof = idio["idio_dof"].values
    raw = np.random.standard_t(dof, size=(n_sim, horizon, len(STOCK_ORDER)))
    scale = np.sqrt((dof - 2.0) / dof)             # единичная дисперсия стандартизованной Стьюдента
    return raw * scale * vol
