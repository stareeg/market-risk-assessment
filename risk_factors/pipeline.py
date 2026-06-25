"""
Этап 2. Выделение риск-факторов и описательная статистика.

Готовим данные из пункта 1 (сплиты, разрывы, переход к приращениям), сжимаем
кривую и акции через PCA, добавляем валютные факторы и сохраняем итоговый набор
из 8 риск-факторов в data/. Тяжёлая логика лежит в factors.py, pca_tools.py и
diagnostics.py, тут только оркестрация и проверка значений.

Запуск: python main.py --stage factors
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from . import factors as F
from . import pca_tools as P
from . import diagnostics as D

pd.options.io.parquet.engine = "fastparquet"


# Короткие подписи факторов для осей и ячеек тепловых карт, чтобы не наезжали.
FACTOR_LABELS = {
    "RATE_PC1": "Ставки 1", "RATE_PC2": "Ставки 2", "RATE_PC3": "Ставки 3",
    "EQ_PC1": "Акции 1", "EQ_PC2": "Акции 2", "EQ_PC3": "Акции 3",
    "FX_USD": "USD", "FX_EUR": "EUR",
}


def _show(title: str, df: pd.DataFrame, ndigits: int = 4) -> None:
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


def _plot_scree(scree_y: pd.Series, scree_e: pd.Series, out_dir: Path) -> Path:
    """
    Scree-график PCA для кривой и для акций.
    По каждой панели столбики это доля дисперсии на компоненту, линия это
    накопленная доля. Вертикаль на третьей компоненте: мы берём ровно три.
    """
    import matplotlib.pyplot as plt
    from viz.style import set_slide_style, COLORS, save_slide

    set_slide_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, scree, title in (
        (axes[0], scree_y, "Кривая доходности"),
        (axes[1], scree_e, "Акции"),
    ):
        n = min(8, len(scree))
        x = range(1, n + 1)
        share = scree.values[:n] * 100.0
        cum = scree.values[:n].cumsum() * 100.0
        ax.bar(x, share, color=COLORS["main"], alpha=0.8, label="доля компоненты")
        ax.plot(x, cum, color=COLORS["accent"], marker="o", linewidth=2,
                label="накопленная доля")
        ax.axvline(3, color=COLORS["grey"], linestyle="--", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Номер компоненты")
        ax.set_xticks(list(x))
    axes[0].set_ylabel("Доля дисперсии, %")
    axes[0].legend(loc="center right")
    fig.suptitle("Scree-график PCA, берём по три компоненты", y=1.02)
    return save_slide(fig, "factors_scree", out_dir)


def _plot_tails(risk_factors: pd.DataFrame, out_dir: Path) -> Path:
    """
    Тяжёлые хвосты на примере самого тяжелохвостого фактора.
    QQ-график стандартизованного фактора против нормального распределения: на
    хвостах точки уходят от прямой, значит крайние движения случаются чаще, чем
    предсказывает нормальное распределение. Это и есть обоснование выбора t в п.3.
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    from viz.style import set_slide_style, COLORS, save_slide

    set_slide_style()
    # Берём фактор с наибольшим эксцессом (самые тяжёлые хвосты).
    kurt = risk_factors.kurtosis()
    name = kurt.idxmax()
    x = risk_factors[name].dropna()
    z = (x - x.mean()) / x.std()

    n = len(z)
    p = (np.arange(1, n + 1) - 0.5) / n
    theo = stats.norm.ppf(p)
    samp = np.sort(z.values)

    fig, ax = plt.subplots()
    lim = max(abs(theo[0]), abs(theo[-1]), abs(samp[0]), abs(samp[-1]))
    ax.plot([-lim, lim], [-lim, lim], color=COLORS["grey"], linestyle="--",
            linewidth=1.5, label="нормальное распределение")
    ax.scatter(theo, samp, s=14, color=COLORS["main"], label="фактор " + name)
    ax.set_title(f"Тяжёлые хвосты фактора {name} (эксцесс {kurt[name]:.1f})")
    ax.set_xlabel("Квантили нормального распределения")
    ax.set_ylabel("Квантили фактора")
    ax.legend(loc="upper left")
    return save_slide(fig, "factors_tails", out_dir)


def _plot_stocks_dynamics(stock_px_adj: pd.DataFrame, out_dir: Path) -> Path:
    """
    Динамика 10 акций портфеля, каждая нормирована к 100 на первый день.
    Рисуем сеткой мелких панелей: так десять рядов читаются со слайда, а
    нормировка делает их сравнимыми по форме. Цены уже очищены от сплитов,
    иначе у Полюса день дробления дал бы ложный обвал в 10 раз.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from viz.style import set_slide_style, COLORS, save_slide

    set_slide_style()
    tickers = C.PORTFOLIO_STOCKS
    px = stock_px_adj[tickers]
    base = px.apply(lambda s: s[s.first_valid_index()])   # цена в первый торговый день
    norm = px.divide(base) * 100.0

    fig, axes = plt.subplots(2, 5, figsize=(14, 6), sharex=True)
    for ax, tk in zip(axes.flat, tickers):
        ax.plot(norm.index, norm[tk], color=COLORS["main"], linewidth=1.3)
        ax.axhline(100, color=COLORS["grey"], linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(tk, fontsize=13)
        ax.tick_params(labelsize=10)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Динамика 10 акций портфеля, нормировано к 100 на старте",
                 y=1.02, fontsize=15)
    return save_slide(fig, "data_stocks_dynamics", out_dir)


def _plot_yield_curve(yields: pd.DataFrame, out_dir: Path) -> Path:
    """
    Кривая бескупонной доходности на нескольких датах.
    Доходность как функция срока в разные моменты: начало периода, шок 2022,
    спокойный год и дата оценки. Видно, как меняются и уровень, и форма кривой.
    Это подводка к PCA кривой: уровень, наклон, кривизна.
    """
    import matplotlib.pyplot as plt
    from viz.style import set_slide_style, COLORS, save_slide
    from .factors import TENOR_YEARS

    set_slide_style()
    idx = yields.index
    targets = ["2021-02-01", "2022-03-31", "2023-09-01", C.EVAL_DATE]
    colors = [COLORS["second"], COLORS["accent"], COLORS["grey"], COLORS["main"]]

    fig, ax = plt.subplots()
    for t, col in zip(targets, colors):
        # берём ближайшую доступную дату, ЦБ публикует по своему календарю
        pos = idx.get_indexer([pd.Timestamp(t)], method="nearest")[0]
        d = idx[pos]
        ax.plot(TENOR_YEARS, yields.loc[d].values, marker="o", color=col,
                linewidth=2, label=d.strftime("%d.%m.%Y"))
    ax.set_title("Кривая бескупонной доходности в разные моменты")
    ax.set_xlabel("Срок, лет")
    ax.set_ylabel("Доходность, % годовых")
    ax.legend(title="дата", loc="best")
    return save_slide(fig, "data_yield_curve", out_dir)


def _plot_kurtosis(risk_factors: pd.DataFrame, out_dir: Path) -> Path:
    """
    Тяжесть хвостов сразу по всем факторам.
    Слева избыточный эксцесс: у нормального он равен нулю, у нас везде заметно
    больше. Справа индекс Хилла: у финансовых рядов он 2-4, чем меньше, тем
    тяжелее хвост. Вместе это обоснование выбора t-распределения в п.3.
    """
    import matplotlib.pyplot as plt
    from viz.style import set_slide_style, COLORS, save_slide
    from . import diagnostics as D

    set_slide_style()
    cols = list(risk_factors.columns)
    labels = [FACTOR_LABELS[c] for c in cols]
    tt = D.tail_table(risk_factors).loc[cols]
    exc = tt["exc_kurt"].values
    hill = tt["hill_alpha"].values
    x = np.arange(len(cols))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].bar(x, exc, color=COLORS["main"])
    axes[0].axhline(0, color=COLORS["accent"], linewidth=1.5, linestyle="--")
    axes[0].text(len(cols) - 1, 0.2, "нормальное распределение",
                 color=COLORS["accent"], va="bottom", ha="right", fontsize=11)
    axes[0].set_title("Избыточный эксцесс (больше - хвост тяжелее)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right")

    axes[1].bar(x, hill, color=COLORS["second"])
    axes[1].axhspan(2, 4, color=COLORS["grey"], alpha=0.2)
    axes[1].set_title("Индекс Хилла (2-4 это тяжёлые хвосты)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")

    fig.suptitle("У всех факторов хвосты тяжелее нормального распределения",
                 y=1.03, fontsize=15)
    return save_slide(fig, "factors_kurtosis", out_dir)


def _plot_acf(risk_factors: pd.DataFrame, out_dir: Path, factor: str = "EQ_PC1",
              nlags: int = 20) -> Path:
    """
    Автокорреляция фактора в доходностях и в их квадратах.
    Слева сами доходности: автокорреляция почти нулевая, линейной памяти нет.
    Справа квадраты доходностей: автокорреляция держится много дней. Это и есть
    кластеризация волатильности, прямой аргумент в пользу GARCH.
    """
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import acf
    from viz.style import set_slide_style, COLORS, save_slide

    set_slide_style()
    x = risk_factors[factor].dropna().values
    n = len(x)
    a_ret = acf(x, nlags=nlags, fft=True)[1:]
    a_sq = acf(x ** 2, nlags=nlags, fft=True)[1:]
    conf = 1.96 / np.sqrt(n)   # граница значимости автокорреляции
    lags = np.arange(1, nlags + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, vals, title in (
        (axes[0], a_ret, "Доходности фактора"),
        (axes[1], a_sq, "Квадраты доходностей"),
    ):
        ax.bar(lags, vals, color=COLORS["main"], width=0.7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(conf, color=COLORS["accent"], linestyle="--", linewidth=1.2)
        ax.axhline(-conf, color=COLORS["accent"], linestyle="--", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Лаг, дней")
    axes[0].set_ylabel("Автокорреляция")
    fig.suptitle(f"Фактор {factor}: в доходностях памяти нет, в квадратах есть "
                 "(кластеризация волатильности)", y=1.02, fontsize=14)
    return save_slide(fig, "factors_acf", out_dir)


def _plot_rolling_corr(risk_factors: pd.DataFrame, out_dir: Path,
                       a: str = "EQ_PC1", b: str = "FX_USD", window: int = 60) -> Path:
    """
    Скользящая корреляция пары факторов во времени.
    Если бы связь была постоянной, линия держалась бы у одного уровня. Она же
    заметно гуляет, поэтому статической корреляции мало и нужна надстройка DCC
    поверх GARCH.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from viz.style import set_slide_style, COLORS, save_slide

    set_slide_style()
    pair = risk_factors[[a, b]].dropna()
    rc = pair[a].rolling(window).corr(pair[b]).dropna()
    full = pair[a].corr(pair[b])

    fig, ax = plt.subplots()
    ax.plot(rc.index, rc.values, color=COLORS["main"], linewidth=1.6)
    ax.axhline(full, color=COLORS["grey"], linestyle="--", linewidth=1.5,
               label=f"корреляция за весь период {full:.2f}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.set_title(f"Скользящая корреляция {a} и {b}, окно {window} дней")
    ax.set_xlabel("Год")
    ax.set_ylabel("Корреляция")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return save_slide(fig, "factors_rolling_corr", out_dir)


def _plot_corr_regimes(risk_factors: pd.DataFrame, out_dir: Path) -> Path:
    """
    Корреляции факторов в кризис и в спокойный период двумя картами рядом.
    Структура связей между режимами разная (в кризис связи обычно сильнее),
    значит постоянная корреляционная матрица неадекватна. Это наглядное
    оправдание DCC.
    """
    import matplotlib.pyplot as plt
    from viz.style import set_slide_style, save_slide

    set_slide_style()
    labels = [FACTOR_LABELS[c] for c in risk_factors.columns]
    crisis = risk_factors.loc["2022-02-01":"2022-12-31"]
    calm = risk_factors.loc["2024-01-01":"2025-12-31"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.set_layout_engine("constrained")   # ровно разложит две карты и общую шкалу
    im = None
    for ax, sub, title in (
        (axes[0], crisis, "Кризис 2022"),
        (axes[1], calm, "Спокойный период 2024-2025"),
    ):
        m = sub.corr().values
        im = ax.imshow(m, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=11)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(m[i, j]) > 0.6 else "black")
        ax.set_title(title)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="корреляция")
    fig.suptitle("Корреляции факторов меняются между режимами, поэтому нужна DCC",
                 fontsize=14)
    return save_slide(fig, "factors_corr_regimes", out_dir)


def _plot_curve_loadings(loadings: pd.DataFrame, interp: pd.DataFrame,
                         out_dir: Path) -> Path:
    """
    Нагрузки первых трёх компонент кривой по срокам.
    Первая компонента сдвигает всю кривую (уровень), вторая разворачивает
    короткий и длинный конец в разные стороны (наклон), третья выгибает
    середину (кривизна). Поэтому трёх компонент хватает.
    """
    import matplotlib.pyplot as plt
    from viz.style import set_slide_style, COLORS, save_slide
    from .factors import TENOR_YEARS

    set_slide_style()
    L = loadings.copy()
    for pc in L.columns:
        # ориентируем знак так, чтобы наибольшая по модулю нагрузка была вверх,
        # иначе PCA может выдать перевёрнутую, но эквивалентную компоненту
        imax = L[pc].abs().idxmax()
        if L.loc[imax, pc] < 0:
            L[pc] = -L[pc]

    colors = [COLORS["main"], COLORS["accent"], COLORS["second"]]
    fig, ax = plt.subplots()
    for pc, col in zip(["PC1", "PC2", "PC3"], colors):
        name = interp.loc[pc, "interpretation"]
        ax.plot(TENOR_YEARS, L[pc].values, marker="o", color=col, linewidth=2,
                label=f"{pc}: {name}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Нагрузки PCA кривой: уровень, наклон, кривизна")
    ax.set_xlabel("Срок, лет")
    ax.set_ylabel("Нагрузка")
    ax.legend(loc="best")
    return save_slide(fig, "factors_curve_loadings", out_dir)


def run(data_dir: str | Path | None = None) -> None:
    """
    Считает риск-факторы по данным из data_dir и сохраняет их туда же.
    По умолчанию работаем с config.DATA_DIR. Можно передать свою папку, чтобы
    проверить расчёт, не трогая уже сохранённые файлы.
    """
    data_dir = Path(data_dir) if data_dir is not None else C.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(C.RANDOM_SEED)   # фиксируем seed (воспроизводимость)
    panels = F.load_panels(data_dir)
    print(f"Каталог данных: {data_dir}")
    print("Размеры загруженных панелей:", {k: v.shape for k, v in panels.items()})

    # Чистим цены акций от сплитов, иначе день дробления даёт ложный обвал
    # доходности в N раз.
    splits = F.detect_splits(panels["stock_px"])
    _show("Найденные сплиты (реальные корпоративные события):",
          splits.set_index("TICKER") if not splits.empty else splits)
    stock_px_adj, _ = F.adjust_splits(panels["stock_px"], splits)

    # Единый торговый календарь MOEX по дням, когда торговался широкий рынок акций.
    # Курсы и кривую ЦБ протягиваем на него: опубликованное значение действует до
    # следующей публикации, иначе праздники ЦБ выбивали бы часть наблюдений.
    master = stock_px_adj.dropna(how="all").index
    fx_lvl = F.to_trading_calendar(panels["fx_px"], master)
    yield_lvl = F.to_trading_calendar(panels["yields"], master)

    # Переходим к приращениям: лог-доходности цен и изменения ставок в б.п.
    stock_ret = F.log_returns(stock_px_adj)
    fx_ret = F.log_returns(fx_lvl)
    brent_ret = F.log_returns(F.to_trading_calendar(panels["brent"], master))
    index_ret = F.log_returns(panels["index_px"])
    dY = F.yield_changes(yield_lvl)

    n_gap = int((~F.mask_calendar_gaps(stock_px_adj)).sum())
    print(f"\nТорговых дней в общем календаре: {len(master)}")
    print(f"Исключено доходностей через длинные разрывы (включая остановку биржи в 2022): {n_gap}")

    # PCA кривой по изменениям ставок (ковариация, единицы одинаковые, б.п.).
    # Три компоненты это уровень, наклон, кривизна.
    scree_y = P.full_scree(dY)
    model_y, scores_y, load_y, evr_y = P.pca_on_changes(dY, n_components=3)
    print("\nPCA кривой, доля объяснённой дисперсии по компонентам:",
          evr_y.round(4).to_dict())
    print(f"  три компоненты вместе объясняют {evr_y.sum():.1%} дисперсии")
    _show("Интерпретация компонент кривой:", P.interpret_curve_pcs(load_y))

    # PCA акций по корреляциям (доходности стандартизуем, волатильности разные).
    # Структуру проекта держим на 8 факторах, поэтому берём ровно 3 компоненты,
    # критерий Кайзера печатаем как справку.
    eq = stock_ret.dropna()
    scree_e = P.full_scree(eq, standardize=True)
    eigvals = scree_e.values * eq.shape[1]
    k_kaiser = int((eigvals > 1).sum())
    cum = {f"PC{i+1}": round(scree_e.head(i + 1).sum(), 3) for i in range(min(6, len(scree_e)))}
    print(f"\nКритерий Кайзера для акций: компонент с собственным числом больше 1: {k_kaiser}")
    print("Накопленная дисперсия акций по компонентам:", cum)

    model_e, scores_e, load_e, evr_e = P.pca_on_changes(eq, n_components=3, standardize=True)
    print("PCA акций, доля объяснённой дисперсии по компонентам:",
          evr_e.round(4).to_dict())
    print(f"  три компоненты вместе объясняют {evr_e.sum():.1%} дисперсии")

    # Проверяем, что PC1 акций это рыночный фактор (сравниваем с доходностью IMOEX)
    imoex_ret = index_ret["IMOEX"].reindex(scores_e.index)
    corr_pc1_imoex = scores_e["PC1"].corr(imoex_ret)
    print(f"corr(PC1 акций, доходность IMOEX) = {corr_pc1_imoex:.3f}. "
          "Значит PC1 это рыночный фактор")

    # Проверяем избыточность РТС и роль Brent (почему их не берём факторами)
    piv = panels["index_px"].join(panels["fx_px"], how="inner").dropna()
    ratio = piv["IMOEX"] / piv["RTSI"]
    print(f"\ncorr(IMOEX/RTSI, USD/RUB) = {ratio.corr(piv['USD']):.3f}. "
          "Значит РТС это IMOEX в долларах, как фактор избыточен")
    common = stock_ret.join(brent_ret, how="inner").dropna()
    bcorr = common.corr()["BRENT"].drop("BRENT").sort_values(ascending=False)
    _show("Корреляция доходности Brent с акциями (топ-5):", bcorr.head(5))

    # Собираем финальный набор из 8 факторов, выравниваем по общему календарю
    rate_f = scores_y.add_prefix("RATE_")
    eq_f = scores_e.add_prefix("EQ_")
    fx_f = fx_ret.rename(columns={"USD": "FX_USD", "EUR": "FX_EUR"})
    aligned = F.align_on_common_dates(rate_f, eq_f, fx_f)
    risk_factors = pd.concat(aligned, axis=1).dropna()
    print(f"\nИтоговая матрица риск-факторов: {risk_factors.shape}")
    print("Факторы:", list(risk_factors.columns))

    _show("Корреляции риск-факторов:", risk_factors.corr(), ndigits=2)

    # Тяжесть хвостов по каждому фактору (важно для выбора модели в п.3)
    tt = D.tail_table(risk_factors)
    _show("Хвосты распределений факторов:",
          tt[["std", "skew", "exc_kurt", "JB_p", "t_dof", "hill_alpha"]], ndigits=3)
    print("Везде JB_p близко к 0, эксцесс заметно больше 0, t_dof мал, значит хвосты тяжёлые.")

    # Стационарность: уровни нестационарны, приращения стационарны
    levels = pd.concat([
        panels["yields"]["10"].rename("yield_10y_level"),
        np.log(panels["brent"]["BRENT"]).rename("log_brent_level"),
        np.log(panels["fx_px"]["USD"]).rename("log_usd_level"),
    ], axis=1)
    changes = pd.concat([
        dY["10"].rename("yield_10y_chg"),
        brent_ret["BRENT"].rename("brent_ret"),
        fx_ret["USD"].rename("usd_ret"),
    ], axis=1)
    _show("Стационарность уровней:", D.stationarity_table(levels))
    _show("Стационарность приращений:", D.stationarity_table(changes))

    # Недельная сезонность доходностей факторов (проверяем, что её почти нет)
    _show("Сезонность факторов по дням недели:",
          D.weekday_seasonality(risk_factors), ndigits=5)

    # Графики для слайдов. Обзор входных данных рисуем здесь же, а не на этапе
    # загрузки: цены уже очищены от сплитов, а перерисовка из parquet быстрая.
    fig_dir = C.PROJECT_DIR / "docs" / "figures"
    figs = [
        ("scree PCA", _plot_scree(scree_y, scree_e, fig_dir)),
        ("тяжёлые хвосты", _plot_tails(risk_factors, fig_dir)),
        ("динамика акций", _plot_stocks_dynamics(stock_px_adj, fig_dir)),
        ("кривая доходности", _plot_yield_curve(panels["yields"], fig_dir)),
        ("эксцесс и Хилл", _plot_kurtosis(risk_factors, fig_dir)),
        ("ACF доходностей", _plot_acf(risk_factors, fig_dir)),
        ("скользящая корреляция", _plot_rolling_corr(risk_factors, fig_dir)),
        ("корреляции по режимам", _plot_corr_regimes(risk_factors, fig_dir)),
        ("нагрузки PCA кривой",
         _plot_curve_loadings(load_y, P.interpret_curve_pcs(load_y), fig_dir)),
    ]
    print()
    for label, path in figs:
        print(f"График {label}: {path}")

    # Сохраняем итог и нагрузки PCA, понадобятся в пунктах 3-5
    rf_path = data_dir / "risk_factors.parquet"
    ry_path = data_dir / "rate_pca_loadings.parquet"
    re_path = data_dir / "equity_pca_loadings.parquet"
    risk_factors.to_parquet(rf_path)
    load_y.to_parquet(ry_path)
    load_e.to_parquet(re_path)

    print("\nИтог, сохранённые файлы:")
    print(f"  risk_factors.parquet         {risk_factors.shape}, колонки {list(risk_factors.columns)}")
    print(f"  rate_pca_loadings.parquet    {load_y.shape}")
    print(f"  equity_pca_loadings.parquet  {load_e.shape}")


if __name__ == "__main__":
    run()
