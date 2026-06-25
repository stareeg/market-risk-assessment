"""
Этап 4. Оценка справедливой стоимости инструментов портфеля.

Читает кривую, цены и риск-факторы из предыдущих этапов и считает цены всех
инструментов как функцию риск-факторов:
  облигации  через дисконтирование потоков по кривой (модуль bonds, curve),
  акции      через факторную модель на EQ-факторах (модуль stocks),
  валюта     прямой переоценкой по курсу (модуль fx).

Тут же проверка точности из задания: модельная цена против рыночной. И тут же
дописываем недостающие для воспроизводимости файлы: коэффициенты факторной
модели, волатильность остатка акций, последние цены, базовая кривая и курсы.

Запуск: python main.py --stage pricing
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from . import bonds as B
from . import stocks as S
from . import fx as X
from . import curve as CV

pd.options.io.parquet.engine = "fastparquet"


def _show(title: str, df, ndigits: int = 4) -> None:
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


def run(data_dir: str | Path | None = None) -> None:
    """
    Оценивает портфель на дату оценки по данным из data_dir и сохраняет туда же
    параметры и снимки цен. По умолчанию работаем с config.DATA_DIR.
    """
    data_dir = Path(data_dir) if data_dir is not None else C.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(C.RANDOM_SEED)   # фиксируем seed (воспроизводимость)
    eval_date = pd.Timestamp(C.EVAL_DATE)
    print(f"Каталог данных: {data_dir}")
    print(f"Дата оценки: {eval_date.date()}")

    # Базовая кривая на дату оценки (12 узлов).
    base_curve = CV.load_base_curve(data_dir, eval_date)
    node_yields = base_curve.values
    print("\nКривая ЦБ на дату оценки (% годовых):")
    print(base_curve.round(2).to_string())

    # Облигации: дисконтирование потоков и сверка с рынком.
    bond_res = B.price_portfolio(data_dir, eval_date, node_yields)
    _show("Облигации, модельная цена против рыночной:", bond_res, 3)
    acc = B.accuracy(bond_res)
    print(f"\nТочность облигаций: RMSE={acc['RMSE_pct']:.2f}%, "
          f"MAPE={acc['MAPE_pct']:.2f}%, макс. ошибка={acc['max_abs_err_pct']:.2f}%")
    worst = bond_res["err_pct"].abs().idxmax()
    print(f"Худший выпуск {worst} (длинный конец), ошибка {bond_res.loc[worst, 'err_pct']:.2f}%. "
          "Это базис между подогнанной кривой ЦБ и ценой конкретной длинной ОФЗ.")

    # Акции: факторная модель и доля объяснённой дисперсии.
    rf = pd.read_parquet(data_dir / "risk_factors.parquet")
    ret = S.portfolio_returns(data_dir, rf.index)
    coeff, idio, resid = S.fit_factor_model(ret, rf)
    _show("Акции, коэффициенты факторной модели и R2:", coeff, 4)
    print(f"\nСредний R2 = {coeff['R2'].mean():.3f}. Около {(1 - coeff['R2'].mean()) * 100:.0f}% "
          "дисперсии это идиосинкразия, её вернём собственным шоком в п.5.")
    _show("Акции, параметры остатка (волатильность и хвост t):", idio, 4)

    # Остатки разных бумаг должны слабо коррелировать (свойство факторной модели).
    rc = resid.corr()
    abs_off = rc.abs().where(~np.eye(len(rc), dtype=bool)).stack()
    a, b = abs_off.idxmax()
    print(f"\nОстатки бумаг: средняя |корреляция| {abs_off.mean():.2f}, "
          f"максимум {abs_off.max():.2f} ({a} и {b}).")
    print("Большинство близко к нулю, но у пары бумаг связь около 0.5 (секторная структура,")
    print("которую три рыночных фактора не снимают). Собственные шоки в п.5 берём независимыми,")
    print("это осознанное упрощение.")

    # Валюта: курсы и количество валюты в позиции.
    fx_rates = X.base_fx(data_dir, eval_date)
    fx_pos = X.positions(fx_rates)
    fx_tab = pd.DataFrame({"rate": fx_rates, "units": fx_pos,
                           "rub": pd.Series(C.FX_NOTIONAL_RUB)})
    _show("Валюта, курс и позиция:", fx_tab, 4)

    # Последние цены для старта симуляции в п.5.
    bond_snap = B.save_last_prices(data_dir, eval_date)
    stock_snap = S.last_prices(data_dir, eval_date)

    # Сохраняем параметры и снимки.
    coeff[["alpha", "beta_EQ1", "beta_EQ2", "beta_EQ3"]].to_parquet(
        data_dir / "stock_factors_coeff.parquet")
    idio.to_parquet(data_dir / "stock_idio_vol.parquet")
    stock_snap.to_parquet(data_dir / "last_stock_prices.parquet")
    base_curve_df = pd.DataFrame({"years": CV.TENOR_YEARS, "yield_pct": base_curve.values},
                                 index=pd.Index(CV.TENOR_LABELS, name="tenor"))
    base_curve_df.to_parquet(data_dir / "base_curve.parquet")
    fx_tab[["rate", "units"]].to_parquet(data_dir / "base_fx.parquet")

    print("\nИтог, сохранённые файлы:")
    print(f"  stock_factors_coeff.parquet  {coeff.shape[0]} акций x (alpha, beta_EQ1, EQ2, EQ3)")
    print(f"  stock_idio_vol.parquet       {idio.shape} (idio_vol, idio_dof)")
    print(f"  last_stock_prices.parquet    {stock_snap.shape}")
    print(f"  last_bond_prices.parquet     {bond_snap.shape}")
    print(f"  base_curve.parquet           {base_curve_df.shape} (кривая на дату оценки)")
    print(f"  base_fx.parquet              курсы USD и EUR на дату оценки")


if __name__ == "__main__":
    run()
