"""
Бонус (пункт 8). Оркестрация двух дополнительных портфелей.

Считает портфель опционов (Блэк-76) и облигации со встроенными опционами (решётка
BDT), печатает сравнение с наблюдаемыми ценами, объясняет, как встроить эти
инструменты в основной Monte Carlo, и сохраняет результаты и графики.

Запуск: python main.py --stage bonus
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config as C
from . import options as OPT
from . import embedded_bonds as EB

pd.options.io.parquet.engine = "fastparquet"


def _print_options(table: pd.DataFrame, info: dict) -> None:
    print("Портфель 1. Опционы на фьючерс Si, модель Блэка-76")
    print(f"Дата оценки {pd.Timestamp(info['trade_day']).date()}, "
          f"фьючерс SiZ5 = {info['F']:.0f}, до экспирации "
          f"{pd.Timestamp(info['expiry']).date()} ({info['T']*365:.0f} дней), "
          f"волатильность на деньгах {info['sigma_atm']*100:.2f}%")
    print(table.to_string(index=False))
    print("Расчёт по своей волатильности страйка повторяет наблюдаемую цену (это "
          "проверка реализации). По единой волатильности на деньгах видно влияние "
          "улыбки: в деньгах волатильность выше, поэтому единая ставка слегка "
          "занижает цену.")


def _print_embedded(table: pd.DataFrame, info: dict) -> None:
    print("\nПортфель 2. Облигации со встроенными опционами, решётка BDT")
    print(f"Базовая бумага ОФЗ {info['bond']}, дата оценки "
          f"{pd.Timestamp(info['eval_date']).date()}, оферта/отзыв "
          f"{pd.Timestamp(info['option_date']).date()} по 100% номинала")
    print(f"Шагов решётки {info['n_steps']}, волатильность короткой ставки "
          f"{info['sigma']*100:.1f}%")
    print(f"Сверка обычной облигации: решётка {info['lattice_straight_pct']:.2f}%, "
          f"кривая (пункт 4) {info['curve_straight_pct']:.2f}%, "
          f"рынок {info['market_pct']:.2f}%")
    print(table.to_string(index=False))
    print(f"Оферта put добавляет {info['put_value_pct']:+.2f} п.п. (инвестор почти "
          f"наверняка предъявит к выкупу, бумага глубоко ниже номинала). Отзыв call "
          f"меняет цену на {info['call_value_pct']:+.2f} п.п. (эмитенту невыгодно "
          f"выкупать дороже рынка, опцион вне денег).")


def _print_integration() -> None:
    """Объяснение интеграции бонусных инструментов в основной Monte Carlo (8c)."""
    print("\nИнтеграция в основной Monte Carlo (пункт 8c):")
    print("- Опционы. В симуляции уже есть фактор FX_USD, он же двигает фьючерс на "
          "доллар. На каждом сценарии берём смоделированный курс, пересчитываем "
          "фьючерсную цену и переоцениваем опцион по Блэку-76. Для горизонта в "
          "несколько дней к этому добавляется риск волатильности (vega): "
          "подразумеваемую волатильность можно вести отдельным фактором или брать "
          "из улыбки на новом уровне курса.")
    print("- Облигации со встроенными опционами. Их переоценка это функция тех же "
          "риск-факторов кривой (RATE_PC1, PC2, PC3). На каждом сценарии двигаем "
          "кривую через PCA-нагрузки (как обычные ОФЗ), пересобираем решётку BDT и "
          "получаем цену с учётом права выкупа. Дальше всё как с обычными "
          "инструментами: считаем P&L, VaR и ES.")
    print("- Так оба портфеля встают в ту же схему, что и основной: единый набор "
          "риск-факторов, единый прогон сценариев, отдельные функции переоценки.")


def run(data_dir: str | Path | None = None) -> None:
    """Запускает оба бонусных расчёта, печатает, сохраняет результаты и графики."""
    data_dir = Path(data_dir) if data_dir is not None else C.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Каталог данных: {data_dir}")

    fig_dir = C.PROJECT_DIR / "docs" / "figures"

    # Портфель опционов.
    opt_table, opt_info, chain = OPT.price_selected(data_dir)
    _print_options(opt_table, opt_info)
    path_opt = OPT.plot_options(opt_table, opt_info, chain, fig_dir)

    # Облигации со встроенными опционами.
    eval_date = pd.Timestamp(C.EVAL_DATE)
    eb_table, eb_info = EB.evaluate(data_dir, eval_date)
    _print_embedded(eb_table, eb_info)
    path_eb = EB.plot_embedded(eb_table, eb_info, fig_dir)

    _print_integration()

    # Сохраняем результаты.
    opt_path = data_dir / "bonus_options.parquet"
    eb_path = data_dir / "bonus_embedded_bonds.parquet"
    opt_table.to_parquet(opt_path)
    eb_table.to_parquet(eb_path)

    print(f"\nГрафик опционов: {path_opt}")
    print(f"График облигаций с опционами: {path_eb}")
    print("\nИтог, сохранённые файлы:")
    print(f"  bonus_options.parquet        {opt_table.shape} "
          "(цена опционов Блэк-76 против наблюдаемой)")
    print(f"  bonus_embedded_bonds.parquet {eb_table.shape} "
          "(цена ОФЗ с офертой put и отзывной call против обычной)")


if __name__ == "__main__":
    run()
