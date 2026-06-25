"""
Единая точка входа в проект по рыночному риску.

Запуск без аргументов прогоняет весь пайплайн по порядку: сбор данных,
риск-факторы, модели, оценка инструментов, дальше VaR/ES, бэктестинг и тесты.
Каждый этап читает то, что сохранил предыдущий, поэтому порядок важен.

Примеры запуска:
    python main.py                    весь пайплайн
    python main.py --stage data       только сбор данных
    python main.py --stage factors    только риск-факторы
    python main.py --list             список этапов и их статус
"""
from __future__ import annotations

import argparse

import config as C


# Этапы пайплайна по порядку: ключ, номер, описание, готов ли, ответственный.
STAGES = [
    ("data",     1, "Сбор данных из MOEX и ЦБ РФ",     True,  "Сева"),
    ("factors",  2, "Риск-факторы и PCA",              True,  "Сева"),
    ("models",   3, "Стохастические модели GARCH-DCC", True,  "Лёша"),
    ("pricing",  4, "Оценка стоимости инструментов",   True,  "Лёша"),
    ("var",      5, "Monte Carlo, VaR и ES",           False, "Егор"),
    ("backtest", 6, "Бэктестинг",                      False, "Вика"),
    ("tests",    7, "Статистические тесты VaR",        False, "Настя"),
]


def _run_data() -> None:
    # Импортируем внутри функции, чтобы тяжёлые зависимости грузились
    # только когда этап реально запускают.
    from data_collection import pipeline
    pipeline.run()


def _run_factors() -> None:
    from risk_factors import pipeline
    pipeline.run()


def _run_models() -> None:
    from models import pipeline
    pipeline.run()


def _run_pricing() -> None:
    from pricing import pipeline
    pipeline.run()


# Готовые этапы. Ключ это функция запуска. Остальные пока не реализованы.
STAGE_RUNNERS = {
    "data": _run_data,
    "factors": _run_factors,
    "models": _run_models,
    "pricing": _run_pricing,
}


def _stage(key: str):
    """Находит описание этапа по ключу."""
    return next((s for s in STAGES if s[0] == key), None)


def _print_intro() -> None:
    """Короткая шапка перед запуском: что считаем и с какими настройками."""
    print("Оценка рыночного риска портфеля из ОФЗ, акций и валюты")
    print(f"Состав портфеля: {len(C.PORTFOLIO_BONDS)} ОФЗ, "
          f"{len(C.PORTFOLIO_STOCKS)} акций, валюта USD и EUR")
    print(f"Дата оценки риска: {C.EVAL_DATE}")
    print(f"Seed для воспроизводимости: {C.RANDOM_SEED}")


def _not_ready(num: int) -> None:
    """Сообщает, что этап ещё не реализован."""
    print(f"  Этап {num} пока не реализован, подробности в README.")


def run_stage(key: str) -> None:
    """Запускает один этап по ключу."""
    info = _stage(key)
    if info is None:
        raise SystemExit(f"Неизвестный этап: {key}. "
                         f"Доступные: {[s[0] for s in STAGES]}")
    _, num, desc, _, who = info
    print(f"\nЭтап {num}. {desc} ({who})")

    runner = STAGE_RUNNERS.get(key)
    if runner is None:
        _not_ready(num)
    else:
        runner()


def run_all() -> None:
    """Прогоняет все этапы по порядку."""
    for key, *_ in STAGES:
        run_stage(key)
    done = sum(1 for s in STAGES if s[3])
    print(f"\nГотовые этапы отработали ({done} из {len(STAGES)}). "
          "Результаты в папке data/, что ещё в работе смотри в README.")


def print_list() -> None:
    """Печатает список этапов и их статус."""
    print("Этапы пайплайна по порядку:")
    for key, num, desc, ready, who in STAGES:
        status = "готово" if ready else "в работе"
        print(f"  {num}  {key:9} {desc:35} {status:9} {who}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Пайплайн оценки рыночного риска портфеля")
    parser.add_argument(
        "--stage",
        choices=[s[0] for s in STAGES] + ["all"],
        default="all",
        help="какой этап запустить, по умолчанию весь пайплайн (all)",
    )
    parser.add_argument("--list", action="store_true",
                        help="показать список этапов и их статус")
    args = parser.parse_args()

    if args.list:
        print_list()
        return

    _print_intro()
    if args.stage == "all":
        run_all()
    else:
        run_stage(args.stage)


if __name__ == "__main__":
    main()
