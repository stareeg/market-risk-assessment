"""
Единая точка входа в проект по рыночному риску.

Запускает пайплайн по этапам. Каждый этап закрывает свой пункт задания и
складывает промежуточные результаты в data/. Этапы идут строго по порядку,
потому что каждый следующий читает то, что сохранил предыдущий.

Примеры запуска:
    python main.py --stage all        # весь пайплайн
    python main.py --stage data       # только сбор данных (п.1)
    python main.py --stage factors    # только риск-факторы (п.2)
    python main.py --list             # показать список этапов и статус

Часть этапов ещё не перенесена из тетрадок в модули, см. refactor-plan.md.
"""
from __future__ import annotations

import argparse

import config as C


# Этапы пайплайна по порядку. Для каждого: ключ, пункт задания, описание,
# готов ли (реализован в модулях) и ответственный.
STAGES = [
    ("data",     "п.1", "Сбор данных из MOEX и ЦБ РФ",        True,  "Сева"),
    ("factors",  "п.2", "Риск-факторы и PCA",                 False, "Сева"),
    ("models",   "п.3", "Стохастические модели (GARCH-DCC)",  False, "Лёша"),
    ("pricing",  "п.4", "Оценка стоимости инструментов",      False, "Лёша"),
    ("var",      "п.5", "Monte Carlo, VaR и ES",              False, "Егор"),
    ("backtest", "п.6", "Бэктестинг",                         False, "Вика"),
    ("tests",    "п.7", "Статистические тесты VaR",           False, "Настя"),
]


def _run_data() -> None:
    # Импортируем внутри функции, чтобы тяжёлые зависимости грузились
    # только когда этап реально запускают.
    from data_collection import pipeline
    pipeline.run()


# Готовые этапы. Ключ это функция запуска. Остальные пока не реализованы.
STAGE_RUNNERS = {
    "data": _run_data,
}


def _not_ready(key: str) -> None:
    """Сообщает, что этап ещё не перенесён в модули."""
    print(f"  Этап '{key}' пока не реализован.")
    print("  Что и в каком порядке делаем, смотри в refactor-plan.md.")


def run_stage(key: str) -> None:
    """Запускает один этап по ключу."""
    info = next((s for s in STAGES if s[0] == key), None)
    if info is None:
        raise SystemExit(f"Неизвестный этап: {key}. Доступные: {[s[0] for s in STAGES]}")

    print(f"\n=== {info[1]}  {info[2]}  ({info[4]}) ===")

    runner = STAGE_RUNNERS.get(key)
    if runner is None:
        _not_ready(key)
    else:
        runner()


def run_all() -> None:
    """Прогоняет все этапы по порядку."""
    for key, *_ in STAGES:
        run_stage(key)


def print_list() -> None:
    """Печатает список этапов и их статус."""
    print("Этапы пайплайна:")
    for key, point, desc, ready, who in STAGES:
        mark = "готов" if ready else "не готов"
        print(f"  [{mark:8}] {point}  {key:9} {desc}  ({who})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Пайплайн оценки рыночного риска")
    parser.add_argument(
        "--stage",
        choices=[s[0] for s in STAGES] + ["all"],
        default="all",
        help="какой этап запустить (по умолчанию all)",
    )
    parser.add_argument("--list", action="store_true", help="показать этапы и статус")
    args = parser.parse_args()

    # Фиксируем seed для воспроизводимости (требование задания).
    print(f"RANDOM_SEED = {C.RANDOM_SEED}")

    if args.list:
        print_list()
        return

    if args.stage == "all":
        run_all()
    else:
        run_stage(args.stage)


if __name__ == "__main__":
    main()
