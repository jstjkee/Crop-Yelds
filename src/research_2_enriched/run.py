from __future__ import annotations

import argparse

from src.research_2_enriched.compare_feature_sets import main as compare_main
from src.research_2_enriched.config import RESEARCH_2_CONFIG
from src.research_2_enriched.eda_dataset import main as eda_main
from src.research_2_enriched.train_enriched import main as train_main
from src.research_2_enriched.tune_optuna import main as tune_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research 2: enriched Russian dataset pipeline"
    )

    parser.add_argument(
        "command",
        choices=["train", "compare", "eda", "tune", "all"],
        help="Какой сценарий запустить",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=RESEARCH_2_CONFIG["model_types"],
        help="Какие модели запускать",
    )

    parser.add_argument(
        "--feature-modes",
        nargs="+",
        default=None,
        choices=RESEARCH_2_CONFIG["feature_modes"],
        help="Какие feature modes запускать",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Один конкретный seed для запуска",
    )

    parser.add_argument(
        "--seed-list",
        nargs="+",
        type=int,
        default=None,
        help="Список seed для серии запусков, например: --seed-list 42 52 62",
    )

    # Аргументы для Optuna
    parser.add_argument(
        "--model",
        default="transformer",
        choices=["transformer", "mlp_resnet", "tab_mlp"],
        help="Модель для Optuna-подбора",
    )

    parser.add_argument(
        "--feature-set",
        default="full",
        choices=RESEARCH_2_CONFIG["feature_sets"],
        help="Какой feature set использовать при подборе",
    )

    parser.add_argument(
        "--split",
        default="year",
        choices=RESEARCH_2_CONFIG["splits"],
        help="Какой split использовать при подборе",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Количество trial для Optuna",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Ограничение по времени в секундах для Optuna",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "eda":
        eda_main()
        return

    if args.command == "train":
        train_main(
            models=args.models,
            feature_modes=args.feature_modes,
            seed=args.seed,
            seed_list=args.seed_list,
        )
        return

    if args.command == "compare":
        compare_main()
        return

    if args.command == "tune":
        tune_main(
            model_type=args.model,
            feature_mode=(args.feature_modes[0] if args.feature_modes else "raw"),
            split_name=args.split,
            feature_set_name=args.feature_set,
            n_trials=args.trials,
            timeout=args.timeout,
        )
        return

    eda_main()
    train_main(
        models=args.models,
        feature_modes=args.feature_modes,
        seed=args.seed,
        seed_list=args.seed_list,
    )
    compare_main()


if __name__ == "__main__":
    main()