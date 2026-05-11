from __future__ import annotations

import argparse

from src.research_2_enriched.compare_feature_sets import main as compare_main
from src.research_2_enriched.config import RESEARCH_2_CONFIG
from src.research_2_enriched.train_enriched import main as train_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research 2: enriched Russian dataset pipeline")

    parser.add_argument(
        "command",
        choices=["train", "compare", "all"],
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

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

    train_main(
        models=args.models,
        feature_modes=args.feature_modes,
        seed=args.seed,
        seed_list=args.seed_list,
    )
    compare_main()


if __name__ == "__main__":
    main()