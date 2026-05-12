from __future__ import annotations

import argparse
from src.research_3_forecast.config import RESEARCH_3_CONFIG
from src.research_3_forecast.train_forecast import main as train_main

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research 3: forecast with weather month cutoff")

    parser.add_argument(
        "command",
        choices=["train"],
        help="Какой сценарий запустить",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=RESEARCH_3_CONFIG["model_types"],
        help="Какие модели запускать",
    )

    parser.add_argument(
        "--feature-modes",
        nargs="+",
        default=None,
        choices=RESEARCH_3_CONFIG["feature_modes"],
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
        help="Список seed для серии запусков",
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


if __name__ == "__main__":
    main()