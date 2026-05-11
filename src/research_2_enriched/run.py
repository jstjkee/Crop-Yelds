from __future__ import annotations

import argparse

from src.research_2_enriched.compare_feature_sets import main as compare_main
from src.research_2_enriched.config import RESEARCH_2_CONFIG
from src.research_2_enriched.eda_dataset import main as eda_main
from src.research_2_enriched.train_enriched import main as train_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research 2: enriched Russian dataset pipeline")

    parser.add_argument(
        "command",
        choices=["train", "compare", "eda", "all"],
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "eda":
        eda_main()
        return

    if args.command == "train":
        train_main(models=args.models, feature_modes=args.feature_modes)
        return

    if args.command == "compare":
        compare_main()
        return

    eda_main()
    train_main(models=args.models, feature_modes=args.feature_modes)
    compare_main()


if __name__ == "__main__":
    main()