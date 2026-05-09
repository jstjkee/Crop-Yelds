from __future__ import annotations

import argparse

from src.research_1_transfer.compare_variants import main as compare_variants_main
from src.research_1_transfer.config import RESEARCH_1_CONFIG
from src.research_1_transfer.evaluate_transfer import main as evaluate_transfer_main
from src.research_1_transfer.finetune_on_russia import main as finetune_on_russia_main
from src.research_1_transfer.train_russia_scratch import main as train_russia_scratch_main
from src.research_1_transfer.train_source import main as train_source_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research 1: transfer learning pipeline")
    parser.add_argument(
        "command",
        choices=[
            "train_source",
            "evaluate_transfer",
            "finetune_on_russia",
            "train_russia_scratch",
            "compare_variants",
            "all",
        ],
        help="Какой сценарий запустить",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=RESEARCH_1_CONFIG["model_types"],
        help="Какие модели запускать",
    )
    parser.add_argument(
        "--feature-modes",
        nargs="+",
        default=None,
        choices=RESEARCH_1_CONFIG["feature_modes"],
        help="Какие feature mode запускать",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "train_source":
        train_source_main(models=args.models, feature_modes=args.feature_modes)
        return

    if args.command == "evaluate_transfer":
        evaluate_transfer_main(models=args.models, feature_modes=args.feature_modes)
        return

    if args.command == "finetune_on_russia":
        finetune_on_russia_main(models=args.models, feature_modes=args.feature_modes)
        return

    if args.command == "train_russia_scratch":
        train_russia_scratch_main(models=args.models, feature_modes=args.feature_modes)
        return

    if args.command == "compare_variants":
        compare_variants_main()
        return

    train_source_main(models=args.models, feature_modes=args.feature_modes)
    evaluate_transfer_main(models=args.models, feature_modes=args.feature_modes)
    finetune_on_russia_main(models=args.models, feature_modes=args.feature_modes)
    train_russia_scratch_main(models=args.models, feature_modes=args.feature_modes)
    compare_variants_main()


if __name__ == "__main__":
    main()