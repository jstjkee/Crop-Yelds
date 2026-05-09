from __future__ import annotations

import argparse

from src.research_1_transfer.config import RESEARCH_1_CONFIG
from src.research_1_transfer.evaluate_transfer import main as evaluate_transfer_main
from src.research_1_transfer.finetune_on_russia import main as finetune_on_russia_main
from src.research_1_transfer.train_source import main as train_source_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research 1: transfer learning pipeline")
    parser.add_argument(
        "command",
        choices=["train_source", "evaluate_transfer", "finetune_on_russia", "all"],
        help="Какой сценарий запустить",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=RESEARCH_1_CONFIG["model_types"],
        help="Какие модели запускать",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "train_source":
        train_source_main(models=args.models)
        return

    if args.command == "evaluate_transfer":
        evaluate_transfer_main(models=args.models)
        return

    if args.command == "finetune_on_russia":
        finetune_on_russia_main(models=args.models)
        return

    train_source_main(models=args.models)
    evaluate_transfer_main(models=args.models)
    finetune_on_russia_main(models=args.models)


if __name__ == "__main__":
    main()