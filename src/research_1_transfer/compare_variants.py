from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.evaluation.reports import print_summary, reorder_summary_columns
from src.research_1_transfer.config import RESEARCH_1_CONFIG


SCENARIO_FILES = {
    "source_training": "source_training_metrics.csv",
    "zero_shot_transfer": "transfer_zero_shot_metrics.csv",
    "finetuned_transfer": "transfer_finetuned_metrics.csv",
    "russia_scratch": "russia_scratch_metrics.csv",
}


def _load_one(path: Path, scenario_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл метрик: {path}")

    df = pd.read_csv(path)
    df["scenario"] = scenario_name
    return df


def main() -> None:
    metrics_dir = RESEARCH_1_CONFIG["results"]["metrics"]

    parts = []
    for scenario_name, filename in SCENARIO_FILES.items():
        parts.append(_load_one(metrics_dir / filename, scenario_name))

    summary_df = pd.concat(parts, ignore_index=True)

    scenario_order = {
        "source_training": 0,
        "zero_shot_transfer": 1,
        "finetuned_transfer": 2,
        "russia_scratch": 3,
    }
    summary_df["scenario_order"] = summary_df["scenario"].map(scenario_order).fillna(999).astype(int)

    summary_df = summary_df.sort_values(
        ["feature_mode", "model", "scenario_order", "rmse"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    summary_df = reorder_summary_columns(summary_df)

    out_path = metrics_dir / "research_1_compare_all_metrics.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print_summary(summary_df)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()