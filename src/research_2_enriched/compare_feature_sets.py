from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.evaluation.reports import print_summary, reorder_summary_columns
from src.research_2_enriched.config import RESEARCH_2_CONFIG


def main() -> None:
    metrics_dir: Path = RESEARCH_2_CONFIG["results"]["metrics"]

    path = metrics_dir / "research_2_enriched_metrics.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл метрик: {path}. "
            f"Сначала запусти: python -m src.research_2_enriched.run train"
        )

    df = pd.read_csv(path)
    df["scenario"] = "ablation"

    df = df.sort_values(
        ["split", "feature_set", "feature_mode", "rmse"]
    ).reset_index(drop=True)

    df = reorder_summary_columns(df)

    out_path = metrics_dir / "research_2_compare_metrics.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print_summary(df)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()