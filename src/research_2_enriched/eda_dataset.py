from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.research_2_enriched.build_dataset import (
    clean_enriched_dataframe,
    load_enriched_dataframe,
)
from src.research_2_enriched.config import RESEARCH_2_CONFIG


sns.set_theme(style="whitegrid")


def _save_missing_ratio(raw_df: pd.DataFrame, out_dir: Path) -> None:
    missing = raw_df.isna().mean().sort_values(ascending=False)
    missing_nonzero = missing[missing > 0]

    missing.to_csv(
        out_dir / "research_2_missing_ratio.csv",
        encoding="utf-8-sig",
        header=["missing_ratio"],
    )

    if missing_nonzero.empty:
        return

    plt.figure(figsize=(12, 7))
    missing_nonzero.head(25).sort_values().plot(kind="barh")
    plt.title("Missing ratio by feature")
    plt.xlabel("missing share")
    plt.tight_layout()
    plt.savefig(out_dir / "research_2_missing_ratio.png", dpi=180, bbox_inches="tight")
    plt.close()


def _save_target_distribution(df: pd.DataFrame, target_col: str, out_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(df[target_col].dropna(), bins=50)
    plt.title("Target distribution")
    plt.xlabel(target_col)
    plt.tight_layout()
    plt.savefig(out_dir / "research_2_target_distribution.png", dpi=180, bbox_inches="tight")
    plt.close()


def _save_target_correlations(df: pd.DataFrame, target_col: str, out_dir: Path) -> pd.Series:
    num_df = df.select_dtypes(include=["number"])
    corr = num_df.corr(numeric_only=True)

    target_corr = (
        corr[target_col]
        .drop(target_col)
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )

    target_corr.to_csv(
        out_dir / "research_2_target_correlations.csv",
        encoding="utf-8-sig",
        header=["corr_with_target"],
    )

    top_corr = target_corr.head(20).sort_values()

    plt.figure(figsize=(12, 8))
    top_corr.plot(kind="barh")
    plt.title("Top correlations with target")
    plt.xlabel("correlation")
    plt.tight_layout()
    plt.savefig(out_dir / "research_2_target_correlations.png", dpi=180, bbox_inches="tight")
    plt.close()

    return target_corr


def _save_corr_heatmap(df: pd.DataFrame, target_col: str, target_corr: pd.Series, out_dir: Path) -> None:
    top_cols = [target_col] + target_corr.head(20).index.tolist()
    corr = df[top_cols].corr(numeric_only=True)

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Target-centered correlation heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "research_2_corr_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close()


def _save_boxplots(df: pd.DataFrame, out_dir: Path) -> None:
    for col in [
        "target_yield_centner_per_ha",
        "mineral_fertilizers_centner",
        "mineral_fertilizers_tons",
        "agricultural_machinery_total_count",
    ]:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue

        plt.figure(figsize=(10, 2.8))
        plt.boxplot(series, vert=False)
        plt.title(f"Boxplot: {col}")
        plt.tight_layout()
        plt.savefig(out_dir / f"research_2_boxplot_{col}.png", dpi=180, bbox_inches="tight")
        plt.close()


def _save_suspicious_rows(raw_df: pd.DataFrame, out_dir: Path) -> None:
    suspicious_cols = [
        "agricultural_machinery_total_count",
        "tractors_count",
        "grain_harvesters_count",
        "mineral_fertilizers_centner",
        "mineral_fertilizers_tons",
        "mineral_fertilizers_cumulative_centner",
    ]

    rows: list[pd.DataFrame] = []

    base_cols = [c for c in ["region", "crop", "year"] if c in raw_df.columns]

    for col in suspicious_cols:
        if col not in raw_df.columns:
            continue

        cur = raw_df[base_cols + [col]].copy()
        cur[col] = pd.to_numeric(cur[col], errors="coerce")
        cur = cur.sort_values(col, ascending=False).head(30)
        cur.insert(0, "feature", col)
        rows.append(cur)

    if rows:
        pd.concat(rows, ignore_index=True).to_csv(
            out_dir / "research_2_suspicious_rows.csv",
            index=False,
            encoding="utf-8-sig",
        )


def main() -> None:
    cfg = RESEARCH_2_CONFIG
    figures_dir = cfg["results"]["figures"]
    tables_dir = cfg["results"]["tables"]

    raw_df = load_enriched_dataframe()
    clean_df = clean_enriched_dataframe(raw_df.copy())

    target_col = cfg["target_col"]

    print("RAW SHAPE:", raw_df.shape)
    print("CLEAN SHAPE:", clean_df.shape)
    print("\nTARGET DESCRIBE:")
    print(clean_df[target_col].describe())

    _save_missing_ratio(raw_df, figures_dir)
    _save_target_distribution(clean_df, target_col, figures_dir)
    target_corr = _save_target_correlations(clean_df, target_col, figures_dir)
    _save_corr_heatmap(clean_df, target_col, target_corr, figures_dir)
    _save_boxplots(clean_df, figures_dir)
    _save_suspicious_rows(raw_df, tables_dir)

    print("\nSaved EDA artifacts:")
    print(figures_dir / "research_2_missing_ratio.png")
    print(figures_dir / "research_2_target_distribution.png")
    print(figures_dir / "research_2_target_correlations.png")
    print(figures_dir / "research_2_corr_heatmap.png")
    print(tables_dir / "research_2_suspicious_rows.csv")


if __name__ == "__main__":
    main()