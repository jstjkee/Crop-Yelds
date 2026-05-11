from __future__ import annotations

import pandas as pd


def reorder_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "model",
        "feature_mode",
        "feature_set",
        "split",
        "dataset",
        "seed",
        "n_seeds",
        "seed_list",
        "mae",
        "mse",
        "rmse",
        "r2",
        "mae_mean",
        "mae_std",
        "rmse_mean",
        "rmse_std",
        "r2_mean",
        "r2_std",
        "rows_train",
        "rows_val",
        "rows_test",
        "num_crops",
        "input_dim",
    ]

    path_cols = [c for c in df.columns if "path" in c.lower()]
    main_cols = [c for c in preferred if c in df.columns]
    other_cols = [c for c in df.columns if c not in main_cols and c not in path_cols]

    return df[main_cols + other_cols + path_cols]


def visible_summary_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "model",
        "feature_mode",
        "feature_set",
        "split",
        "dataset",
        "seed",
        "n_seeds",
        "mae",
        "rmse",
        "r2",
        "mae_mean",
        "mae_std",
        "rmse_mean",
        "rmse_std",
        "r2_mean",
        "r2_std",
        "rows_train",
        "rows_val",
        "rows_test",
        "num_crops",
    ]
    return [c for c in preferred if c in df.columns]


def print_summary(df: pd.DataFrame) -> None:
    visible = visible_summary_columns(df)
    if not visible:
        print(df.to_string(index=False))
        return
    print(df[visible].to_string(index=False))