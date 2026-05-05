from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV файл не найден: {csv_path}")
    return pd.read_csv(csv_path, **kwargs)


def validate_required_columns(
    df: pd.DataFrame,
    target_col: str,
    crop_col: str,
) -> None:
    missing = {target_col, crop_col} - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {sorted(missing)}")


def summarize_dataframe(
    df: pd.DataFrame,
    target_col: str,
    crop_col: str,
) -> dict:
    validate_required_columns(df, target_col, crop_col)

    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "n_missing_total": int(df.isna().sum().sum()),
        "n_crops": int(df[crop_col].nunique(dropna=True)),
        "crop_counts": df[crop_col].value_counts(dropna=False).to_dict(),
        "target_missing": int(df[target_col].isna().sum()),
        "crop_missing": int(df[crop_col].isna().sum()),
    }


def load_and_validate_source_data(
    path: str | Path,
    target_col: str,
    crop_col: str,
    **kwargs,
) -> pd.DataFrame:
    df = load_csv(path, **kwargs)
    validate_required_columns(df, target_col, crop_col)
    return df