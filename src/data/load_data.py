from __future__ import annotations
import pandas as pd

def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def clean_dataset(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    result = df.copy()

    result = result.drop_duplicates()
    result = result.dropna(subset=[target_col])

    if pd.api.types.is_numeric_dtype(result[target_col]):
        result = result[result[target_col] >= 0]

    return result.reset_index(drop=True)

def basic_info(df: pd.DataFrame) -> dict:
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "columns": df.columns.tolist(),
        "missing_total": int(df.isna().sum().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }