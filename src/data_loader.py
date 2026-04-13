from pathlib import Path

import pandas as pd


def load_csv(file) -> pd.DataFrame:
    """
    Загружает CSV из пути или загруженного файла Streamlit.
    """
    return pd.read_csv(file)


def basic_info(df: pd.DataFrame) -> dict:
    """
    Возвращает базовую информацию о датасете.
    """
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "columns": df.columns.tolist(),
        "missing_total": int(df.isna().sum().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }