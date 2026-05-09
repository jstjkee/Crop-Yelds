from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    RUSSIAN_FERT_DATA_PATH,
    RUSSIAN_REAL_PROJECT_PATH,
    SOURCE_DATA_PATH,
)
from src.data.load_data import load_csv
from src.data.russian_mapping import build_russian_real_project_view


def prepare_russian_real_dataset(
    input_path: str | Path = RUSSIAN_FERT_DATA_PATH,
    output_path: str | Path = RUSSIAN_REAL_PROJECT_PATH,
    min_crop_count: int = 3,
    force: bool = False,
    filter_to_source_crops: bool = True,
) -> pd.DataFrame:
    output_path = Path(output_path)
    if output_path.exists() and not force:
        return pd.read_csv(output_path)

    raw_df = load_csv(input_path)
    prepared_df = build_russian_real_project_view(
        raw_df,
        min_crop_count=min_crop_count,
    )

    if filter_to_source_crops:
        source_df = load_csv(SOURCE_DATA_PATH)
        if "Crop" not in source_df.columns:
            raise ValueError("В source crop_yield.csv нет колонки 'Crop'")

        source_crops = set(source_df["Crop"].astype(str).str.strip())
        before_rows = len(prepared_df)
        before_crops = prepared_df["Crop"].nunique()

        prepared_df = prepared_df[
            prepared_df["Crop"].astype(str).str.strip().isin(source_crops)
        ].reset_index(drop=True)

        print("=" * 100)
        print("Фильтрация russian real по культурам source-датасета")
        print(f"Строк до фильтрации: {before_rows}")
        print(f"Строк после фильтрации: {len(prepared_df)}")
        print(f"Культур до фильтрации: {before_crops}")
        print(f"Культур после фильтрации: {prepared_df['Crop'].nunique()}")
        print(f"Оставшиеся культуры: {sorted(prepared_df['Crop'].astype(str).unique().tolist())}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return prepared_df