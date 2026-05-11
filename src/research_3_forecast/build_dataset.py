from __future__ import annotations

from typing import Any

import pandas as pd

from src.research_2_enriched.build_dataset import (
    PreparedEnrichedDataset,
    add_missing_indicators,
    add_yield_history_features,
    apply_train_fitted_quantile_clip_and_log,
    build_preprocessor,
    clean_enriched_dataframe,
    load_enriched_dataframe,
    split_by_year,
    split_random,
)
from src.research_3.config import RESEARCH_3_CONFIG


def _is_weather_month_col(col: str) -> bool:
    # Ищем погодные колонки с суффиксом месяца: _04 ... _09
    weather_prefixes = [
        "dry_days_",
        "hot_days_",
        "humidity_mean_",
        "precip_sum_",
        "rain_days_",
        "rain_sum_",
        "temp_max_",
        "temp_mean_",
        "temp_min_",
    ]
    col_lower = col.lower()
    return any(col_lower.startswith(prefix) for prefix in weather_prefixes)


def _extract_month_from_col(col: str) -> int | None:
    parts = col.rsplit("_", 1)
    if len(parts) != 2:
        return None
    suffix = parts[-1]
    if suffix.isdigit():
        month = int(suffix)
        if 1 <= month <= 12:
            return month
    return None


def apply_weather_month_cutoff(df: pd.DataFrame) -> pd.DataFrame:
    cfg = RESEARCH_3_CONFIG
    cutoff = cfg.get("weather_month_cutoff", 9)

    df = df.copy()

    if cutoff is None:
        # Убираем всю погоду
        weather_cols = [c for c in df.columns if _is_weather_month_col(c)]
        if weather_cols:
            df = df.drop(columns=weather_cols, errors="ignore")
        return df

    cols_to_drop: list[str] = []

    for col in df.columns:
        if not _is_weather_month_col(col):
            continue

        month = _extract_month_from_col(col)
        if month is None:
            continue

        if month > int(cutoff):
            cols_to_drop.append(col)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors="ignore")

    return df


def prepare_forecast_dataset(
    split_name: str,
    feature_set_name: str = "full",
) -> PreparedEnrichedDataset:
    cfg = RESEARCH_3_CONFIG

    df = load_enriched_dataframe()
    df = clean_enriched_dataframe(df)
    df = add_yield_history_features(df)
    df = add_missing_indicators(df)
    df = apply_weather_month_cutoff(df)

    if split_name == "year":
        train_df, val_df, test_df = split_by_year(df)
    elif split_name == "random":
        train_df, val_df, test_df = split_random(df)
    else:
        raise ValueError(f"Неизвестный split_name: {split_name}")

    train_df, val_df, test_df = apply_train_fitted_quantile_clip_and_log(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    from src.research_2_enriched.build_dataset import prepare_enriched_dataset  # noqa

    raise NotImplementedError("Ниже надо вставить local version of prepare_enriched_dataset under RESEARCH_3_CONFIG")