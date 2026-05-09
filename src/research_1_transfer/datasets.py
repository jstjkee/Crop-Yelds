from __future__ import annotations

from typing import Any

import pandas as pd

from src.core.config import RANDOM_STATE
from src.core.data.io import load_csv, summarize_dataframe, validate_required_columns
from src.core.data.preprocess import PreparedDataset, prepare_dataset, transform_external_dataset
from src.research_1_transfer.config import RESEARCH_1_CONFIG


def _clean_common(df: pd.DataFrame, target_col: str, crop_col: str) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=[target_col, crop_col])
    df = df[df[crop_col].astype(str).str.strip() != ""]
    df = df[df[target_col] >= 0]
    return df.reset_index(drop=True)


def _drop_rare_crops(df: pd.DataFrame, crop_col: str, min_crop_count: int) -> pd.DataFrame:
    counts = df[crop_col].astype(str).value_counts()
    keep = counts[counts >= min_crop_count].index.tolist()
    return df[df[crop_col].astype(str).isin(keep)].reset_index(drop=True)


def load_source_dataframe() -> pd.DataFrame:
    cfg = RESEARCH_1_CONFIG
    df = load_csv(cfg["source_dataset_path"])
    validate_required_columns(df, cfg["target_col"], cfg["crop_col"])
    df = _clean_common(df, cfg["target_col"], cfg["crop_col"])
    df = _drop_rare_crops(df, cfg["crop_col"], int(cfg.get("min_crop_count", 10)))
    return df


def load_target_dataframe() -> pd.DataFrame:
    cfg = RESEARCH_1_CONFIG
    df = load_csv(cfg["target_dataset_path"])
    validate_required_columns(df, cfg["target_col"], cfg["crop_col"])
    df = _clean_common(df, cfg["target_col"], cfg["crop_col"])
    return df


def prepare_source_dataset() -> PreparedDataset:
    cfg = RESEARCH_1_CONFIG
    source_df = load_source_dataframe()

    return prepare_dataset(
        df=source_df,
        target_col=cfg["target_col"],
        crop_col=cfg["crop_col"],
        test_size=float(cfg.get("test_size", 0.2)),
        random_state=RANDOM_STATE,
        val_size_from_train=float(cfg.get("val_size_from_train", 0.2)),
    )


def prepare_target_external(prepared_source: PreparedDataset) -> dict[str, Any]:
    cfg = RESEARCH_1_CONFIG
    target_df = load_target_dataframe()

    return transform_external_dataset(
        df=target_df,
        preprocessor=prepared_source.preprocessor,
        crop_to_id=prepared_source.crop_to_id,
        target_col=cfg["target_col"],
        crop_col=cfg["crop_col"],
    )


def describe_stage1_datasets() -> dict[str, dict[str, Any]]:
    cfg = RESEARCH_1_CONFIG
    source_df = load_source_dataframe()
    target_df = load_target_dataframe()

    return {
        "source": summarize_dataframe(source_df, cfg["target_col"], cfg["crop_col"]),
        "target": summarize_dataframe(target_df, cfg["target_col"], cfg["crop_col"]),
    }

def prepare_target_dataset_for_scratch() -> PreparedDataset:
    cfg = RESEARCH_1_CONFIG
    target_df = load_target_dataframe()
    target_df = _drop_rare_crops(
        target_df,
        crop_col=cfg["crop_col"],
        min_crop_count=int(cfg.get("min_crop_count", 10)),
    )

    return prepare_dataset(
        df=target_df,
        target_col=cfg["target_col"],
        crop_col=cfg["crop_col"],
        test_size=float(cfg.get("test_size", 0.2)),
        random_state=RANDOM_STATE,
        val_size_from_train=float(cfg.get("val_size_from_train", 0.2)),
    )