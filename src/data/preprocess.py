from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreparedDataset:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray

    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    crop_train: np.ndarray
    crop_val: np.ndarray
    crop_test: np.ndarray

    crop_to_id: dict[str, int]
    id_to_crop: dict[int, str]

    preprocessor: ColumnTransformer

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame

    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]


def validate_required_columns(
    df: pd.DataFrame,
    target_col: str,
    crop_col: str,
) -> None:
    missing = {target_col, crop_col} - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {sorted(missing)}")


def drop_invalid_rows(
    df: pd.DataFrame,
    target_col: str,
    crop_col: str,
) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=[target_col, crop_col])
    cleaned = cleaned[cleaned[crop_col].astype(str).str.strip() != ""]
    return cleaned.reset_index(drop=True)


def build_crop_mapping(series: pd.Series) -> tuple[dict[str, int], dict[int, str]]:
    crop_values = sorted(series.astype(str).unique().tolist())
    crop_to_id = {crop: idx for idx, crop in enumerate(crop_values)}
    id_to_crop = {idx: crop for crop, idx in crop_to_id.items()}
    return crop_to_id, id_to_crop


def encode_crop_ids(
    series: pd.Series,
    crop_to_id: dict[str, int],
) -> np.ndarray:
    encoded = series.astype(str).map(crop_to_id)
    if encoded.isna().any():
        unknown = sorted(series.astype(str)[encoded.isna()].unique().tolist())
        raise ValueError(f"Найдены неизвестные культуры при кодировании: {unknown}")
    return encoded.to_numpy(dtype=np.int64)


def get_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    crop_col: str,
) -> tuple[list[str], list[str], list[str]]:
    feature_cols = [col for col in df.columns if col not in {target_col, crop_col}]
    numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]
    return feature_cols, numeric_cols, categorical_cols


def build_preprocessor(
    train_df: pd.DataFrame,
    target_col: str,
    crop_col: str,
) -> tuple[ColumnTransformer, list[str], list[str], list[str]]:
    feature_cols, numeric_cols, categorical_cols = get_feature_columns(
        train_df,
        target_col=target_col,
        crop_col=crop_col,
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    if not transformers:
        raise ValueError("После исключения target и crop не осталось признаков для обучения")

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor, feature_cols, numeric_cols, categorical_cols


def _validate_crop_counts_for_split(
    df: pd.DataFrame,
    crop_col: str,
    min_count: int = 3,
) -> None:
    crop_counts = df[crop_col].astype(str).value_counts()
    rare_crops = crop_counts[crop_counts < min_count].index.tolist()
    if rare_crops:
        raise ValueError(
            f"Для train/val/test разбиения у каждой культуры должно быть минимум {min_count} записей. "
            f"Проблемные культуры: {rare_crops}"
        )


def prepare_dataset(
    df: pd.DataFrame,
    target_col: str,
    crop_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    val_size_from_train: float = 0.2,
) -> PreparedDataset:
    """
    Схема разбиения:
        1) полный датасет -> train_full / test
        2) train_full -> train / val

    test_size:
        доля test от полного датасета

    val_size_from_train:
        доля validation от train_full
    """
    validate_required_columns(df, target_col, crop_col)
    df = drop_invalid_rows(df, target_col, crop_col)
    _validate_crop_counts_for_split(df, crop_col=crop_col, min_count=3)

    train_full_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[crop_col].astype(str),
    )

    train_df, val_df = train_test_split(
        train_full_df,
        test_size=val_size_from_train,
        random_state=random_state,
        stratify=train_full_df[crop_col].astype(str),
    )

    crop_to_id, id_to_crop = build_crop_mapping(train_df[crop_col])

    unknown_in_val = set(val_df[crop_col].astype(str).unique()) - set(crop_to_id.keys())
    if unknown_in_val:
        raise ValueError(
            "В validation попали культуры, которых нет в train: "
            f"{sorted(unknown_in_val)}"
        )

    unknown_in_test = set(test_df[crop_col].astype(str).unique()) - set(crop_to_id.keys())
    if unknown_in_test:
        raise ValueError(
            "В test попали культуры, которых нет в train: "
            f"{sorted(unknown_in_test)}"
        )

    preprocessor, feature_cols, numeric_cols, categorical_cols = build_preprocessor(
        train_df=train_df,
        target_col=target_col,
        crop_col=crop_col,
    )

    X_train = preprocessor.fit_transform(train_df.drop(columns=[target_col]))
    X_val = preprocessor.transform(val_df.drop(columns=[target_col]))
    X_test = preprocessor.transform(test_df.drop(columns=[target_col]))

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_val, "toarray"):
        X_val = X_val.toarray()
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    y_train = train_df[target_col].to_numpy(dtype=np.float32)
    y_val = val_df[target_col].to_numpy(dtype=np.float32)
    y_test = test_df[target_col].to_numpy(dtype=np.float32)

    crop_train = encode_crop_ids(train_df[crop_col], crop_to_id)
    crop_val = encode_crop_ids(val_df[crop_col], crop_to_id)
    crop_test = encode_crop_ids(test_df[crop_col], crop_to_id)

    return PreparedDataset(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        crop_train=crop_train,
        crop_val=crop_val,
        crop_test=crop_test,
        crop_to_id=crop_to_id,
        id_to_crop=id_to_crop,
        preprocessor=preprocessor,
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )