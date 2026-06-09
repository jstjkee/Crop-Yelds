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

from src.core.config import RANDOM_STATE
from src.core.data.io import load_csv, summarize_dataframe, validate_required_columns
from src.research_2_enriched.config import RESEARCH_2_CONFIG


@dataclass
class PreparedEnrichedDataset:
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

    split_name: str
    feature_set_name: str


def load_enriched_dataframe() -> pd.DataFrame:
    cfg = RESEARCH_2_CONFIG
    df = load_csv(cfg["final_dataset_path"])

    validate_required_columns(
        df,
        cfg["target_col"],
        cfg["crop_col"],
    )

    return df


def clean_enriched_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cfg = RESEARCH_2_CONFIG
    target_col = cfg["target_col"]
    crop_col = cfg["crop_col"]

    df = df.copy()

    df = df.dropna(subset=[target_col, crop_col])
    df = df[df[crop_col].astype(str).str.strip() != ""]
    df = df[df[target_col] >= 0]

    suspicious_numeric_cols = [
        "agricultural_machinery_total_count",
        "tractors_count",
        "grain_harvesters_count",
        "forage_harvesters_count",
        "potato_harvesters_count",
        "corn_harvesters_count",
        "beet_harvesters_count",
        "mineral_fertilizers_centner",
        "mineral_fertilizers_tons",
        "mineral_fertilizers_cumulative_centner",
        "fertilizer_municipality_count",
        "fertilizer_records_count",
        "valid_fertilizer_values_count",
        "sown_area",
    ]

    for col in suspicious_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    suspicious_upper_bounds = {
        "agricultural_machinery_total_count": 1_000_000,
        "tractors_count": 1_000_000,
        "grain_harvesters_count": 1_000_000,
        "forage_harvesters_count": 1_000_000,
        "potato_harvesters_count": 1_000_000,
        "corn_harvesters_count": 1_000_000,
        "beet_harvesters_count": 1_000_000,
        "mineral_fertilizers_centner": 10_000_000,
        "mineral_fertilizers_tons": 1_000_000,
    }

    for col, upper_bound in suspicious_upper_bounds.items():
        if col not in df.columns:
            continue

        bad_mask = df[col] > upper_bound
        bad_count = int(bad_mask.sum())

        if bad_count > 0:
            print(
                f"[research_2][clean] {col}: values > {upper_bound:,} -> NaN | rows={bad_count}"
            )
            df.loc[bad_mask, col] = np.nan

    if bool(cfg.get("zero_fill_all_nulls", False)):
        df = df.fillna(0)

    counts = df[crop_col].astype(str).value_counts()
    keep_crops = counts[counts >= int(cfg.get("min_crop_count", 10))].index
    df = df[df[crop_col].astype(str).isin(keep_crops)]

    return df.reset_index(drop=True)

def add_yield_history_features(df: pd.DataFrame) -> pd.DataFrame:
    cfg = RESEARCH_2_CONFIG
    target_col = cfg["target_col"]
    year_col = cfg["year_col"]

    if not bool(cfg.get("use_yield_history_features", False)):
        return df

    group_cols = list(cfg.get("yield_history_group_cols", ["region", "crop"]))
    lag_steps = list(cfg.get("yield_lags", [1, 2, 3]))
    roll_windows = list(cfg.get("yield_roll_windows", [3]))

    required_cols = [*group_cols, year_col, target_col]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Для yield history features не хватает колонок: {missing_required}"
        )

    df = df.copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")

    df = df.sort_values(group_cols + [year_col]).reset_index(drop=True)

    grouped = df.groupby(group_cols, sort=False)[target_col]

    # Простые лаги
    for lag in lag_steps:
        df[f"yield_lag_{lag}"] = grouped.shift(lag)

    # Rolling mean / std только по прошлым значениям
    for window in roll_windows:
        shifted = grouped.shift(1)

        df[f"yield_roll_mean_{window}"] = (
            shifted.groupby([df[c] for c in group_cols], sort=False)
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=list(range(len(group_cols))), drop=True)
        )

        df[f"yield_roll_std_{window}"] = (
            shifted.groupby([df[c] for c in group_cols], sort=False)
            .rolling(window=window, min_periods=2)
            .std()
            .reset_index(level=list(range(len(group_cols))), drop=True)
        )

    # Простая динамика
    if "yield_lag_1" in df.columns and "yield_lag_2" in df.columns:
        df["yield_growth_1"] = df["yield_lag_1"] - df["yield_lag_2"]

    return df

def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cols_for_missing_flags = [
        "mineral_fertilizers_centner",
        "mineral_fertilizers_tons",
        "fertilizer_municipality_count",
        "fertilizer_records_count",
        "valid_fertilizer_values_count",
        "agricultural_machinery_total_count",
        "tractors_count",
        "grain_harvesters_count",
        "forage_harvesters_count",
        "potato_harvesters_count",
        "corn_harvesters_count",
        "beet_harvesters_count",
        "sown_area",
    ]

    for col in cols_for_missing_flags:
        if col in df.columns:
            df[f"{col}_was_missing"] = df[col].isna().astype(int)

    return df


def split_random(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = RESEARCH_2_CONFIG
    crop_col = cfg["crop_col"]

    train_full_df, test_df = train_test_split(
        df,
        test_size=float(cfg.get("test_size", 0.2)),
        random_state=RANDOM_STATE,
        stratify=df[crop_col].astype(str),
    )

    train_df, val_df = train_test_split(
        train_full_df,
        test_size=float(cfg.get("val_size_from_train", 0.2)),
        random_state=RANDOM_STATE,
        stratify=train_full_df[crop_col].astype(str),
    )

    print("=" * 100)
    print("Research 2 | RANDOM SPLIT")
    print(f"train rows: {len(train_df)}")
    print(f"val rows:   {len(val_df)}")
    print(f"test rows:  {len(test_df)}")
    print("=" * 100)

    return train_df, val_df, test_df


def split_by_year(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = RESEARCH_2_CONFIG
    year_col = cfg["year_col"]

    if year_col not in df.columns:
        raise ValueError(f"Для split='year' нужна колонка {year_col}")

    df = df.copy()
    df[year_col] = df[year_col].astype(int)

    train_start, train_end = cfg["train_years_range"]
    val_start, val_end = cfg["val_years_range"]
    test_start, test_end = cfg["test_years_range"]

    train_df = df[
        (df[year_col] >= train_start)
        & (df[year_col] <= train_end)
    ].copy()

    val_df = df[
        (df[year_col] >= val_start)
        & (df[year_col] <= val_end)
    ].copy()

    test_df = df[
        (df[year_col] >= test_start)
        & (df[year_col] <= test_end)
    ].copy()

    if len(train_df) == 0:
        raise ValueError("Пустой train split для year split")

    if len(val_df) == 0:
        raise ValueError("Пустой val split для year split")

    if len(test_df) == 0:
        raise ValueError("Пустой test split для year split")

    print("=" * 100)
    print("Research 2 | YEAR SPLIT")
    print(f"train years: {train_start}-{train_end} | rows={len(train_df)}")
    print(f"val years:   {val_start}-{val_end} | rows={len(val_df)}")
    print(f"test years:  {test_start}-{test_end} | rows={len(test_df)}")
    print("=" * 100)

    return train_df, val_df, test_df


def _column_belongs_to_group(column_name: str, group_name: str) -> bool:
    cfg = RESEARCH_2_CONFIG
    keywords = cfg["feature_group_keywords"].get(group_name, [])
    col = column_name.lower()
    return any(keyword.lower() in col for keyword in keywords)


def filter_feature_columns_by_set(
    all_feature_cols: list[str],
    feature_set_name: str,
) -> list[str]:
    cfg = RESEARCH_2_CONFIG

    if feature_set_name not in cfg["feature_set_groups"]:
        raise ValueError(f"Неизвестный feature_set_name: {feature_set_name}")

    selected_groups = set(cfg["feature_set_groups"][feature_set_name])

    result_cols: list[str] = []

    for col in all_feature_cols:
        is_group_col = False

        for group_name in cfg["feature_group_keywords"].keys():
            if _column_belongs_to_group(col, group_name):
                is_group_col = True

                if group_name in selected_groups:
                    result_cols.append(col)

                break

        if not is_group_col:
            result_cols.append(col)

    if not result_cols:
        raise ValueError(f"Для feature_set={feature_set_name} не осталось признаков")

    removed = sorted(set(all_feature_cols) - set(result_cols))

    print("=" * 100)
    print(f"Research 2 | FEATURE SET: {feature_set_name}")
    print(f"features kept:    {len(result_cols)}")
    print(f"features removed: {len(removed)}")

    if removed:
        print("removed columns:")
        for col in removed:
            print(f"  - {col}")

    print("=" * 100)

    return result_cols


def apply_train_fitted_quantile_clip_and_log(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = RESEARCH_2_CONFIG

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    cols = [c for c in cfg.get("fert_mach_clip_cols", []) if c in train_df.columns]
    clip_enabled = bool(cfg.get("quantile_clip_enabled", False))
    low_q = float(cfg.get("quantile_clip_low", 0.005))
    high_q = float(cfg.get("quantile_clip_high", 0.995))
    log_enabled = bool(cfg.get("log1p_fert_mach_enabled", False))

    for col in cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        val_df[col] = pd.to_numeric(val_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

        if clip_enabled:
            lower = train_df[col].quantile(low_q)
            upper = train_df[col].quantile(high_q)

            if pd.notna(lower) and pd.notna(upper):
                train_df[col] = train_df[col].clip(lower=lower, upper=upper)
                val_df[col] = val_df[col].clip(lower=lower, upper=upper)
                test_df[col] = test_df[col].clip(lower=lower, upper=upper)

                print(
                    f"[research_2][clip] {col}: "
                    f"train q{low_q:.3f}={lower:.4f}, q{high_q:.3f}={upper:.4f}"
                )

        if log_enabled:
            train_df[col] = train_df[col].clip(lower=0)
            val_df[col] = val_df[col].clip(lower=0)
            test_df[col] = test_df[col].clip(lower=0)

            train_df[col] = np.log1p(train_df[col])
            val_df[col] = np.log1p(val_df[col])
            test_df[col] = np.log1p(test_df[col])

            print(f"[research_2][log1p] {col}")

    return train_df, val_df, test_df


def build_preprocessor(
    train_df: pd.DataFrame,
    feature_set_name: str,
) -> tuple[ColumnTransformer, list[str], list[str], list[str]]:
    cfg = RESEARCH_2_CONFIG

    target_col = cfg["target_col"]
    crop_col = cfg["crop_col"]
    exclude_cols = set(cfg.get("exclude_cols", []))

    all_feature_cols = [
        c for c in train_df.columns
        if c not in {target_col, crop_col, *exclude_cols}
    ]

    feature_cols = filter_feature_columns_by_set(
        all_feature_cols=all_feature_cols,
        feature_set_name=feature_set_name,
    )

    numeric_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(train_df[c])
    ]

    categorical_cols = [
        c for c in feature_cols
        if c not in numeric_cols
    ]

    weather_numeric_cols = [
        c for c in numeric_cols
        if _column_belongs_to_group(c, "weather")
    ]

    fert_mach_numeric_cols = [
        c for c in numeric_cols
        if _column_belongs_to_group(c, "fertilizers")
        or _column_belongs_to_group(c, "machinery")
        or c.endswith("_was_missing")
    ]

    other_numeric_cols = [
        c for c in numeric_cols
        if c not in weather_numeric_cols and c not in fert_mach_numeric_cols
    ]

    transformers: list[tuple[str, Any, list[str]]] = []

    if weather_numeric_cols:
        transformers.append(
            (
                "weather_num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                weather_numeric_cols,
            )
        )

    if fert_mach_numeric_cols:
        transformers.append(
            (
                "fert_mach_num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                fert_mach_numeric_cols,
            )
        )

    if other_numeric_cols:
        transformers.append(
            (
                "other_num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                other_numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError("После исключения колонок не осталось признаков")

    preprocessor = ColumnTransformer(transformers=transformers)

    return preprocessor, feature_cols, numeric_cols, categorical_cols


def prepare_enriched_dataset(
    split_name: str,
    feature_set_name: str = "full",
) -> PreparedEnrichedDataset:
    cfg = RESEARCH_2_CONFIG

    df = load_enriched_dataframe()
    df = clean_enriched_dataframe(df)
    df = add_yield_history_features(df)
    df = add_missing_indicators(df)

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

    crop_col = cfg["crop_col"]
    target_col = cfg["target_col"]

    crop_values = sorted(train_df[crop_col].astype(str).unique().tolist())
    crop_to_id = {crop: idx for idx, crop in enumerate(crop_values)}
    id_to_crop = {idx: crop for crop, idx in crop_to_id.items()}

    before_val = len(val_df)
    before_test = len(test_df)

    val_df = val_df[
        val_df[crop_col].astype(str).isin(crop_to_id.keys())
    ].copy()

    test_df = test_df[
        test_df[crop_col].astype(str).isin(crop_to_id.keys())
    ].copy()

    print(
        f"[research_2][{split_name}][{feature_set_name}] removed unseen crops | "
        f"val_removed={before_val - len(val_df)} | "
        f"test_removed={before_test - len(test_df)}"
    )

    preprocessor, feature_cols, numeric_cols, categorical_cols = build_preprocessor(
        train_df=train_df,
        feature_set_name=feature_set_name,
    )

    X_train = preprocessor.fit_transform(train_df[feature_cols])
    X_val = preprocessor.transform(val_df[feature_cols])
    X_test = preprocessor.transform(test_df[feature_cols])

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_val, "toarray"):
        X_val = X_val.toarray()
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    return PreparedEnrichedDataset(
        X_train=np.asarray(X_train, dtype=np.float32),
        X_val=np.asarray(X_val, dtype=np.float32),
        X_test=np.asarray(X_test, dtype=np.float32),

        y_train=train_df[target_col].to_numpy(dtype=np.float32),
        y_val=val_df[target_col].to_numpy(dtype=np.float32),
        y_test=test_df[target_col].to_numpy(dtype=np.float32),

        crop_train=train_df[crop_col].astype(str).map(crop_to_id).to_numpy(dtype=np.int64),
        crop_val=val_df[crop_col].astype(str).map(crop_to_id).to_numpy(dtype=np.int64),
        crop_test=test_df[crop_col].astype(str).map(crop_to_id).to_numpy(dtype=np.int64),

        crop_to_id=crop_to_id,
        id_to_crop=id_to_crop,

        preprocessor=preprocessor,

        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),

        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,

        split_name=split_name,
        feature_set_name=feature_set_name,
    )


def describe_research_2_dataset() -> dict[str, Any]:
    cfg = RESEARCH_2_CONFIG
    df = load_enriched_dataframe()
    df = clean_enriched_dataframe(df)

    return summarize_dataframe(
        df=df,
        target_col=cfg["target_col"],
        crop_col=cfg["crop_col"],
    )