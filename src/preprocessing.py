from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # обычный bool
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype("int64")

    # pandas nullable boolean
    nullable_bool_cols = df.select_dtypes(include=["boolean"]).columns.tolist()
    for col in nullable_bool_cols:
        df[col] = df[col].astype("Int64")

    return df

def detect_feature_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    X = df.drop(columns=[target_col])

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    return numeric_features, categorical_features

def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:

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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor

def maybe_add_pca(preprocessor, use_pca: bool, n_components: int | None = None) -> Pipeline:
    steps = [("preprocessor", preprocessor)]

    if use_pca:
        if n_components is None:
            n_components = 7
        steps.append(("pca", PCA(n_components=n_components)))

    return Pipeline(steps=steps)