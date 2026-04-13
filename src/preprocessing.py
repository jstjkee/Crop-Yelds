from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def detect_feature_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str], List[str]]:
    X = df.drop(columns=[target_col])

    bool_features = X.select_dtypes(include=["bool"]).columns.tolist()
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # bool не дублируем в numeric
    numeric_features = [col for col in numeric_features if col not in bool_features]

    return numeric_features, categorical_features, bool_features


def cast_bool_to_int(df: pd.DataFrame, bool_features: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in bool_features:
        df[col] = df[col].astype(int)
    return df


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    bool_features: List[str],
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

    bool_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
            ("bool", bool_pipeline, bool_features),
        ]
    )

    return preprocessor


def maybe_add_pca(preprocessor, use_pca: bool, n_components: int | None = None) -> Pipeline:
    """
    Возвращает pipeline с optional PCA.
    """
    steps = [("preprocessor", preprocessor)]

    if use_pca:
        if n_components is None:
            n_components = 7
        steps.append(("pca", PCA(n_components=n_components)))

    return Pipeline(steps=steps)