from __future__ import annotations

from typing import Dict

import joblib
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from src.config import MODELS_DIR, RANDOM_STATE, TEST_SIZE
from src.preprocessing import (
    build_preprocessor,
    build_transform_pipeline,
    detect_feature_types,
    sanitize_dataframe,
)

def get_models(selected_models: list[str]) -> Dict[str, object]:
    all_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_depth=12,
            min_samples_leaf=5,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=150,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_depth=14,
            min_samples_leaf=3,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=RANDOM_STATE,
            max_depth=10,
            learning_rate=0.08,
            max_iter=200,
        ),
        "KNN Regressor": KNeighborsRegressor(
            n_neighbors=30,
            weights="distance",
        ),
        "SVR": SVR(
            C=1.0,
            epsilon=0.1,
            kernel="rbf",
        ),
    }

    return {name: model for name, model in all_models.items() if name in selected_models}

def prepare_data(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

def train_all_models(
    df: pd.DataFrame,
    target_col: str,
    selected_models: list[str],
    use_pca: bool = False,
    pca_components: int = 7,
    sample_size: int | None = None,
):
    result = df.copy()

    if sample_size is not None and len(result) > sample_size:
        result = result.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)

    result = sanitize_dataframe(result)

    numeric_features, categorical_features = detect_feature_types(result, target_col)

    X_train, X_test, y_train, y_test = prepare_data(result, target_col)

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    trained_models = {}

    for model_name, model in get_models(selected_models).items():
        transform_pipeline = build_transform_pipeline(
            preprocessor=preprocessor,
            use_pca=use_pca,
            pca_components=pca_components,
        )

        pipeline = Pipeline(
            steps=[
                *transform_pipeline.steps,
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        trained_models[model_name] = pipeline

    return trained_models, X_train, y_train, X_test, y_test

def save_model(model, model_name: str) -> str:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")
    path = MODELS_DIR / f"{safe_name}.joblib"
    joblib.dump(model, path)
    return str(path)