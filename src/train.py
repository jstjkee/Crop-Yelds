from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import MODELS_DIR, RANDOM_STATE, TEST_SIZE
from src.preprocessing import build_preprocessor, detect_feature_types


def get_models() -> Dict[str, object]:
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_depth=12,
            min_samples_leaf=5,
        ),
    }


def prepare_data(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


def train_all_models(df: pd.DataFrame, target_col: str) -> Tuple[dict, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=RANDOM_STATE)

    X_train, X_test, y_train, y_test = prepare_data(df, target_col)

    numeric_features, categorical_features = detect_feature_types(df, target_col)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    trained_models = {}

    for model_name, model in get_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
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