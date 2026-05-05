from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from src.config import BASELINE_CONFIG
from src.evaluation.metrics import regression_metrics


@dataclass
class BaselineResult:
    model_name: str
    metrics: dict[str, float]
    model: object


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestRegressor:
    cfg = BASELINE_CONFIG["random_forest"]

    model = RandomForestRegressor(
        n_estimators=cfg.get("n_estimators", 300),
        max_depth=cfg.get("max_depth", None),
        n_jobs=cfg.get("n_jobs", -1),
        random_state=cfg.get("random_state", 42),
    )
    model.fit(X_train, y_train)
    return model


def train_extra_trees(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ExtraTreesRegressor:
    cfg = BASELINE_CONFIG["extra_trees"]

    model = ExtraTreesRegressor(
        n_estimators=cfg.get("n_estimators", 300),
        max_depth=cfg.get("max_depth", None),
        n_jobs=cfg.get("n_jobs", -1),
        random_state=cfg.get("random_state", 42),
    )
    model.fit(X_train, y_train)
    return model


def evaluate_sklearn_regressor(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    preds = model.predict(X_test)
    return regression_metrics(y_test, preds)


def train_all_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> list[BaselineResult]:
    results: list[BaselineResult] = []

    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_sklearn_regressor(rf_model, X_test, y_test)
    results.append(
        BaselineResult(
            model_name="random_forest",
            metrics=rf_metrics,
            model=rf_model,
        )
    )

    et_model = train_extra_trees(X_train, y_train)
    et_metrics = evaluate_sklearn_regressor(et_model, X_test, y_test)
    results.append(
        BaselineResult(
            model_name="extra_trees",
            metrics=et_metrics,
            model=et_model,
        )
    )

    return results