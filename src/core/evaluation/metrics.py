from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def regression_metrics_by_crop(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    crop_ids: np.ndarray,
    id_to_crop: dict[int, str],
) -> pd.DataFrame:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    crop_ids = np.asarray(crop_ids).ravel()

    if not (len(y_true) == len(y_pred) == len(crop_ids)):
        raise ValueError("Длины y_true, y_pred и crop_ids должны совпадать")

    rows: list[dict[str, float | int | str]] = []

    for crop_id in sorted(np.unique(crop_ids).tolist()):
        mask = crop_ids == crop_id
        crop_name = id_to_crop.get(int(crop_id), f"crop_{crop_id}")

        metrics = regression_metrics(y_true[mask], y_pred[mask])
        rows.append(
            {
                "crop_id": int(crop_id),
                "crop": crop_name,
                "count": int(mask.sum()),
                "mae": metrics["mae"],
                "mse": metrics["mse"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
            }
        )

    return pd.DataFrame(rows).sort_values(["crop"]).reset_index(drop=True)


def build_metrics_row(
    model_name: str,
    feature_mode: str,
    split_name: str,
    metrics: dict[str, float],
    extra: dict | None = None,
) -> dict:
    row = {
        "model": model_name,
        "feature_mode": feature_mode,
        "split": split_name,
        "mae": metrics["mae"],
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
    }

    if extra:
        row.update(extra)

    return row