from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


EPS = 1e-8


def _safe_mean_abs(y_true: np.ndarray) -> float:
    value = float(np.mean(np.abs(y_true)))
    return value if value > EPS else np.nan


def _safe_mape_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mask = np.abs(y_true) > EPS
    if not np.any(mask):
        return np.nan

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError("Длины y_true и y_pred должны совпадать")

    if len(y_true) == 0:
        raise ValueError("Нельзя посчитать метрики для пустого массива")

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    normalizer = _safe_mean_abs(y_true)

    nmae = float(mae / normalizer * 100.0) if not np.isnan(normalizer) else np.nan
    nrmse = float(rmse / normalizer * 100.0) if not np.isnan(normalizer) else np.nan
    nmape = _safe_mape_percent(y_true, y_pred)

    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "nmae_percent": nmae,
        "nrmse_percent": nrmse,
        "nmape_percent": nmape,
        "target_mean_abs": normalizer,
    }


def regression_metrics_by_crop(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    crop_ids: np.ndarray,
    id_to_crop: dict[int, str],
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    crop_ids = np.asarray(crop_ids).ravel()

    if not (len(y_true) == len(y_pred) == len(crop_ids)):
        raise ValueError("Длины y_true, y_pred и crop_ids должны совпадать")

    if len(y_true) == 0:
        raise ValueError("Нельзя посчитать метрики для пустого массива")

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

                "nmae_percent": metrics["nmae_percent"],
                "nrmse_percent": metrics["nrmse_percent"],
                "nmape_percent": metrics["nmape_percent"],

                "target_mean_abs": metrics["target_mean_abs"],
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

        # Абсолютные метрики
        "mae": metrics["mae"],
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],

        # Относительные / нормализованные метрики
        "nmae_percent": metrics.get("nmae_percent", np.nan),
        "nrmse_percent": metrics.get("nrmse_percent", np.nan),
        "nmape_percent": metrics.get("nmape_percent", np.nan),
        "target_mean_abs": metrics.get("target_mean_abs", np.nan),
    }

    if extra:
        row.update(extra)

    return row