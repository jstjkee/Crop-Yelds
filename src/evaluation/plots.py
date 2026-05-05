from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR


def _ensure_figures_dir() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_model_comparison(
    summary_df: pd.DataFrame,
    metric: str = "rmse",
    save_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    required_cols = {"model", "feature_mode", metric}
    missing = required_cols - set(summary_df.columns)
    if missing:
        raise ValueError(f"В summary_df отсутствуют колонки: {sorted(missing)}")

    plot_df = summary_df.copy()
    plot_df["label"] = plot_df["model"] + " | " + plot_df["feature_mode"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(plot_df["label"], plot_df[metric])
    ax.set_xlabel("Модель")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"Сравнение моделей по метрике {metric.upper()}")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    if save_path is not None:
        _ensure_figures_dir()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_transformer_modes_only(
    summary_df: pd.DataFrame,
    metric: str = "rmse",
    save_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    required_cols = {"model", "feature_mode", metric}
    missing = required_cols - set(summary_df.columns)
    if missing:
        raise ValueError(f"В summary_df отсутствуют колонки: {sorted(missing)}")

    plot_df = summary_df[summary_df["model"] == "multihead_transformer"].copy()
    if plot_df.empty:
        raise ValueError("В summary_df нет строк для модели multihead_transformer")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_df["feature_mode"], plot_df[metric])
    ax.set_xlabel("Режим признаков")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"Transformer: сравнение режимов по {metric.upper()}")
    fig.tight_layout()

    if save_path is not None:
        _ensure_figures_dir()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_by_crop_metric(
    by_crop_df: pd.DataFrame,
    metric: str = "rmse",
    save_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    required_cols = {"crop", metric}
    missing = required_cols - set(by_crop_df.columns)
    if missing:
        raise ValueError(f"В by_crop_df отсутствуют колонки: {sorted(missing)}")

    plot_df = by_crop_df.copy().sort_values("crop")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(plot_df["crop"], plot_df[metric])
    ax.set_xlabel("Культура")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"Метрика {metric.upper()} по культурам")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    if save_path is not None:
        _ensure_figures_dir()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_true_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError("y_true и y_pred должны иметь одинаковую длину")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.6)

    min_val = float(min(y_true.min(), y_pred.min()))
    max_val = float(max(y_true.max(), y_pred.max()))
    ax.plot([min_val, max_val], [min_val, max_val])

    ax.set_xlabel("Истинная урожайность")
    ax.set_ylabel("Предсказанная урожайность")
    ax.set_title(title or "True vs Predicted")
    fig.tight_layout()

    if save_path is not None:
        _ensure_figures_dir()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def save_default_source_plots(
    summary_df: pd.DataFrame,
    by_crop_tables: dict[str, pd.DataFrame],
) -> list[Path]:
    _ensure_figures_dir()
    saved_paths: list[Path] = []

    path_1 = FIGURES_DIR / "model_comparison_rmse.png"
    fig_1 = plot_model_comparison(summary_df, metric="rmse", save_path=path_1)
    plt.close(fig_1)
    saved_paths.append(path_1)

    path_2 = FIGURES_DIR / "transformer_modes_r2.png"
    fig_2 = plot_transformer_modes_only(summary_df, metric="r2", save_path=path_2)
    plt.close(fig_2)
    saved_paths.append(path_2)

    for key, table_df in by_crop_tables.items():
        safe_key = str(key).replace(" ", "_").lower()
        path = FIGURES_DIR / f"by_crop_rmse_{safe_key}.png"
        fig = plot_by_crop_metric(
            table_df,
            metric="rmse",
            save_path=path,
            title=f"RMSE по культурам: {key}",
        )
        plt.close(fig)
        saved_paths.append(path)

    return saved_paths