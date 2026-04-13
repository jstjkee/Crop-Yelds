import io

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_actual_vs_predicted(y_true: pd.Series, y_pred: pd.Series, title: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, ax=ax)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    return fig


def plot_target_distribution(df: pd.DataFrame, target_col: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[target_col], kde=True, ax=ax)
    ax.set_title(f"Target distribution: {target_col}")
    return fig


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Подготовка DataFrame для скачивания в Streamlit.
    """
    return df.to_csv(index=False).encode("utf-8")