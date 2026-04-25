from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_target_distribution(df: pd.DataFrame, target_col: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[target_col], kde=True, ax=ax)
    ax.set_title(f"Target distribution: {target_col}")
    ax.set_xlabel(target_col)
    return fig

def plot_actual_vs_predicted(y_true: pd.Series, y_pred: pd.Series, title: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, ax=ax)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)

    return fig

def plot_correlation_matrix(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation matrix")
    return fig

def plot_boxplots(df: pd.DataFrame, columns: list[str]):
    numeric_cols = [col for col in columns if col in df.columns]
    if not numeric_cols:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df[numeric_cols], ax=ax)
    ax.set_title("Boxplots")
    ax.tick_params(axis="x", rotation=45)
    return fig

def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")