import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def analyze_missing(df):
    missing = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
    )

    print(missing[missing > 0])


def analyze_target(df, target):
    print(df[target].describe())

    plt.figure(figsize=(10, 5))
    sns.histplot(df[target], bins=50)
    plt.title("Target distribution")
    plt.show()


def analyze_correlations(df, target):
    num_df = df.select_dtypes(include=["number"])

    corr = num_df.corr(numeric_only=True)

    target_corr = (
        corr[target]
        .drop(target)
        .sort_values(key=lambda x: x.abs(), ascending=False)
    )

    print(target_corr.head(30))

    plt.figure(figsize=(10, 8))
    sns.barplot(
        x=target_corr.head(20).values,
        y=target_corr.head(20).index
    )
    plt.title("Top correlations with target")
    plt.show()


def run_eda(df, target):
    analyze_missing(df)
    analyze_target(df, target)
    analyze_correlations(df, target)