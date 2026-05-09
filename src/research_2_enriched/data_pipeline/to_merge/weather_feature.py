from pathlib import Path

import pandas as pd


RAW_WEATHER_DIR = Path("data/raw/weather")
OUTPUT_PATH = Path("data/processed/weather_monthly_features.csv")

MONTHS = [4, 5, 6, 7, 8, 9]


def main():
    files = list(RAW_WEATHER_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError("Нет файлов в data/raw/weather")

    print("Файлов погоды:", len(files))

    all_daily = []

    for file in files:
        df = pd.read_csv(file)

        df["time"] = pd.to_datetime(df["time"])
        df["month"] = df["time"].dt.month

        df = df[df["month"].isin(MONTHS)].copy()

        df["hot_day"] = df["temperature_2m_max"] >= 27
        df["dry_day"] = df["precipitation_sum"] <= 1
        df["rain_day"] = df["precipitation_sum"] > 1

        all_daily.append(df)

    daily = pd.concat(all_daily, ignore_index=True)

    monthly = (
        daily.groupby(["region", "year", "month"])
        .agg(
            temp_mean=("temperature_2m_mean", "mean"),
            temp_max=("temperature_2m_max", "max"),
            temp_min=("temperature_2m_min", "min"),
            precip_sum=("precipitation_sum", "sum"),
            rain_sum=("rain_sum", "sum"),
            humidity_mean=("relative_humidity_2m_mean", "mean"),
            hot_days=("hot_day", "sum"),
            dry_days=("dry_day", "sum"),
            rain_days=("rain_day", "sum"),
        )
        .reset_index()
    )

    wide = monthly.pivot_table(
        index=["region", "year"],
        columns="month",
        values=[
            "temp_mean",
            "temp_max",
            "temp_min",
            "precip_sum",
            "rain_sum",
            "humidity_mean",
            "hot_days",
            "dry_days",
            "rain_days",
        ],
    )

    wide.columns = [
        f"{feature}_{month:02d}"
        for feature, month in wide.columns
    ]

    wide = wide.reset_index()

    season = (
        daily.groupby(["region", "year"])
        .agg(
            season_temp_mean=("temperature_2m_mean", "mean"),
            season_temp_max=("temperature_2m_max", "max"),
            season_temp_min=("temperature_2m_min", "min"),
            season_precip_sum=("precipitation_sum", "sum"),
            season_rain_sum=("rain_sum", "sum"),
            season_humidity_mean=("relative_humidity_2m_mean", "mean"),
            season_hot_days=("hot_day", "sum"),
            season_dry_days=("dry_day", "sum"),
            season_rain_days=("rain_day", "sum"),
        )
        .reset_index()
    )

    result = wide.merge(
        season,
        on=["region", "year"],
        how="left",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Сохранено:", OUTPUT_PATH)
    print("Размер:", result.shape)
    print(result.head().to_string())


if __name__ == "__main__":
    main()