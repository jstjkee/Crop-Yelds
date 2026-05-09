from pathlib import Path
import pandas as pd


print("СТАРТ merge_agro_weather.py", flush=True)

AGRO_PATH = Path("data/processed/rosstat_agro_panel.csv")
WEATHER_PATH = Path("data/processed/weather_monthly_features.csv")
OUTPUT_PATH = Path("data/processed/agro_weather_dataset.csv")


def main():
    print("Запущен main()", flush=True)

    print("Проверяю файлы...", flush=True)
    print("AGRO exists:", AGRO_PATH.exists(), AGRO_PATH, flush=True)
    print("WEATHER exists:", WEATHER_PATH.exists(), WEATHER_PATH, flush=True)

    if not AGRO_PATH.exists():
        raise FileNotFoundError(f"Не найден файл Росстата: {AGRO_PATH}")

    if not WEATHER_PATH.exists():
        raise FileNotFoundError(f"Не найден файл погоды: {WEATHER_PATH}")

    print("Читаю Росстат...", flush=True)
    agro = pd.read_csv(AGRO_PATH)
    print("Росстат прочитан:", agro.shape, flush=True)
    print("Колонки Росстата:", agro.columns.tolist(), flush=True)

    print("Читаю погоду...", flush=True)
    weather = pd.read_csv(WEATHER_PATH)
    print("Погода прочитана:", weather.shape, flush=True)
    print("Колонки погоды:", weather.columns.tolist()[:20], "...", flush=True)

    print("Проверяю ключи...", flush=True)

    for col in ["region", "year"]:
        if col not in agro.columns:
            raise ValueError(f"В Росстате нет колонки: {col}")
        if col not in weather.columns:
            raise ValueError(f"В погоде нет колонки: {col}")

    agro["region"] = agro["region"].astype(str).str.strip()
    weather["region"] = weather["region"].astype(str).str.strip()

    agro["year"] = agro["year"].astype(int)
    weather["year"] = weather["year"].astype(int)

    print("Уникальных region-year в Росстате:", agro[["region", "year"]].drop_duplicates().shape[0], flush=True)
    print("Уникальных region-year в погоде:", weather[["region", "year"]].drop_duplicates().shape[0], flush=True)

    print("Merge...", flush=True)
    result = agro.merge(
        weather,
        on=["region", "year"],
        how="left",
    )

    print("Итог:", result.shape, flush=True)

    print("\nПропуски по ключевым колонкам:", flush=True)
    check_cols = [
        "target_yield_centner_per_ha",
        "sown_area",
        "gross_harvest",
        "season_temp_mean",
        "season_precip_sum",
        "season_humidity_mean",
    ]

    for col in check_cols:
        if col in result.columns:
            print(f"{col}: {result[col].isna().sum()}", flush=True)
        else:
            print(f"{col}: колонки нет", flush=True)

    print("\nПример регионов без погоды:", flush=True)
    if "season_temp_mean" in result.columns:
        no_weather = (
            result[result["season_temp_mean"].isna()]["region"]
            .drop_duplicates()
            .head(30)
        )
        print(no_weather.to_string(index=False), flush=True)

    print("Сохраняю...", flush=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Сохранено:", OUTPUT_PATH, flush=True)
    print(result.head(10).to_string(), flush=True)


if __name__ == "__main__":
    main()