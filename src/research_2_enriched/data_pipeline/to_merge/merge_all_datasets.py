from pathlib import Path
import pandas as pd


BASE_DATASET_PATH = Path("data/processed/agro_weather_dataset.csv")
FERTILIZERS_PATH = Path("data/processed/mineral_fertilizers_region_year.csv")
TECH_PATH = Path("data/processed/agricultural_tech_region_year.csv")

OUTPUT_PATH = Path("data/processed/russian.csv")


def normalize_region(value):
    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .replace("ё", "е")
        .replace("Ё", "Е")
    )


def read_csv_checked(path: Path, name: str) -> pd.DataFrame:
    print(f"\nЧитаю {name}: {path}", flush=True)

    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}")

    df = pd.read_csv(path, low_memory=False)

    print(f"{name} shape:", df.shape, flush=True)
    print(f"{name} columns:", df.columns.tolist(), flush=True)

    if "region" not in df.columns:
        raise ValueError(f"В {name} нет колонки region")

    if "year" not in df.columns:
        raise ValueError(f"В {name} нет колонки year")

    df["region"] = df["region"].apply(normalize_region)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    df = df.dropna(subset=["region", "year"])
    df = df[df["region"] != ""].copy()
    df["year"] = df["year"].astype(int)

    return df


def main():
    print("СТАРТ merge_all_datasets.py", flush=True)

    base = read_csv_checked(BASE_DATASET_PATH, "base agro+weather")
    fert = read_csv_checked(FERTILIZERS_PATH, "fertilizers")
    tech = read_csv_checked(TECH_PATH, "tech")

    print("\nПроверка дублей по ключу region+year", flush=True)

    fert_duplicates = fert.duplicated(subset=["region", "year"]).sum()
    tech_duplicates = tech.duplicated(subset=["region", "year"]).sum()

    print("fert duplicates:", fert_duplicates, flush=True)
    print("tech duplicates:", tech_duplicates, flush=True)

    if fert_duplicates > 0:
        print("Агрегирую дубли в удобрениях...", flush=True)

        numeric_cols = fert.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "year"]

        fert = (
            fert.groupby(["region", "year"], as_index=False)[numeric_cols]
            .sum(min_count=1)
        )

    if tech_duplicates > 0:
        print("Агрегирую дубли в технике...", flush=True)

        numeric_cols = tech.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "year"]

        tech = (
            tech.groupby(["region", "year"], as_index=False)[numeric_cols]
            .sum(min_count=1)
        )

    print("\nMerge base + fertilizers...", flush=True)

    result = base.merge(
        fert,
        on=["region", "year"],
        how="left",
    )

    print("После удобрений:", result.shape, flush=True)

    print("\nMerge + tech...", flush=True)

    result = result.merge(
        tech,
        on=["region", "year"],
        how="left",
    )

    print("Финальный размер:", result.shape, flush=True)

    print("\nПропуски по ключевым группам:", flush=True)

    check_cols = [
        "target_yield_centner_per_ha",
        "sown_area",
        "gross_harvest",
        "season_temp_mean",
        "season_precip_sum",
        "mineral_fertilizers_centner",
        "mineral_fertilizers_tons",
        "tractors_count",
        "grain_harvesters_count",
        "agricultural_machinery_total_count",
    ]

    for col in check_cols:
        if col in result.columns:
            missing = result[col].isna().sum()
            pct = round(missing / len(result) * 100, 2)
            print(f"{col}: {missing} ({pct}%)", flush=True)
        else:
            print(f"{col}: колонки нет", flush=True)

    print("\nПокрытие удобрений по регионам:", flush=True)
    if "mineral_fertilizers_centner" in result.columns:
        fert_regions = result.loc[
            result["mineral_fertilizers_centner"].notna(),
            "region",
        ].nunique()
        print(fert_regions, "/", result["region"].nunique(), flush=True)

    print("\nПокрытие техники по регионам:", flush=True)
    if "agricultural_machinery_total_count" in result.columns:
        tech_regions = result.loc[
            result["agricultural_machinery_total_count"].notna(),
            "region",
        ].nunique()
        print(tech_regions, "/", result["region"].nunique(), flush=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nСохранено:", OUTPUT_PATH, flush=True)
    print(result.head(20).to_string(), flush=True)


if __name__ == "__main__":
    main()