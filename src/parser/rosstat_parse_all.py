import re
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw/rosstat")
OUT_DIR = Path("data/processed")

FILES = [
    {
        "path": RAW_DIR / "rosstat_yield_31533.xls",
        "value_col": "target_yield_centner_per_ha",
    },
    {
        "path": RAW_DIR / "rosstat_area.xls",
        "value_col": "sown_area",
    },
    {
        "path": RAW_DIR / "rosstat_harvest.xls",
        "value_col": "gross_harvest",
    },
]


def parse_year(value):
    if pd.isna(value):
        return None

    match = re.search(r"(19\d{2}|20\d{2})", str(value))
    return int(match.group(1)) if match else None


def parse_float(value):
    if pd.isna(value):
        return None

    text = str(value).replace(",", ".")
    match = re.search(r"-?\d+(\.\d+)?", text)
    return float(match.group(0)) if match else None


def clean_region(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()
    lower = text.lower()

    bad_fragments = [
        "российская федерация",
        "федеральный округ",
        "по 2009 год",
    ]

    for bad in bad_fragments:
        if bad in lower:
            return ""

    return text


def clean_crop(value):
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip().lower()

    bad_fragments = [
        "хозяйства",
        "категорий",
        "фермерские",
        "индивидуальные",
        "предприниматели",
        "сельскохозяйственных организаций",
        "населения",
        "валовой сбор",
        "урожайность",
        "посевные площади",
        "сельскохозяйственных культур",
    ]

    for bad in bad_fragments:
        if bad in text:
            return ""

    return text


def find_year_rows(df):
    rows = []

    for row in range(min(40, df.shape[0])):
        years = []

        for col in range(1, df.shape[1]):
            year = parse_year(df.iloc[row, col])
            if year is not None:
                years.append(year)

        if len(years) >= 3:
            rows.append(
                {
                    "row": row,
                    "count": len(years),
                    "min_year": min(years),
                    "max_year": max(years),
                }
            )

    if not rows:
        raise ValueError("Не найдены строки с годами")

    return pd.DataFrame(rows)


def find_crop_for_column(df, col, year_row):
    filled = df.iloc[:year_row, :].ffill(axis=1)

    for row in range(year_row - 1, -1, -1):
        value = filled.iloc[row, col]

        crop = clean_crop(value)

        if crop:
            return crop

    return ""

def parse_file(path: Path, value_col: str) -> pd.DataFrame:
    print(f"\n==============================")
    print(f"Читаю файл: {path}", flush=True)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    df = pd.read_excel(path, header=None, engine="xlrd")
    print(f"Размер файла: {df.shape}", flush=True)

    year_rows = find_year_rows(df)

    print("\nНайденные возможные строки годов:")
    print(year_rows.to_string(index=False))

    year_row = int(year_rows.sort_values("count", ascending=False).iloc[0]["row"])
    data_start_row = year_row + 1

    print(f"\nИспользуем строку годов: {year_row}", flush=True)

    column_specs = []

    for col in range(1, df.shape[1]):
        year = parse_year(df.iloc[year_row, col])

        if year is None:
            continue

        crop = find_crop_for_column(df, col, year_row)

        if not crop:
            continue

        column_specs.append(
            {
                "col": col,
                "year": year,
                "crop": crop,
            }
        )

    spec_df = pd.DataFrame(column_specs)

    print("\nНайденные годы по культурам в структуре файла:")
    print(
        spec_df.groupby("crop")["year"]
        .agg(["min", "max", "nunique", "count"])
        .sort_values(["nunique", "crop"])
        .to_string()
    )

    regions = []

    for row in range(data_start_row, df.shape[0]):
        region = clean_region(df.iloc[row, 0])
        if region:
            regions.append(
                {
                    "row": row,
                    "region": region,
                }
            )

    region_df = pd.DataFrame(regions)

    records = []

    for _, region_item in region_df.iterrows():
        row = int(region_item["row"])
        region = region_item["region"]

        for spec in column_specs:
            value = parse_float(df.iloc[row, spec["col"]])

            records.append(
                {
                    "region": region,
                    "year": spec["year"],
                    "crop": spec["crop"],
                    value_col: value,
                }
            )

    result = pd.DataFrame(records)

    result["crop"] = result["crop"].apply(clean_crop)
    result = result[result["crop"] != ""].copy()

    result = result.drop_duplicates(
        subset=["region", "year", "crop"],
        keep="first",
    )

    print("\nРаспарсенный файл БЕЗ удаления NaN:")
    print("Строк:", len(result))
    print("Регионов:", result["region"].nunique())
    print("Культур:", result["crop"].nunique())
    print("Годы:", result["year"].min(), "-", result["year"].max())

    print("\nПропуски по показателю:")
    print(result[value_col].isna().sum())

    print("\nФактическая сводка по культурам и годам:")
    print(
        result.groupby("crop")["year"]
        .agg(["min", "max", "nunique", "count"])
        .sort_values(["nunique", "crop"])
        .to_string()
    )

    return result

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = []

    for item in FILES:
        parsed = parse_file(
            path=item["path"],
            value_col=item["value_col"],
        )
        datasets.append(parsed)

    result = datasets[0]

    for df in datasets[1:]:
        result = result.merge(
            df,
            on=["region", "year", "crop"],
            how="outer",
        )

    result["crop"] = result["crop"].apply(clean_crop)
    result = result[result["crop"] != ""].copy()

    result = result.drop_duplicates(
        subset=["region", "year", "crop"],
        keep="first",
    )

    result = result.sort_values(["region", "crop", "year"])

    output_path = OUT_DIR / "rosstat_agro_full.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n==============================")
    print("ИТОГОВЫЙ РОССТАТ-ДАТАСЕТ")
    print(result.head(40).to_string())
    print("\nРазмер:", result.shape)
    print("Регионов:", result["region"].nunique())
    print("Культур:", result["crop"].nunique())
    print("Годы:", result["year"].min(), "-", result["year"].max())

    print("\nФинальная сводка по культурам и годам:")
    print(
        result.groupby("crop")["year"]
        .agg(["min", "max", "nunique", "count"])
        .sort_values(["nunique", "crop"])
        .to_string()
    )

    print("\nСтрок по годам:")
    print(result["year"].value_counts().sort_index().to_string())

    print("\nСохранено:", output_path)


if __name__ == "__main__":
    main()