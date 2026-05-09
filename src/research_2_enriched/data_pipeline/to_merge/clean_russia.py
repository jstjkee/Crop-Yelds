from pathlib import Path
import pandas as pd


INPUT_PATH = Path("data/processed/russian_clean2.csv")
OUTPUT_PATH = Path("data/processed/russian_final.csv")


TECH_COLS = [
    "tractors_count",
    "grain_harvesters_count",
    "forage_harvesters_count",
    "potato_harvesters_count",
    "beet_harvesters_count",
    "corn_harvesters_count",
    "flax_harvesters_count",
]


def main():
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    existing_tech_cols = [c for c in TECH_COLS if c in df.columns]

    print("Размер исходного:", df.shape)
    print("Найдены колонки техники:", existing_tech_cols)

    # Заполняем null по отдельным типам техники нулями
    for col in existing_tech_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Пересчитываем общий столбец техники
    df["agricultural_machinery_total_count"] = df[existing_tech_cols].sum(axis=1)

    # Заполняем null в итоговом столбце техники
    df["agricultural_machinery_total_count"] = (
        df["agricultural_machinery_total_count"]
        .fillna(0)
        .round(0)
        .astype(int)
    )

    # Удаляем отдельные типы техники, оставляем только общий показатель
    df = df.drop(columns=existing_tech_cols)

    # Если есть служебные колонки покрытия техники — тоже можно убрать
    service_cols = [
        "tech_municipality_count",
        "tech_records_count",
    ]

    df = df.drop(
        columns=[c for c in service_cols if c in df.columns],
        errors="ignore",
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("Размер после:", df.shape)
    print("Сохранено:", OUTPUT_PATH)

    print("\nПроверка:")
    print(df[["region", "year", "crop", "agricultural_machinery_total_count"]].head(20).to_string())
    print("\nПропуски в agricultural_machinery_total_count:")
    print(df["agricultural_machinery_total_count"].isna().sum())


if __name__ == "__main__":
    main()