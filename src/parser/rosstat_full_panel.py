from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/rosstat_agro_full.csv")
OUTPUT_PATH = Path("data/processed/rosstat_agro_panel.csv")


START_YEAR = 2000
END_YEAR = 2025


def main():
    print("Чтение исходного датасета...")

    df = pd.read_csv(INPUT_PATH)

    print("Размер исходного:", df.shape)

    # -----------------------------
    # Уникальные сущности
    # -----------------------------

    regions = sorted(df["region"].dropna().unique())
    crops = sorted(df["crop"].dropna().unique())
    years = list(range(START_YEAR, END_YEAR + 1))

    print("Регионов:", len(regions))
    print("Культур:", len(crops))
    print("Лет:", len(years))

    # -----------------------------
    # Полная сетка
    # -----------------------------

    print("\nСоздаем полную panel-сетку...")

    full_index = pd.MultiIndex.from_product(
        [regions, crops, years],
        names=["region", "crop", "year"],
    )

    panel = pd.DataFrame(index=full_index).reset_index()

    print("Размер полной сетки:", panel.shape)

    # -----------------------------
    # Merge с реальными данными
    # -----------------------------

    print("\nОбъединяем с Росстатом...")

    result = panel.merge(
        df,
        on=["region", "crop", "year"],
        how="left",
    )

    # -----------------------------
    # Диагностика пропусков
    # -----------------------------

    print("\nПропуски:")

    for col in [
        "target_yield_centner_per_ha",
        "sown_area",
        "gross_harvest",
    ]:
        if col in result.columns:
            missing = result[col].isna().sum()
            pct = round(missing / len(result) * 100, 2)

            print(f"{col}: {missing} ({pct}%)")

    # -----------------------------
    # Сводка
    # -----------------------------

    print("\nИтоговая panel-структура:")
    print("Размер:", result.shape)

    print("\nГоды:")
    print(result["year"].min(), "-", result["year"].max())

    print("\nПример:")
    print(result.head(30).to_string())

    # -----------------------------
    # Сохранение
    # -----------------------------

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nСохранено: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()