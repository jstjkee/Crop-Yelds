from __future__ import annotations

import pandas as pd

RUSSIAN_REAL_REQUIRED_COLS = {
    "region_name",
    "year",
    "crop_name_norm_v2",
    "sown_area_ha",
    "gross_harvest_ton",
    "harvested_area_ha_est",
    "target_yield_ton_per_ha",
    "mineral_fertilizer_ton_total",
    "organic_fertilizer_ton_total",
    "has_mineral_fertilizer_data",
    "has_organic_fertilizer_data",
    "mineral_fertilizer_kg_per_ha_proxy",
    "organic_fertilizer_ton_per_ha_proxy",
    "log_mineral_fertilizer_kg_total",
    "log_organic_fertilizer_ton_total",
    "log_mineral_fertilizer_kg_per_ha_proxy",
    "log_organic_fertilizer_ton_per_ha_proxy",
}

CROP_DISPLAY_MAP = {
    "wheat": "Wheat",
    "spring_wheat": "Wheat",
    "winter_wheat": "Wheat",
    "barley": "Barley",
    "spring_barley": "Barley",
    "winter_barley": "Barley",
    "corn": "Maize",
    "maize": "Maize",
    "soybean": "Soybean",
    "rice": "Rice",
    "oats": "Oats",
    "buckwheat": "Buckwheat",
    "millet": "Millet",
    "sunflower": "Sunflower",
    "potato": "Potato",
    "pea": "Pea",
    "rye": "Rye",
    "winter_rye": "Rye",
    "spring_rye": "Rye",
    "rapeseed": "Rapeseed",
    "winter_rapeseed": "Rapeseed",
    "spring_rapeseed": "Rapeseed",
    "sugar_beet": "Sugar Beet",
}


def validate_russian_real_columns(df: pd.DataFrame) -> None:
    missing = sorted(RUSSIAN_REAL_REQUIRED_COLS - set(df.columns))
    if missing:
        raise ValueError(f"В russian_fert.csv не хватает колонок: {missing}")


def to_display_crop_name(value: object) -> str:
    raw = str(value).strip()
    if not raw:
        return "Unknown"
    normalized = raw.lower()
    return CROP_DISPLAY_MAP.get(normalized, raw.replace("_", " ").title())


def build_russian_real_project_view(
    df: pd.DataFrame,
    min_crop_count: int = 3,
) -> pd.DataFrame:
    validate_russian_real_columns(df)

    work = df.copy()
    if "quality_flag_v4" in work.columns:
        work = work[work["quality_flag_v4"].fillna("ok") == "ok"]

    work["Crop"] = work["crop_name_norm_v2"].map(to_display_crop_name)
    work["Yield_tons_per_hectare"] = pd.to_numeric(work["target_yield_ton_per_ha"], errors="coerce")

    project_df = pd.DataFrame(
        {
            "Region": work["region_name"].astype(str),
            "Year": pd.to_numeric(work["year"], errors="coerce"),
            "Crop": work["Crop"].astype(str),
            "Sown_Area_Ha": pd.to_numeric(work["sown_area_ha"], errors="coerce"),
            "Gross_Harvest_Ton": pd.to_numeric(work["gross_harvest_ton"], errors="coerce"),
            "Harvested_Area_Ha_Est": pd.to_numeric(work["harvested_area_ha_est"], errors="coerce"),
            "Mineral_Fertilizer_Ton_Total": pd.to_numeric(work["mineral_fertilizer_ton_total"], errors="coerce"),
            "Organic_Fertilizer_Ton_Total": pd.to_numeric(work["organic_fertilizer_ton_total"], errors="coerce"),
            "Mineral_Fertilizer_Kg_Per_Ha_Proxy": pd.to_numeric(work["mineral_fertilizer_kg_per_ha_proxy"], errors="coerce"),
            "Organic_Fertilizer_Ton_Per_Ha_Proxy": pd.to_numeric(work["organic_fertilizer_ton_per_ha_proxy"], errors="coerce"),
            "Has_Mineral_Fertilizer_Data": pd.to_numeric(work["has_mineral_fertilizer_data"], errors="coerce").fillna(0).astype(int),
            "Has_Organic_Fertilizer_Data": pd.to_numeric(work["has_organic_fertilizer_data"], errors="coerce").fillna(0).astype(int),
            "Log_Mineral_Fertilizer_Kg_Total": pd.to_numeric(work["log_mineral_fertilizer_kg_total"], errors="coerce"),
            "Log_Organic_Fertilizer_Ton_Total": pd.to_numeric(work["log_organic_fertilizer_ton_total"], errors="coerce"),
            "Log_Mineral_Fertilizer_Kg_Per_Ha_Proxy": pd.to_numeric(work["log_mineral_fertilizer_kg_per_ha_proxy"], errors="coerce"),
            "Log_Organic_Fertilizer_Ton_Per_Ha_Proxy": pd.to_numeric(work["log_organic_fertilizer_ton_per_ha_proxy"], errors="coerce"),
            "Yield_tons_per_hectare": pd.to_numeric(work["Yield_tons_per_hectare"], errors="coerce"),
        }
    )

    project_df = project_df.dropna(subset=["Crop", "Yield_tons_per_hectare"])
    project_df = project_df[project_df["Crop"].astype(str).str.strip() != ""]
    project_df = project_df[project_df["Yield_tons_per_hectare"] > 0]

    crop_counts = project_df["Crop"].value_counts()
    valid_crops = crop_counts[crop_counts >= min_crop_count].index
    project_df = project_df[project_df["Crop"].isin(valid_crops)]

    return project_df.reset_index(drop=True)