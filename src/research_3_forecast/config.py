from __future__ import annotations

from src.core.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RESEARCH_3_RESULTS


RESEARCH_3_CONFIG = {
    "name": "research_3_forecast",
    "base_dataset_path": INTERIM_DATA_DIR / "research_2" / "rosstat_agro_full.csv",
    "final_dataset_path": PROCESSED_DATA_DIR / "research_2" / "russian_final_cleaned.csv",

    "target_col": "target_yield_centner_per_ha",
    "crop_col": "crop",
    "year_col": "year",

    "model_types": ["mlp_resnet", "transformer"],
    "feature_modes": ["raw"],
    "feature_sets": ["full"],
    "splits": ["year"],
    "test_years": 2,
    "val_years": 1,
    "min_crop_count": 10,

    "train_years_range": [2000, 2016],
    "val_years_range": [2017, 2020],
    "test_years_range": [2021, 2024],

    "always_exclude_cols": [
        "gross_harvest",
    ],

    "zero_fill_cols": [
        "mineral_fertilizers_centner",
        "mineral_fertilizers_tons",
        "mineral_fertilizers_cumulative_centner",
        "fertilizer_municipality_count",
        "fertilizer_records_count",
        "valid_fertilizer_values_count",
        "agricultural_machinery_total_count",
    ],

    "use_log_target": True,
    "balanced_sampler": True,

    "exclude_cols": [
        "gross_harvest",
        "mineral_fertilizers_cumulative_centner",
        "season_temp_mean",
        "season_temp_max",
        "season_temp_min",
        "season_precip_sum",
        "season_rain_sum",
        "season_humidity_mean",
        "season_hot_days",
        "season_dry_days",
        "season_rain_days",
    ],

    "horizons": [
        {
            "name": "forecast_june",
            "available_months": [4, 5, 6],
        },
        {
            "name": "forecast_july",
            "available_months": [4, 5, 6, 7],
        },
        {
            "name": "forecast_august",
            "available_months": [4, 5, 6, 7, 8],
        },
    ],

    "zero_fill_all_nulls": False,

    "epochs": 40,
    "patience": 10,

    "feature_group_keywords": {
        "weather": [
            "temp",
            "temperature",
            "precip",
            "rain",
            "weather",
            "season",
            "dry_days",
            "hot_days",
            "humidity",
        ],
        "fertilizers": [
            "fertilizer",
            "fertilizers",
            "mineral",
            "удобр",
        ],
        "machinery": [
            "machinery",
            "machine",
            "tractor",
            "combine",
            "tech",
            "equipment",
            "техник",
        ],
    },

    "feature_set_groups": {
        "full": ["weather", "fertilizers", "machinery"],
    },

    "use_yield_history_features": True,
    "yield_history_group_cols": ["region", "crop"],
    "yield_lags": [1, 2, 3],
    "yield_roll_windows": [3],

    "weather_month_cutoff": 9,

    "quality_threshold_r2": 0.90,
    "quality_threshold_rmse": 17.0,

    "dataset_label": "russian_final_cleaned",
    "results": RESEARCH_3_RESULTS,


    "monthly_feature_prefixes": [
        "temp_",
        "precip_",
        "ndvi_",
    ],

}