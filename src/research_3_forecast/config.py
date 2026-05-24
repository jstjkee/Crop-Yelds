from __future__ import annotations

from src.core.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RESEARCH_3_RESULTS


RESEARCH_3_CONFIG = {
    "name": "research_3_forecast",

    "base_dataset_path": INTERIM_DATA_DIR / "research_2" / "rosstat_agro_full.csv",
    "final_dataset_path": PROCESSED_DATA_DIR / "research_2" / "russian_final_cleaned.csv",

    "target_col": "target_yield_centner_per_ha",
    "crop_col": "crop",
    "year_col": "year",

    "model_types": ["mlp_resnet", "transformer", "tab_mlp", "wide_deep"],
    "feature_modes": ["raw"],
    "splits": ["year"],
    "feature_sets": ["full"],

    "train_years_range": [2000, 2016],
    "val_years_range": [2017, 2020],
    "test_years_range": [2021, 2024],

    "min_crop_count": 10,

    "exclude_cols": [
        "gross_harvest",
        "mineral_fertilizers_cumulative_centner",
        #"season_temp_mean",
       # "season_temp_max",
        #"season_temp_min",
        #"season_precip_sum",
        #"season_rain_sum",
        #"season_rain_days",
        #"season_humidity_mean",
        #"season_hot_days",
        #"season_dry_days",

        #"dry_days_04", "dry_days_05", "dry_days_06", "dry_days_07", "dry_days_08", "dry_days_09",
        #"hot_days_04", "hot_days_05", "hot_days_06", "hot_days_07", "hot_days_08", "hot_days_09",
        #"humidity_mean_04", "humidity_mean_05", "humidity_mean_06", "humidity_mean_07", "humidity_mean_08", "humidity_mean_09",
        #"precip_sum_04", "precip_sum_05", "precip_sum_06", "precip_sum_07", "precip_sum_08", "precip_sum_09",
        #"rain_days_04", "rain_days_05", "rain_days_06", "rain_days_07", "rain_days_08", "rain_days_09",
        #"rain_sum_04", "rain_sum_05", "rain_sum_06", "rain_sum_07", "rain_sum_08", "rain_sum_09",
        #"temp_max_04", "temp_max_05", "temp_max_06", "temp_max_07", "temp_max_08", "temp_max_09",
        #"temp_mean_04", "temp_mean_05", "temp_mean_06", "temp_mean_07", "temp_mean_08", "temp_mean_09",
        #"temp_min_04", "temp_min_05", "temp_min_06", "temp_min_07", "temp_min_08", "temp_min_09",
        ],

    "zero_fill_all_nulls": False,
    "use_log_target": True,
    "balanced_sampler": True,
    "use_agro_weather_windows": False,

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

    "scenario_name": "F0_full_nowcast",

    "scenarios": {
        # Полный nowcast
        "F0_full_nowcast": {
            "weather_month_cutoff": 9,
            "keep_sown_area": True,
            "keep_fertilizers": True,
            "keep_machinery": True,
            "keep_fert_coverage": True,
        },

        "F1_mid_july": {
            "weather_month_cutoff": 7,
            "keep_sown_area": True,
            "keep_fertilizers": True,
            "keep_machinery": True,
            "keep_fert_coverage": True,
        },
        "F1_mid_june": {
            "weather_month_cutoff": 6,
            "keep_sown_area": True,
            "keep_fertilizers": True,
            "keep_machinery": True,
            "keep_fert_coverage": True,
        },

        "F2_early_may": {
            "weather_month_cutoff": 5,
            "keep_sown_area": True,
            "keep_fertilizers": False,
            "keep_machinery": False,
            "keep_fert_coverage": False,
        },
        "F2_early_april": {
            "weather_month_cutoff": 4,
            "keep_sown_area": True,
            "keep_fertilizers": False,
            "keep_machinery": False,
            "keep_fert_coverage": False,
        },

        "F3_preseason_operational": {
            "weather_month_cutoff": None,
            "keep_sown_area": True,
            "keep_fertilizers": False,
            "keep_machinery": False,
            "keep_fert_coverage": False,
        },
        "F3_preseason_strict": {
            "weather_month_cutoff": None,
            "keep_sown_area": False,
            "keep_fertilizers": False,
            "keep_machinery": False,
            "keep_fert_coverage": False,
        },
        "F2_windows_only": {
            "weather_month_cutoff": None,
            "keep_weather_windows": True,
            "keep_sown_area": True,
            "keep_fertilizers": False,
            "keep_machinery": False,
            "keep_fert_coverage": False,
        },
    },

    "quality_thresholds": {
        "F0_full_nowcast": {"r2": 0.90, "rmse": 17.0},
        "F1_mid_july": {"r2": 0.90, "rmse": 17.0},
        "F1_mid_june": {"r2": 0.90, "rmse": 17.0},
        "F2_early_may": {"r2": 0.88, "rmse": 18.0},
        "F2_early_april": {"r2": 0.88, "rmse": 18.0},
        "F3_preseason_operational": {"r2": 0.85, "rmse": 19.0},
        "F3_preseason_strict": {"r2": 0.85, "rmse": 19.0},
    },

    "agro_weather_windows": [
    ("precip_sum", [4, 5], "precip_04_05"),
    ("precip_sum", [5, 6], "precip_05_06"),
    ("precip_sum", [6, 7], "precip_06_07"),
    ("hot_days", [6, 7], "hot_days_06_07"),
    ("dry_days", [5, 6], "dry_days_05_06"),
    ("dry_days", [6, 7], "dry_days_06_07"),
    ("temp_mean", [5, 6], "temp_mean_05_06"),
    ("temp_mean", [6, 7], "temp_mean_06_07"),
    ("humidity_mean", [5, 6], "humidity_mean_05_06"),
    ("humidity_mean", [6, 7], "humidity_mean_06_07"),
        ],

    "dataset_label": "russian_final_cleaned",
    "results": RESEARCH_3_RESULTS,
}