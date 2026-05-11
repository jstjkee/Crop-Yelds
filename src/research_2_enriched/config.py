from __future__ import annotations

from src.core.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RESEARCH_2_RESULTS


RESEARCH_2_CONFIG = {
    "name": "research_2_enriched",

    "base_dataset_path": INTERIM_DATA_DIR / "research_2" / "rosstat_agro_full.csv",
    "final_dataset_path": PROCESSED_DATA_DIR / "research_2" / "russian_final_cleaned.csv",

    "target_col": "target_yield_centner_per_ha",
    "crop_col": "crop",
    "year_col": "year",

    "model_types": ["mlp_resnet", "transformer", "tab_mlp"],
    "feature_modes": ["raw"],

    # Во 2 исследовании концентрируемся на честной схеме
    "splits": ["year"],

    "test_size": 0.2,
    "val_size_from_train": 0.2,

    "train_years_range": [2000, 2016],
    "val_years_range": [2017, 2020],
    "test_years_range": [2021, 2024],

    "min_crop_count": 10,

    "exclude_cols": [
        "gross_harvest",
        "mineral_fertilizers_cumulative_centner",
    ],

    # Не делаем глобальный fillna(0) по всему датасету
    "zero_fill_all_nulls": False,

    "use_log_target": True,
    "balanced_sampler": True,

    "epochs": 25,
    "patience": 8,

    # Во 2 исследовании работаем только с полным датасетом
    "feature_sets": [
        "full",
    ],

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

    "dataset_label": "russian_final_cleaned",
    "results": RESEARCH_2_RESULTS,
}