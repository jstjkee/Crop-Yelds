from __future__ import annotations

from src.core.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RESEARCH_2_RESULTS


RESEARCH_2_CONFIG = {
    "name": "research_2_enriched",

    # datasets
    "base_dataset_path": INTERIM_DATA_DIR / "research_2" / "rosstat_agro_full.csv",
    "final_dataset_path": PROCESSED_DATA_DIR / "research_2" / "russian_final_cleaned.csv",

    # columns
    "target_col": "target_yield_centner_per_ha",
    "crop_col": "crop",

    # model run
    "model_types": ["mlp_resnet", "transformer"],
    "split": "random",          # можно потом переключить на "year"
    "test_size": 0.2,
    "val_size_from_train": 0.2,
    "test_years": 2,
    "val_years": 1,
    "min_crop_count": 10,

    # preprocessing
    # gross_harvest лучше исключать сразу, если он есть в финальной таблице
    "exclude_cols": [
        "gross_harvest",
    ],

    # столбцы, где пропуск = скорее отсутствие значения / события
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

    # сравнение наборов признаков
    "feature_sets": {
        "baseline": [],
        "baseline_plus_weather": ["weather"],
        "baseline_plus_weather_fertilizers": ["weather", "fertilizers"],
        "full": ["weather", "fertilizers", "machinery"],
    },

    # outputs
    "results": RESEARCH_2_RESULTS,
    "dataset_label": "russian_final_cleaned",
}