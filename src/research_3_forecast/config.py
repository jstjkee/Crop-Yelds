from __future__ import annotations

from src.core.config import PROCESSED_DATA_DIR, RESEARCH_3_RESULTS


RESEARCH_3_CONFIG = {
    "name": "research_3_forecast",

    # base dataset for building forecast views
    "base_dataset_path": PROCESSED_DATA_DIR / "research_2" / "russian_final_cleaned.csv",

    # output dir for generated forecast datasets
    "forecast_data_dir": PROCESSED_DATA_DIR / "research_3",

    # columns
    "target_col": "target_yield_centner_per_ha",
    "crop_col": "crop",
    "year_col": "year",

    # training
    "model_types": ["mlp_resnet", "transformer"],
    "split": "year",
    "test_years": 2,
    "val_years": 1,
    "min_crop_count": 10,

    # leak-prone columns
    "always_exclude_cols": [
        "gross_harvest",
    ],

    # пропуски
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

    # forecast horizons
    # пример: прогноз на конец июня, июля, августа
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

    # правила отбора месячных признаков
    # здесь потом подстроим под реальные названия колонок
    "monthly_feature_prefixes": [
        "temp_",
        "precip_",
        "ndvi_",
    ],

    # outputs
    "results": RESEARCH_3_RESULTS,
    "dataset_label": "forecast",
}