from __future__ import annotations
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"

SOURCE_DATA_PATH = RAW_DATA_DIR / "crop_yield.csv"
LEGACY_RUSSIAN_DATA_PATH = RAW_DATA_DIR / "russian_crop_yield_clean.csv"
RUSSIAN_FERT_DATA_PATH = RAW_DATA_DIR / "russian_fert.csv"
RUSSIAN_REAL_PROJECT_PATH = PROCESSED_DATA_DIR / "russian_fert_project.csv"
RUSSIAN_FINAL_CLEANED_PATH = PROCESSED_DATA_DIR / "russian_final_cleaned.csv"

DATASET_PATHS = {
    "source": SOURCE_DATA_PATH,
    "russian_legacy": LEGACY_RUSSIAN_DATA_PATH,
    "russian_real_raw": RUSSIAN_FERT_DATA_PATH,
    "russian_real_project": RUSSIAN_REAL_PROJECT_PATH,
    "russian_final_cleaned": RUSSIAN_FINAL_CLEANED_PATH,
}

MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

TARGET_COL = "Yield_tons_per_hectare"
CROP_COL = "Crop"

TEST_SIZE = 0.2
RANDOM_STATE = 42

FEATURE_MODES = [
    "raw",
    "pca",
    "umap",
    "autoencoder",
]

TRANSFORMER_CONFIG = {
    "d_model": 64,
    "nhead": 4,
    "num_layers": 2,
    "dim_feedforward": 256,
    "dropout": 0.1,
}

AUTOENCODER_CONFIG = {
    "latent_dim": 18,
    "hidden_dim_1": 128,
    "hidden_dim_2": 64,
    "epochs": 8,
    "batch_size": 1024,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "dropout": 0.0,
}

TRAIN_CONFIG = {
    "batch_size": 512,
    "epochs": 8,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "transform_components": 8,
    "num_workers": 0,
    "huber_delta": 1.0,
    "finetune_epochs": 8,
    "finetune_lr": 5e-4,
}

BASELINE_CONFIG = {
    "random_forest": {
        "n_estimators": 150,
        "max_depth": None,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    },
    "extra_trees": {
        "n_estimators": 150,
        "max_depth": None,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    },
}

MODEL_TYPES = [
    "transformer",
    "mlp_resnet",
]

DEFAULT_MODEL_TYPE = "transformer"

MLP_RESNET_CONFIG = {
    "d_model": 128,
    "hidden_dim": 256,
    "num_blocks": 4,
    "dropout": 0.15,
    "head_hidden_dim": 64,
}

RUSSIAN_FINAL_CONFIG = {
    "dataset_path": RUSSIAN_FINAL_CLEANED_PATH,
    "target_col": "target_yield_centner_per_ha",
    "crop_col": "crop",
    "dataset_label": "russian_final_cleaned",
    "results_prefix": "russian_final_cleaned",

    "feature_mode": "raw",

    "model_types": ["mlp_resnet", "transformer"],

    "split": "random",

    "test_size": 0.2,
    "val_size_from_train": 0.2,

    "test_years": 2,
    "val_years": 1,

    "min_crop_count": 10,

    "exclude_cols": [
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

    "epochs": 20,
    "patience": 6,
}


def ensure_project_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)