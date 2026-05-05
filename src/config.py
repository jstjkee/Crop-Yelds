from __future__ import annotations
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

SOURCE_DATA_PATH = DATA_DIR / "raw/crop_yield.csv"

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
    "latent_dim": 12,
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
    "transform_components": 10,
    "num_workers": 0,
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

def ensure_project_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)