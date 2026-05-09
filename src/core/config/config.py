from __future__ import annotations

from pathlib import Path


# -----------------------------------------------------------------------------
# ROOT PATHS
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# -----------------------------------------------------------------------------
# RESULTS PATHS
# -----------------------------------------------------------------------------
RESULTS_RESEARCH_1_DIR = RESULTS_DIR / "research_1"
RESULTS_RESEARCH_2_DIR = RESULTS_DIR / "research_2"
RESULTS_RESEARCH_3_DIR = RESULTS_DIR / "research_3"


def build_results_dirs(base_dir: Path) -> dict[str, Path]:
    return {
        "base": base_dir,
        "models": base_dir / "models",
        "metrics": base_dir / "metrics",
        "tables": base_dir / "tables",
        "figures": base_dir / "figures",
    }


RESEARCH_1_RESULTS = build_results_dirs(RESULTS_RESEARCH_1_DIR)
RESEARCH_2_RESULTS = build_results_dirs(RESULTS_RESEARCH_2_DIR)
RESEARCH_3_RESULTS = build_results_dirs(RESULTS_RESEARCH_3_DIR)


# -----------------------------------------------------------------------------
# GLOBAL TRAIN SETTINGS
# -----------------------------------------------------------------------------
RANDOM_STATE = 42

TRAIN_CONFIG = {
    "batch_size": 256,
    "epochs": 25,
    "patience": 8,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "huber_delta": 1.0,
    "num_workers": 0,
    "clip_grad_norm": 2.0,
}


# -----------------------------------------------------------------------------
# MODEL CONFIGS
# -----------------------------------------------------------------------------
MLP_RESNET_CONFIG = {
    "d_model": 64,
    "hidden_dim": 128,
    "num_blocks": 2,
    "dropout": 0.15,
    "head_hidden_dim": 32,
}

TRANSFORMER_CONFIG = {
    "d_model": 32,
    "nhead": 4,
    "num_layers": 1,
    "dim_feedforward": 128,
    "dropout": 0.10,
}


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def ensure_project_dirs() -> None:
    dirs = [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        DOCS_DIR,
        *RESEARCH_1_RESULTS.values(),
        *RESEARCH_2_RESULTS.values(),
        *RESEARCH_3_RESULTS.values(),
    ]

    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)