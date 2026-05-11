from __future__ import annotations

from pathlib import Path


# ROOT PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[3]

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


def build_results_dirs(base_dir: Path) -> dict[str, Path]:
    return {
        "base": base_dir,
        "models": base_dir / "models",
        "metrics": base_dir / "metrics",
        "tables": base_dir / "tables",
        "figures": base_dir / "figures",
    }


# RESULTS PATHS
RESEARCH_1_RESULTS = build_results_dirs(RESULTS_DIR / "research_1")
RESEARCH_2_RESULTS = build_results_dirs(RESULTS_DIR / "research_2")
RESEARCH_3_RESULTS = build_results_dirs(RESULTS_DIR / "research_3")


# GLOBAL TRAIN SETTINGS
RANDOM_STATE = 42

TRAIN_CONFIG = {
    "batch_size": 64,
    "epochs": 40,
    "patience": 10,
    "lr": 0.00035092801102042304,
    "weight_decay": 0.000557849801675742,
    "huber_delta": 1.2784627139505054,
    "num_workers": 0,
    "clip_grad_norm": 4.462744873698036,
    "finetune_epochs": 8,
    "finetune_patience": 4,
    "finetune_lr": 5e-4,
}


# MODEL CONFIGS
MLP_RESNET_CONFIG = {
    "d_model": 128,
    "hidden_dim": 384,
    "num_blocks": 3,
    "dropout": 0.06560227549662564,
    "head_hidden_dim": 128,
}

TRANSFORMER_CONFIG = {
    "d_model": 32,
    "nhead": 4,
    "num_layers": 1,
    "dim_feedforward": 128,
    "dropout": 0.10,
}

WIDE_DEEP_CONFIG = {
    "deep_hidden_dim": 64,
    "deep_num_layers": 2,
    "deep_dropout": 0.25,
    "head_hidden_dim": 32,
}

TAB_MLP_CONFIG = {
    "hidden_dims": [128, 64, 32],
    "dropout": 0.20,
    "head_hidden_dim": 32,
}

MODEL_TYPES = ["mlp_resnet", "transformer", "wide_deep", "tab_mlp"]


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