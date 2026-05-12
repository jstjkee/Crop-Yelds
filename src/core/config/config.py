from __future__ import annotations

from pathlib import Path


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


RESEARCH_1_RESULTS = build_results_dirs(RESULTS_DIR / "research_1")
RESEARCH_2_RESULTS = build_results_dirs(RESULTS_DIR / "research_2")
RESEARCH_3_RESULTS = build_results_dirs(RESULTS_DIR / "research_3")

RANDOM_STATE = 42

TRAIN_DEFAULTS = {
    "batch_size": 128,
    "epochs": 25,
    "patience": 8,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "huber_delta": 1.0,
    "num_workers": 0,
    "clip_grad_norm": 2.0,
    "finetune_epochs": 8,
    "finetune_patience": 4,
    "finetune_lr": 5e-4,
}

MODEL_TRAIN_CONFIGS = {
    "mlp_resnet": {
        "batch_size": 64,
        "epochs": 40,
        "patience": 10,
        "lr": 0.00035092801102042304,
        "weight_decay": 0.000557849801675742,
        "huber_delta": 1.2784627139505054,
        "clip_grad_norm": 4.462744873698036,
        "num_workers": 0,
    },
    "transformer": {
        "batch_size": 256,
        "epochs": 20,
        "patience": 10,
        "lr": 0.0007071400074107647,
        "weight_decay": 0.0009004715095952081,
        "huber_delta": 1.0478466680779337,
        "clip_grad_norm": 0.9949174167959878,
        "num_workers": 0,
    },
    "wide_deep": {},
    "tab_mlp": {},
}

MLP_RESNET_CONFIG = {
    "d_model": 128,
    "hidden_dim": 384,
    "num_blocks": 3,
    "dropout": 0.06560227549662564,
    "head_hidden_dim": 128,
}

TRANSFORMER_CONFIG = {
    "d_model": 96,
    "nhead": 8,
    "num_layers": 1,
    "dim_feedforward": 576,
    "dropout": 0.3257351676672524,
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