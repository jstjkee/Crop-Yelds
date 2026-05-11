from __future__ import annotations

from src.core.config import RAW_DATA_DIR, RESEARCH_1_RESULTS


RESEARCH_1_CONFIG = {
    "name": "research_1_transfer",
    "source_dataset_path": RAW_DATA_DIR / "research_1" / "crop_yield.csv",
    "target_dataset_path": RAW_DATA_DIR / "research_1" / "russian_crop_yield_clean.csv",
    "target_col": "Yield_tons_per_hectare",
    "crop_col": "Crop",

    "model_types": ["mlp_resnet", "transformer", "wide_deep"],
    "feature_modes": ["raw", "pca", "autoencoder"],

    "test_size": 0.2,
    "val_size_from_train": 0.2,
    "min_crop_count": 10,

    "use_log_target": True,
    "balanced_sampler": True,

    "epochs": 12,
    "patience": 8,
    "finetune_epochs": 8,
    "finetune_patience": 4,
    "finetune_lr": 5e-4,

    "feature_config": {
        "pca_n_components": 0.95,
        "pca_random_state": 42,

        "autoencoder_latent_dim": 16,
        "autoencoder_hidden_dim_1": 128,
        "autoencoder_hidden_dim_2": 64,
        "autoencoder_dropout": 0.0,
        "autoencoder_epochs": 12,
        "autoencoder_batch_size": 256,
        "autoencoder_lr": 1e-3,
        "autoencoder_weight_decay": 0.0,
    },

    "results": RESEARCH_1_RESULTS,
}