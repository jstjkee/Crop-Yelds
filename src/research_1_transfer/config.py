from __future__ import annotations

from src.core.config import RAW_DATA_DIR, RESEARCH_1_RESULTS


RESEARCH_1_CONFIG = {
    "name": "research_1_transfer",
    "source_dataset_path": RAW_DATA_DIR / "research_1" / "crop_yield.csv",
    "target_dataset_path": RAW_DATA_DIR / "research_1" / "russian_crop_yield_clean.csv",
    "target_col": "Yield_tons_per_hectare",
    "crop_col": "Crop",
    "model_types": ["mlp_resnet", "transformer"],
    "test_size": 0.2,
    "val_size_from_train": 0.2,
    "min_crop_count": 10,
    "use_log_target": True,
    "balanced_sampler": True,
    "epochs": 25,
    "patience": 8,
    "finetune_epochs": 8,
    "finetune_patience": 4,
    "results": RESEARCH_1_RESULTS,
}