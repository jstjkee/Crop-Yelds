from __future__ import annotations

import pandas as pd

from src.config import (
    RANDOM_STATE,
    RUSSIAN_FERT_DATA_PATH,
    RUSSIAN_REAL_PROJECT_PATH,
    TARGET_COL,
    CROP_COL,
    ensure_project_dirs,
)
from src.data.load_data import load_and_validate_source_data
from src.data.russian_parser import prepare_russian_real_dataset
from src.training.train_source import (
    get_device,
    run_training_pipeline,
    seed_everything,
)


def main() -> None:
    ensure_project_dirs()
    seed_everything(RANDOM_STATE)
    device = get_device()

    print(f"Device: {device}")
    print(f"Raw russian fert path: {RUSSIAN_FERT_DATA_PATH}")

    prepare_russian_real_dataset(
        input_path=RUSSIAN_FERT_DATA_PATH,
        output_path=RUSSIAN_REAL_PROJECT_PATH,
        min_crop_count=3,
        force=True,
        filter_to_source_crops=True,
    )

    df = load_and_validate_source_data(
        path=RUSSIAN_REAL_PROJECT_PATH,
        target_col=TARGET_COL,
        crop_col=CROP_COL,
    )

    run_training_pipeline(
        df=df,
        dataset_label="russian_real_v4_overlap",
        target_col=TARGET_COL,
        crop_col=CROP_COL,
        results_prefix="russia_real_v4_overlap",
        device=device,
    )


if __name__ == "__main__":
    main()