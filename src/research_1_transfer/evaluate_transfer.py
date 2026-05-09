from __future__ import annotations

import pandas as pd
import torch

from src.core.config import MODEL_TYPES, RANDOM_STATE, ensure_project_dirs
from src.core.data.target_scaler import TargetScaler
from src.core.evaluation.metrics import build_metrics_row, regression_metrics, regression_metrics_by_crop
from src.core.training.dl_trainer import (
    get_device,
    load_model_from_checkpoint,
    make_loader,
    predict_unscaled,
    seed_everything,
)
from src.research_1_transfer.config import RESEARCH_1_CONFIG
from src.research_1_transfer.datasets import prepare_source_dataset, prepare_target_external


def _checkpoint_path(model_type: str):
    return RESEARCH_1_CONFIG["results"]["models"] / f"source_{model_type}.pt"


def evaluate_one_model(model_type: str, prepared_source, external, device: str) -> dict:
    if len(external["X"]) == 0:
        raise ValueError("После фильтрации не осталось строк для zero-shot transfer")

    checkpoint_path = _checkpoint_path(model_type)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Сначала обучи source-модель: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = load_model_from_checkpoint(checkpoint=checkpoint, device=device)
    target_scaler = TargetScaler.from_dict(checkpoint["target_scaler"])

    loader = make_loader(
        X=external["X"],
        y=target_scaler.transform(external["y"]),
        crop_ids=external["crop_ids"],
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    y_true, y_pred, crop_ids = predict_unscaled(
        model=model,
        loader=loader,
        device=device,
        target_scaler=target_scaler,
    )
    metrics = regression_metrics(y_true, y_pred)

    by_crop_df = regression_metrics_by_crop(
        y_true=y_true,
        y_pred=y_pred,
        crop_ids=crop_ids,
        id_to_crop=checkpoint["id_to_crop"],
    )
    by_crop_df.to_csv(
        RESEARCH_1_CONFIG["results"]["tables"] / f"transfer_zero_shot_metrics_by_crop_{model_type}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pred_df = external["df"].copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred
    pred_df["abs_error"] = (pred_df["y_true"] - pred_df["y_pred"]).abs()
    pred_df.to_csv(
        RESEARCH_1_CONFIG["results"]["tables"] / f"transfer_zero_shot_predictions_{model_type}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return build_metrics_row(
        model_name=model_type,
        feature_mode="raw",
        split_name="target_zero_shot",
        metrics=metrics,
        extra={
            "dataset": "russian_crop_yield_clean",
            "num_rows": int(len(pred_df)),
            "num_known_crops": int(pred_df[RESEARCH_1_CONFIG["crop_col"]].nunique()),
            "checkpoint_path": str(checkpoint_path),
        },
    )


def main(models: list[str] | None = None) -> None:
    ensure_project_dirs()
    seed_everything(RANDOM_STATE)
    device = get_device()
    selected_models = models or list(RESEARCH_1_CONFIG.get("model_types", MODEL_TYPES))

    prepared_source = prepare_source_dataset()
    external = prepare_target_external(prepared_source)

    rows = []
    for model_type in selected_models:
        print("=" * 100)
        print(f"Research 1 | zero-shot transfer | model={model_type} | device={device}")
        rows.append(
            evaluate_one_model(
                model_type=model_type,
                prepared_source=prepared_source,
                external=external,
                device=device,
            )
        )

    summary_df = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    out_path = RESEARCH_1_CONFIG["results"]["metrics"] / "transfer_zero_shot_metrics.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(summary_df)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()