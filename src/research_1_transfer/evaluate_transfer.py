from __future__ import annotations

import pandas as pd
import torch

from src.core.config import MODEL_TYPES, RANDOM_STATE, ensure_project_dirs
from src.core.data.target_scaler import TargetScaler
from src.core.evaluation.metrics import build_metrics_row, regression_metrics, regression_metrics_by_crop
from src.core.evaluation.reports import print_summary, reorder_summary_columns
from src.core.training.dl_trainer import (
    get_device,
    load_model_from_checkpoint,
    make_loader,
    predict_unscaled,
    seed_everything,
)
from src.research_1_transfer.config import RESEARCH_1_CONFIG
from src.research_1_transfer.datasets import prepare_source_dataset, prepare_target_external
from src.research_1_transfer.features import transform_with_feature_artifact


def _checkpoint_path(model_type: str, feature_mode: str):
    return RESEARCH_1_CONFIG["results"]["models"] / f"source_{model_type}_{feature_mode}.pt"


def evaluate_one_model(
    model_type: str,
    feature_mode: str,
    prepared_source,
    external,
    device: str,
) -> dict:
    if len(external["X"]) == 0:
        raise ValueError("После фильтрации не осталось строк для zero-shot transfer")

    checkpoint_path = _checkpoint_path(model_type, feature_mode)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Сначала обучи source-модель: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = load_model_from_checkpoint(checkpoint=checkpoint, device=device)
    target_scaler = TargetScaler.from_dict(checkpoint["target_scaler"])

    X_external = transform_with_feature_artifact(
        artifact=checkpoint["feature_artifact"],
        X=external["X"],
        device=device,
    )

    loader = make_loader(
        X=X_external,
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
        RESEARCH_1_CONFIG["results"]["tables"] / f"transfer_zero_shot_metrics_by_crop_{model_type}_{feature_mode}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pred_df = external["df"].copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred
    pred_df["abs_error"] = (pred_df["y_true"] - pred_df["y_pred"]).abs()
    pred_df.to_csv(
        RESEARCH_1_CONFIG["results"]["tables"] / f"transfer_zero_shot_predictions_{model_type}_{feature_mode}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return build_metrics_row(
        model_name=model_type,
        feature_mode=feature_mode,
        split_name="target_zero_shot",
        metrics=metrics,
        extra={
            "dataset": "russian_crop_yield_clean",
            "num_rows": int(len(pred_df)),
            "num_known_crops": int(pred_df[RESEARCH_1_CONFIG["crop_col"]].nunique()),
            "checkpoint_path": str(checkpoint_path),
        },
    )


def main(
    models: list[str] | None = None,
    feature_modes: list[str] | None = None,
) -> None:
    ensure_project_dirs()
    seed_everything(RANDOM_STATE)
    device = get_device()
    selected_models = models or list(RESEARCH_1_CONFIG.get("model_types", MODEL_TYPES))
    selected_feature_modes = feature_modes or list(RESEARCH_1_CONFIG.get("feature_modes", ["raw"]))

    prepared_source = prepare_source_dataset()
    external = prepare_target_external(prepared_source)

    rows = []
    for feature_mode in selected_feature_modes:
        for model_type in selected_models:
            print("=" * 100)
            print(f"Research 1 | zero-shot transfer | model={model_type} | feature_mode={feature_mode} | device={device}")
            rows.append(
                evaluate_one_model(
                    model_type=model_type,
                    feature_mode=feature_mode,
                    prepared_source=prepared_source,
                    external=external,
                    device=device,
                )
            )

    summary_df = pd.DataFrame(rows).sort_values(["feature_mode", "rmse"]).reset_index(drop=True)
    summary_df = reorder_summary_columns(summary_df)

    out_path = RESEARCH_1_CONFIG["results"]["metrics"] / "transfer_zero_shot_metrics.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print_summary(summary_df)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()