from __future__ import annotations

import json
from typing import Any

import joblib
import pandas as pd
import torch
import torch.nn as nn

from src.core.config import (
    MLP_RESNET_CONFIG,
    MODEL_TYPES,
    RANDOM_STATE,
    TRAIN_CONFIG,
    TRANSFORMER_CONFIG,
    ensure_project_dirs,
)
from src.core.data.target_scaler import TargetScaler
from src.core.evaluation.metrics import build_metrics_row, regression_metrics, regression_metrics_by_crop
from src.core.training.dl_trainer import (
    build_model,
    fit_with_early_stopping,
    get_device,
    make_loader,
    predict_unscaled,
    seed_everything,
)
from src.research_1_transfer.config import RESEARCH_1_CONFIG
from src.research_1_transfer.datasets import describe_stage1_datasets, prepare_source_dataset


def _model_config(model_type: str) -> dict[str, Any]:
    if model_type == "mlp_resnet":
        return MLP_RESNET_CONFIG
    if model_type == "transformer":
        return TRANSFORMER_CONFIG
    raise ValueError(f"Неизвестный model_type: {model_type}")


def _checkpoint_path(model_type: str):
    return RESEARCH_1_CONFIG["results"]["models"] / f"source_{model_type}.pt"


def train_one_model(model_type: str, prepared, device: str) -> dict[str, Any]:
    cfg = RESEARCH_1_CONFIG

    target_scaler = TargetScaler.fit(
        prepared.y_train,
        use_log_target=bool(cfg.get("use_log_target", True)),
    )

    batch_size = int(TRAIN_CONFIG.get("batch_size", 256))
    num_workers = int(TRAIN_CONFIG.get("num_workers", 0))
    balanced_sampler = bool(cfg.get("balanced_sampler", False))

    train_loader = make_loader(
        X=prepared.X_train,
        y=target_scaler.transform(prepared.y_train),
        crop_ids=prepared.crop_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        balanced_sampler=balanced_sampler,
    )
    val_loader = make_loader(
        X=prepared.X_val,
        y=target_scaler.transform(prepared.y_val),
        crop_ids=prepared.crop_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = make_loader(
        X=prepared.X_test,
        y=target_scaler.transform(prepared.y_test),
        crop_ids=prepared.crop_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_model(
        model_type=model_type,
        input_dim=int(prepared.X_train.shape[1]),
        num_crops=len(prepared.crop_to_id),
        model_config=_model_config(model_type),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(TRAIN_CONFIG.get("lr", 1e-3)),
        weight_decay=float(TRAIN_CONFIG.get("weight_decay", 1e-5)),
    )
    criterion = nn.HuberLoss(delta=float(TRAIN_CONFIG.get("huber_delta", 1.0)))

    history, _ = fit_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        target_scaler=target_scaler,
        epochs=int(cfg.get("epochs", TRAIN_CONFIG.get("epochs", 25))),
        patience=int(cfg.get("patience", TRAIN_CONFIG.get("patience", 8))),
        clip_grad_norm=float(TRAIN_CONFIG.get("clip_grad_norm", 2.0)),
        verbose_prefix=f"[research_1][source][{model_type}] ",
    )

    y_test_true, y_test_pred, crop_test = predict_unscaled(
        model=model,
        loader=test_loader,
        device=device,
        target_scaler=target_scaler,
    )

    test_metrics = regression_metrics(y_test_true, y_test_pred)

    by_crop_df = regression_metrics_by_crop(
        y_true=y_test_true,
        y_pred=y_test_pred,
        crop_ids=crop_test,
        id_to_crop=prepared.id_to_crop,
    )

    predictions_df = prepared.test_df.copy()
    predictions_df["y_true"] = y_test_true
    predictions_df["y_pred"] = y_test_pred
    predictions_df["abs_error"] = (predictions_df["y_true"] - predictions_df["y_pred"]).abs()

    torch.save(
        {
            "model_type": model_type,
            "model_state_dict": model.state_dict(),
            "model_config": _model_config(model_type),
            "input_dim": int(prepared.X_train.shape[1]),
            "feature_cols": prepared.feature_cols,
            "numeric_cols": prepared.numeric_cols,
            "categorical_cols": prepared.categorical_cols,
            "crop_to_id": prepared.crop_to_id,
            "id_to_crop": prepared.id_to_crop,
            "target_scaler": target_scaler.to_dict(),
            "research": "research_1_transfer",
            "source_dataset_path": str(cfg["source_dataset_path"]),
            "target_col": cfg["target_col"],
            "crop_col": cfg["crop_col"],
            "random_state": RANDOM_STATE,
            "test_size": cfg["test_size"],
            "val_size_from_train": cfg["val_size_from_train"],
        },
        _checkpoint_path(model_type),
    )

    pd.DataFrame(history).to_csv(
        cfg["results"]["metrics"] / f"source_history_{model_type}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_crop_df.to_csv(
        cfg["results"]["tables"] / f"source_metrics_by_crop_{model_type}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions_df.to_csv(
        cfg["results"]["tables"] / f"source_predictions_{model_type}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return build_metrics_row(
        model_name=model_type,
        feature_mode="raw",
        split_name="source_test",
        metrics=test_metrics,
        extra={
            "dataset": "crop_yield_source",
            "rows_train": int(len(prepared.train_df)),
            "rows_val": int(len(prepared.val_df)),
            "rows_test": int(len(prepared.test_df)),
            "num_crops": len(prepared.crop_to_id),
            "checkpoint_path": str(_checkpoint_path(model_type)),
        },
    )


def main(models: list[str] | None = None) -> None:
    ensure_project_dirs()
    seed_everything(RANDOM_STATE)
    device = get_device()
    cfg = RESEARCH_1_CONFIG

    selected_models = models or list(cfg.get("model_types", MODEL_TYPES))

    info = describe_stage1_datasets()
    print(json.dumps(info["source"], ensure_ascii=False, indent=2))
    print(json.dumps(info["target"], ensure_ascii=False, indent=2))

    prepared = prepare_source_dataset()

    joblib.dump(
        prepared.preprocessor,
        cfg["results"]["models"] / "source_preprocessor.joblib",
    )
    with open(cfg["results"]["models"] / "source_crop_mapping.json", "w", encoding="utf-8") as f:
        json.dump(prepared.crop_to_id, f, ensure_ascii=False, indent=2)

    rows = []
    for model_type in selected_models:
        print("=" * 100)
        print(f"Research 1 | train source | model={model_type} | device={device}")
        rows.append(train_one_model(model_type=model_type, prepared=prepared, device=device))

    summary_df = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    out_path = cfg["results"]["metrics"] / "source_training_metrics.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(summary_df)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()