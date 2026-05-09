from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.core.config import MODEL_TYPES, RANDOM_STATE, TRAIN_CONFIG, ensure_project_dirs
from src.core.data.target_scaler import TargetScaler
from src.core.evaluation.metrics import build_metrics_row, regression_metrics, regression_metrics_by_crop
from src.core.evaluation.reports import print_summary, reorder_summary_columns
from src.core.training.dl_trainer import (
    fit_with_early_stopping,
    get_device,
    load_model_from_checkpoint,
    make_loader,
    predict_unscaled,
    seed_everything,
)
from src.research_1_transfer.config import RESEARCH_1_CONFIG
from src.research_1_transfer.datasets import prepare_source_dataset, prepare_target_external
from src.research_1_transfer.features import transform_with_feature_artifact


def _source_checkpoint_path(model_type: str, feature_mode: str):
    return RESEARCH_1_CONFIG["results"]["models"] / f"source_{model_type}_{feature_mode}.pt"


def _finetuned_checkpoint_path(model_type: str, feature_mode: str):
    return RESEARCH_1_CONFIG["results"]["models"] / f"finetuned_russia_{model_type}_{feature_mode}.pt"


def _split_external(
    X: np.ndarray,
    y: np.ndarray,
    crop_ids: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    idx = np.arange(len(X))

    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=crop_ids,
    )

    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=crop_ids[train_idx],
    )

    return {
        "train": (X[train_idx], y[train_idx], crop_ids[train_idx], train_idx),
        "val": (X[val_idx], y[val_idx], crop_ids[val_idx], val_idx),
        "test": (X[test_idx], y[test_idx], crop_ids[test_idx], test_idx),
    }


def finetune_one_model(
    model_type: str,
    feature_mode: str,
    external,
    device: str,
) -> dict:
    if len(external["X"]) < 30:
        raise ValueError("Слишком мало строк для fine-tuning на российском датасете")

    checkpoint_path = _source_checkpoint_path(model_type, feature_mode)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Сначала обучи source-модель: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = load_model_from_checkpoint(checkpoint=checkpoint, device=device)

    X_external = transform_with_feature_artifact(
        artifact=checkpoint["feature_artifact"],
        X=external["X"],
        device=device,
    )

    split = _split_external(X_external, external["y"], external["crop_ids"])

    target_scaler = TargetScaler.fit(
        split["train"][1],
        use_log_target=bool(RESEARCH_1_CONFIG.get("use_log_target", True)),
    )

    balanced_sampler = bool(RESEARCH_1_CONFIG.get("balanced_sampler", False))

    train_loader = make_loader(
        X=split["train"][0],
        y=target_scaler.transform(split["train"][1]),
        crop_ids=split["train"][2],
        batch_size=256,
        shuffle=True,
        num_workers=0,
        balanced_sampler=balanced_sampler,
    )
    val_loader = make_loader(
        X=split["val"][0],
        y=target_scaler.transform(split["val"][1]),
        crop_ids=split["val"][2],
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )
    test_loader = make_loader(
        X=split["test"][0],
        y=target_scaler.transform(split["test"][1]),
        crop_ids=split["test"][2],
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(RESEARCH_1_CONFIG.get("finetune_lr", TRAIN_CONFIG.get("finetune_lr", 5e-4))),
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
        epochs=int(RESEARCH_1_CONFIG.get("finetune_epochs", TRAIN_CONFIG.get("finetune_epochs", 8))),
        patience=int(RESEARCH_1_CONFIG.get("finetune_patience", TRAIN_CONFIG.get("finetune_patience", 4))),
        clip_grad_norm=float(TRAIN_CONFIG.get("clip_grad_norm", 2.0)),
        verbose_prefix=f"[research_1][finetune][{model_type}][{feature_mode}] ",
    )

    y_test_true, y_test_pred, crop_ids = predict_unscaled(
        model=model,
        loader=test_loader,
        device=device,
        target_scaler=target_scaler,
    )
    metrics = regression_metrics(y_test_true, y_test_pred)

    id_to_crop = checkpoint["id_to_crop"]
    by_crop_df = regression_metrics_by_crop(
        y_true=y_test_true,
        y_pred=y_test_pred,
        crop_ids=crop_ids,
        id_to_crop=id_to_crop,
    )
    by_crop_df.to_csv(
        RESEARCH_1_CONFIG["results"]["tables"] / f"transfer_finetuned_metrics_by_crop_{model_type}_{feature_mode}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pred_df = external["df"].iloc[split["test"][3]].copy().reset_index(drop=True)
    pred_df["y_true"] = y_test_true
    pred_df["y_pred"] = y_test_pred
    pred_df["abs_error"] = (pred_df["y_true"] - pred_df["y_pred"]).abs()
    pred_df.to_csv(
        RESEARCH_1_CONFIG["results"]["tables"] / f"transfer_finetuned_predictions_{model_type}_{feature_mode}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    finetuned_path = _finetuned_checkpoint_path(model_type, feature_mode)

    torch.save(
        {
            **checkpoint,
            "model_state_dict": model.state_dict(),
            "research": "research_1_transfer_finetuned",
            "source_checkpoint_path": str(checkpoint_path),
            "target_scaler": target_scaler.to_dict(),
        },
        finetuned_path,
    )

    pd.DataFrame(history).to_csv(
        RESEARCH_1_CONFIG["results"]["metrics"] / f"transfer_finetuned_history_{model_type}_{feature_mode}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return build_metrics_row(
        model_name=model_type,
        feature_mode=feature_mode,
        split_name="target_finetuned_test",
        metrics=metrics,
        extra={
            "dataset": "russian_crop_yield_clean",
            "checkpoint_path": str(finetuned_path),
            "source_checkpoint_path": str(checkpoint_path),
            "num_rows_test": int(len(split["test"][3])),
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
            print(f"Research 1 | finetune on Russia | model={model_type} | feature_mode={feature_mode} | device={device}")
            rows.append(
                finetune_one_model(
                    model_type=model_type,
                    feature_mode=feature_mode,
                    external=external,
                    device=device,
                )
            )

    summary_df = pd.DataFrame(rows).sort_values(["feature_mode", "rmse"]).reset_index(drop=True)
    summary_df = reorder_summary_columns(summary_df)

    out_path = RESEARCH_1_CONFIG["results"]["metrics"] / "transfer_finetuned_metrics.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print_summary(summary_df)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()