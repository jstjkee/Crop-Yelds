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
    WIDE_DEEP_CONFIG,
    ensure_project_dirs,
)

from src.core.data.target_scaler import TargetScaler
from src.core.evaluation.metrics import build_metrics_row, regression_metrics, regression_metrics_by_crop
from src.core.evaluation.reports import print_summary, reorder_summary_columns
from src.core.training.dl_trainer import (
    build_model,
    fit_with_early_stopping,
    get_device,
    make_loader,
    predict_unscaled,
    seed_everything,
)
from src.core.config import (
    MLP_RESNET_CONFIG,
    MODEL_TYPES,
    RANDOM_STATE,
    TAB_MLP_CONFIG,
    TRAIN_CONFIG,
    TRANSFORMER_CONFIG,
    ensure_project_dirs,
)
from src.research_2_enriched.build_dataset import (
    describe_research_2_dataset,
    prepare_enriched_dataset,
)
from src.research_2_enriched.config import RESEARCH_2_CONFIG


def _model_config(model_type: str) -> dict[str, Any]:
    if model_type == "mlp_resnet":
        return MLP_RESNET_CONFIG
    if model_type == "transformer":
        return TRANSFORMER_CONFIG
    if model_type == "wide_deep":
        return WIDE_DEEP_CONFIG
    if model_type == "tab_mlp":
        return TAB_MLP_CONFIG
    raise ValueError(f"Неизвестный model_type: {model_type}")


def _checkpoint_path(
    split_name: str,
    feature_set_name: str,
    model_type: str,
    feature_mode: str,
):
    return (
        RESEARCH_2_CONFIG["results"]["models"]
        / f"research_2_{split_name}_{feature_set_name}_{model_type}_{feature_mode}.pt"
    )


def train_one_model(
    split_name: str,
    feature_set_name: str,
    model_type: str,
    feature_mode: str,
    prepared,
    device: str,
) -> dict[str, Any]:
    cfg = RESEARCH_2_CONFIG

    if feature_mode != "raw":
        raise ValueError("В Research 2 пока поддерживается только feature_mode='raw'")

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
        verbose_prefix=(
            f"[research_2][{split_name}][{feature_set_name}]"
            f"[enriched][{model_type}][{feature_mode}] "
        ),
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

    checkpoint_path = _checkpoint_path(
        split_name=split_name,
        feature_set_name=feature_set_name,
        model_type=model_type,
        feature_mode=feature_mode,
    )

    torch.save(
        {
            "model_type": model_type,
            "model_state_dict": model.state_dict(),
            "model_config": _model_config(model_type),
            "input_dim": int(prepared.X_train.shape[1]),
            "feature_mode": feature_mode,
            "feature_set_name": feature_set_name,
            "feature_cols": prepared.feature_cols,
            "numeric_cols": prepared.numeric_cols,
            "categorical_cols": prepared.categorical_cols,
            "crop_to_id": prepared.crop_to_id,
            "id_to_crop": prepared.id_to_crop,
            "target_scaler": target_scaler.to_dict(),
            "research": "research_2_enriched",
            "dataset_path": str(cfg["final_dataset_path"]),
            "target_col": cfg["target_col"],
            "crop_col": cfg["crop_col"],
            "random_state": RANDOM_STATE,
            "split": split_name,
        },
        checkpoint_path,
    )

    pd.DataFrame(history).to_csv(
        cfg["results"]["metrics"]
        / f"research_2_history_{split_name}_{feature_set_name}_{model_type}_{feature_mode}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_crop_df.to_csv(
        cfg["results"]["tables"]
        / f"research_2_metrics_by_crop_{split_name}_{feature_set_name}_{model_type}_{feature_mode}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions_df.to_csv(
        cfg["results"]["tables"]
        / f"research_2_predictions_{split_name}_{feature_set_name}_{model_type}_{feature_mode}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return build_metrics_row(
        model_name=model_type,
        feature_mode=feature_mode,
        split_name=f"research_2_{split_name}_test",
        metrics=test_metrics,
        extra={
            "dataset": cfg["dataset_label"],
            "feature_set": feature_set_name,
            "rows_train": int(len(prepared.train_df)),
            "rows_val": int(len(prepared.val_df)),
            "rows_test": int(len(prepared.test_df)),
            "num_crops": len(prepared.crop_to_id),
            "input_dim": int(prepared.X_train.shape[1]),
            "checkpoint_path": str(checkpoint_path),
        },
    )


def main(
    models: list[str] | None = None,
    feature_modes: list[str] | None = None,
) -> None:
    ensure_project_dirs()
    seed_everything(RANDOM_STATE)

    cfg = RESEARCH_2_CONFIG
    device = get_device()

    selected_models = models or list(cfg.get("model_types", MODEL_TYPES))
    selected_feature_modes = feature_modes or list(cfg.get("feature_modes", ["raw"]))
    selected_splits = list(cfg.get("splits", ["random"]))
    selected_feature_sets = list(cfg.get("feature_sets", ["full"]))

    info = describe_research_2_dataset()
    print(json.dumps(info, ensure_ascii=False, indent=2))

    rows = []

    for split_name in selected_splits:
        for feature_set_name in selected_feature_sets:
            prepared = prepare_enriched_dataset(
                split_name=split_name,
                feature_set_name=feature_set_name,
            )

            joblib.dump(
                prepared.preprocessor,
                cfg["results"]["models"]
                / f"research_2_preprocessor_{split_name}_{feature_set_name}.joblib",
            )

            with open(
                cfg["results"]["models"]
                / f"research_2_crop_mapping_{split_name}_{feature_set_name}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(prepared.crop_to_id, f, ensure_ascii=False, indent=2)

            for feature_mode in selected_feature_modes:
                for model_type in selected_models:
                    print("=" * 100)
                    print(
                        f"Research 2 | train enriched | "
                        f"split={split_name} | feature_set={feature_set_name} | "
                        f"model={model_type} | feature_mode={feature_mode} | device={device}"
                    )

                    rows.append(
                        train_one_model(
                            split_name=split_name,
                            feature_set_name=feature_set_name,
                            model_type=model_type,
                            feature_mode=feature_mode,
                            prepared=prepared,
                            device=device,
                        )
                    )

    summary_df = pd.DataFrame(rows).sort_values(
        ["split", "feature_set", "feature_mode", "rmse"]
    ).reset_index(drop=True)

    summary_df = reorder_summary_columns(summary_df)

    out_path = cfg["results"]["metrics"] / "research_2_enriched_metrics.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print_summary(summary_df)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()