from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURE_MODES,
    LEGACY_RUSSIAN_DATA_PATH,
    METRICS_DIR,
    MODEL_TYPES,
    MODELS_DIR,
    RANDOM_STATE,
    SOURCE_DATA_PATH,
    TARGET_COL,
    CROP_COL,
    TRAIN_CONFIG,
    ensure_project_dirs,
)
from src.data.load_data import load_csv, validate_required_columns
from src.data.preprocess import prepare_dataset, transform_external_dataset
from src.evaluation.metrics import regression_metrics
from src.features.transforms import build_feature_view
from src.models.mlp_resnet_multihead import build_multihead_mlp_resnet
from src.models.multihead_transformer import build_multihead_transformer
from src.training.target_scaler import TargetScaler
from src.training.train_source import (
    get_device,
    make_loader,
    predict_unscaled,
    seed_everything,
    train_one_epoch,
)


def build_model_from_checkpoint(checkpoint: dict, device: str) -> torch.nn.Module:
    model_type = checkpoint.get("model_type", "transformer")
    input_dim = int(checkpoint["input_dim"])
    num_crops = len(checkpoint["crop_to_id"])

    if model_type == "transformer":
        config = checkpoint.get("model_config") or checkpoint.get("transformer_config")
        if config is None:
            raise ValueError("В checkpoint transformer отсутствует model_config/transformer_config")
        model = build_multihead_transformer(
            input_dim=input_dim,
            num_crops=num_crops,
            config=config,
        ).to(device)

    elif model_type == "mlp_resnet":
        config = checkpoint.get("model_config")
        if config is None:
            raise ValueError("В checkpoint mlp_resnet отсутствует model_config")
        model = build_multihead_mlp_resnet(
            input_dim=input_dim,
            num_crops=num_crops,
            config=config,
        ).to(device)

    else:
        raise ValueError(f"Неизвестный тип модели в checkpoint: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def resolve_source_checkpoint_path(model_type: str, mode: str):
    return MODELS_DIR / f"{model_type}_{mode}.pt"


def resolve_finetuned_checkpoint_path(model_type: str, mode: str):
    return MODELS_DIR / f"finetuned_russia_legacy_{model_type}_{mode}.pt"


def _split_external(
    X: np.ndarray,
    y: np.ndarray,
    crop_ids: np.ndarray,
):
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
        "train": (X[train_idx], y[train_idx], crop_ids[train_idx]),
        "val": (X[val_idx], y[val_idx], crop_ids[val_idx]),
        "test": (X[test_idx], y[test_idx], crop_ids[test_idx]),
    }


def finetune_model_and_mode(
    model_type: str,
    mode: str,
    device: str,
) -> dict:
    source_df = load_csv(SOURCE_DATA_PATH)
    russian_df = load_csv(LEGACY_RUSSIAN_DATA_PATH)

    validate_required_columns(source_df, TARGET_COL, CROP_COL)
    validate_required_columns(russian_df, TARGET_COL, CROP_COL)

    prepared_source = prepare_dataset(
        df=source_df,
        target_col=TARGET_COL,
        crop_col=CROP_COL,
    )

    external = transform_external_dataset(
        df=russian_df,
        preprocessor=prepared_source.preprocessor,
        crop_to_id=prepared_source.crop_to_id,
        target_col=TARGET_COL,
        crop_col=CROP_COL,
    )

    if len(external["X"]) < 30:
        raise ValueError("Слишком мало строк для fine-tune на russian legacy dataset")

    feature_view = build_feature_view(
        mode=mode,
        X_train=prepared_source.X_train,
        X_val=external["X"],
        X_test=external["X"],
        device=device,
    )
    X_ext = feature_view.X_test

    split = _split_external(
        X=X_ext,
        y=external["y"],
        crop_ids=external["crop_ids"],
    )

    checkpoint_path = resolve_source_checkpoint_path(model_type=model_type, mode=mode)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Сначала обучи source-модель: {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model_from_checkpoint(checkpoint=checkpoint, device=device)

    scaler = TargetScaler(
        mean_=float(checkpoint.get("target_scaler_mean", prepared_source.y_train.mean())),
        std_=float(checkpoint.get("target_scaler_std", prepared_source.y_train.std() or 1.0)),
    )

    train_loader = make_loader(
        X=split["train"][0],
        y=scaler.transform(split["train"][1]),
        crop_ids=split["train"][2],
        batch_size=256,
        shuffle=True,
        num_workers=0,
    )
    val_loader = make_loader(
        X=split["val"][0],
        y=scaler.transform(split["val"][1]),
        crop_ids=split["val"][2],
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )
    test_loader = make_loader(
        X=split["test"][0],
        y=scaler.transform(split["test"][1]),
        crop_ids=split["test"][2],
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(TRAIN_CONFIG.get("finetune_lr", 5e-4)),
        weight_decay=1e-5,
    )
    criterion = nn.HuberLoss(delta=float(TRAIN_CONFIG.get("huber_delta", 1.0)))

    best_rmse = float("inf")
    best_state = None

    for epoch in range(int(TRAIN_CONFIG.get("finetune_epochs", 4))):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        y_val_true, y_val_pred = predict_unscaled(
            model=model,
            loader=val_loader,
            device=device,
            target_scaler=scaler,
        )
        val_metrics = regression_metrics(y_val_true, y_val_pred)

        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[finetune][{model_type}][{mode}] "
            f"epoch={epoch + 1} "
            f"train_loss={train_loss:.4f} "
            f"val_mae={val_metrics['mae']:.4f} "
            f"val_rmse={val_metrics['rmse']:.4f} "
            f"val_r2={val_metrics['r2']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    y_test_true, y_test_pred = predict_unscaled(
        model=model,
        loader=test_loader,
        device=device,
        target_scaler=scaler,
    )
    metrics = regression_metrics(y_test_true, y_test_pred)

    output_ckpt = resolve_finetuned_checkpoint_path(model_type=model_type, mode=mode)
    torch.save(
        {
            "model_type": checkpoint.get("model_type", model_type),
            "model_state_dict": model.state_dict(),
            "mode": mode,
            "input_dim": checkpoint["input_dim"],
            "crop_to_id": checkpoint["crop_to_id"],
            "id_to_crop": checkpoint["id_to_crop"],
            "model_config": checkpoint.get("model_config") or checkpoint.get("transformer_config"),
            "target_scaler_mean": scaler.mean_,
            "target_scaler_std": scaler.std_,
        },
        output_ckpt,
    )

    return {
        "model_type": model_type,
        "mode": mode,
        "num_rows": int(len(external["df"])),
        "mae": metrics["mae"],
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "output_checkpoint": str(output_ckpt),
    }


def main() -> None:
    ensure_project_dirs()
    seed_everything(RANDOM_STATE)
    device = get_device()

    rows = []
    for model_type in MODEL_TYPES:
        for mode in FEATURE_MODES:
            print("=" * 80)
            print(f"Fine-tune russian legacy | model_type={model_type} | mode={mode}")
            rows.append(
                finetune_model_and_mode(
                    model_type=model_type,
                    mode=mode,
                    device=device,
                )
            )

    df = pd.DataFrame(rows)
    output = METRICS_DIR / "russian_legacy_finetune_metrics.csv"
    df.to_csv(output, index=False, encoding="utf-8-sig")

    print(df)
    print(f"CSV сохранен: {output}")


if __name__ == "__main__":
    main()