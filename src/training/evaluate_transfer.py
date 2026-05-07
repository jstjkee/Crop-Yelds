from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

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
    TEST_SIZE,
    ensure_project_dirs,
)
from src.data.load_data import load_csv, validate_required_columns
from src.data.preprocess import prepare_dataset, transform_external_dataset
from src.evaluation.metrics import regression_metrics
from src.features.transforms import build_feature_view
from src.models.mlp_resnet_multihead import build_multihead_mlp_resnet
from src.models.multihead_transformer import build_multihead_transformer
from src.training.target_scaler import TargetScaler


class YieldDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, crop_ids: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        self.crop_ids = torch.tensor(crop_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.crop_ids[idx]


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def predict_scaled(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true_scaled = []
    y_pred_scaled = []

    for xb, yb, cb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        cb = cb.to(device)

        preds = model(xb, cb)

        y_true_scaled.append(yb.cpu().numpy())
        y_pred_scaled.append(preds.cpu().numpy())

    return np.concatenate(y_true_scaled).ravel(), np.concatenate(y_pred_scaled).ravel()


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


def resolve_checkpoint_path(model_type: str, mode: str) -> str:
    return str(MODELS_DIR / f"{model_type}_{mode}.pt")


def evaluate_transfer_for_model_and_mode(
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
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    external = transform_external_dataset(
        df=russian_df,
        preprocessor=prepared_source.preprocessor,
        crop_to_id=prepared_source.crop_to_id,
        target_col=TARGET_COL,
        crop_col=CROP_COL,
    )

    if len(external["X"]) == 0:
        raise ValueError(
            "После фильтрации в russian_crop_yield_clean.csv не осталось строк с известными культурами"
        )

    feature_view = build_feature_view(
        mode=mode,
        X_train=prepared_source.X_train,
        X_val=external["X"],
        X_test=external["X"],
        device=device,
    )
    X_external = feature_view.X_test

    checkpoint_path = resolve_checkpoint_path(model_type=model_type, mode=mode)
    checkpoint_path_obj = MODELS_DIR / f"{model_type}_{mode}.pt"
    if not checkpoint_path_obj.exists():
        raise FileNotFoundError(
            f"Не найден checkpoint для model_type='{model_type}', mode='{mode}': {checkpoint_path_obj}. "
            f"Сначала запусти обучение source-модели."
        )

    checkpoint = torch.load(checkpoint_path_obj, map_location=device)
    model = build_model_from_checkpoint(checkpoint=checkpoint, device=device)

    scaler = TargetScaler(
        mean_=float(checkpoint.get("target_scaler_mean", prepared_source.y_train.mean())),
        std_=float(checkpoint.get("target_scaler_std", prepared_source.y_train.std() or 1.0)),
    )

    y_external_scaled = scaler.transform(external["y"])

    loader = DataLoader(
        YieldDataset(X_external, y_external_scaled, external["crop_ids"]),
        batch_size=256,
        shuffle=False,
    )

    y_true_scaled, y_pred_scaled = predict_scaled(model, loader, device)
    y_true = scaler.inverse_transform(y_true_scaled)
    y_pred = scaler.inverse_transform(y_pred_scaled)

    metrics = regression_metrics(y_true, y_pred)

    return {
        "model_type": model_type,
        "mode": mode,
        "dataset": "russian_legacy_transfer",
        "num_rows": int(len(external["df"])),
        "num_crops": int(external["df"][CROP_COL].nunique()),
        "mae": metrics["mae"],
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "checkpoint_path": checkpoint_path,
    }


def main() -> None:
    ensure_project_dirs()
    device = get_device()
    results = []

    for model_type in MODEL_TYPES:
        for mode in FEATURE_MODES:
            print("=" * 80)
            print(f"Transfer evaluation | model_type={model_type} | mode={mode}")
            results.append(
                evaluate_transfer_for_model_and_mode(
                    model_type=model_type,
                    mode=mode,
                    device=device,
                )
            )

    results_df = pd.DataFrame(results)
    output_path = METRICS_DIR / "russian_legacy_transfer_metrics.csv"
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(results_df)
    print(f"CSV сохранен: {output_path}")


if __name__ == "__main__":
    main()