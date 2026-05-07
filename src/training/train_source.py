from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.config import (
    CROP_COL,
    DEFAULT_MODEL_TYPE,
    FEATURE_MODES,
    METRICS_DIR,
    MLP_RESNET_CONFIG,
    MODELS_DIR,
    MODEL_TYPES,
    RANDOM_STATE,
    SOURCE_DATA_PATH,
    TABLES_DIR,
    TARGET_COL,
    TEST_SIZE,
    TRAIN_CONFIG,
    TRANSFORMER_CONFIG,
    ensure_project_dirs,
)
from src.data.load_data import load_and_validate_source_data, summarize_dataframe
from src.data.preprocess import PreparedDataset, prepare_dataset
from src.evaluation.metrics import (
    build_metrics_row,
    regression_metrics,
    regression_metrics_by_crop,
)
from src.features.transforms import FeatureTransformResult, build_feature_view
from src.models.baselines import train_all_baselines
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


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    crop_ids: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = YieldDataset(X=X, y=y, crop_ids=crop_ids)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

def get_model_config(model_type: str) -> dict:
    if model_type == "transformer":
        return TRANSFORMER_CONFIG
    if model_type == "mlp_resnet":
        return MLP_RESNET_CONFIG
    raise ValueError(f"Неизвестный model_type: {model_type}")


def build_model(
    model_type: str,
    input_dim: int,
    num_crops: int,
) -> nn.Module:
    if model_type == "transformer":
        return build_multihead_transformer(
            input_dim=input_dim,
            num_crops=num_crops,
            config=TRANSFORMER_CONFIG,
        )
    if model_type == "mlp_resnet":
        return build_multihead_mlp_resnet(
            input_dim=input_dim,
            num_crops=num_crops,
            config=MLP_RESNET_CONFIG,
        )
    raise ValueError(f"Неизвестный model_type: {model_type}")


def resolve_model_filename(
    model_type: str,
    mode: str,
    model_prefix: str,
) -> Path:
    if model_prefix == "source":
        return MODELS_DIR / f"{model_type}_{mode}.pt"
    return MODELS_DIR / f"{model_prefix}_{model_type}_{mode}.pt"

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    model.train()
    losses: list[float] = []

    for xb, yb, cb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        cb = cb.to(device)

        preds = model(xb, cb)
        loss = criterion(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def predict_scaled(
    model: nn.Module,
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

    return (
        np.concatenate(y_true_scaled).ravel(),
        np.concatenate(y_pred_scaled).ravel(),
    )


@torch.no_grad()
def predict_unscaled(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    target_scaler: TargetScaler,
) -> tuple[np.ndarray, np.ndarray]:
    y_true_scaled, y_pred_scaled = predict_scaled(model, loader, device)
    y_true = target_scaler.inverse_transform(y_true_scaled)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    return y_true, y_pred


def _resolve_model_path(model_type: str, mode: str, model_prefix: str) -> Path:
    return resolve_model_filename(
        model_type=model_type,
        mode=mode,
        model_prefix=model_prefix,
    )

def train_dl_model_for_mode(
    model_type: str,
    mode: str,
    prepared: PreparedDataset,
    feature_result: FeatureTransformResult,
    device: str,
    model_prefix: str = "source",
) -> tuple[dict[str, float], pd.DataFrame, str]:
    batch_size = int(TRAIN_CONFIG.get("batch_size", 128))
    epochs = int(TRAIN_CONFIG.get("epochs", 40))
    lr = float(TRAIN_CONFIG.get("lr", 1e-3))
    weight_decay = float(TRAIN_CONFIG.get("weight_decay", 1e-5))
    num_workers = int(TRAIN_CONFIG.get("num_workers", 0))
    huber_delta = float(TRAIN_CONFIG.get("huber_delta", 1.0))

    target_scaler = TargetScaler.fit(prepared.y_train)

    y_train_scaled = target_scaler.transform(prepared.y_train)
    y_val_scaled = target_scaler.transform(prepared.y_val)
    y_test_scaled = target_scaler.transform(prepared.y_test)

    train_loader = make_loader(
        X=feature_result.X_train,
        y=y_train_scaled,
        crop_ids=prepared.crop_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = make_loader(
        X=feature_result.X_val,
        y=y_val_scaled,
        crop_ids=prepared.crop_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = make_loader(
        X=feature_result.X_test,
        y=y_test_scaled,
        crop_ids=prepared.crop_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_model(
        model_type=model_type,
        input_dim=feature_result.X_train.shape[1],
        num_crops=len(prepared.crop_to_id),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.HuberLoss(delta=huber_delta)

    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(epochs):
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
            target_scaler=target_scaler,
        )
        val_metrics = regression_metrics(y_val_true, y_val_pred)

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[{model_type}][{model_prefix}][{mode}] Epoch {epoch + 1:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_mae={val_metrics['mae']:.4f} | "
            f"val_rmse={val_metrics['rmse']:.4f} | "
            f"val_r2={val_metrics['r2']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    y_test_true, y_test_pred = predict_unscaled(
        model=model,
        loader=test_loader,
        device=device,
        target_scaler=target_scaler,
    )
    final_metrics = regression_metrics(y_test_true, y_test_pred)

    by_crop_df = regression_metrics_by_crop(
        y_true=y_test_true,
        y_pred=y_test_pred,
        crop_ids=prepared.crop_test,
        id_to_crop=prepared.id_to_crop,
    )

    model_path = _resolve_model_path(
        model_type=model_type,
        mode=mode,
        model_prefix=model_prefix,
    )
    checkpoint = {
        "model_type": model_type,
        "model_state_dict": model.state_dict(),
        "mode": mode,
        "input_dim": int(feature_result.X_train.shape[1]),
        "crop_to_id": prepared.crop_to_id,
        "id_to_crop": prepared.id_to_crop,
        "model_config": get_model_config(model_type),
        "feature_metadata": feature_result.metadata or {},
        "target_scaler_mean": target_scaler.mean_,
        "target_scaler_std": target_scaler.std_,
    }
    torch.save(checkpoint, model_path)

    return final_metrics, by_crop_df, str(model_path)

def run_dl_experiments(
    prepared: PreparedDataset,
    device: str,
    model_prefix: str = "source",
    model_types: list[str] | None = None,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    if model_types is None:
        model_types = MODEL_TYPES

    results: list[dict] = []
    by_crop_tables: dict[str, pd.DataFrame] = {}

    for model_type in model_types:
        for mode in FEATURE_MODES:
            print("=" * 100)
            print(f"Запуск DL-модели: {model_type} | feature mode: {mode}")

            feature_result = build_feature_view(
                mode=mode,
                X_train=prepared.X_train,
                X_val=prepared.X_val,
                X_test=prepared.X_test,
                device=device,
            )

            final_metrics, by_crop_df, model_path = train_dl_model_for_mode(
                model_type=model_type,
                mode=mode,
                prepared=prepared,
                feature_result=feature_result,
                device=device,
                model_prefix=model_prefix,
            )

            extra = {
                "input_dim": int(feature_result.X_train.shape[1]),
                "num_crops": len(prepared.crop_to_id),
                "model_path": model_path,
            }
            if feature_result.metadata:
                extra.update(feature_result.metadata)

            results.append(
                build_metrics_row(
                    model_name=model_type,
                    feature_mode=mode,
                    split_name="test",
                    metrics=final_metrics,
                    extra=extra,
                )
            )

            table_key = f"{model_type}_{mode}"
            by_crop_df.insert(0, "feature_mode", mode)
            by_crop_df.insert(0, "model", model_type)
            by_crop_tables[table_key] = by_crop_df

    return results, by_crop_tables

def run_baseline_experiments(
    prepared: PreparedDataset,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    print("=" * 100)
    print("Запуск baseline-моделей на raw признаках")

    baseline_results = train_all_baselines(
        X_train=prepared.X_train,
        y_train=prepared.y_train,
        X_test=prepared.X_test,
        y_test=prepared.y_test,
    )

    rows: list[dict] = []
    by_crop_tables: dict[str, pd.DataFrame] = {}

    for result in baseline_results:
        preds = result.model.predict(prepared.X_test)
        by_crop_df = regression_metrics_by_crop(
            y_true=prepared.y_test,
            y_pred=preds,
            crop_ids=prepared.crop_test,
            id_to_crop=prepared.id_to_crop,
        )

        by_crop_df.insert(0, "feature_mode", "raw")
        by_crop_df.insert(0, "model", result.model_name)
        by_crop_tables[result.model_name] = by_crop_df

        rows.append(
            build_metrics_row(
                model_name=result.model_name,
                feature_mode="raw",
                split_name="test",
                metrics=result.metrics,
                extra={
                    "input_dim": int(prepared.X_train.shape[1]),
                    "num_crops": len(prepared.crop_to_id),
                    "model_path": None,
                },
            )
        )

    return rows, by_crop_tables


def save_summary_tables(
    results_rows: list[dict],
    by_crop_tables: dict[str, pd.DataFrame],
    prefix: str = "source",
) -> None:
    results_df = pd.DataFrame(results_rows)
    summary_path = METRICS_DIR / f"{prefix}_metrics_summary.csv"
    results_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    for key, table_df in by_crop_tables.items():
        safe_key = str(key).replace(" ", "_").lower()
        table_path = TABLES_DIR / f"{prefix}_metrics_by_crop_{safe_key}.csv"
        table_df.to_csv(table_path, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("Итоговая таблица результатов:")
    print(results_df)
    print(f"Summary CSV: {summary_path}")


def run_training_pipeline(
    df: pd.DataFrame,
    dataset_label: str,
    target_col: str,
    crop_col: str,
    results_prefix: str,
    device: str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    val_size_from_train: float = 0.2,
) -> PreparedDataset:
    dataset_info = summarize_dataframe(
        df=df,
        target_col=target_col,
        crop_col=crop_col,
    )
    print("=" * 100)
    print(f"Информация о датасете: {dataset_label}")
    for key, value in dataset_info.items():
        print(f"{key}: {value}")

    prepared = prepare_dataset(
        df=df,
        target_col=target_col,
        crop_col=crop_col,
        test_size=test_size,
        random_state=random_state,
        val_size_from_train=val_size_from_train,
    )

    print("=" * 100)
    print("Подготовка датасета завершена")
    print(f"X_train shape: {prepared.X_train.shape}")
    print(f"X_val shape: {prepared.X_val.shape}")
    print(f"X_test shape: {prepared.X_test.shape}")
    print(f"Количество культур: {len(prepared.crop_to_id)}")
    print(f"Культуры: {prepared.crop_to_id}")

    dl_rows, dl_by_crop = run_dl_experiments(
        prepared=prepared,
        device=device,
        model_prefix=results_prefix,
    )

    baseline_rows, baseline_by_crop = run_baseline_experiments(prepared=prepared)

    all_rows = dl_rows + baseline_rows
    all_by_crop = {}
    all_by_crop.update(dl_by_crop)
    all_by_crop.update(baseline_by_crop)

    save_summary_tables(
        results_rows=all_rows,
        by_crop_tables=all_by_crop,
        prefix=results_prefix,
    )
    return prepared


def main() -> None:
    ensure_project_dirs()
    seed_everything(RANDOM_STATE)
    device = get_device()

    print(f"Device: {device}")
    print(f"Source dataset path: {SOURCE_DATA_PATH}")

    df = load_and_validate_source_data(
        path=SOURCE_DATA_PATH,
        target_col=TARGET_COL,
        crop_col=CROP_COL,
    )

    run_training_pipeline(
        df=df,
        dataset_label="source",
        target_col=TARGET_COL,
        crop_col=CROP_COL,
        results_prefix="source",
        device=device,
    )


if __name__ == "__main__":
    main()