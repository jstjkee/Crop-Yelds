from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.config import (
    CROP_COL,
    FEATURE_MODES,
    METRICS_DIR,
    MODELS_DIR,
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
from src.models.multihead_transformer import build_multihead_transformer


@dataclass
class SourceRunResult:
    mode: str
    model_name: str
    mae: float
    mse: float
    rmse: float
    r2: float
    input_dim: int
    num_crops: int
    model_path: str | None = None


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
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    y_true = []
    y_pred = []

    for xb, yb, cb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        cb = cb.to(device)

        preds = model(xb, cb)

        y_true.append(yb.cpu().numpy())
        y_pred.append(preds.cpu().numpy())

    return np.concatenate(y_true).ravel(), np.concatenate(y_pred).ravel()


def train_transformer_for_mode(
    mode: str,
    prepared: PreparedDataset,
    feature_result: FeatureTransformResult,
    device: str,
) -> tuple[dict[str, float], pd.DataFrame, str]:
    batch_size = int(TRAIN_CONFIG.get("batch_size", 128))
    epochs = int(TRAIN_CONFIG.get("epochs", 40))
    lr = float(TRAIN_CONFIG.get("lr", 1e-3))
    weight_decay = float(TRAIN_CONFIG.get("weight_decay", 1e-5))
    num_workers = int(TRAIN_CONFIG.get("num_workers", 0))

    train_loader = make_loader(
        X=feature_result.X_train,
        y=prepared.y_train,
        crop_ids=prepared.crop_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = make_loader(
        X=feature_result.X_test,
        y=prepared.y_test,
        crop_ids=prepared.crop_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_multihead_transformer(
        input_dim=feature_result.X_train.shape[1],
        num_crops=len(prepared.crop_to_id),
        config=TRANSFORMER_CONFIG,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()

    best_rmse = float("inf")
    best_state = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        y_true, y_pred = predict(model, test_loader, device)
        metrics = regression_metrics(y_true, y_pred)

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[transformer][{mode}] Epoch {epoch + 1:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"mae={metrics['mae']:.4f} | "
            f"rmse={metrics['rmse']:.4f} | "
            f"r2={metrics['r2']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred = predict(model, test_loader, device)
    final_metrics = regression_metrics(y_true, y_pred)
    by_crop_df = regression_metrics_by_crop(
        y_true=y_true,
        y_pred=y_pred,
        crop_ids=prepared.crop_test,
        id_to_crop=prepared.id_to_crop,
    )

    model_path = MODELS_DIR / f"multihead_transformer_{mode}.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "mode": mode,
        "input_dim": int(feature_result.X_train.shape[1]),
        "crop_to_id": prepared.crop_to_id,
        "id_to_crop": prepared.id_to_crop,
        "transformer_config": TRANSFORMER_CONFIG,
        "feature_metadata": feature_result.metadata or {},
    }
    torch.save(checkpoint, model_path)

    return final_metrics, by_crop_df, str(model_path)


def run_transformer_experiments(
    prepared: PreparedDataset,
    device: str,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    results: list[dict] = []
    by_crop_tables: dict[str, pd.DataFrame] = {}

    for mode in FEATURE_MODES:
        print("=" * 100)
        print(f"Запуск transformer-режима: {mode}")

        feature_result = build_feature_view(
            mode=mode,
            X_train=prepared.X_train,
            X_test=prepared.X_test,
            device=device,
        )

        final_metrics, by_crop_df, model_path = train_transformer_for_mode(
            mode=mode,
            prepared=prepared,
            feature_result=feature_result,
            device=device,
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
                model_name="multihead_transformer",
                feature_mode=mode,
                split_name="test",
                metrics=final_metrics,
                extra=extra,
            )
        )

        by_crop_df.insert(0, "feature_mode", mode)
        by_crop_df.insert(0, "model", "multihead_transformer")
        by_crop_tables[mode] = by_crop_df

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
) -> None:
    results_df = pd.DataFrame(results_rows)
    summary_path = METRICS_DIR / "source_metrics_summary.csv"
    results_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    for key, table_df in by_crop_tables.items():
        safe_key = str(key).replace(" ", "_").lower()
        table_path = TABLES_DIR / f"source_metrics_by_crop_{safe_key}.csv"
        table_df.to_csv(table_path, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("Итоговая таблица результатов:")
    print(results_df)
    print(f"Summary CSV: {summary_path}")


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

    dataset_info = summarize_dataframe(
        df=df,
        target_col=TARGET_COL,
        crop_col=CROP_COL,
    )
    print("=" * 100)
    print("Информация о датасете:")
    for key, value in dataset_info.items():
        print(f"{key}: {value}")

    prepared = prepare_dataset(
        df=df,
        target_col=TARGET_COL,
        crop_col=CROP_COL,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("=" * 100)
    print("Подготовка датасета завершена")
    print(f"X_train shape: {prepared.X_train.shape}")
    print(f"X_test shape: {prepared.X_test.shape}")
    print(f"Количество культур: {len(prepared.crop_to_id)}")
    print(f"Культуры: {prepared.crop_to_id}")

    transformer_rows, transformer_by_crop = run_transformer_experiments(
        prepared=prepared,
        device=device,
    )

    baseline_rows, baseline_by_crop = run_baseline_experiments(prepared=prepared)

    all_rows = transformer_rows + baseline_rows
    all_by_crop = {}
    all_by_crop.update(transformer_by_crop)
    all_by_crop.update(baseline_by_crop)

    save_summary_tables(
        results_rows=all_rows,
        by_crop_tables=all_by_crop,
    )


if __name__ == "__main__":
    main()