from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.core.config import MLP_RESNET_CONFIG, TRANSFORMER_CONFIG
from src.core.data.target_scaler import TargetScaler
from src.core.evaluation.metrics import regression_metrics
from src.core.models.mlp_resnet_multihead import build_multihead_mlp_resnet
from src.core.models.multihead_transformer import build_multihead_transformer


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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(
    model_type: str,
    input_dim: int,
    num_crops: int,
    model_config: dict[str, Any] | None = None,
) -> nn.Module:
    if model_type == "mlp_resnet":
        return build_multihead_mlp_resnet(
            input_dim=input_dim,
            num_crops=num_crops,
            config=model_config or MLP_RESNET_CONFIG,
        )

    if model_type == "transformer":
        return build_multihead_transformer(
            input_dim=input_dim,
            num_crops=num_crops,
            config=model_config or TRANSFORMER_CONFIG,
        )

    raise ValueError(f"Неизвестный model_type: {model_type}")


def load_model_from_checkpoint(checkpoint: dict[str, Any], device: str) -> nn.Module:
    model_type = str(checkpoint["model_type"])
    input_dim = int(checkpoint["input_dim"])
    num_crops = len(checkpoint["crop_to_id"])
    model_config = checkpoint.get("model_config")

    model = build_model(
        model_type=model_type,
        input_dim=input_dim,
        num_crops=num_crops,
        model_config=model_config,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    crop_ids: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    balanced_sampler: bool = False,
) -> DataLoader:
    dataset = YieldDataset(X=X, y=y, crop_ids=crop_ids)

    if balanced_sampler and shuffle and len(crop_ids) > 0:
        counts = np.bincount(crop_ids)
        weights = np.array([1.0 / max(counts[c], 1) for c in crop_ids], dtype=np.float32)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
        )

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
    clip_grad_norm: float = 2.0,
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def predict_scaled(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    y_true_scaled = []
    y_pred_scaled = []
    crop_ids = []

    for xb, yb, cb in loader:
        xb = xb.to(device)
        cb = cb.to(device)

        preds = model(xb, cb)

        y_true_scaled.append(yb.cpu().numpy())
        y_pred_scaled.append(preds.cpu().numpy())
        crop_ids.append(cb.cpu().numpy())

    return (
        np.concatenate(y_true_scaled).ravel(),
        np.concatenate(y_pred_scaled).ravel(),
        np.concatenate(crop_ids).ravel(),
    )


@torch.no_grad()
def predict_unscaled(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    target_scaler: TargetScaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true_scaled, y_pred_scaled, crop_ids = predict_scaled(model, loader, device)
    y_true = target_scaler.inverse_transform(y_true_scaled)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    return y_true, y_pred, crop_ids


def fit_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    target_scaler: TargetScaler,
    epochs: int,
    patience: int,
    clip_grad_norm: float = 2.0,
    verbose_prefix: str = "",
) -> tuple[list[dict[str, float]], dict[str, float]]:
    history: list[dict[str, float]] = []
    best_state = None
    best_val_rmse = float("inf")
    best_val_metrics: dict[str, float] = {}
    patience_left = patience

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            clip_grad_norm=clip_grad_norm,
        )

        y_val_true, y_val_pred, _ = predict_unscaled(
            model=model,
            loader=val_loader,
            device=device,
            target_scaler=target_scaler,
        )
        val_metrics = regression_metrics(y_val_true, y_val_pred)

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_mae": float(val_metrics["mae"]),
            "val_mse": float(val_metrics["mse"]),
            "val_rmse": float(val_metrics["rmse"]),
            "val_r2": float(val_metrics["r2"]),
        }
        history.append(row)

        print(
            f"{verbose_prefix}epoch={epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_mae={val_metrics['mae']:.4f} | "
            f"val_rmse={val_metrics['rmse']:.4f} | "
            f"val_r2={val_metrics['r2']:.4f}"
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_val_metrics = val_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"{verbose_prefix}early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_val_metrics