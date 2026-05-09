from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class FeatureTransformResult:
    mode: str
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    artifact: dict[str, Any]
    fit_info: dict[str, Any]


class _Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        enc_layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
        ]
        if dropout > 0:
            enc_layers.append(nn.Dropout(dropout))
        enc_layers.extend(
            [
                nn.Linear(hidden_dim_1, hidden_dim_2),
                nn.ReLU(),
            ]
        )
        if dropout > 0:
            enc_layers.append(nn.Dropout(dropout))
        enc_layers.append(nn.Linear(hidden_dim_2, latent_dim))

        dec_layers: list[nn.Module] = [
            nn.Linear(latent_dim, hidden_dim_2),
            nn.ReLU(),
        ]
        if dropout > 0:
            dec_layers.append(nn.Dropout(dropout))
        dec_layers.extend(
            [
                nn.Linear(hidden_dim_2, hidden_dim_1),
                nn.ReLU(),
                nn.Linear(hidden_dim_1, input_dim),
            ]
        )

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return z, reconstructed


def _build_autoencoder(input_dim: int, cfg: dict[str, Any]) -> _Autoencoder:
    return _Autoencoder(
        input_dim=input_dim,
        latent_dim=int(cfg.get("autoencoder_latent_dim", 16)),
        hidden_dim_1=int(cfg.get("autoencoder_hidden_dim_1", 128)),
        hidden_dim_2=int(cfg.get("autoencoder_hidden_dim_2", 64)),
        dropout=float(cfg.get("autoencoder_dropout", 0.0)),
    )


def _train_autoencoder(
    model: _Autoencoder,
    X_train: np.ndarray,
    cfg: dict[str, Any],
    device: str,
) -> tuple[_Autoencoder, float]:
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("autoencoder_batch_size", 256)),
        shuffle=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.get("autoencoder_lr", 1e-3)),
        weight_decay=float(cfg.get("autoencoder_weight_decay", 0.0)),
    )
    criterion = nn.MSELoss()

    model = model.to(device)

    last_loss = 0.0
    epochs = int(cfg.get("autoencoder_epochs", 20))
    for epoch in range(epochs):
        model.train()
        losses: list[float] = []

        for (xb,) in loader:
            xb = xb.to(device)

            _, reconstructed = model(xb)
            loss = criterion(reconstructed, xb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

        last_loss = float(np.mean(losses)) if losses else 0.0
        print(f"[research_1][autoencoder] epoch={epoch + 1:03d} | recon_loss={last_loss:.6f}")

    return model, last_loss


@torch.no_grad()
def _encode_autoencoder(
    model: _Autoencoder,
    X: np.ndarray,
    device: str,
    batch_size: int = 1024,
) -> np.ndarray:
    model = model.to(device)
    model.eval()

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    encoded_parts: list[np.ndarray] = []

    for (xb,) in loader:
        xb = xb.to(device)
        z, _ = model(xb)
        encoded_parts.append(z.cpu().numpy())

    return np.concatenate(encoded_parts, axis=0).astype(np.float32)


def build_feature_view(
    mode: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    device: str,
    config: dict[str, Any],
) -> FeatureTransformResult:
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    if mode == "raw":
        return FeatureTransformResult(
            mode="raw",
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            artifact={"mode": "raw"},
            fit_info={"output_dim": int(X_train.shape[1])},
        )

    if mode == "pca":
        pca = PCA(
            n_components=config.get("pca_n_components", 0.95),
            random_state=int(config.get("pca_random_state", 42)),
        )
        X_train_pca = pca.fit_transform(X_train).astype(np.float32)
        X_val_pca = pca.transform(X_val).astype(np.float32)
        X_test_pca = pca.transform(X_test).astype(np.float32)

        return FeatureTransformResult(
            mode="pca",
            X_train=X_train_pca,
            X_val=X_val_pca,
            X_test=X_test_pca,
            artifact={
                "mode": "pca",
                "pca": pca,
            },
            fit_info={
                "output_dim": int(X_train_pca.shape[1]),
                "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            },
        )

    if mode == "autoencoder":
        model = _build_autoencoder(input_dim=int(X_train.shape[1]), cfg=config)
        model, last_loss = _train_autoencoder(model=model, X_train=X_train, cfg=config, device=device)

        X_train_ae = _encode_autoencoder(model, X_train, device=device)
        X_val_ae = _encode_autoencoder(model, X_val, device=device)
        X_test_ae = _encode_autoencoder(model, X_test, device=device)

        return FeatureTransformResult(
            mode="autoencoder",
            X_train=X_train_ae,
            X_val=X_val_ae,
            X_test=X_test_ae,
            artifact={
                "mode": "autoencoder",
                "input_dim": int(X_train.shape[1]),
                "config": {
                    "autoencoder_latent_dim": int(config.get("autoencoder_latent_dim", 16)),
                    "autoencoder_hidden_dim_1": int(config.get("autoencoder_hidden_dim_1", 128)),
                    "autoencoder_hidden_dim_2": int(config.get("autoencoder_hidden_dim_2", 64)),
                    "autoencoder_dropout": float(config.get("autoencoder_dropout", 0.0)),
                },
                "state_dict": model.state_dict(),
            },
            fit_info={
                "output_dim": int(X_train_ae.shape[1]),
                "final_recon_loss": float(last_loss),
            },
        )

    raise ValueError(f"Неизвестный feature mode: {mode}")


def transform_with_feature_artifact(
    artifact: dict[str, Any],
    X: np.ndarray,
    device: str,
) -> np.ndarray:
    mode = str(artifact["mode"])
    X = np.asarray(X, dtype=np.float32)

    if mode == "raw":
        return X.astype(np.float32)

    if mode == "pca":
        pca = artifact["pca"]
        return pca.transform(X).astype(np.float32)

    if mode == "autoencoder":
        input_dim = int(artifact["input_dim"])
        cfg = dict(artifact["config"])
        model = _Autoencoder(
            input_dim=input_dim,
            latent_dim=int(cfg.get("autoencoder_latent_dim", 16)),
            hidden_dim_1=int(cfg.get("autoencoder_hidden_dim_1", 128)),
            hidden_dim_2=int(cfg.get("autoencoder_hidden_dim_2", 64)),
            dropout=float(cfg.get("autoencoder_dropout", 0.0)),
        )
        model.load_state_dict(artifact["state_dict"])
        return _encode_autoencoder(model, X, device=device)

    raise ValueError(f"Неизвестный feature mode в artifact: {mode}")