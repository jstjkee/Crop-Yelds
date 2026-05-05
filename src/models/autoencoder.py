from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim должен быть > 0")
        if latent_dim <= 0:
            raise ValueError("latent_dim должен быть > 0")

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        encoder_layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
        ]
        if dropout > 0:
            encoder_layers.append(nn.Dropout(dropout))
        encoder_layers.extend(
            [
                nn.Linear(hidden_dim_1, hidden_dim_2),
                nn.ReLU(),
            ]
        )
        if dropout > 0:
            encoder_layers.append(nn.Dropout(dropout))
        encoder_layers.append(nn.Linear(hidden_dim_2, latent_dim))

        decoder_layers: list[nn.Module] = [
            nn.Linear(latent_dim, hidden_dim_2),
            nn.ReLU(),
        ]
        if dropout > 0:
            decoder_layers.append(nn.Dropout(dropout))
        decoder_layers.extend(
            [
                nn.Linear(hidden_dim_2, hidden_dim_1),
                nn.ReLU(),
                nn.Linear(hidden_dim_1, input_dim),
            ]
        )

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return z, reconstructed


def build_autoencoder(
    input_dim: int,
    config: dict,
) -> Autoencoder:
    return Autoencoder(
        input_dim=input_dim,
        latent_dim=config.get("latent_dim", 16),
        hidden_dim_1=config.get("hidden_dim_1", 128),
        hidden_dim_2=config.get("hidden_dim_2", 64),
        dropout=config.get("dropout", 0.0),
    )


def train_autoencoder(
    model: Autoencoder,
    X_train: np.ndarray,
    device: str = "cpu",
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    verbose: bool = True,
) -> Autoencoder:
    if X_train.ndim != 2:
        raise ValueError("X_train должен быть двумерным массивом [n_samples, n_features]")

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []

        for (xb,) in loader:
            xb = xb.to(device)

            _, reconstructed = model(xb)
            loss = criterion(reconstructed, xb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))

        if verbose:
            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            print(f"[Autoencoder] Epoch {epoch + 1:03d} | loss={mean_loss:.6f}")

    return model


@torch.no_grad()
def encode_features(
    model: Autoencoder,
    X: np.ndarray,
    device: str = "cpu",
    batch_size: int = 1024,
) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("X должен быть двумерным массивом [n_samples, n_features]")

    model.eval()
    model = model.to(device)

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    encoded_batches: list[np.ndarray] = []

    for (xb,) in loader:
        xb = xb.to(device)
        z, _ = model(xb)
        encoded_batches.append(z.cpu().numpy())

    return np.concatenate(encoded_batches, axis=0).astype(np.float32)


@torch.no_grad()
def reconstruct_features(
    model: Autoencoder,
    X: np.ndarray,
    device: str = "cpu",
    batch_size: int = 1024,
) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("X должен быть двумерным массивом [n_samples, n_features]")

    model.eval()
    model = model.to(device)

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    reconstructed_batches: list[np.ndarray] = []

    for (xb,) in loader:
        xb = xb.to(device)
        _, reconstructed = model(xb)
        reconstructed_batches.append(reconstructed.cpu().numpy())

    return np.concatenate(reconstructed_batches, axis=0).astype(np.float32)


@torch.no_grad()
def autoencoder_reconstruction_mse(
    model: Autoencoder,
    X: np.ndarray,
    device: str = "cpu",
) -> float:
    reconstructed = reconstruct_features(model, X, device=device)
    mse = np.mean((X.astype(np.float32) - reconstructed) ** 2)
    return float(mse)