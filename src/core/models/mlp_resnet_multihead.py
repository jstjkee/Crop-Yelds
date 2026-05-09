from __future__ import annotations

import torch
import torch.nn as nn


class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return residual + x


class CropSpecificHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        head_hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadMLPResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_crops: int,
        d_model: int = 128,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        dropout: float = 0.1,
        head_hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(
                    d_model=d_model,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

        self.crop_heads = nn.ModuleList(
            [
                CropSpecificHead(
                    d_model=d_model,
                    head_hidden_dim=head_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_crops)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        crop_ids: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Ожидался 2D tensor [batch, features], получено: {tuple(x.shape)}")

        h = self.input_proj(x)

        for block in self.blocks:
            h = block(h)

        h = self.final_norm(h)

        outputs = torch.zeros((x.size(0), 1), dtype=h.dtype, device=h.device)

        unique_crop_ids = torch.unique(crop_ids)
        for crop_id in unique_crop_ids:
            crop_idx = int(crop_id.item())
            mask = crop_ids == crop_id
            if mask.any():
                outputs[mask] = self.crop_heads[crop_idx](h[mask])

        return outputs


def build_multihead_mlp_resnet(
    input_dim: int,
    num_crops: int,
    config: dict,
) -> MultiHeadMLPResNet:
    return MultiHeadMLPResNet(
        input_dim=input_dim,
        num_crops=num_crops,
        d_model=int(config.get("d_model", 128)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        num_blocks=int(config.get("num_blocks", 4)),
        dropout=float(config.get("dropout", 0.1)),
        head_hidden_dim=int(config.get("head_hidden_dim", 64)),
    )