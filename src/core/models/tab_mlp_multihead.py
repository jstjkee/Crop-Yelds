from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class MultiHeadTabMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_crops: int,
        hidden_dims: list[int],
        dropout: float,
        head_hidden_dim: int,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(current_dim, head_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden_dim, 1),
                )
                for _ in range(num_crops)
            ]
        )

    def forward(self, x: torch.Tensor, crop_ids: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)

        outputs = torch.zeros(
            (x.size(0), 1),
            dtype=x.dtype,
            device=x.device,
        )

        for crop_id, head in enumerate(self.heads):
            mask = crop_ids == crop_id
            if mask.any():
                outputs[mask] = head(features[mask])

        return outputs


def build_multihead_tab_mlp(
    input_dim: int,
    num_crops: int,
    config: dict[str, Any] | None = None,
) -> MultiHeadTabMLP:
    config = config or {}

    return MultiHeadTabMLP(
        input_dim=input_dim,
        num_crops=num_crops,
        hidden_dims=list(config.get("hidden_dims", [128, 64, 32])),
        dropout=float(config.get("dropout", 0.20)),
        head_hidden_dim=int(config.get("head_hidden_dim", 32)),
    )