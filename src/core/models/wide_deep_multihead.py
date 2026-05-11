from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class MultiHeadWideDeep(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_crops: int,
        deep_hidden_dim: int = 128,
        deep_num_layers: int = 3,
        deep_dropout: float = 0.15,
        head_hidden_dim: int = 32,
    ) -> None:
        super().__init__()

        deep_layers: list[nn.Module] = []
        current_dim = input_dim

        for _ in range(deep_num_layers):
            deep_layers.extend(
                [
                    nn.Linear(current_dim, deep_hidden_dim),
                    nn.BatchNorm1d(deep_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(deep_dropout),
                ]
            )
            current_dim = deep_hidden_dim

        self.deep = nn.Sequential(*deep_layers)

        self.wide = nn.Linear(input_dim, 1)

        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(deep_hidden_dim, head_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(deep_dropout),
                    nn.Linear(head_hidden_dim, 1),
                )
                for _ in range(num_crops)
            ]
        )

    def forward(self, x: torch.Tensor, crop_ids: torch.Tensor) -> torch.Tensor:
        deep_features = self.deep(x)

        deep_outputs = torch.zeros(
            (x.size(0), 1),
            dtype=x.dtype,
            device=x.device,
        )

        for crop_id, head in enumerate(self.heads):
            mask = crop_ids == crop_id
            if mask.any():
                deep_outputs[mask] = head(deep_features[mask])

        wide_output = self.wide(x)

        return 0.1 * wide_output + deep_outputs


def build_multihead_wide_deep(
    input_dim: int,
    num_crops: int,
    config: dict[str, Any] | None = None,
) -> MultiHeadWideDeep:
    config = config or {}

    return MultiHeadWideDeep(
        input_dim=input_dim,
        num_crops=num_crops,
        deep_hidden_dim=int(config.get("deep_hidden_dim", 128)),
        deep_num_layers=int(config.get("deep_num_layers", 3)),
        deep_dropout=float(config.get("deep_dropout", 0.15)),
        head_hidden_dim=int(config.get("head_hidden_dim", 32)),
    )