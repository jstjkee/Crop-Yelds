from __future__ import annotations
import torch
import torch.nn as nn

class NumericFeatureTokenizer(nn.Module):
    def __init__(self, num_features: int, d_model: int) -> None:
        super().__init__()

        if num_features <= 0:
            raise ValueError("num_features должен быть > 0")
        if d_model <= 0:
            raise ValueError("d_model должен быть > 0")

        self.num_features = num_features
        self.d_model = d_model

        self.weight = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_features, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"x должен иметь форму [batch_size, num_features], получено: {tuple(x.shape)}"
            )
        if x.size(1) != self.num_features:
            raise ValueError(
                f"Ожидалось {self.num_features} признаков, получено {x.size(1)}"
            )

        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class CropHead(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()

        hidden_dim = max(d_model // 2, 8)

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadTabularTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_crops: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim должен быть > 0")
        if num_crops <= 0:
            raise ValueError("num_crops должен быть > 0")
        if d_model <= 0:
            raise ValueError("d_model должен быть > 0")
        if nhead <= 0:
            raise ValueError("nhead должен быть > 0")
        if d_model % nhead != 0:
            raise ValueError("d_model должен делиться на nhead без остатка")

        ff_dim = dim_feedforward if dim_feedforward is not None else d_model * 4

        self.input_dim = input_dim
        self.num_crops = num_crops
        self.d_model = d_model

        self.tokenizer = NumericFeatureTokenizer(
            num_features=input_dim,
            d_model=d_model,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.output_norm = nn.LayerNorm(d_model)

        self.heads = nn.ModuleList(
            [CropHead(d_model=d_model, dropout=dropout) for _ in range(num_crops)]
        )

    def pool_tokens(self, encoded_tokens: torch.Tensor) -> torch.Tensor:

        if encoded_tokens.ndim != 3:
            raise ValueError(
                "encoded_tokens должен иметь форму [batch_size, num_features, d_model]"
            )

        pooled = encoded_tokens.mean(dim=1)
        return self.output_norm(pooled)

    def forward(self, x: torch.Tensor, crop_ids: torch.Tensor) -> torch.Tensor:

        if x.ndim != 2:
            raise ValueError(
                f"x должен иметь размерность [batch_size, input_dim], получено: {tuple(x.shape)}"
            )

        if crop_ids.ndim != 1:
            raise ValueError(
                f"crop_ids должен иметь размерность [batch_size], получено: {tuple(crop_ids.shape)}"
            )

        if x.size(0) != crop_ids.size(0):
            raise ValueError("Размер батча в x и crop_ids должен совпадать")

        tokens = self.tokenizer(x)
        encoded = self.encoder(tokens)
        pooled = self.pool_tokens(encoded)

        outputs = torch.zeros(
            (x.size(0), 1),
            device=x.device,
            dtype=pooled.dtype,
        )

        for crop_id, head in enumerate(self.heads):
            mask = crop_ids == crop_id
            if mask.any():
                outputs[mask] = head(pooled[mask])

        return outputs

def build_multihead_transformer(
    input_dim: int,
    num_crops: int,
    config: dict,
) -> MultiHeadTabularTransformer:
    return MultiHeadTabularTransformer(
        input_dim=input_dim,
        num_crops=num_crops,
        d_model=config.get("d_model", 64),
        nhead=config.get("nhead", 4),
        num_layers=config.get("num_layers", 2),
        dim_feedforward=config.get("dim_feedforward"),
        dropout=config.get("dropout", 0.1),
    )