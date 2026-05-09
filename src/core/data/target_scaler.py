from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TargetScaler:
    mean_: float
    std_: float
    use_log_target: bool = True

    @classmethod
    def fit(cls, y: np.ndarray, use_log_target: bool = True) -> "TargetScaler":
        y = np.asarray(y, dtype=np.float32).ravel()
        z = np.log1p(y) if use_log_target else y
        mean_ = float(np.mean(z))
        std_ = float(np.std(z)) or 1.0
        return cls(mean_=mean_, std_=std_, use_log_target=use_log_target)

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32).ravel()
        z = np.log1p(y) if self.use_log_target else y
        return ((z - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        y_scaled = np.asarray(y_scaled, dtype=np.float32).ravel()
        z = y_scaled * self.std_ + self.mean_
        y = np.expm1(z) if self.use_log_target else z
        return np.maximum(y, 0.0).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean_,
            "std": self.std_,
            "use_log_target": self.use_log_target,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TargetScaler":
        return cls(
            mean_=float(payload.get("mean", 0.0)),
            std_=float(payload.get("std", 1.0)) or 1.0,
            use_log_target=bool(payload.get("use_log_target", True)),
        )