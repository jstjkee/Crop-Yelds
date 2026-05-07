from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TargetScaler:
    mean_: float
    std_: float

    @classmethod
    def fit(cls, y: np.ndarray) -> "TargetScaler":
        y = np.asarray(y).astype(np.float32).ravel()
        mean_ = float(np.mean(y))
        std_ = float(np.std(y))

        if std_ == 0.0:
            std_ = 1.0

        return cls(mean_=mean_, std_=std_)

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).astype(np.float32).ravel()
        return ((y - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        y_scaled = np.asarray(y_scaled).astype(np.float32).ravel()
        return (y_scaled * self.std_ + self.mean_).astype(np.float32)