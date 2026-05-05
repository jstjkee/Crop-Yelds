from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

try:
    import umap
except Exception:
    umap = None

from src.config import AUTOENCODER_CONFIG, RANDOM_STATE, TRAIN_CONFIG
from src.models.autoencoder import (
    Autoencoder,
    build_autoencoder,
    encode_features,
    train_autoencoder,
)


@dataclass
class FeatureTransformResult:
    mode: str
    X_train: np.ndarray
    X_test: np.ndarray
    transformer: Any | None = None
    metadata: dict[str, Any] | None = None


def _ensure_2d_float32(X: np.ndarray, name: str) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"{name} должен быть двумерным массивом [n_samples, n_features]")
    return X.astype(np.float32)


def _resolve_n_components(X_train: np.ndarray) -> int:
    requested = int(TRAIN_CONFIG.get("transform_components", 16))
    return max(1, min(requested, X_train.shape[0], X_train.shape[1]))


def apply_raw_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> FeatureTransformResult:
    X_train = _ensure_2d_float32(X_train, "X_train")
    X_test = _ensure_2d_float32(X_test, "X_test")

    return FeatureTransformResult(
        mode="raw",
        X_train=X_train,
        X_test=X_test,
        transformer=None,
        metadata={
            "input_dim_before": int(X_train.shape[1]),
            "input_dim_after": int(X_train.shape[1]),
        },
    )


def apply_pca_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> FeatureTransformResult:
    X_train = _ensure_2d_float32(X_train, "X_train")
    X_test = _ensure_2d_float32(X_test, "X_test")

    n_components = _resolve_n_components(X_train)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)

    X_train_pca = pca.fit_transform(X_train).astype(np.float32)
    X_test_pca = pca.transform(X_test).astype(np.float32)

    explained_variance_ratio = float(np.sum(pca.explained_variance_ratio_))

    return FeatureTransformResult(
        mode="pca",
        X_train=X_train_pca,
        X_test=X_test_pca,
        transformer=pca,
        metadata={
            "input_dim_before": int(X_train.shape[1]),
            "input_dim_after": int(X_train_pca.shape[1]),
            "n_components": int(n_components),
            "explained_variance_ratio_sum": explained_variance_ratio,
        },
    )


def apply_umap_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> FeatureTransformResult:
    X_train = _ensure_2d_float32(X_train, "X_train")
    X_test = _ensure_2d_float32(X_test, "X_test")

    if umap is None:
        raise ImportError("UMAP не установлен. Установи пакет: pip install umap-learn")

    n_components = max(1, min(int(TRAIN_CONFIG.get("transform_components", 16)), X_train.shape[1]))

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        random_state=RANDOM_STATE,
    )

    X_train_umap = reducer.fit_transform(X_train).astype(np.float32)
    X_test_umap = reducer.transform(X_test).astype(np.float32)

    return FeatureTransformResult(
        mode="umap",
        X_train=X_train_umap,
        X_test=X_test_umap,
        transformer=reducer,
        metadata={
            "input_dim_before": int(X_train.shape[1]),
            "input_dim_after": int(X_train_umap.shape[1]),
            "n_components": int(n_components),
            "n_neighbors": 15,
            "min_dist": 0.1,
        },
    )


def apply_autoencoder_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    device: str,
) -> FeatureTransformResult:
    X_train = _ensure_2d_float32(X_train, "X_train")
    X_test = _ensure_2d_float32(X_test, "X_test")

    ae: Autoencoder = build_autoencoder(
        input_dim=X_train.shape[1],
        config=AUTOENCODER_CONFIG,
    )

    ae = train_autoencoder(
        model=ae,
        X_train=X_train,
        device=device,
        epochs=int(AUTOENCODER_CONFIG.get("epochs", 30)),
        batch_size=int(AUTOENCODER_CONFIG.get("batch_size", 256)),
        lr=float(AUTOENCODER_CONFIG.get("lr", 1e-3)),
        weight_decay=float(AUTOENCODER_CONFIG.get("weight_decay", 0.0)),
        verbose=True,
    )

    X_train_ae = encode_features(
        model=ae,
        X=X_train,
        device=device,
        batch_size=int(AUTOENCODER_CONFIG.get("batch_size", 256)),
    )

    X_test_ae = encode_features(
        model=ae,
        X=X_test,
        device=device,
        batch_size=int(AUTOENCODER_CONFIG.get("batch_size", 256)),
    )

    return FeatureTransformResult(
        mode="autoencoder",
        X_train=X_train_ae,
        X_test=X_test_ae,
        transformer=ae,
        metadata={
            "input_dim_before": int(X_train.shape[1]),
            "input_dim_after": int(X_train_ae.shape[1]),
            "latent_dim": int(AUTOENCODER_CONFIG.get("latent_dim", 16)),
        },
    )


def build_feature_view(
    mode: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    device: str,
) -> FeatureTransformResult:
    normalized_mode = mode.lower().strip()

    if normalized_mode == "raw":
        return apply_raw_features(X_train, X_test)

    if normalized_mode == "pca":
        return apply_pca_features(X_train, X_test)

    if normalized_mode == "umap":
        return apply_umap_features(X_train, X_test)

    if normalized_mode == "autoencoder":
        return apply_autoencoder_features(X_train, X_test, device=device)

    raise ValueError(f"Неизвестный режим признаков: {mode}")