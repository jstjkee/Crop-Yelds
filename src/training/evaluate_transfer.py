from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

try:
    import umap
except Exception:
    umap = None

from torch.utils.data import DataLoader, Dataset

from src.config import (
    AUTOENCODER_CONFIG,
    CROP_COL,
    RANDOM_STATE,
    RESULTS_DIR,
    RUSSIAN_DATA_PATH,
    SOURCE_DATA_PATH,
    TARGET_COL,
    TEST_SIZE,
    TRANSFORMER_CONFIG,
    TRAIN_CONFIG,
)
from src.data.load_data import load_csv, validate_required_columns
from src.data.preprocess import prepare_dataset, transform_external_dataset
from src.evaluation.metrics import regression_metrics
from src.models.autoencoder import build_autoencoder, encode_features, train_autoencoder
from src.models.multihead_transformer import build_multihead_transformer


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


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    y_true = []
    y_pred = []

    for xb, yb, cb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        cb = cb.to(device)

        preds = model(xb, cb)

        y_true.append(yb.cpu().numpy())
        y_pred.append(preds.cpu().numpy())

    return (
        np.concatenate(y_true).ravel(),
        np.concatenate(y_pred).ravel(),
    )


def build_feature_view_for_transfer(
    mode: str,
    X_source_train: np.ndarray,
    X_external: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    mode = mode.lower()

    if mode == "raw":
        return X_source_train, X_external

    if mode == "pca":
        n_components = min(
            TRAIN_CONFIG.get("transform_components", 16),
            X_source_train.shape[0],
            X_source_train.shape[1],
        )
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_source_train_mode = pca.fit_transform(X_source_train).astype(np.float32)
        X_external_mode = pca.transform(X_external).astype(np.float32)
        return X_source_train_mode, X_external_mode

    if mode == "umap":
        if umap is None:
            raise ImportError("UMAP не установлен. Установи пакет: pip install umap-learn")

        n_components = min(
            TRAIN_CONFIG.get("transform_components", 16),
            X_source_train.shape[1],
        )

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=15,
            min_dist=0.1,
            random_state=RANDOM_STATE,
        )
        X_source_train_mode = reducer.fit_transform(X_source_train).astype(np.float32)
        X_external_mode = reducer.transform(X_external).astype(np.float32)
        return X_source_train_mode, X_external_mode

    if mode == "autoencoder":
        ae = build_autoencoder(
            input_dim=X_source_train.shape[1],
            config=AUTOENCODER_CONFIG,
        )

        ae = train_autoencoder(
            model=ae,
            X_train=X_source_train,
            device=device,
            epochs=AUTOENCODER_CONFIG.get("epochs", 30),
            batch_size=AUTOENCODER_CONFIG.get("batch_size", 256),
            lr=AUTOENCODER_CONFIG.get("lr", 1e-3),
            weight_decay=AUTOENCODER_CONFIG.get("weight_decay", 0.0),
        )

        X_source_train_mode = encode_features(ae, X_source_train, device=device)
        X_external_mode = encode_features(ae, X_external, device=device)
        return X_source_train_mode, X_external_mode

    raise ValueError(f"Неизвестный режим признаков: {mode}")


def evaluate_transfer_for_mode(mode: str, device: str) -> dict:
    source_df = load_csv(SOURCE_DATA_PATH)
    russian_df = load_csv(RUSSIAN_DATA_PATH)

    validate_required_columns(source_df, TARGET_COL, CROP_COL)
    validate_required_columns(russian_df, TARGET_COL, CROP_COL)

    prepared_source = prepare_dataset(
        df=source_df,
        target_col=TARGET_COL,
        crop_col=CROP_COL,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    source_X_train = prepared_source["X_train"]
    crop_to_id = prepared_source["crop_to_id"]
    preprocessor = prepared_source["preprocessor"]

    prepared_russian = transform_external_dataset(
        df=russian_df,
        preprocessor=preprocessor,
        crop_to_id=crop_to_id,
        target_col=TARGET_COL,
        crop_col=CROP_COL,
    )

    X_russian = prepared_russian["X"]
    y_russian = prepared_russian["y"]
    crop_russian = prepared_russian["crop_ids"]
    russian_filtered_df = prepared_russian["df"]

    if len(X_russian) == 0:
        raise ValueError("После фильтрации в российском датасете не осталось строк с известными культурами")

    _, X_russian_mode = build_feature_view_for_transfer(
        mode=mode,
        X_source_train=source_X_train,
        X_external=X_russian,
        device=device,
    )

    test_dataset = YieldDataset(X_russian_mode, y_russian, crop_russian)
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN_CONFIG.get("batch_size", 128),
        shuffle=False,
    )

    checkpoint_path = Path(RESULTS_DIR) / "models" / f"multihead_transformer_{mode}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Не найден checkpoint для режима '{mode}': {checkpoint_path}. "
            f"Сначала запусти обучение source-модели."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_multihead_transformer(
        input_dim=checkpoint["input_dim"],
        num_crops=len(checkpoint["crop_to_id"]),
        config=checkpoint.get("transformer_config", TRANSFORMER_CONFIG),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred = predict(model, test_loader, device)
    metrics = regression_metrics(y_true, y_pred)

    result = {
        "mode": mode,
        "dataset": "russian_transfer",
        "num_rows": len(russian_filtered_df),
        "num_crops": russian_filtered_df[CROP_COL].nunique(),
        "mae": metrics["mae"],
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "checkpoint_path": str(checkpoint_path),
    }
    return result


def main() -> None:
    device = get_device()
    print(f"Device: {device}")

    modes = ["raw", "pca", "umap", "autoencoder"]
    results = []

    for mode in modes:
        print("=" * 80)
        print(f"Transfer evaluation для режима: {mode}")
        result = evaluate_transfer_for_mode(mode=mode, device=device)
        results.append(result)

    results_df = pd.DataFrame(results)

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "transfer_metrics.csv"
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("Итоговые transfer-результаты:")
    print(results_df)
    print(f"CSV сохранен: {output_path}")


if __name__ == "__main__":
    main()