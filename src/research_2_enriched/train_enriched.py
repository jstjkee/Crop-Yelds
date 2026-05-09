from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.config import (
    METRICS_DIR,
    MODELS_DIR,
    MLP_RESNET_CONFIG,
    RANDOM_STATE,
    RUSSIAN_FINAL_CONFIG,
    TABLES_DIR,
    TRAIN_CONFIG,
    TRANSFORMER_CONFIG,
    ensure_project_dirs,
)
from src.data.load_data import load_csv, summarize_dataframe, validate_required_columns
from src.evaluation.metrics import build_metrics_row, regression_metrics, regression_metrics_by_crop
from src.models.mlp_resnet_multihead import build_multihead_mlp_resnet
from src.models.multihead_transformer import build_multihead_transformer
from src.training.train_source import get_device, seed_everything


@dataclass
class PreparedRussianFinalDataset:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray

    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    crop_train: np.ndarray
    crop_val: np.ndarray
    crop_test: np.ndarray

    crop_to_id: dict[str, int]
    id_to_crop: dict[int, str]

    preprocessor: ColumnTransformer

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame

    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]


class YieldDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, crop_ids: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        self.crop_ids = torch.tensor(crop_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.crop_ids[idx]


class LogTargetScaler:
    def __init__(self, use_log: bool = True) -> None:
        self.use_log = use_log
        self.mean_: float = 0.0
        self.std_: float = 1.0

    @classmethod
    def fit(cls, y: np.ndarray, use_log: bool = True) -> "LogTargetScaler":
        obj = cls(use_log=use_log)
        z = np.asarray(y, dtype=np.float32).ravel()
        if use_log:
            z = np.log1p(z)

        obj.mean_ = float(np.mean(z))
        obj.std_ = float(np.std(z)) or 1.0
        return obj

    def transform(self, y: np.ndarray) -> np.ndarray:
        z = np.asarray(y, dtype=np.float32).ravel()
        if self.use_log:
            z = np.log1p(z)
        return ((z - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        z = np.asarray(y_scaled, dtype=np.float32).ravel()
        z = z * self.std_ + self.mean_
        if self.use_log:
            z = np.expm1(z)
        return np.maximum(z, 0.0).astype(np.float32)


def build_model(model_type: str, input_dim: int, num_crops: int) -> nn.Module:
    if model_type == "transformer":
        return build_multihead_transformer(
            input_dim=input_dim,
            num_crops=num_crops,
            config=TRANSFORMER_CONFIG,
        )

    if model_type == "mlp_resnet":
        return build_multihead_mlp_resnet(
            input_dim=input_dim,
            num_crops=num_crops,
            config=MLP_RESNET_CONFIG,
        )

    raise ValueError(f"Неизвестный model_type: {model_type}")


def get_model_config(model_type: str) -> dict[str, Any]:
    if model_type == "transformer":
        return TRANSFORMER_CONFIG
    if model_type == "mlp_resnet":
        return MLP_RESNET_CONFIG
    raise ValueError(f"Неизвестный model_type: {model_type}")


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    crop_ids: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    balanced_sampler: bool = False,
) -> DataLoader:
    dataset = YieldDataset(X=X, y=y, crop_ids=crop_ids)

    if balanced_sampler and shuffle:
        counts = np.bincount(crop_ids)
        weights = np.array([1.0 / max(counts[c], 1) for c in crop_ids], dtype=np.float32)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    model.train()
    losses: list[float] = []

    for xb, yb, cb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        cb = cb.to(device)

        preds = model(xb, cb)
        loss = criterion(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def predict_unscaled(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    target_scaler: LogTargetScaler,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    y_true_scaled = []
    y_pred_scaled = []

    for xb, yb, cb in loader:
        xb = xb.to(device)
        cb = cb.to(device)

        preds = model(xb, cb)

        y_true_scaled.append(yb.cpu().numpy())
        y_pred_scaled.append(preds.cpu().numpy())

    y_true = target_scaler.inverse_transform(np.concatenate(y_true_scaled).ravel())
    y_pred = target_scaler.inverse_transform(np.concatenate(y_pred_scaled).ravel())
    return y_true, y_pred


def _drop_invalid_rows(
    df: pd.DataFrame,
    target_col: str,
    crop_col: str,
) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=[target_col, crop_col])
    df = df[df[crop_col].astype(str).str.strip() != ""]
    df = df[df[target_col] >= 0]
    return df.reset_index(drop=True)


def _filter_rare_crops(
    df: pd.DataFrame,
    crop_col: str,
    min_crop_count: int,
) -> pd.DataFrame:
    counts = df[crop_col].astype(str).value_counts()
    keep = counts[counts >= min_crop_count].index.tolist()
    return df[df[crop_col].astype(str).isin(keep)].reset_index(drop=True)


def _split_random(
    df: pd.DataFrame,
    crop_col: str,
    test_size: float,
    val_size_from_train: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_full_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df[crop_col].astype(str),
    )

    train_df, val_df = train_test_split(
        train_full_df,
        test_size=val_size_from_train,
        random_state=RANDOM_STATE,
        stratify=train_full_df[crop_col].astype(str),
    )

    return train_df, val_df, test_df


def _split_by_year(
    df: pd.DataFrame,
    year_col: str,
    test_years: int,
    val_years: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = sorted(df[year_col].dropna().astype(int).unique().tolist())
    if len(years) < test_years + val_years + 1:
        raise ValueError(f"Слишком мало лет для split='year': {years}")

    test_set = set(years[-test_years:])
    val_set = set(years[-test_years - val_years : -test_years])

    train_df = df[~df[year_col].astype(int).isin(test_set | val_set)]
    val_df = df[df[year_col].astype(int).isin(val_set)]
    test_df = df[df[year_col].astype(int).isin(test_set)]

    return train_df, val_df, test_df


def _build_preprocessor(
    train_df: pd.DataFrame,
    target_col: str,
    crop_col: str,
    exclude_cols: list[str],
    zero_fill_cols: list[str],
) -> tuple[ColumnTransformer, list[str], list[str], list[str]]:
    feature_cols = [c for c in train_df.columns if c not in {target_col, crop_col, *exclude_cols}]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    zero_fill_numeric_cols = [c for c in numeric_cols if c in zero_fill_cols]
    median_numeric_cols = [c for c in numeric_cols if c not in zero_fill_numeric_cols]

    transformers: list[tuple[str, Any, list[str]]] = []

    if median_numeric_cols:
        transformers.append(
            (
                "num_median",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                median_numeric_cols,
            )
        )

    if zero_fill_numeric_cols:
        transformers.append(
            (
                "num_zero",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                zero_fill_numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError("После исключения колонок не осталось признаков")

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor, feature_cols, numeric_cols, categorical_cols


def prepare_russian_final_dataset(df: pd.DataFrame) -> PreparedRussianFinalDataset:
    cfg = RUSSIAN_FINAL_CONFIG
    target_col = cfg["target_col"]
    crop_col = cfg["crop_col"]

    validate_required_columns(df, target_col, crop_col)

    df = _drop_invalid_rows(df, target_col, crop_col)
    df = _filter_rare_crops(df, crop_col, int(cfg.get("min_crop_count", 10)))

    split_name = str(cfg.get("split", "random"))
    if split_name == "year":
        train_df, val_df, test_df = _split_by_year(
            df=df,
            year_col="year",
            test_years=int(cfg.get("test_years", 2)),
            val_years=int(cfg.get("val_years", 1)),
        )
    else:
        train_df, val_df, test_df = _split_random(
            df=df,
            crop_col=crop_col,
            test_size=float(cfg.get("test_size", 0.2)),
            val_size_from_train=float(cfg.get("val_size_from_train", 0.2)),
        )

    crop_values = sorted(train_df[crop_col].astype(str).unique().tolist())
    crop_to_id = {crop: idx for idx, crop in enumerate(crop_values)}
    id_to_crop = {idx: crop for crop, idx in crop_to_id.items()}

    # Для year split отбрасываем культуры, которых не было в train
    val_df = val_df[val_df[crop_col].astype(str).isin(crop_to_id.keys())].copy()
    test_df = test_df[test_df[crop_col].astype(str).isin(crop_to_id.keys())].copy()

    preprocessor, feature_cols, numeric_cols, categorical_cols = _build_preprocessor(
        train_df=train_df,
        target_col=target_col,
        crop_col=crop_col,
        exclude_cols=list(cfg.get("exclude_cols", [])),
        zero_fill_cols=list(cfg.get("zero_fill_cols", [])),
    )

    X_train = preprocessor.fit_transform(train_df[feature_cols])
    X_val = preprocessor.transform(val_df[feature_cols])
    X_test = preprocessor.transform(test_df[feature_cols])

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_val, "toarray"):
        X_val = X_val.toarray()
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    return PreparedRussianFinalDataset(
        X_train=np.asarray(X_train, dtype=np.float32),
        X_val=np.asarray(X_val, dtype=np.float32),
        X_test=np.asarray(X_test, dtype=np.float32),

        y_train=train_df[target_col].to_numpy(dtype=np.float32),
        y_val=val_df[target_col].to_numpy(dtype=np.float32),
        y_test=test_df[target_col].to_numpy(dtype=np.float32),

        crop_train=train_df[crop_col].astype(str).map(crop_to_id).to_numpy(dtype=np.int64),
        crop_val=val_df[crop_col].astype(str).map(crop_to_id).to_numpy(dtype=np.int64),
        crop_test=test_df[crop_col].astype(str).map(crop_to_id).to_numpy(dtype=np.int64),

        crop_to_id=crop_to_id,
        id_to_crop=id_to_crop,

        preprocessor=preprocessor,

        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),

        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )


def train_model(
    model_type: str,
    prepared: PreparedRussianFinalDataset,
    device: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    cfg = RUSSIAN_FINAL_CONFIG
    prefix = str(cfg["results_prefix"])
    mode = str(cfg.get("feature_mode", "raw"))

    scaler = LogTargetScaler.fit(
        prepared.y_train,
        use_log=bool(cfg.get("use_log_target", True)),
    )

    train_loader = make_loader(
        X=prepared.X_train,
        y=scaler.transform(prepared.y_train),
        crop_ids=prepared.crop_train,
        batch_size=int(TRAIN_CONFIG.get("batch_size", 256)),
        shuffle=True,
        num_workers=int(TRAIN_CONFIG.get("num_workers", 0)),
        balanced_sampler=bool(cfg.get("balanced_sampler", False)),
    )

    val_loader = make_loader(
        X=prepared.X_val,
        y=scaler.transform(prepared.y_val),
        crop_ids=prepared.crop_val,
        batch_size=int(TRAIN_CONFIG.get("batch_size", 256)),
        shuffle=False,
        num_workers=int(TRAIN_CONFIG.get("num_workers", 0)),
        balanced_sampler=False,
    )

    test_loader = make_loader(
        X=prepared.X_test,
        y=scaler.transform(prepared.y_test),
        crop_ids=prepared.crop_test,
        batch_size=int(TRAIN_CONFIG.get("batch_size", 256)),
        shuffle=False,
        num_workers=int(TRAIN_CONFIG.get("num_workers", 0)),
        balanced_sampler=False,
    )

    model = build_model(
        model_type=model_type,
        input_dim=prepared.X_train.shape[1],
        num_crops=len(prepared.crop_to_id),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(TRAIN_CONFIG.get("lr", 1e-3)),
        weight_decay=float(TRAIN_CONFIG.get("weight_decay", 1e-5)),
    )
    criterion = nn.HuberLoss(delta=float(TRAIN_CONFIG.get("huber_delta", 1.0)))

    epochs = int(cfg.get("epochs", TRAIN_CONFIG.get("epochs", 20)))
    patience = int(cfg.get("patience", 6))

    best_val_rmse = float("inf")
    best_state = None
    patience_left = patience
    history_rows: list[dict[str, Any]] = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        y_val_true, y_val_pred = predict_unscaled(
            model=model,
            loader=val_loader,
            device=device,
            target_scaler=scaler,
        )
        val_metrics = regression_metrics(y_val_true, y_val_pred)

        history_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_mae": val_metrics["mae"],
                "val_mse": val_metrics["mse"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
            }
        )

        print(
            f"[{model_type}][{mode}] "
            f"epoch={epoch + 1:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_mae={val_metrics['mae']:.4f} | "
            f"val_rmse={val_metrics['rmse']:.4f} | "
            f"val_r2={val_metrics['r2']:.4f}"
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[{model_type}][{mode}] early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_test_true, y_test_pred = predict_unscaled(
        model=model,
        loader=test_loader,
        device=device,
        target_scaler=scaler,
    )

    test_metrics = regression_metrics(y_test_true, y_test_pred)

    by_crop_df = regression_metrics_by_crop(
        y_true=y_test_true,
        y_pred=y_test_pred,
        crop_ids=prepared.crop_test,
        id_to_crop=prepared.id_to_crop,
    )

    predictions_df = prepared.test_df.copy()
    predictions_df["y_true"] = y_test_true
    predictions_df["y_pred"] = y_test_pred
    predictions_df["abs_error"] = np.abs(y_test_true - y_test_pred)

    checkpoint_path = MODELS_DIR / f"{prefix}_{model_type}_{mode}.pt"
    torch.save(
        {
            "model_type": model_type,
            "model_state_dict": model.state_dict(),
            "mode": mode,
            "input_dim": int(prepared.X_train.shape[1]),
            "crop_to_id": prepared.crop_to_id,
            "id_to_crop": prepared.id_to_crop,
            "model_config": get_model_config(model_type),
            "target_scaler_mean": scaler.mean_,
            "target_scaler_std": scaler.std_,
            "target_scaler_use_log": scaler.use_log,
            "feature_cols": prepared.feature_cols,
            "numeric_cols": prepared.numeric_cols,
            "categorical_cols": prepared.categorical_cols,
        },
        checkpoint_path,
    )

    pd.DataFrame(history_rows).to_csv(
        METRICS_DIR / f"{prefix}_history_{model_type}_{mode}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_crop_df.to_csv(
        TABLES_DIR / f"{prefix}_metrics_by_crop_{model_type}_{mode}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions_df.to_csv(
        TABLES_DIR / f"{prefix}_predictions_{model_type}_{mode}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    row = build_metrics_row(
        model_name=model_type,
        feature_mode=mode,
        split_name="test",
        metrics=test_metrics,
        extra={
            "input_dim": int(prepared.X_train.shape[1]),
            "num_crops": len(prepared.crop_to_id),
            "model_path": str(checkpoint_path),
        },
    )

    return row, by_crop_df


def main() -> None:
    ensure_project_dirs()
    seed_everything(RANDOM_STATE)
    device = get_device()

    cfg = RUSSIAN_FINAL_CONFIG
    dataset_path = Path(cfg["dataset_path"])
    target_col = str(cfg["target_col"])
    crop_col = str(cfg["crop_col"])
    prefix = str(cfg["results_prefix"])

    print(f"Device: {device}")
    print(f"Dataset path: {dataset_path}")

    df = load_csv(dataset_path)
    validate_required_columns(df, target_col, crop_col)

    dataset_info = summarize_dataframe(
        df=df,
        target_col=target_col,
        crop_col=crop_col,
    )

    print("=" * 100)
    print(f"Информация о датасете: {cfg['dataset_label']}")
    for key, value in dataset_info.items():
        if key == "columns":
            continue
        print(f"{key}: {value}")

    prepared = prepare_russian_final_dataset(df)

    print("=" * 100)
    print("Подготовка датасета завершена")
    print(f"X_train shape: {prepared.X_train.shape}")
    print(f"X_val shape: {prepared.X_val.shape}")
    print(f"X_test shape: {prepared.X_test.shape}")
    print(f"Количество культур: {len(prepared.crop_to_id)}")
    print(f"Культуры: {prepared.crop_to_id}")

    joblib.dump(
        prepared.preprocessor,
        MODELS_DIR / f"{prefix}_preprocessor.joblib",
    )

    with open(MODELS_DIR / f"{prefix}_crop_mapping.json", "w", encoding="utf-8") as f:
        json.dump(prepared.crop_to_id, f, ensure_ascii=False, indent=2)

    summary_rows = []
    for model_type in cfg.get("model_types", ["mlp_resnet", "transformer"]):
        print("=" * 100)
        print(f"Запуск модели: {model_type}")
        row, _ = train_model(
            model_type=model_type,
            prepared=prepared,
            device=device,
        )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("rmse").reset_index(drop=True)
    summary_path = METRICS_DIR / f"{prefix}_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("Итоговая таблица результатов:")
    print(summary_df)
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()