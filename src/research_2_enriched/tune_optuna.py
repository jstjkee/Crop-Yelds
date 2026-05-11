from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import optuna

from src.core.config import (
    MLP_RESNET_CONFIG,
    RANDOM_STATE,
    RESEARCH_2_RESULTS,
    TAB_MLP_CONFIG,
    TRAIN_CONFIG,
    TRANSFORMER_CONFIG,
    ensure_project_dirs,
)
from src.core.data.target_scaler import TargetScaler
from src.core.evaluation.metrics import regression_metrics
from src.core.training.dl_trainer import (
    build_model,
    fit_with_early_stopping,
    get_device,
    make_loader,
    predict_unscaled,
    seed_everything,
)
from src.research_2_enriched.build_dataset import prepare_enriched_dataset
from src.research_2_enriched.config import RESEARCH_2_CONFIG


def _base_model_config(model_type: str) -> dict[str, Any]:
    if model_type == "transformer":
        return dict(TRANSFORMER_CONFIG)
    if model_type == "mlp_resnet":
        return dict(MLP_RESNET_CONFIG)
    if model_type == "tab_mlp":
        return dict(TAB_MLP_CONFIG)
    raise ValueError(f"Optuna пока поддерживает только transformer / mlp_resnet / tab_mlp, получено: {model_type}")


def _suggest_model_config(trial: "optuna.Trial", model_type: str) -> dict[str, Any]:
    if model_type == "transformer":
        d_model = trial.suggest_categorical("d_model", [16, 32, 64, 96, 128])

        # Делаем nhead статическим search space
        nhead = trial.suggest_categorical("nhead", [1, 2, 4, 8])

        # Отдельно выбираем множитель, а не готовое значение dim_feedforward
        ff_mult = trial.suggest_categorical("ff_mult", [2, 4, 6])

        return {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "dim_feedforward": d_model * ff_mult,
            "dropout": trial.suggest_float("dropout", 0.05, 0.35),
        }

    if model_type == "mlp_resnet":
        d_model = trial.suggest_categorical("d_model", [64, 96, 128, 192, 256])

        return {
            "d_model": d_model,
            "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 384, 512]),
            "num_blocks": trial.suggest_int("num_blocks", 1, 4),
            "dropout": trial.suggest_float("dropout", 0.05, 0.35),
            "head_hidden_dim": trial.suggest_categorical("head_hidden_dim", [16, 32, 64, 128]),
        }

    if model_type == "tab_mlp":
        hidden_dims_key = trial.suggest_categorical(
            "hidden_dims_key",
            [
                "128_64_32",
                "256_128_64",
                "256_128",
                "512_256_128",
            ],
        )

        hidden_dims_map = {
            "128_64_32": [128, 64, 32],
            "256_128_64": [256, 128, 64],
            "256_128": [256, 128],
            "512_256_128": [512, 256, 128],
        }

        return {
            "hidden_dims": hidden_dims_map[hidden_dims_key],
            "dropout": trial.suggest_float("dropout", 0.05, 0.35),
            "head_hidden_dim": trial.suggest_categorical("head_hidden_dim", [16, 32, 64, 128]),
        }

    raise ValueError(f"Неизвестный model_type: {model_type}")


def _suggest_train_config(trial: "optuna.Trial") -> dict[str, Any]:
    return {
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "huber_delta": trial.suggest_float("huber_delta", 0.5, 2.0),
        "clip_grad_norm": trial.suggest_float("clip_grad_norm", 0.5, 5.0),
        "epochs": trial.suggest_categorical("epochs", [20, 25, 30, 35, 40]),
        "patience": trial.suggest_categorical("patience", [5, 6, 8, 10]),
        "num_workers": int(TRAIN_CONFIG.get("num_workers", 0)),
    }


def _build_artifact_paths(
    model_type: str,
    split_name: str,
    feature_set_name: str,
) -> dict[str, Path]:
    metrics_dir = RESEARCH_2_RESULTS["metrics"]
    study_prefix = f"research_2_{model_type}_{split_name}_{feature_set_name}"

    return {
        "db": metrics_dir / f"{study_prefix}.db",
        "trials_csv": metrics_dir / f"{study_prefix}_trials.csv",
        "best_json": metrics_dir / f"{study_prefix}_best_params.json",
    }


def _trials_to_df(study: "optuna.Study") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for trial in study.trials:
        row: dict[str, Any] = {
            "trial_number": trial.number,
            "state": str(trial.state),
            "objective_val_rmse": trial.value,
        }

        for k, v in trial.params.items():
            row[f"param__{k}"] = v

        for k, v in trial.user_attrs.items():
            row[f"user__{k}"] = v

        rows.append(row)

    return pd.DataFrame(rows)


def main(
    model_type: str = "transformer",
    feature_mode: str = "raw",
    split_name: str = "year",
    feature_set_name: str = "full",
    n_trials: int = 20,
    timeout: int | None = None,
) -> None:
    ensure_project_dirs()
    seed_everything(RANDOM_STATE)

    if feature_mode != "raw":
        raise ValueError("Optuna-подбор сейчас поддерживается только для feature_mode='raw'")

    device = get_device()
    cfg = RESEARCH_2_CONFIG

    print("=" * 100)
    print(
        f"[research_2][optuna] model={model_type} | "
        f"split={split_name} | feature_set={feature_set_name} | "
        f"feature_mode={feature_mode} | device={device}"
    )
    print("=" * 100)

    prepared = prepare_enriched_dataset(
        split_name=split_name,
        feature_set_name=feature_set_name,
    )

    target_scaler = TargetScaler.fit(
        prepared.y_train,
        use_log_target=bool(cfg.get("use_log_target", True)),
    )

    artifact_paths = _build_artifact_paths(
        model_type=model_type,
        split_name=split_name,
        feature_set_name=feature_set_name,
    )

    storage = f"sqlite:///{artifact_paths['db'].as_posix()}"
    study_name = f"research_2_{model_type}_{split_name}_{feature_set_name}"

    def objective(trial: "optuna.Trial") -> float:
        seed_everything(RANDOM_STATE)

        model_config = _suggest_model_config(trial, model_type)
        train_config = _suggest_train_config(trial)

        train_loader = make_loader(
            X=prepared.X_train,
            y=target_scaler.transform(prepared.y_train),
            crop_ids=prepared.crop_train,
            batch_size=int(train_config["batch_size"]),
            shuffle=True,
            num_workers=int(train_config["num_workers"]),
            balanced_sampler=bool(cfg.get("balanced_sampler", False)),
        )

        val_loader = make_loader(
            X=prepared.X_val,
            y=target_scaler.transform(prepared.y_val),
            crop_ids=prepared.crop_val,
            batch_size=int(train_config["batch_size"]),
            shuffle=False,
            num_workers=int(train_config["num_workers"]),
        )

        test_loader = make_loader(
            X=prepared.X_test,
            y=target_scaler.transform(prepared.y_test),
            crop_ids=prepared.crop_test,
            batch_size=int(train_config["batch_size"]),
            shuffle=False,
            num_workers=int(train_config["num_workers"]),
        )

        model = build_model(
            model_type=model_type,
            input_dim=int(prepared.X_train.shape[1]),
            num_crops=len(prepared.crop_to_id),
            model_config=model_config,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_config["lr"]),
            weight_decay=float(train_config["weight_decay"]),
        )

        criterion = nn.HuberLoss(delta=float(train_config["huber_delta"]))

        history, best_val_metrics = fit_with_early_stopping(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            target_scaler=target_scaler,
            epochs=int(train_config["epochs"]),
            patience=int(train_config["patience"]),
            clip_grad_norm=float(train_config["clip_grad_norm"]),
            verbose_prefix=(
                f"[research_2][optuna][{model_type}]"
                f"[trial={trial.number}] "
            ),
        )

        y_test_true, y_test_pred, _ = predict_unscaled(
            model=model,
            loader=test_loader,
            device=device,
            target_scaler=target_scaler,
        )
        test_metrics = regression_metrics(y_test_true, y_test_pred)

        trial.set_user_attr("val_mae", float(best_val_metrics["mae"]))
        trial.set_user_attr("val_rmse", float(best_val_metrics["rmse"]))
        trial.set_user_attr("val_r2", float(best_val_metrics["r2"]))

        trial.set_user_attr("test_mae", float(test_metrics["mae"]))
        trial.set_user_attr("test_rmse", float(test_metrics["rmse"]))
        trial.set_user_attr("test_r2", float(test_metrics["r2"]))

        trial.set_user_attr("input_dim", int(prepared.X_train.shape[1]))
        trial.set_user_attr("rows_train", int(len(prepared.train_df)))
        trial.set_user_attr("rows_val", int(len(prepared.val_df)))
        trial.set_user_attr("rows_test", int(len(prepared.test_df)))
        trial.set_user_attr("num_epochs_run", int(len(history)))

        return float(best_val_metrics["rmse"])

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=int(n_trials),
        timeout=timeout,
        show_progress_bar=False,
    )

    trials_df = _trials_to_df(study).sort_values(
        by=["objective_val_rmse", "trial_number"],
        ascending=[True, True],
    )
    trials_df.to_csv(artifact_paths["trials_csv"], index=False, encoding="utf-8-sig")

    best_trial = study.best_trial
    best_model_config = _suggest_model_config(best_trial, model_type)
    best_train_config = _suggest_train_config(best_trial)

    best_payload = {
        "model_type": model_type,
        "feature_mode": feature_mode,
        "split_name": split_name,
        "feature_set_name": feature_set_name,
        "objective": "val_rmse",
        "best_trial_number": int(best_trial.number),
        "best_objective_value": float(best_trial.value),
        "best_params_flat": best_trial.params,
        "model_config": best_model_config,
        "train_config": best_train_config,
        "device_used": device,
    }

    with open(artifact_paths["best_json"], "w", encoding="utf-8") as f:
        json.dump(best_payload, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("[research_2][optuna] FINISHED")
    print(f"Best trial: {best_trial.number}")
    print(f"Best val RMSE: {best_trial.value:.6f}")
    print(f"Saved trials: {artifact_paths['trials_csv']}")
    print(f"Saved best params: {artifact_paths['best_json']}")
    print(f"Study DB: {artifact_paths['db']}")
    print("=" * 100)


if __name__ == "__main__":
    main()