from typing import Dict, Tuple

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def _calc_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }

def evaluate_model(model, X_train, y_train, X_test, y_test) -> Tuple[dict, pd.Series, pd.Series]:
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_metrics = _calc_metrics(y_train, y_pred_train)
    test_metrics = _calc_metrics(y_test, y_pred_test)

    metrics = {
        "Train MAE": train_metrics["MAE"],
        "Train MSE": train_metrics["MSE"],
        "Train R2": train_metrics["R2"],
        "Test MAE": test_metrics["MAE"],
        "Test MSE": test_metrics["MSE"],
        "Test R2": test_metrics["R2"],
    }

    return (
        metrics,
        pd.Series(y_pred_train, index=y_train.index, name="train_prediction"),
        pd.Series(y_pred_test, index=y_test.index, name="test_prediction"),
    )

def evaluate_all_models(models: Dict[str, object], X_train, y_train, X_test, y_test):
    rows = []
    predictions = {}

    for model_name, model in models.items():
        metrics, y_pred_train, y_pred_test = evaluate_model(model, X_train, y_train, X_test, y_test)

        rows.append(
            {
                "Model": model_name,
                **metrics,
            }
        )

        predictions[model_name] = {
            "train": y_pred_train,
            "test": y_pred_test,
        }

    results_df = pd.DataFrame(rows).sort_values(by="Test R2", ascending=False).reset_index(drop=True)
    return results_df, predictions