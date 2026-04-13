from typing import Dict, Tuple

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test) -> Tuple[dict, pd.Series]:
    """
    Считает метрики одной модели.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "MSE": float(mean_squared_error(y_test, y_pred)),
        "R2": float(r2_score(y_test, y_pred)),
    }

    return metrics, pd.Series(y_pred, index=y_test.index, name="prediction")


def evaluate_all_models(models: Dict[str, object], X_test, y_test):
    """
    Возвращает таблицу метрик по всем моделям и предсказания.
    """
    rows = []
    predictions = {}

    for model_name, model in models.items():
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        rows.append(
            {
                "Model": model_name,
                **metrics,
            }
        )
        predictions[model_name] = y_pred

    results_df = pd.DataFrame(rows).sort_values(by="R2", ascending=False).reset_index(drop=True)
    return results_df, predictions