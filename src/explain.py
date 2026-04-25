from __future__ import annotations
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def _extract_transformed_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    return [str(name) for name in feature_names]


def get_linear_regression_coefficients(model_pipeline: Pipeline) -> pd.DataFrame:
    model = model_pipeline.named_steps["model"]
    feature_names = _extract_transformed_feature_names(model_pipeline)

    coef = model.coef_

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coef,
            "abs_coefficient": abs(coef),
        }
    ).sort_values("abs_coefficient", ascending=False)

    return coef_df.reset_index(drop=True)


def get_random_forest_feature_importance(model_pipeline: Pipeline) -> pd.DataFrame:
    model = model_pipeline.named_steps["model"]
    feature_names = _extract_transformed_feature_names(model_pipeline)

    importances = model.feature_importances_

    feature_importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    return feature_importance_df.reset_index(drop=True)


def build_shap_summary_plot(model_pipeline: Pipeline, X_sample: pd.DataFrame, max_display: int = 20):
    transformer = Pipeline(model_pipeline.steps[:-1])
    model = model_pipeline.named_steps["model"]

    X_transformed = transformer.transform(X_sample)
    feature_names = _extract_transformed_feature_names(model_pipeline)

    explainer = shap.TreeExplainer(model)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(X_transformed)

    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    return fig