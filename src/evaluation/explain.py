from __future__ import annotations
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

def _extract_feature_names_for_model(pipeline: Pipeline) -> list[str]:
    model = pipeline.named_steps["model"]

    if "pca" in pipeline.named_steps:
        n_components = getattr(model, "n_features_in_", None)
        if n_components is None:
            pca = pipeline.named_steps["pca"]
            n_components = pca.n_components_
        return [f"PC{i + 1}" for i in range(int(n_components))]

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    return [str(name) for name in feature_names]

def get_linear_regression_coefficients(model_pipeline: Pipeline) -> pd.DataFrame:
    model = model_pipeline.named_steps["model"]
    feature_names = _extract_feature_names_for_model(model_pipeline)

    coef = model.coef_

    min_len = min(len(feature_names), len(coef))
    feature_names = feature_names[:min_len]
    coef = coef[:min_len]

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coef,
        }
    )
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    return coef_df


def get_random_forest_feature_importance(model_pipeline: Pipeline) -> pd.DataFrame:
    model = model_pipeline.named_steps["model"]
    feature_names = _extract_feature_names_for_model(model_pipeline)

    importances = model.feature_importances_

    min_len = min(len(feature_names), len(importances))
    feature_names = feature_names[:min_len]
    importances = importances[:min_len]

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
    feature_names = _extract_feature_names_for_model(model_pipeline)

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