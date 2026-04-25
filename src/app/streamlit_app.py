from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import basic_info, clean_dataset, load_csv
from src.evaluate import evaluate_all_models
from src.explain import (
    build_shap_summary_plot,
    get_linear_regression_coefficients,
    get_random_forest_feature_importance,
)
from src.train import save_model, train_all_models
from src.visualize import (
    build_predictions_dataframe,
    dataframe_to_csv_bytes,
    plot_actual_vs_predicted,
    plot_boxplots,
    plot_correlation_matrix,
    plot_feature_importance,
    plot_linear_coefficients,
    plot_target_distribution,
)

st.set_page_config(page_title="Crop Yield Forecast", layout="wide")

st.title("Прогнозирование урожайности")
st.write("Прототип приложения для загрузки датасета, анализа данных и сравнения моделей.")

uploaded_file = st.file_uploader("Загрузи CSV-файл", type=["csv"])

if uploaded_file is None:
    st.info("Сначала загрузи CSV-файл.")
else:
    try:
        raw_df = load_csv(uploaded_file)

        default_index = (
            raw_df.columns.tolist().index("Yield_tons_per_hectare")
            if "Yield_tons_per_hectare" in raw_df.columns
            else len(raw_df.columns) - 1
        )

        with st.sidebar:
            st.header("Настройки")

            target_col = st.selectbox(
                "Целевая колонка",
                options=raw_df.columns.tolist(),
                index=default_index,
            )

            selected_models = st.multiselect(
                "Модели",
                options=["Linear Regression",
                         "Random Forest",
                         "Extra Trees",
                         "HistGradientBoosting",
                         "KNN Regressor",
                         "SVR"],
                default=["Linear Regression",
                         "Random Forest",
                         "Extra Trees",
                         "HistGradientBoosting"],
            )

            sample_size = st.selectbox(
                "Размер выборки",
                options=[5000, 10000, 20000, 50000, None],
                index=1,
                format_func=lambda x: "Весь датасет" if x is None else str(x),
            )
            compare_pca_modes = st.checkbox("Сравнить PCA и без PCA", value=True)
            use_pca = st.checkbox("Использовать PCA", value=False)

            pca_components = 7
            if use_pca:
                pca_components = st.slider(
                    "Количество компонент PCA",
                    min_value=2,
                    max_value=20,
                    value=7,
                )

            if "SVR" in selected_models and sample_size is not None and sample_size > 20000:
                st.warning("SVR может обучаться очень долго на больших выборках.")

            if "KNN Regressor" in selected_models and sample_size is not None and sample_size > 50000:
                st.warning("KNN может заметно замедляться на больших выборках.")

            run_training = st.button("Обучить модели", key="train_button_sidebar")

        df = clean_dataset(raw_df, target_col)
        info = basic_info(df)

        tab_data, tab_train, tab_explain = st.tabs(
            ["Данные", "Обучение", "Интерпретация"]
        )

        with tab_data:
            st.subheader("Предпросмотр данных")
            st.dataframe(df.head(), width="stretch")

            c1, c2, c3 = st.columns(3)
            c1.metric("Количество строк", info["rows"])
            c2.metric("Количество столбцов", info["cols"])
            c3.metric("Всего пропусков", info["missing_total"])

            with st.expander("Показать список колонок"):
                st.write(info["columns"])

            st.subheader("Базовый анализ данных")

            fig_target = plot_target_distribution(df, target_col)
            st.pyplot(fig_target, width="content")

            numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
            if len(numeric_columns) >= 2:
                fig_corr = plot_correlation_matrix(df)
                st.pyplot(fig_corr, width="content")

            boxplot_candidates = [col for col in numeric_columns if col != target_col][:5]
            if boxplot_candidates:
                fig_box = plot_boxplots(df, boxplot_candidates)
                if fig_box is not None:
                    st.pyplot(fig_box, width="content")

        with tab_train:
            st.subheader("Обучение и результаты")

            st.markdown("### Текущая конфигурация")
            c1, c2, c3 = st.columns(3)
            c1.write(f"**Target:** {target_col}")
            c2.write(f"**Сравнение PCA:** {'Да' if compare_pca_modes else 'Нет'}")
            c3.write(f"**Размер выборки:** {'Весь датасет' if sample_size is None else sample_size}")

            st.write(f"**Модели:** {', '.join(selected_models) if selected_models else 'не выбраны'}")

            if run_training:
                if not selected_models:
                    st.error("Выбери хотя бы одну модель.")
                else:
                    with st.spinner("Обучение моделей..."):
                        all_results = []
                        all_predictions = {}
                        all_models_store = {}

                        experiment_modes = []
                        if compare_pca_modes:
                            experiment_modes = [
                                ("No PCA", False),
                                ("With PCA", True),
                            ]
                        else:
                            experiment_modes = [
                                ("With PCA" if use_pca else "No PCA", use_pca),
                            ]

                        shared_data = None

                        for mode_name, current_use_pca in experiment_modes:
                            models, X_train, y_train, X_test, y_test = train_all_models(
                                df=df,
                                target_col=target_col,
                                selected_models=selected_models,
                                use_pca=current_use_pca,
                                pca_components=pca_components,
                                sample_size=sample_size,
                            )

                            results_df_mode, predictions_mode = evaluate_all_models(
                                models=models,
                                X_train=X_train,
                                y_train=y_train,
                                X_test=X_test,
                                y_test=y_test,
                                mode_name=mode_name,
                            )

                            all_results.append(results_df_mode)
                            all_predictions.update(predictions_mode)

                            for model_name, model in models.items():
                                all_models_store[(mode_name, model_name)] = model

                            if shared_data is None:
                                shared_data = {
                                    "X_train": X_train,
                                    "y_train": y_train,
                                    "X_test": X_test,
                                    "y_test": y_test,
                                }

                        final_results_df = pd.concat(all_results, ignore_index=True)
                        final_results_df = final_results_df.sort_values(by="Test R2", ascending=False).reset_index(
                            drop=True)

                    st.session_state["models"] = all_models_store
                    st.session_state["X_train"] = shared_data["X_train"]
                    st.session_state["y_train"] = shared_data["y_train"]
                    st.session_state["X_test"] = shared_data["X_test"]
                    st.session_state["y_test"] = shared_data["y_test"]
                    st.session_state["results_df"] = final_results_df
                    st.session_state["predictions"] = all_predictions
                    st.session_state["target_col"] = target_col

                    st.success("Обучение завершено")

            if "results_df" in st.session_state:
                results_df = st.session_state["results_df"]
                models_store = st.session_state["models"]
                y_test = st.session_state["y_test"]
                predictions = st.session_state["predictions"]

                if not results_df.empty:
                    best_row = results_df.iloc[0]
                    best_mode = best_row["Mode"]
                    best_model_name = best_row["Model"]
                    best_model = models_store[(best_mode, best_model_name)]
                    best_pred_test = predictions[(best_mode, best_model_name)]["test"]

                    st.markdown("### Лучшая конфигурация")

                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Режим", best_mode)
                    s2.metric("Модель", best_model_name)
                    s3.metric("Test R²", f"{best_row['Test R2']:.4f}")
                    s4.metric("Test MAE", f"{best_row['Test MAE']:.4f}")

                    st.info(
                        f"На текущем запуске лучшей конфигурацией стала **{best_model_name}** "
                        f"в режиме **{best_mode}**. "
                        f"Она показала Test R² = **{best_row['Test R2']:.4f}**, "
                        f"Test MAE = **{best_row['Test MAE']:.4f}** и "
                        f"Test MSE = **{best_row['Test MSE']:.4f}**."
                    )

                    st.subheader("Таблица результатов")
                    st.dataframe(results_df, width="stretch")

                    metrics_csv = dataframe_to_csv_bytes(results_df)
                    st.download_button(
                        label="Скачать метрики в CSV",
                        data=metrics_csv,
                        file_name="model_metrics_comparison.csv",
                        mime="text/csv",
                    )

                    predictions_df = build_predictions_dataframe(
                        y_true=y_test,
                        y_pred=best_pred_test,
                    )

                    st.subheader(f"График: {best_model_name} ({best_mode})")
                    fig = plot_actual_vs_predicted(
                        y_true=y_test.reset_index(drop=True),
                        y_pred=best_pred_test.reset_index(drop=True),
                        title=f"Actual vs Predicted — {best_model_name} ({best_mode})",
                    )
                    st.pyplot(fig, width="content")

                    st.subheader("Предсказания лучшей конфигурации")
                    st.dataframe(predictions_df.head(50), width="stretch")

                    predictions_csv = dataframe_to_csv_bytes(predictions_df)
                    st.download_button(
                        label="Скачать предсказания в CSV",
                        data=predictions_csv,
                        file_name="best_configuration_predictions.csv",
                        mime="text/csv",
                    )

                    if st.button("Сохранить лучшую модель", key="save_model_button"):
                        path = save_model(best_model, f"{best_mode}_{best_model_name}")
                        st.success(f"Модель сохранена: {path}")
                else:
                    st.warning("После обучения таблица результатов оказалась пустой.")
            else:
                st.info("Нажми «Обучить модели» в боковой панели.")

        with tab_explain:
            st.subheader("Интерпретация модели")

            if "results_df" not in st.session_state:
                st.info("Сначала обучи модели.")
            else:
                results_df = st.session_state["results_df"]
                models = st.session_state["models"]
                X_test = st.session_state["X_test"]

                if results_df.empty:
                    st.warning("Нет результатов для интерпретации.")
                else:
                    best_row = results_df.iloc[0]
                    best_mode = best_row["Mode"]
                    best_model_name = best_row["Model"]
                    best_model = models[(best_mode, best_model_name)]

                    st.write(f"Текущая лучшая конфигурация: **{best_model_name} ({best_mode})**")

                    if best_model_name == "Linear Regression":
                        st.markdown(
                            "Для линейной регрессии интерпретация выполняется через анализ коэффициентов признаков."
                        )
                        coef_df = get_linear_regression_coefficients(best_model)
                        st.dataframe(coef_df.head(20), width="stretch")

                        fig_coef = plot_linear_coefficients(coef_df, top_n=20)
                        st.pyplot(fig_coef, width="content")

                    elif best_model_name == "Random Forest":
                        st.markdown(
                            "Для Random Forest интерпретация выполняется через feature importance и SHAP."
                        )

                        fi_df = get_random_forest_feature_importance(best_model)
                        st.dataframe(fi_df.head(20), width="stretch")

                        fig_fi = plot_feature_importance(fi_df, top_n=20)
                        st.pyplot(fig_fi, width="content")

                        shap_sample_size = min(len(X_test), 1000)
                        X_shap = X_test.sample(n=shap_sample_size, random_state=42)

                        with st.spinner("Строится SHAP summary plot..."):
                            shap_fig = build_shap_summary_plot(best_model, X_shap, max_display=20)
                        st.pyplot(shap_fig, width="stretch")

                    else:
                        st.info(
                            "Для текущей лучшей модели отдельный блок интерпретации пока не реализован."
                        )

    except Exception as e:
        st.error(f"Ошибка при обработке файла: {e}")