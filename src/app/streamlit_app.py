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
from src.train import train_all_models, save_model
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


def find_demo_csv() -> Path | None:
    raw_dir = PROJECT_ROOT / "data" / "raw"
    if not raw_dir.exists():
        return None

    csv_files = sorted(raw_dir.glob("*.csv"))
    return csv_files[0] if csv_files else None


def build_experiment_conclusion(results_df: pd.DataFrame) -> list[str]:
    if results_df.empty:
        return ["Результаты эксперимента отсутствуют."]

    best_row = results_df.iloc[0]
    best_mode = best_row["Mode"]
    best_model = best_row["Model"]

    conclusions = [
        f"Лучшая конфигурация: **{best_model}** в режиме **{best_mode}**.",
        (
            f"На тестовой выборке модель показала "
            f"**R² = {best_row['Test R2']:.4f}**, "
            f"**MAE = {best_row['Test MAE']:.4f}**, "
            f"**RMSE = {best_row['Test RMSE']:.4f}**, "
            f"**MSE = {best_row['Test MSE']:.4f}**."
        ),
    ]

    no_pca_df = results_df[results_df["Mode"] == "No PCA"]
    with_pca_df = results_df[results_df["Mode"] == "With PCA"]

    if not no_pca_df.empty and not with_pca_df.empty:
        best_no_pca = no_pca_df.iloc[0]
        best_with_pca = with_pca_df.iloc[0]

        if best_with_pca["Test R2"] > best_no_pca["Test R2"]:
            conclusions.append(
                "В текущем запуске режим **With PCA** дал лучший максимальный результат по Test R², чем режим **No PCA**."
            )
        elif best_with_pca["Test R2"] < best_no_pca["Test R2"]:
            conclusions.append(
                "В текущем запуске режим **No PCA** дал лучший максимальный результат по Test R², чем режим **With PCA**."
            )
        else:
            conclusions.append(
                "В текущем запуске лучшие результаты режимов **With PCA** и **No PCA** по Test R² оказались очень близкими."
            )

    overfit_candidates = results_df.copy()
    overfit_candidates["R2_gap"] = overfit_candidates["Train R2"] - overfit_candidates["Test R2"]
    worst_gap_row = overfit_candidates.sort_values("R2_gap", ascending=False).iloc[0]

    if worst_gap_row["R2_gap"] > 0.08:
        conclusions.append(
            f"У конфигурации **{worst_gap_row['Model']} ({worst_gap_row['Mode']})** заметен разрыв между Train R² и Test R², что может указывать на переобучение."
        )
    else:
        conclusions.append(
            "Сильного разрыва между train- и test-метриками на текущем запуске не наблюдается."
        )

    return conclusions


def style_results_table(df: pd.DataFrame):
    return df.style.format(
        {
            col: "{:.4f}"
            for col in [
                "Train MAE",
                "Train MSE",
                "Train RMSE",
                "Train R2",
                "Test MAE",
                "Test MSE",
                "Test RMSE",
                "Test R2",
            ]
            if col in df.columns
        }
    )


def filter_results_for_display(
    results_df: pd.DataFrame,
    selected_mode_filter: str,
    sort_by: str,
    top_n: int,
) -> pd.DataFrame:
    result = results_df.copy()

    if selected_mode_filter != "Все":
        result = result[result["Mode"] == selected_mode_filter]

    ascending = sort_by in ["Test MAE", "Test MSE"]
    result = result.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)

    if top_n is not None and top_n > 0:
        result = result.head(top_n)

    return result


def load_demo_dataset_if_requested() -> pd.DataFrame | None:
    demo_path = find_demo_csv()
    if demo_path is None:
        return None
    return pd.read_csv(demo_path)

st.title("Прогнозирование урожайности")
st.write(
    "Аналитическое приложение для загрузки агроданных, сравнения моделей машинного обучения, "
    "оценки влияния PCA и интерпретации результатов."
)

uploaded_file = st.file_uploader("Загрузи CSV-файл", type=["csv"])

demo_df = None
demo_path = find_demo_csv()

with st.sidebar:
    st.header("Источник данных")

    use_demo = False
    if demo_path is not None:
        st.caption(f"Найден demo-датасет: `{demo_path.name}`")
        use_demo = st.checkbox("Использовать demo-датасет", value=False)
    else:
        st.caption("Demo-датасет не найден в папке `data/raw`.")

if uploaded_file is None and not use_demo:
    st.info("Загрузи CSV-файл или включи demo-датасет в боковой панели.")

    with st.container(border=True):
        st.subheader("Что умеет приложение")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
1. Загружать пользовательский CSV-датасет  
2. Выполнять базовый анализ данных  
3. Обучать несколько моделей регрессии  
4. Сравнивать режимы **PCA / No PCA**
                """
            )
        with col2:
            st.markdown(
                """
5. Показывать метрики качества  
6. Сохранять лучшую модель  
7. Экспортировать метрики и предсказания  
8. Интерпретировать результаты модели
                """
            )

    with st.container(border=True):
        st.subheader("Как пользоваться")
        st.markdown(
            """
**Шаг 1.** Загрузи CSV или включи demo-датасет  
**Шаг 2.** Выбери целевую колонку и модели  
**Шаг 3.** Настрой PCA и размер выборки  
**Шаг 4.** Запусти обучение и изучи результаты
            """
        )

else:
    try:
        if use_demo and demo_path is not None:
            raw_df = pd.read_csv(demo_path)
            st.success(f"Загружен demo-датасет: {demo_path.name}")
        else:
            raw_df = load_csv(uploaded_file)

        default_index = (
            raw_df.columns.tolist().index("Yield_tons_per_hectare")
            if "Yield_tons_per_hectare" in raw_df.columns
            else len(raw_df.columns) - 1
        )

        with st.sidebar:
            st.header("Данные")
            target_col = st.selectbox(
                "Целевая колонка",
                options=raw_df.columns.tolist(),
                index=default_index,
            )

            st.header("Модели")
            selected_models = st.multiselect(
                "Выбор моделей",
                options=[
                    "Linear Regression",
                    "Random Forest",
                    "Extra Trees",
                    "HistGradientBoosting",
                    "KNN Regressor",
                    "SVR",
                ],
                default=[
                    "Linear Regression",
                    "Random Forest",
                    "Extra Trees",
                    "HistGradientBoosting",
                ],
            )

            sample_size = st.selectbox(
                "Размер выборки",
                options=[5000, 10000, 20000, 50000, None],
                index=1,
                format_func=lambda x: "Весь датасет" if x is None else str(x),
            )

            st.header("Преобразования")
            compare_pca_modes = st.checkbox("Сравнить PCA и без PCA", value=True)
            use_pca = st.checkbox("Использовать PCA", value=False, disabled=compare_pca_modes)

            pca_components = 7
            if use_pca or compare_pca_modes:
                pca_components = st.slider(
                    "Количество компонент PCA",
                    min_value=2,
                    max_value=20,
                    value=7,
                )

            st.header("Запуск")
            if "SVR" in selected_models and sample_size is not None and sample_size > 20000:
                st.warning("SVR может обучаться очень долго на больших выборках.")

            if "KNN Regressor" in selected_models and sample_size is not None and sample_size > 50000:
                st.warning("KNN может заметно замедляться на больших выборках.")

            run_training = st.button("Обучить модели", key="train_button_sidebar")

        df = clean_dataset(raw_df, target_col)
        info = basic_info(df)

        tab_data, tab_train, tab_explain = st.tabs(["Данные", "Обучение", "Интерпретация"])

        with tab_data:
            left_col, right_col = st.columns([1.2, 1])

            with left_col:
                st.subheader("Предпросмотр данных")
                st.dataframe(df.head(), width="stretch")

                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                metrics_col1.metric("Количество строк", info["rows"])
                metrics_col2.metric("Количество столбцов", info["cols"])
                metrics_col3.metric("Всего пропусков", info["missing_total"])

                with st.expander("Список колонок"):
                    st.write(info["columns"])

            with right_col:
                st.subheader("Распределение целевой переменной")
                fig_target = plot_target_distribution(df, target_col)
                st.pyplot(fig_target, width="content")

            st.subheader("Дополнительный анализ")
            extra_left, extra_right = st.columns(2)

            numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

            with extra_left:
                if len(numeric_columns) >= 2:
                    fig_corr = plot_correlation_matrix(df)
                    st.pyplot(fig_corr, width="content")

            with extra_right:
                boxplot_candidates = [col for col in numeric_columns if col != target_col][:5]
                if boxplot_candidates:
                    fig_box = plot_boxplots(df, boxplot_candidates)
                    if fig_box is not None:
                        st.pyplot(fig_box, width="content")

        with tab_train:
            st.subheader("Обучение и результаты")

            config_left, config_right = st.columns(2)

            with config_left:
                with st.container(border=True):
                    st.markdown("### Текущая конфигурация")
                    st.write(f"**Целевая колонка:** {target_col}")
                    st.write(
                        f"**Размер выборки:** {'Весь датасет' if sample_size is None else sample_size}"
                    )
                    st.write(
                        f"**Сравнение PCA / No PCA:** {'Да' if compare_pca_modes else 'Нет'}"
                    )
                    if not compare_pca_modes:
                        st.write(f"**PCA:** {'Да' if use_pca else 'Нет'}")

            with config_right:
                with st.container(border=True):
                    st.markdown("### Выбранные модели")
                    if selected_models:
                        for model_name in selected_models:
                            st.write(f"- {model_name}")
                    else:
                        st.write("Модели не выбраны.")

            if run_training:
                if not selected_models:
                    st.error("Выбери хотя бы одну модель.")
                else:
                    progress_placeholder = st.empty()
                    status_placeholder = st.empty()

                    all_results = []
                    all_predictions = {}
                    all_models_store = {}
                    shared_data = None

                    experiment_modes = (
                        [("No PCA", False), ("With PCA", True)]
                        if compare_pca_modes
                        else [("With PCA" if use_pca else "No PCA", use_pca)]
                    )

                    total_steps = len(experiment_modes) * len(selected_models)
                    current_step = 0

                    for mode_name, current_use_pca in experiment_modes:
                        status_placeholder.info(f"Подготовка режима: {mode_name}")

                        models, X_train, y_train, X_test, y_test = train_all_models(
                            df=df,
                            target_col=target_col,
                            selected_models=selected_models,
                            use_pca=current_use_pca,
                            pca_components=pca_components,
                            sample_size=sample_size,
                        )

                        rows_mode = []
                        preds_mode = {}

                        for model_name, model in models.items():
                            current_step += 1
                            progress_placeholder.progress(current_step / total_steps)
                            status_placeholder.info(f"Обрабатывается {model_name} ({mode_name})")

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
                    final_results_df = final_results_df.sort_values(
                        by="Test R2", ascending=False
                    ).reset_index(drop=True)

                    st.session_state["models"] = all_models_store
                    st.session_state["X_train"] = shared_data["X_train"]
                    st.session_state["y_train"] = shared_data["y_train"]
                    st.session_state["X_test"] = shared_data["X_test"]
                    st.session_state["y_test"] = shared_data["y_test"]
                    st.session_state["results_df"] = final_results_df
                    st.session_state["predictions"] = all_predictions
                    st.session_state["target_col"] = target_col

                    progress_placeholder.empty()
                    status_placeholder.success("Обучение и оценка завершены.")

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

                    st.markdown("### Итог эксперимента")

                    summary_col1, summary_col2, summary_col3, summary_col4, summary_col5, summary_col6 = st.columns(6)
                    summary_col1.metric("Режим", best_mode)
                    summary_col2.metric("Модель", best_model_name)
                    summary_col3.metric("Test R²", f"{best_row['Test R2']:.4f}")
                    summary_col4.metric("Test MAE", f"{best_row['Test MAE']:.4f}")
                    summary_col5.metric("Test RMSE", f"{best_row['Test RMSE']:.4f}")
                    summary_col6.metric("Test MSE", f"{best_row['Test MSE']:.4f}")

                    with st.container(border=True):
                        st.markdown("### Краткие выводы")
                        for line in build_experiment_conclusion(results_df):
                            st.write(f"- {line}")

                    display_col1, display_col2 = st.columns([1.1, 1])

                    with display_col1:
                        st.subheader("Результаты сравнения")

                        filter_col1, filter_col2, filter_col3 = st.columns(3)
                        selected_mode_filter = filter_col1.selectbox(
                            "Фильтр по режиму",
                            options=["Все", "No PCA", "With PCA"],
                            index=0,
                        )
                        sort_by = filter_col2.selectbox(
                            "Сортировка",
                            options=["Test R2", "Test MAE", "Test MSE"],
                            index=0,
                        )
                        top_n = filter_col3.selectbox(
                            "Показать top-N",
                            options=[3, 5, 10],
                            index=1,
                        )

                        display_results_df = filter_results_for_display(
                            results_df=results_df,
                            selected_mode_filter=selected_mode_filter,
                            sort_by=sort_by,
                            top_n=top_n,
                        )

                        st.dataframe(style_results_table(display_results_df), width="stretch")

                        with st.expander("Полная таблица результатов"):
                            st.dataframe(style_results_table(results_df), width="stretch")

                        metrics_csv = dataframe_to_csv_bytes(results_df)
                        st.download_button(
                            label="Скачать метрики в CSV",
                            data=metrics_csv,
                            file_name="model_metrics_comparison.csv",
                            mime="text/csv",
                        )

                    with display_col2:
                        st.subheader("График лучшей конфигурации")
                        best_true_test = predictions[(best_mode, best_model_name)]["y_test"]

                        fig = plot_actual_vs_predicted(
                            y_true=best_true_test.reset_index(drop=True),
                            y_pred=best_pred_test.reset_index(drop=True),
                            title=f"Actual vs Predicted — {best_model_name} ({best_mode})",
                        )
                        st.pyplot(fig, width="content")

                        if st.button("Сохранить лучшую модель", key="save_model_button"):
                            path = save_model(best_model, f"{best_mode}_{best_model_name}")
                            st.success(f"Модель сохранена: {path}")

                    best_true_test = predictions[(best_mode, best_model_name)]["y_test"]

                    predictions_df = build_predictions_dataframe(
                        y_true=best_true_test,
                        y_pred=best_pred_test.reset_index(drop=True),
                    )

                    with st.expander("Предсказания лучшей конфигурации"):
                        st.dataframe(predictions_df.head(50), width="stretch")

                    predictions_csv = dataframe_to_csv_bytes(predictions_df)
                    st.download_button(
                        label="Скачать предсказания в CSV",
                        data=predictions_csv,
                        file_name="best_configuration_predictions.csv",
                        mime="text/csv",
                    )
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

                    st.write(
                        f"Текущая лучшая конфигурация: **{best_model_name} ({best_mode})**"
                    )

                    explain_left, explain_right = st.columns([1, 1])

                    if best_model_name == "Linear Regression":
                        st.markdown(
                            "Для линейной регрессии интерпретация выполняется через анализ коэффициентов признаков."
                        )

                        coef_df = get_linear_regression_coefficients(best_model)

                        with explain_left:
                            with st.expander("Таблица коэффициентов", expanded=True):
                                st.dataframe(coef_df.head(20), width="stretch")

                        with explain_right:
                            fig_coef = plot_linear_coefficients(coef_df, top_n=20)
                            st.pyplot(fig_coef, width="content")

                    elif best_model_name == "Random Forest":
                        st.markdown(
                            "Для Random Forest интерпретация выполняется через feature importance и SHAP."
                        )

                        fi_df = get_random_forest_feature_importance(best_model)

                        with explain_left:
                            with st.expander("Таблица важности признаков", expanded=True):
                                st.dataframe(fi_df.head(20), width="stretch")

                        with explain_right:
                            fig_fi = plot_feature_importance(fi_df, top_n=20)
                            st.pyplot(fig_fi, width="content")

                        st.subheader("SHAP summary plot")
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