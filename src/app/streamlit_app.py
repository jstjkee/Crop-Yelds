from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import basic_info, clean_dataset, load_csv
from src.evaluate import evaluate_all_models
from src.train import save_model, train_all_models
from src.visualize import (
    dataframe_to_csv_bytes,
    plot_actual_vs_predicted,
    plot_boxplots,
    plot_correlation_matrix,
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
        df = load_csv(uploaded_file)

        default_index = (
            df.columns.tolist().index("Yield_tons_per_hectare")
            if "Yield_tons_per_hectare" in df.columns
            else len(df.columns) - 1
        )

        target_col = st.selectbox(
            "Выбери целевую колонку",
            options=df.columns.tolist(),
            index=default_index,
        )

        df = clean_dataset(df, target_col)
        info = basic_info(df)

        st.subheader("Предпросмотр данных")
        st.dataframe(df.head())

        c1, c2, c3 = st.columns(3)
        c1.metric("Количество строк", info["rows"])
        c2.metric("Количество столбцов", info["cols"])
        c3.metric("Всего пропусков", info["missing_total"])

        with st.expander("Показать список колонок"):
            st.write(info["columns"])

        st.subheader("Базовый анализ данных")

        fig_target = plot_target_distribution(df, target_col)
        st.pyplot(fig_target)

        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_columns) >= 2:
            fig_corr = plot_correlation_matrix(df)
            st.pyplot(fig_corr)

        boxplot_candidates = [col for col in numeric_columns if col != target_col][:5]
        if boxplot_candidates:
            fig_box = plot_boxplots(df, boxplot_candidates)
            if fig_box is not None:
                st.pyplot(fig_box)

        st.subheader("Настройки обучения")

        selected_models = st.multiselect(
            "Выбери модели",
            options=["Linear Regression", "Random Forest", "KNN Regressor", "SVR"],
            default=["Linear Regression", "Random Forest"],
        )

        sample_size = st.selectbox(
            "Размер выборки для обучения",
            options=[5000, 10000, 20000, 50000, None],
            index=1,
            format_func=lambda x: "Весь датасет" if x is None else str(x),
        )

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

        run_training = st.button("Обучить модели")

        if run_training:
            if not selected_models:
                st.error("Выбери хотя бы одну модель.")
            else:
                with st.spinner("Обучение моделей..."):
                    models, X_train, y_train, X_test, y_test = train_all_models(
                        df=df,
                        target_col=target_col,
                        selected_models=selected_models,
                        use_pca=use_pca,
                        pca_components=pca_components,
                        sample_size=sample_size,
                    )

                    results_df, predictions = evaluate_all_models(
                        models=models,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                    )

                st.success("Обучение завершено")

                st.subheader("Результаты моделей")
                st.dataframe(results_df, use_container_width=True)

                csv_bytes = dataframe_to_csv_bytes(results_df)
                st.download_button(
                    label="Скачать метрики в CSV",
                    data=csv_bytes,
                    file_name="model_metrics.csv",
                    mime="text/csv",
                )

                if not results_df.empty:
                    best_model_name = results_df.iloc[0]["Model"]
                    best_model = models[best_model_name]
                    best_pred_test = predictions[best_model_name]["test"]

                    st.subheader(f"Лучшая модель: {best_model_name}")
                    fig = plot_actual_vs_predicted(
                        y_true=y_test.reset_index(drop=True),
                        y_pred=best_pred_test.reset_index(drop=True),
                        title=f"Actual vs Predicted — {best_model_name}",
                    )
                    st.pyplot(fig)

                    if st.button("Сохранить лучшую модель"):
                        path = save_model(best_model, best_model_name)
                        st.success(f"Модель сохранена: {path}")

    except Exception as e:
        st.error(f"Ошибка при обработке файла: {e}")