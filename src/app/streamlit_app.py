import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import basic_info, load_csv
from src.evaluate import evaluate_all_models
from src.train import save_model, train_all_models
from src.visualize import plot_actual_vs_predicted, plot_target_distribution


st.set_page_config(page_title="Crop Yield Forecast", layout="wide")

st.title("Прогнозирование урожайности")
st.write("MVP-приложение для загрузки датасета, обучения моделей и сравнения результатов.")

uploaded_file = st.file_uploader("Загрузи CSV-файл", type=["csv"])

if uploaded_file is not None:
    try:
        df = load_csv(uploaded_file)

        st.subheader("Предпросмотр данных")
        st.dataframe(df.head())

        info = basic_info(df)
        col1, col2, col3 = st.columns(3)
        col1.metric("Количество строк", info["rows"])
        col2.metric("Количество столбцов", info["cols"])
        col3.metric("Всего пропусков", info["missing_total"])

        st.subheader("Колонки")
        st.write(info["columns"])

        target_col = st.selectbox(
            "Выбери целевую колонку",
            options=df.columns.tolist(),
            index=df.columns.tolist().index("Yield_tons_per_hectare")
            if "Yield_tons_per_hectare" in df.columns
            else len(df.columns) - 1,
        )

        st.subheader("Распределение целевой переменной")
        fig_target = plot_target_distribution(df, target_col)
        st.pyplot(fig_target)

        if st.button("Обучить модели"):
            with st.spinner("Обучение моделей..."):
                models, X_train, y_train, X_test, y_test = train_all_models(df, target_col)
                results_df, predictions = evaluate_all_models(models, X_test, y_test)

            st.success("Обучение завершено")

            st.subheader("Результаты моделей")
            st.dataframe(results_df, use_container_width=True)

            best_model_name = results_df.iloc[0]["Model"]
            best_model = models[best_model_name]
            best_pred = predictions[best_model_name]

            st.subheader(f"Лучшая модель: {best_model_name}")
            fig = plot_actual_vs_predicted(
                y_true=y_test.reset_index(drop=True),
                y_pred=best_pred.reset_index(drop=True),
                title=f"Actual vs Predicted — {best_model_name}",
            )
            st.pyplot(fig)

            if st.button("Сохранить лучшую модель"):
                path = save_model(best_model, best_model_name)
                st.success(f"Модель сохранена: {path}")

    except Exception as e:
        st.error(f"Ошибка при обработке файла: {e}")
else:
    st.info("Сначала загрузи CSV-файл.")