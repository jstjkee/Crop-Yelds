from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from src.config import (
    CROP_COL,
    FEATURE_MODES,
    METRICS_DIR,
    RANDOM_STATE,
    SOURCE_DATA_PATH,
    TARGET_COL,
    TEST_SIZE,
    ensure_project_dirs,
)
from src.data.load_data import load_and_validate_source_data, summarize_dataframe
from src.data.preprocess import prepare_dataset
from src.features.transforms import build_feature_view
from src.models.baselines import train_all_baselines
from src.training.train_source import (
    get_device,
    run_transformer_experiments,
)


st.set_page_config(
    page_title="Прогноз урожайности",
    page_icon="🌾",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_source_dataframe(path: str | Path, target_col: str, crop_col: str) -> pd.DataFrame:
    return load_and_validate_source_data(
        path=path,
        target_col=target_col,
        crop_col=crop_col,
    )


@st.cache_data(show_spinner=False)
def build_prepared_dataset(
    path: str | Path,
    target_col: str,
    crop_col: str,
    test_size: float,
    random_state: int,
):
    df = load_source_dataframe(path, target_col, crop_col)
    return prepare_dataset(
        df=df,
        target_col=target_col,
        crop_col=crop_col,
        test_size=test_size,
        random_state=random_state,
    )


def render_dataset_info(df: pd.DataFrame) -> None:
    info = summarize_dataframe(
        df=df,
        target_col=TARGET_COL,
        crop_col=CROP_COL,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Строк", info["n_rows"])
    c2.metric("Колонок", info["n_columns"])
    c3.metric("Культур", info["n_crops"])
    c4.metric("Пропусков", info["n_missing_total"])

    with st.expander("Распределение по культурам", expanded=False):
        crop_counts = pd.DataFrame(
            {
                "crop": list(info["crop_counts"].keys()),
                "count": list(info["crop_counts"].values()),
            }
        )
        st.dataframe(crop_counts, use_container_width=True)

    with st.expander("Первые строки датасета", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)


def render_prepared_info(prepared) -> None:
    st.subheader("Подготовленные данные")

    c1, c2, c3 = st.columns(3)
    c1.metric("X_train shape", str(prepared.X_train.shape))
    c2.metric("X_test shape", str(prepared.X_test.shape))
    c3.metric("Культур в модели", len(prepared.crop_to_id))

    with st.expander("Сопоставление культур", expanded=False):
        crop_map_df = pd.DataFrame(
            {
                "crop": list(prepared.crop_to_id.keys()),
                "crop_id": list(prepared.crop_to_id.values()),
            }
        )
        st.dataframe(crop_map_df, use_container_width=True)

    with st.expander("Колонки признаков", expanded=False):
        st.write("**Числовые признаки**")
        st.write(prepared.numeric_cols if prepared.numeric_cols else "Нет")
        st.write("**Категориальные признаки**")
        st.write(prepared.categorical_cols if prepared.categorical_cols else "Нет")


def render_feature_preview(prepared, selected_mode: str) -> None:
    device = get_device()
    feature_result = build_feature_view(
        mode=selected_mode,
        X_train=prepared.X_train,
        X_test=prepared.X_test,
        device=device,
    )

    st.subheader("Преобразование признаков")

    c1, c2, c3 = st.columns(3)
    c1.metric("Режим", selected_mode)
    c2.metric("Input dim before", prepared.X_train.shape[1])
    c3.metric("Input dim after", feature_result.X_train.shape[1])

    metadata = feature_result.metadata or {}
    if metadata:
        with st.expander("Метаданные преобразования", expanded=True):
            meta_df = pd.DataFrame(
                {"parameter": list(metadata.keys()), "value": list(metadata.values())}
            )
            st.dataframe(meta_df, use_container_width=True)


def render_saved_results() -> None:
    st.subheader("Сохраненные результаты")

    summary_path = METRICS_DIR / "source_metrics_summary.csv"
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        st.write("**Итоговая таблица метрик**")
        st.dataframe(summary_df, use_container_width=True)
    else:
        st.info("Файл source_metrics_summary.csv пока не найден. Сначала запусти обучение.")

    results_dir = Path("results") / "tables"
    if results_dir.exists():
        crop_files = sorted(results_dir.glob("source_metrics_by_crop_*.csv"))
        if crop_files:
            selected_file = st.selectbox(
                "Выбери таблицу метрик по культурам",
                options=[f.name for f in crop_files],
            )
            selected_path = results_dir / selected_file
            crop_df = pd.read_csv(selected_path)
            st.dataframe(crop_df, use_container_width=True)


def run_training_pipeline(prepared) -> None:
    device = get_device()

    with st.spinner("Обучение transformer-моделей..."):
        transformer_rows, transformer_by_crop = run_transformer_experiments(
            prepared=prepared,
            device=device,
        )

    with st.spinner("Обучение baseline-моделей..."):
        baseline_results = train_all_baselines(
            X_train=prepared.X_train,
            y_train=prepared.y_train,
            X_test=prepared.X_test,
            y_test=prepared.y_test,
        )

    baseline_rows = []
    for result in baseline_results:
        baseline_rows.append(
            {
                "model": result.model_name,
                "feature_mode": "raw",
                "split": "test",
                "mae": result.metrics["mae"],
                "mse": result.metrics["mse"],
                "rmse": result.metrics["rmse"],
                "r2": result.metrics["r2"],
            }
        )

    transformer_df = pd.DataFrame(transformer_rows)
    baseline_df = pd.DataFrame(baseline_rows)

    st.success("Обучение завершено")

    st.subheader("Transformer results")
    st.dataframe(transformer_df, use_container_width=True)

    st.subheader("Baseline results")
    st.dataframe(baseline_df, use_container_width=True)

    st.subheader("Transformer metrics by crop")
    available_modes = list(transformer_by_crop.keys())
    selected_mode = st.selectbox(
        "Выбери режим transformer для просмотра по культурам",
        options=available_modes,
        key="transformer_mode_results",
    )
    st.dataframe(transformer_by_crop[selected_mode], use_container_width=True)


def main() -> None:
    ensure_project_dirs()

    st.title("🌾 Прогноз урожайности")
    st.caption("Multi-head Tabular Transformer + PCA / UMAP / Autoencoder")

    with st.sidebar:
        st.header("Параметры")
        st.write(f"Device: **{get_device()}**")
        st.write(f"Source path: `{SOURCE_DATA_PATH}`")
        st.write(f"Target column: `{TARGET_COL}`")
        st.write(f"Crop column: `{CROP_COL}`")
        selected_mode = st.selectbox("Режим признаков", FEATURE_MODES, index=0)
        run_training = st.button("Запустить обучение", type="primary", use_container_width=True)

    try:
        df = load_source_dataframe(SOURCE_DATA_PATH, TARGET_COL, CROP_COL)
    except Exception as exc:
        st.error(f"Ошибка загрузки датасета: {exc}")
        st.stop()

    render_dataset_info(df)

    try:
        prepared = build_prepared_dataset(
            path=SOURCE_DATA_PATH,
            target_col=TARGET_COL,
            crop_col=CROP_COL,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    except Exception as exc:
        st.error(f"Ошибка подготовки датасета: {exc}")
        st.stop()

    render_prepared_info(prepared)
    render_feature_preview(prepared, selected_mode)
    render_saved_results()

    st.divider()

    if run_training:
        try:
            run_training_pipeline(prepared)
        except Exception as exc:
            st.exception(exc)

    st.divider()
    st.caption("Пока интерфейс работает только с исходным датасетом.")


if __name__ == "__main__":
    main()