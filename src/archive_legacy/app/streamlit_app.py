from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.config import (
    CROP_COL,
    DATASET_PATHS,
    FEATURE_MODES,
    METRICS_DIR,
    TARGET_COL,
    ensure_project_dirs,
)
from src.data.load_data import load_and_validate_source_data, summarize_dataframe
from src.data.preprocess import prepare_dataset
from src.data.russian_parser import prepare_russian_real_dataset
from src.features.transforms import build_feature_view
from src.training.train_russia_only import main as run_russia_real_training
from src.training.train_source import get_device, main as run_source_training, run_training_pipeline
from src.training.train_source import seed_everything
from src.config import RANDOM_STATE

st.set_page_config(
    page_title="Прогноз урожайности",
    page_icon="🌾",
    layout="wide",
)

DATASET_OPTIONS = {
    "source": {
        "label": "Исходный датасет",
        "path": DATASET_PATHS["source"],
        "results_prefix": "source",
    },
    "russian_legacy": {
        "label": "Россия exact-match",
        "path": DATASET_PATHS["russian_legacy"],
        "results_prefix": "russian_legacy",
    },
    "russian_real": {
        "label": "Россия реальные признаки (v4)",
        "path": DATASET_PATHS["russian_real_project"],
        "results_prefix": "russia_real_v4",
    },
}


@st.cache_data(show_spinner=False)
def load_dataframe(path: str | Path, target_col: str, crop_col: str) -> pd.DataFrame:
    return load_and_validate_source_data(path=path, target_col=target_col, crop_col=crop_col)


@st.cache_data(show_spinner=False)
def build_prepared_dataset(path: str | Path, target_col: str, crop_col: str):
    df = load_dataframe(path, target_col, crop_col)
    return prepare_dataset(df=df, target_col=target_col, crop_col=crop_col)


def resolve_dataset_path(dataset_key: str) -> Path:
    if dataset_key == "russian_real":
        prepare_russian_real_dataset(force=False)
    return Path(DATASET_OPTIONS[dataset_key]["path"])


def render_dataset_info(df: pd.DataFrame) -> None:
    info = summarize_dataframe(df=df, target_col=TARGET_COL, crop_col=CROP_COL)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Строк", info["n_rows"])
    c2.metric("Колонок", info["n_columns"])
    c3.metric("Культур", info["n_crops"])
    c4.metric("Пропусков", info["n_missing_total"])

    with st.expander("Распределение по культурам", expanded=False):
        crop_counts = pd.DataFrame({"crop": list(info["crop_counts"].keys()), "count": list(info["crop_counts"].values())})
        st.dataframe(crop_counts, use_container_width=True)

    with st.expander("Первые строки датасета", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)


def render_prepared_info(prepared) -> None:
    st.subheader("Подготовленные данные")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("X_train", str(prepared.X_train.shape))
    c2.metric("X_val", str(prepared.X_val.shape))
    c3.metric("X_test", str(prepared.X_test.shape))
    c4.metric("Культур", len(prepared.crop_to_id))

    with st.expander("Сопоставление культур", expanded=False):
        crop_map_df = pd.DataFrame({"crop": list(prepared.crop_to_id.keys()), "crop_id": list(prepared.crop_to_id.values())})
        st.dataframe(crop_map_df, use_container_width=True)

    with st.expander("Колонки признаков", expanded=False):
        st.write("**Числовые признаки**")
        st.write(prepared.numeric_cols if prepared.numeric_cols else "Нет")
        st.write("**Категориальные признаки**")
        st.write(prepared.categorical_cols if prepared.categorical_cols else "Нет")


def render_feature_preview(prepared, selected_mode: str) -> None:
    feature_result = build_feature_view(
        mode=selected_mode,
        X_train=prepared.X_train,
        X_val=prepared.X_val,
        X_test=prepared.X_test,
        device=get_device(),
    )

    st.subheader("Преобразование признаков")
    c1, c2, c3 = st.columns(3)
    c1.metric("Режим", selected_mode)
    c2.metric("Input dim before", prepared.X_train.shape[1])
    c3.metric("Input dim after", feature_result.X_train.shape[1])

    metadata = feature_result.metadata or {}
    if metadata:
        with st.expander("Метаданные преобразования", expanded=True):
            meta_df = pd.DataFrame({"parameter": list(metadata.keys()), "value": list(metadata.values())})
            st.dataframe(meta_df, use_container_width=True)


def render_saved_results(results_prefix: str) -> None:
    st.subheader("Сохраненные результаты")
    summary_path = METRICS_DIR / f"{results_prefix}_metrics_summary.csv"
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        st.write("**Итоговая таблица метрик**")
        st.dataframe(summary_df, use_container_width=True)
    else:
        st.info("Файл с метриками пока не найден. Сначала запусти обучение.")

    tables_dir = Path("results") / "tables"
    if tables_dir.exists():
        crop_files = sorted(tables_dir.glob(f"{results_prefix}_metrics_by_crop_*.csv"))
        if crop_files:
            selected_file = st.selectbox(
                "Выбери таблицу метрик по культурам",
                options=[f.name for f in crop_files],
            )
            selected_path = tables_dir / selected_file
            crop_df = pd.read_csv(selected_path)
            st.dataframe(crop_df, use_container_width=True)


def run_selected_training(dataset_key: str, dataset_path: Path) -> None:
    if dataset_key == "source":
        run_source_training()
        return

    if dataset_key == "russian_real":
        run_russia_real_training()
        return

    seed_everything(RANDOM_STATE)
    df = load_dataframe(dataset_path, TARGET_COL, CROP_COL)
    run_training_pipeline(
        df=df,
        dataset_label=dataset_key,
        target_col=TARGET_COL,
        crop_col=CROP_COL,
        results_prefix="russian_legacy",
        device=get_device(),
    )


def main() -> None:
    ensure_project_dirs()

    st.title("🌾 Прогноз урожайности")
    st.caption("Multi-head Tabular Transformer + raw / PCA / UMAP / Autoencoder")

    with st.sidebar:
        st.header("Параметры")
        st.write(f"Device: **{get_device()}**")
        dataset_key = st.selectbox(
            "Датасет",
            options=list(DATASET_OPTIONS.keys()),
            format_func=lambda key: DATASET_OPTIONS[key]["label"],
            index=0,
        )
        dataset_path = resolve_dataset_path(dataset_key)
        st.write(f"Path: `{dataset_path}`")
        st.write(f"Target column: `{TARGET_COL}`")
        st.write(f"Crop column: `{CROP_COL}`")
        selected_mode = st.selectbox("Режим признаков", FEATURE_MODES, index=0)
        run_training = st.button("Запустить обучение", type="primary", use_container_width=True)

    try:
        df = load_dataframe(dataset_path, TARGET_COL, CROP_COL)
    except Exception as exc:
        st.error(f"Ошибка загрузки датасета: {exc}")
        st.stop()

    render_dataset_info(df)

    try:
        prepared = build_prepared_dataset(dataset_path, TARGET_COL, CROP_COL)
    except Exception as exc:
        st.error(f"Ошибка подготовки датасета: {exc}")
        st.stop()

    render_prepared_info(prepared)
    render_feature_preview(prepared, selected_mode)
    render_saved_results(DATASET_OPTIONS[dataset_key]["results_prefix"])

    st.divider()

    if run_training:
        try:
            with st.spinner("Обучение запущено. Это может занять время..."):
                run_selected_training(dataset_key, dataset_path)
            st.success("Обучение завершено")
            st.rerun()
        except Exception as exc:
            st.exception(exc)

    st.divider()
    st.caption("Теперь интерфейс поддерживает исходный датасет, russian legacy exact-match и russian real v4.")


if __name__ == "__main__":
    main()