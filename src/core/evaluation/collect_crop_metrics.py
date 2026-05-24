from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def collect_metrics_by_crop(input_dir: Path, output_path: Path) -> pd.DataFrame:
    files = sorted(input_dir.rglob("*metrics_by_crop*.csv"))

    if not files:
        raise FileNotFoundError(
            f"В папке {input_dir} не найдено файлов *metrics_by_crop*.csv"
        )

    frames = []

    for path in files:
        df = pd.read_csv(path)
        df["source_file"] = path.name
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)

    preferred_cols = [
        "model",
        "feature_mode",
        "feature_set",
        "forecast_scenario",
        "weather_month_cutoff",
        "split",
        "dataset",
        "seed",
        "crop_id",
        "crop",
        "count",
        "mae",
        "mse",
        "rmse",
        "r2",
        "nmae_percent",
        "nrmse_percent",
        "nmape_percent",
        "target_mean_abs",
        "source_file",
    ]

    main_cols = [c for c in preferred_cols if c in result.columns]
    other_cols = [c for c in result.columns if c not in main_cols]

    result = result[main_cols + other_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Папка, где лежат CSV с metrics_by_crop",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Путь для сохранения общей таблицы",
    )

    args = parser.parse_args()

    result = collect_metrics_by_crop(
        input_dir=args.input_dir,
        output_path=args.output,
    )

    print(f"Найдено строк: {len(result)}")
    print(f"Таблица сохранена: {args.output}")

    visible_cols = [
        c for c in [
            "model",
            "feature_mode",
            "feature_set",
            "split",
            "seed",
            "crop",
            "count",
            "mae",
            "rmse",
            "r2",
            "nmae_percent",
            "nrmse_percent",
            "nmape_percent",
        ]
        if c in result.columns
    ]

    print()
    print(result[visible_cols].to_string(index=False))


if __name__ == "__main__":
    main()