from pathlib import Path
import time
import re

import pandas as pd
import requests
from tqdm import tqdm


REGIONS_PATH = Path("data/reference/regions.csv")
RAW_WEATHER_DIR = Path("data/raw/weather")

START_YEAR = 2000
END_YEAR = 2025

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def safe_filename(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^\wа-яА-ЯёЁ]+", "_", text)
    return text.strip("_")


def download_region_year(region, lat, lon, year, max_retries=5):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "daily": ",".join([
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "rain_sum",
            "relative_humidity_2m_mean",
        ]),
        "timezone": "Europe/Moscow",
    }

    for attempt in range(1, max_retries + 1):
        response = requests.get(
            OPEN_METEO_URL,
            params=params,
            timeout=60,
        )

        if response.status_code == 429:
            wait_seconds = 5 * attempt
            print(
                f"429 Too Many Requests: ждем {wait_seconds} сек. "
                f"Попытка {attempt}/{max_retries}",
                flush=True,
            )
            time.sleep(wait_seconds)
            continue

        response.raise_for_status()

        data = response.json()

        if "daily" not in data:
            raise ValueError(f"Нет daily данных: {data}")

        df = pd.DataFrame(data["daily"])
        df["region"] = region
        df["year"] = year
        df["lat"] = lat
        df["lon"] = lon

        return df

    raise RuntimeError(f"Не удалось скачать после {max_retries} попыток: {region}, {year}")


def main():
    print("СТАРТ weather_download.py", flush=True)

    RAW_WEATHER_DIR.mkdir(parents=True, exist_ok=True)

    regions = pd.read_csv(REGIONS_PATH)

    regions = regions.dropna(subset=["lat", "lon"])

    print("Регионов:", len(regions), flush=True)

    total_expected = len(regions) * (END_YEAR - START_YEAR + 1)

    existing_files = list(RAW_WEATHER_DIR.glob("*.csv"))

    print("Уже скачано файлов:", len(existing_files), flush=True)
    print("Ожидается файлов:", total_expected, flush=True)

    downloaded = 0
    skipped = 0
    errors = 0

    for _, row in tqdm(regions.iterrows(), total=len(regions)):
        region = row["region"]
        lat = float(row["lat"])
        lon = float(row["lon"])

        safe_region = safe_filename(region)

        for year in range(START_YEAR, END_YEAR + 1):
            output_path = RAW_WEATHER_DIR / f"{safe_region}_{year}.csv"

            # Уже скачано
            if output_path.exists():
                skipped += 1
                continue

            try:
                print(f"Скачиваю: {region} {year}", flush=True)

                df = download_region_year(
                    region=region,
                    lat=lat,
                    lon=lon,
                    year=year,
                )

                df.to_csv(
                    output_path,
                    index=False,
                    encoding="utf-8-sig",
                )

                downloaded += 1

                print(
                    f"OK: {output_path.name} ({downloaded} новых)",
                    flush=True,
                )

                time.sleep(0.7)

            except Exception as e:
                errors += 1

                print(
                    f"ОШИБКА: {region} {year}: {repr(e)}",
                    flush=True,
                )

    print("\n===================")
    print("ГОТОВО")
    print("Новых файлов:", downloaded)
    print("Пропущено:", skipped)
    print("Ошибок:", errors)

    final_files = list(RAW_WEATHER_DIR.glob("*.csv"))

    print("ИТОГО файлов:", len(final_files))


if __name__ == "__main__":
    main()