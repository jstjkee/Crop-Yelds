# Crop Yield Forecast

Проект для магистерской работы по теме **прогнозирования урожайности сельскохозяйственных культур на основе табличных данных**.

Текущая версия проекта содержит не старое MVP-приложение, а набор воспроизводимых исследовательских пайплайнов:

1. **Research 1 — Transfer learning**: обучение на большом открытом датасете и перенос на российские данные.
2. **Research 2 — Enriched Russian dataset**: обучение на расширенном российском датасете с погодой, удобрениями и техникой.
3. **Research 3 — Forecast scenarios**: проверка качества прогноза при разной заблаговременности, то есть при ограничении доступных погодных и управленческих признаков.

Основная целевая задача — регрессия урожайности.

---

## 1. Структура проекта

```text
crop-yeld/
├─ data/
│  ├─ raw/
│  │  ├─ research_1/
│  │  │  ├─ crop_yield.csv
│  │  │  └─ russian_crop_yield_clean.csv
│  │  ├─ research_2/
│  │  ├─ research_3/
│  │  ├─ rosstat/
│  │  └─ weather/
│  ├─ interim/
│  │  ├─ research_1/
│  │  ├─ research_2/
│  │  └─ research_3/
│  └─ processed/
│     ├─ research_1/
│     ├─ research_2/
│     │  └─ russian_final_cleaned.csv
│     └─ research_3/
│        └─ russian_final_forecast.csv
│
├─ src/
│  ├─ core/
│  │  ├─ config/
│  │  ├─ data/
│  │  ├─ evaluation/
│  │  ├─ models/
│  │  └─ training/
│  ├─ research_1_transfer/
│  ├─ research_2_enriched/
│  └─ research_3_forecast/
│
├─ results/
│  ├─ research_1/
│  │  ├─ models/
│  │  ├─ metrics/
│  │  ├─ tables/
│  │  └─ figures/
│  ├─ research_2/
│  └─ research_3/
│
├─ docs/
├─ tests/
├─ requirements.txt
└─ README.md
```

Ключевая логика находится в папках `src/research_1_transfer`, `src/research_2_enriched`, `src/research_3_forecast`. Общие модели, обучение, метрики и конфигурация вынесены в `src/core`.

---

## 2. Установка и подготовка окружения

Перейти в корень проекта:

```bash
cd crop-yeld
```

Создать виртуальное окружение:

```bash
python -m venv .venv
```

Активировать окружение.

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Установить зависимости:

```bash
pip install -r requirements.txt
```
---

## 3. Данные

### Research 1

Используются два датасета, приведённые к единому формату признаков:

```text
data/raw/research_1/crop_yield.csv
data/raw/research_1/russian_crop_yield_clean.csv
```

Оба датасета имеют одинаковые колонки:

```text
Region
Soil_Type
Crop
Rainfall_mm
Temperature_Celsius
Fertilizer_Used
Irrigation_Used
Weather_Condition
Days_to_Harvest
Yield_tons_per_hectare
```

Целевая переменная:

```text
Yield_tons_per_hectare
```

### Research 2

Основной очищенный российский датасет:

```text
data/processed/research_2/russian_final_cleaned.csv
```

Целевая переменная:

```text
target_yield_centner_per_ha
```

В датасет включены:

- базовые признаки: регион, культура, год, площадь посева;
- погодные признаки по месяцам с апреля по сентябрь;
- агрегированные признаки по осадкам, температуре, влажности, сухим и жарким дням;
- признаки по минеральным удобрениям;
- признаки по сельскохозяйственной технике;
- лаговые признаки урожайности за прошлые годы.

Текущий очищенный датасет содержит примерно:

```text
12517 строк
76 колонок
8 культур
```

### Research 3

Используется тот же очищенный российский датасет, но признаки дополнительно фильтруются в зависимости от сценария прогноза:

```text
data/processed/research_2/russian_final_cleaned.csv
```

Также в проекте есть подготовленный файл:

```text
data/processed/research_3/russian_final_forecast.csv
```

---

## 4. Модели

В проекте реализованы нейросетевые модели для табличной регрессии с отдельными выходными головами по культурам:

```text
mlp_resnet
transformer
wide_deep
tab_mlp
```

Основные модели для защиты и итогового сравнения:

```text
mlp_resnet
transformer
```

`mlp_resnet` — остаточная MLP-модель с crop-specific heads.

`transformer` — табличный Transformer, где числовые признаки токенизируются и обрабатываются encoder-блоком.

`wide_deep` и `tab_mlp` используются как дополнительные DL-бейзлайны во втором и третьем исследованиях.

---

## 5. Research 1 — Transfer learning

Папка:

```text
src/research_1_transfer/
```

Смысл исследования:

1. обучить модель на большом открытом датасете `crop_yield.csv`;
2. проверить её на российском датасете без дообучения — zero-shot transfer;
3. дообучить модель на российском датасете — fine-tuning;
4. сравнить с обучением на российском датасете с нуля.

Поддерживаемые команды:

```bash
python -m src.research_1_transfer.run train_source
python -m src.research_1_transfer.run evaluate_transfer
python -m src.research_1_transfer.run finetune_on_russia
python -m src.research_1_transfer.run train_russia_scratch
python -m src.research_1_transfer.run compare_variants
python -m src.research_1_transfer.run all
```

### Рекомендуемый запуск первого исследования

```bash
python -m src.research_1_transfer.run all --models mlp_resnet transformer --feature-modes raw autoencoder
```

Можно добавить PCA как вспомогательный режим признаков:

```bash
python -m src.research_1_transfer.run all --models mlp_resnet transformer --feature-modes raw pca autoencoder
```

### Быстрый запуск только одной модели

```bash
python -m src.research_1_transfer.run all --models mlp_resnet --feature-modes raw
```

### Запуск по шагам

Обучение на открытом датасете:

```bash
python -m src.research_1_transfer.run train_source --models mlp_resnet transformer --feature-modes raw autoencoder
```

Zero-shot перенос на российский датасет:

```bash
python -m src.research_1_transfer.run evaluate_transfer --models mlp_resnet transformer --feature-modes raw autoencoder
```

Дообучение на российском датасете:

```bash
python -m src.research_1_transfer.run finetune_on_russia --models mlp_resnet transformer --feature-modes raw autoencoder
```

Обучение на российском датасете с нуля:

```bash
python -m src.research_1_transfer.run train_russia_scratch --models mlp_resnet transformer --feature-modes raw autoencoder
```

Сбор итоговой таблицы:

```bash
python -m src.research_1_transfer.run compare_variants
```

### Результаты Research 1

Основная итоговая таблица:

```text
results/research_1/metrics/research_1_compare_all_metrics.csv
```

Отдельные таблицы:

```text
results/research_1/metrics/source_training_metrics.csv
results/research_1/metrics/transfer_zero_shot_metrics.csv
results/research_1/metrics/transfer_finetuned_metrics.csv
results/research_1/metrics/russia_scratch_metrics.csv
```

Предсказания и метрики по культурам:

```text
results/research_1/tables/
```

Сохранённые модели:

```text
results/research_1/models/
```
---

## 6. Research 2 — Enriched Russian dataset

Папка:

```text
src/research_2_enriched/
```

Смысл исследования:

- перейти от упрощённого сопоставленного датасета к полноценному российскому датасету;
- добавить погодные признаки, удобрения, технику и лаги урожайности;
- оценить качество моделей на временном разделении по годам.

Разделение по годам задаётся в `src/research_2_enriched/config.py`:

```text
train: 2000–2016
val:   2017–2020
test:  2021–2024
```

Поддерживаемые команды:

```bash
python -m src.research_2_enriched.run eda
python -m src.research_2_enriched.run train
python -m src.research_2_enriched.run compare
python -m src.research_2_enriched.run tune
python -m src.research_2_enriched.run all
```

### EDA и визуальный анализ

```bash
python -m src.research_2_enriched.run eda
```

Артефакты сохраняются в:

```text
results/research_2/figures/
results/research_2/tables/
```

Например:

```text
research_2_missing_ratio.png
research_2_target_distribution.png
research_2_target_correlations.png
research_2_corr_heatmap.png
```

### Обучение моделей

Рекомендуемый запуск второго исследования:

```bash
python -m src.research_2_enriched.run train --models mlp_resnet transformer tab_mlp wide_deep --feature-modes raw --seed 52
```

Запуск только основной модели:

```bash
python -m src.research_2_enriched.run train --models mlp_resnet --feature-modes raw --seed 52
```

Запуск по нескольким seed:

```bash
python -m src.research_2_enriched.run train --models mlp_resnet --feature-modes raw --seed-list 42 52 62
```

Сбор сравнительной таблицы:

```bash
python -m src.research_2_enriched.run compare
```

### Подбор гиперпараметров Optuna

Для MLP-ResNet:

```bash
python -m src.research_2_enriched.run tune --model mlp_resnet --feature-set full --split year --trials 20
```

Для Transformer:

```bash
python -m src.research_2_enriched.run tune --model transformer --feature-set full --split year --trials 20
```

### Результаты Research 2

Основные таблицы:

```text
results/research_2/metrics/research_2_enriched_metrics.csv
results/research_2/metrics/research_2_enriched_metrics_by_seed.csv
results/research_2/metrics/research_2_enriched_metrics_seed_agg.csv
results/research_2/metrics/research_2_compare_metrics.csv
```

Предсказания и метрики по культурам:

```text
results/research_2/tables/
```

Модели и препроцессоры:

```text
results/research_2/models/
```

---

## 7. Research 3 — Forecast scenarios

Папка:

```text
src/research_3_forecast/
```

Смысл исследования:

- превратить задачу из nowcast в forecast;
- проверить, насколько рано можно делать прогноз урожайности;
- сравнить сценарии с разным набором доступных признаков.

Сценарии задаются в `src/research_3_forecast/config.py`.

Текущие сценарии:

```text
F0_full_nowcast
F1_mid_july
F1_mid_june
F2_early_may
F2_early_april
F3_preseason_operational
F3_preseason_strict
F2_windows_only
```

Краткая логика сценариев:

| Сценарий | Смысл |
|---|---|
| `F0_full_nowcast` | Доступны признаки за весь сезон до сентября |
| `F1_mid_july` | Доступна погода примерно до июля |
| `F1_mid_june` | Доступна погода примерно до июня |
| `F2_early_may` | Ранний прогноз, погода только до мая |
| `F2_early_april` | Очень ранний прогноз, погода только до апреля |
| `F3_preseason_operational` | Предсезонный прогноз с частью оперативных признаков |
| `F3_preseason_strict` | Строгий предсезонный прогноз без текущих оперативных признаков |
| `F2_windows_only` | Прогноз на основе агрономических погодных окон |

### Запуск одного сценария

```bash
python -m src.research_3_forecast.run train --scenario F0_full_nowcast --models mlp_resnet --feature-modes raw --seed 52
```

Другой пример:

```bash
python -m src.research_3_forecast.run train --scenario F2_windows_only --models mlp_resnet --feature-modes raw --seed-list 42 52 62
```

### Запуск всех сценариев

PowerShell:

```powershell
$scenarios = @(
  "F0_full_nowcast",
  "F1_mid_july",
  "F1_mid_june",
  "F2_early_may",
  "F2_early_april",
  "F3_preseason_operational",
  "F3_preseason_strict",
  "F2_windows_only"
)

foreach ($s in $scenarios) {
  python -m src.research_3_forecast.run train --scenario $s --models mlp_resnet --feature-modes raw --seed 52
}
```

Bash:

```bash
for s in \
  F0_full_nowcast \
  F1_mid_july \
  F1_mid_june \
  F2_early_may \
  F2_early_april \
  F3_preseason_operational \
  F3_preseason_strict \
  F2_windows_only
 do
  python -m src.research_3_forecast.run train --scenario "$s" --models mlp_resnet --feature-modes raw --seed 52
 done
```

### Результаты Research 3

Для каждого сценария создаются отдельные файлы:

```text
results/research_3/metrics/research_3_forecast_metrics_<scenario>.csv
results/research_3/metrics/research_3_forecast_metrics_by_seed_<scenario>.csv
results/research_3/metrics/research_3_forecast_metrics_seed_agg_<scenario>.csv
```

Примеры:

```text
results/research_3/metrics/research_3_forecast_metrics_F0_full_nowcast.csv
results/research_3/metrics/research_3_forecast_metrics_F2_windows_only.csv
results/research_3/metrics/research_3_forecast_metrics_seed_agg_F2_windows_only.csv
```

Предсказания и метрики по культурам:

```text
results/research_3/tables/
```

Модели:

```text
results/research_3/models/
```

---

## 8. Метрики

Во всех исследованиях используются регрессионные метрики:

```text
MAE
MSE
RMSE
R2
```

Интерпретация:

- `MAE` — средняя абсолютная ошибка прогноза;
- `MSE` — среднеквадратичная ошибка;
- `RMSE` — корень из MSE, ошибка в единицах целевой переменной;
- `R2` — доля объяснённой дисперсии, чем ближе к 1, тем лучше.

Единицы измерения зависят от исследования:

```text
Research 1: Yield_tons_per_hectare
Research 2–3: target_yield_centner_per_ha
```

---

## 9. Где смотреть итоговые результаты

### Для презентации первого исследования

```text
results/research_1/metrics/research_1_compare_all_metrics.csv
```

Эта таблица нужна для сравнения:

- обучение на открытом датасете;
- zero-shot перенос на российский датасет;
- fine-tuning на российском датасете;
- обучение на российском датасете с нуля.

### Для презентации второго исследования

```text
results/research_2/metrics/research_2_enriched_metrics.csv
results/research_2/metrics/research_2_enriched_metrics_seed_agg.csv
```

Эти таблицы показывают качество моделей на расширенном российском датасете.

### Для презентации третьего исследования

```text
results/research_3/metrics/research_3_forecast_metrics_*.csv
```

Эти таблицы показывают качество прогнозирования при разной заблаговременности.

---

## 10. Типовой порядок запуска всего проекта

Если нужно воспроизвести весь пайплайн с нуля, порядок такой:

```bash
python -m src.research_1_transfer.run all --models mlp_resnet transformer --feature-modes raw autoencoder
```

```bash
python -m src.research_2_enriched.run eda
```

```bash
python -m src.research_2_enriched.run train --models mlp_resnet transformer tab_mlp wide_deep --feature-modes raw --seed 52
```

```bash
python -m src.research_2_enriched.run compare
```

```bash
python -m src.research_3_forecast.run train --scenario F0_full_nowcast --models mlp_resnet --feature-modes raw --seed 52
```

```bash
python -m src.research_3_forecast.run train --scenario F1_mid_july --models mlp_resnet --feature-modes raw --seed 52
```

```bash
python -m src.research_3_forecast.run train --scenario F1_mid_june --models mlp_resnet --feature-modes raw --seed 52
```

```bash
python -m src.research_3_forecast.run train --scenario F2_early_may --models mlp_resnet --feature-modes raw --seed 52
```

```bash
python -m src.research_3_forecast.run train --scenario F2_early_april --models mlp_resnet --feature-modes raw --seed 52
```

```bash
python -m src.research_3_forecast.run train --scenario F3_preseason_operational --models mlp_resnet --feature-modes raw --seed 52
```

```bash
python -m src.research_3_forecast.run train --scenario F3_preseason_strict --models mlp_resnet --feature-modes raw --seed 52
```

```bash
python -m src.research_3_forecast.run train --scenario F2_windows_only --models mlp_resnet --feature-modes raw --seed-list 42 52 62
```

---


## 11. Краткое описание исследовательской логики

### Research 1

Проверяется возможность переноса знаний с большого открытого датасета на российские данные. Главный вывод строится на сравнении zero-shot и fine-tuning: если zero-shot работает плохо, а fine-tuning улучшает качество, значит простого переноса недостаточно, но предварительное обучение может быть полезным после адаптации к российскому распределению данных.

### Research 2

Проверяется, насколько расширение российского датасета погодой, удобрениями, техникой и историческими лагами урожайности повышает качество прогноза. Основное разделение делается по годам, чтобы эксперимент был ближе к реальной задаче прогнозирования будущих сезонов.

### Research 3

Проверяется, как меняется качество прогноза при уменьшении объёма доступной информации о текущем сезоне. Это позволяет определить компромисс между качеством и заблаговременностью прогноза.
