# Crop-Yelds

Проект для магистерской работы по теме **прогнозирования урожайности сельскохозяйственных культур на основе табличных данных**.

Основная задача проекта — построение воспроизводимого исследовательского пайплайна, который позволяет обучать и сравнивать модели прогнозирования урожайности на открытых и российских аграрных данных.

1. **Research 1 — Transfer learning**
   Проверка переноса обучения с большого открытого датасета на российские данные.

2. **Research 2 — Enriched Russian dataset**
   Построение модели на расширенном российском датасете с погодными, агротехническими и лаговыми признаками.


---

## Содержание

* [Описание проекта](#описание-проекта)
* [Структура репозитория](#структура-репозитория)
* [Установка](#установка)
* [Данные](#данные)
* [Модели](#модели)
* [Research 1 — Transfer learning](#research-1--transfer-learning)
* [Research 2 — Enriched Russian dataset](#research-2--enriched-russian-dataset)
* [Результаты](#результаты)
* [Метрики](#метрики)
* [Работа с GitHub](#работа-с-github)

---

## Описание проекта

Проект решает задачу регрессии: по набору признаков для региона, культуры и года необходимо спрогнозировать урожайность сельскохозяйственной культуры.

Целевая переменная во втором исследовании:

```text
target_yield_centner_per_ha
```

То есть модель прогнозирует урожайность в центнерах с гектара.

Ключевая идея проекта — не просто обучить одну модель, а построить полный пайплайн:

1. загрузка и проверка данных;
2. очистка и нормализация признаков;
3. формирование обучающей, валидационной и тестовой выборок;
4. построение признакового пространства;
5. обучение нейросетевых моделей для табличных данных;
6. использование многоголовой архитектуры с отдельными выходами по культурам;
7. расчет общих метрик и метрик по каждой культуре;
8. сохранение моделей, предсказаний и таблиц результатов.

---

## Структура репозитория

```text
Crop-Yelds/
├─ data/
│  ├─ raw/                         # исходные датасеты, не хранятся в Git
│  ├─ interim/                     # промежуточные датасеты, не хранятся в Git
│  ├─ processed/                   # обработанные датасеты, не хранятся в Git
│  └─ reference/                   # справочные таблицы
├─ results/                        # результаты экспериментов, не хранятся в Git
│  ├─ research_1/
│  │  ├─ figures/
│  │  ├─ metrics/
│  │  ├─ models/
│  │  └─ tables/
│  └─ research_2/
│     ├─ figures/
│     ├─ metrics/
│     ├─ models/
│     └─ tables/
│
├─ src/
│  ├─ core/                        # общая логика проекта
│  │  ├─ config/                   # пути, настройки моделей и обучения
│  │  ├─ data/                     # загрузка, предобработка, target scaling
│  │  ├─ evaluation/               # метрики и отчеты
│  │  ├─ models/                   # нейросетевые модели
│  │  └─ training/                 # training loop, prediction, dataloader
│  │
│  ├─ research_1_transfer/         # первое исследование: transfer learning
│  │  ├─ compare_variants.py
│  │  ├─ config.py
│  │  ├─ datasets.py
│  │  ├─ evaluate_transfer.py
│  │  ├─ features.py
│  │  ├─ finetune_on_russia.py
│  │  ├─ run.py
│  │  ├─ train_russia_scratch.py
│  │  └─ train_source.py
│  │
│  └─ research_2_enriched/         # второе исследование: расширенный российский датасет
│     ├─ build_dataset.py
│     ├─ compare_feature_sets.py
│     ├─ config.py
│     ├─ eda_dataset.py
│     ├─ run.py
│     ├─ train_enriched.py
│     └─ tune_optuna.py
│ 
├─ .gitignore
├─ requirements.txt
└─ README.md
```

Основная переиспользуемая логика находится в `src/core`. Исследовательские сценарии вынесены в отдельные папки `src/research_1_transfer` и `src/research_2_enriched`.

---

## Установка

### 1. Клонировать репозиторий

```bash
git clone https://github.com/jstjkee/Crop-Yelds.git
cd Crop-Yelds
```

### 2. Создать виртуальное окружение

```bash
python -m venv .venv
```

### 3. Активировать окружение


```powershell
.venv\Scripts\Activate.ps1
```

### 4. Установить зависимости

```bash
pip install -r requirements.txt
```

---

## Данные

Датасеты не должны храниться в GitHub. После клонирования проекта их нужно положить локально в папку `data/`.

### Research 1

Для первого исследования используются два датасета, приведенные к единому формату колонок.

Открытый датасет:

```text
data/raw/research_1/crop_yield.csv
```

Российский датасет в формате первого исследования:

```text
data/raw/research_1/russian_crop_yield_clean.csv
```

Ожидаемые колонки:

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

Основной датасет второго исследования:

```text
data/processed/research_2/russian_final_cleaned.csv
```

Целевая переменная:

```text
target_yield_centner_per_ha
```

В текущей версии датасет содержит:

```text
12 517 строк
76 колонок
8 культур
период: 2000–2024
```

Культуры, используемые во втором исследовании:

```text
картофель
зерновые и зернобобовые культуры
горох
пшеница озимая
озимые зерновые культуры
подсолнечник
кукуруза на зерно
просо
```

В датасет входят следующие группы признаков:

* регион, культура и год;
* площадь посева;
* погодные признаки по месяцам с апреля по сентябрь;
* агрегированные сезонные погодные признаки;
* признаки по минеральным удобрениям;
* признаки по сельскохозяйственной технике;
* лаговые признаки урожайности за предыдущие годы.


---

## Модели

В проекте реализованы нейросетевые модели для табличной регрессии:


```text
mlp_resnet
transformer
```

### MLP-ResNet

`mlp_resnet` — многоголовая MLP-модель с остаточными блоками. Общие слои извлекают признаки из табличных данных, после чего для каждой культуры используется отдельная регрессионная голова.

Файл модели:

```text
src/core/models/mlp_resnet_multihead.py
```

### Tabular Transformer

`transformer` — табличная Transformer-модель. Числовые признаки токенизируются, проходят через encoder-блоки, после чего результат передается в crop-specific heads.

Файл модели:

```text
src/core/models/multihead_transformer.py
```

---

## Research 1 — Transfer learning

Папка исследования:

```text
src/research_1_transfer/
```

Цель исследования — проверить, насколько обучение на большом открытом датасете переносится на российские данные.

Логика исследования:

1. обучить модель на открытом датасете `crop_yield.csv`;
2. проверить zero-shot перенос на российский датасет;
3. выполнить fine-tuning на российских данных;
4. обучить модель на российских данных с нуля;
5. сравнить все варианты между собой.

Поддерживаемые команды:

```bash
python -m src.research_1_transfer.run train_source
python -m src.research_1_transfer.run evaluate_transfer
python -m src.research_1_transfer.run finetune_on_russia
python -m src.research_1_transfer.run train_russia_scratch
python -m src.research_1_transfer.run compare_variants
python -m src.research_1_transfer.run all
```

Рекомендуемый полный запуск:

```bash
python -m src.research_1_transfer.run all --models mlp_resnet transformer --feature-modes raw
```

Основные результаты сохраняются в папку:

```text
results/research_1/
```

Ключевые файлы метрик:

```text
results/research_1/metrics/source_training_metrics.csv
results/research_1/metrics/transfer_zero_shot_metrics.csv
results/research_1/metrics/transfer_finetuned_metrics.csv
results/research_1/metrics/russia_scratch_metrics.csv
results/research_1/metrics/research_1_compare_all_metrics.csv
```

---

## Research 2 — Enriched Russian dataset

Папка исследования:

```text
src/research_2_enriched/
```

Цель исследования — построить модель прогнозирования урожайности на расширенном российском датасете.

В отличие от первого исследования, здесь используются не только базовые признаки, но и расширенное описание аграрных условий:

* погодные данные;
* удобрения;
* сельскохозяйственная техника;
* лаговые признаки урожайности;
* разбиение по годам.

### Разбиение данных

Во втором исследовании используется `year split`:

```text
train: 2000–2016
val:   2017–2020
test:  2021–2024
```

Такое разбиение ближе к реальной задаче прогнозирования, потому что модель проверяется на будущих годах, которые не использовались при обучении.

### EDA

Запуск разведочного анализа данных:

```bash
python -m src.research_2_enriched.run eda
```

После запуска сохраняются таблицы и графики:

```text
results/research_2/figures/research_2_missing_ratio.png
results/research_2/figures/research_2_target_distribution.png
results/research_2/figures/research_2_target_correlations.png
results/research_2/figures/research_2_corr_heatmap.png
results/research_2/tables/research_2_suspicious_rows.csv
```

### Обучение моделей

Рекомендуемый запуск основных моделей:


```bash
python -m src.research_2_enriched.run train --models mlp_resnet transformer --feature-modes raw --seed-list 42 52 62
```

Запуск всех доступных моделей второго исследования:

```bash
python -m src.research_2_enriched.run train --models mlp_resnet transformer wide_deep tab_mlp --feature-modes raw --seed-list 42 52 62
```

Сравнение результатов:

```bash
python -m src.research_2_enriched.run compare
```


### Optuna-подбор гиперпараметров

В проекте предусмотрен подбор гиперпараметров через Optuna.

Пример запуска для Transformer:

```bash
python -m src.research_2_enriched.run tune --model transformer --feature-set full --split year --trials 20
```

Пример запуска для MLP-ResNet:

```bash
python -m src.research_2_enriched.run tune --model mlp_resnet --feature-set full --split year --trials 20
```

---

## Результаты

Все результаты экспериментов сохраняются в папку `results/`.

Для первого исследования:

```text
results/research_1/models/      # сохраненные модели и препроцессоры
results/research_1/metrics/     # итоговые метрики и история обучения
results/research_1/tables/      # предсказания и метрики по культурам
results/research_1/figures/     # графики, если формируются
```

Для второго исследования:

```text
results/research_2/models/      # сохраненные модели и препроцессоры
results/research_2/metrics/     # итоговые метрики и история обучения
results/research_2/tables/      # предсказания и метрики по культурам
results/research_2/figures/     # EDA-графики
```

---

## Метрики

В проекте рассчитываются абсолютные и нормализованные метрики качества.

Абсолютные метрики:

```text
MAE
MSE
RMSE
R2
```

Нормализованные метрики:

```text
NMAE, %
NRMSE, %
NMAPE, %
```

Метрики рассчитываются:

1. по всей тестовой выборке;
2. отдельно по каждой культуре;
3. отдельно по каждому seed при серии запусков;
4. в агрегированном виде по нескольким seed.

Метрики по культурам важны для анализа многоголовой архитектуры, потому что каждая культура имеет собственную выходную голову.

---

