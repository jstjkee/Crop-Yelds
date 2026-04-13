# Crop Yield Forecast

MVP-приложение для прогнозирования урожайности на основе табличного датасета.

## Что умеет
- загружать CSV-файл;
- автоматически определять числовые и категориальные признаки;
- обучать несколько моделей:
  - Linear Regression
  - Random Forest Regressor
  - KNN Regressor
  - SVR
- сравнивать модели по метрикам:
  - MAE
  - MSE
  - R²
- показывать график "факт vs прогноз";
- сохранять лучшую модель.

## Структура проекта
```text
crop-yield-forecast/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ notebooks/
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data_loader.py
│  ├─ preprocessing.py
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ visualize.py
│  └─ app/
│     └─ streamlit_app.py
├─ models/
├─ reports/
├─ tests/
├─ .gitignore
├─ requirements.txt
└─ README.md