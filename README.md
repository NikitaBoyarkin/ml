# ML Fundamentals

> Пайплайн из курса Machine Learning A-Z на scikit-learn: препроцессинг без
> утечки данных (импутация → масштабирование → one-hot), обученный только на
> тренировочной части. **21 тест**, Train (8, 5) / Test (2, 5).

## Навигация
- 🌐 [Сайт-портфолио](https://nikitaboyarkin.github.io)
- 🏦 [Флагманский кейс: необанк Volta](https://github.com/NikitaBoyarkin/volta-banking)
- 🧪 Рядом: [churn-prediction](https://github.com/NikitaBoyarkin/churn-prediction) · [python](https://github.com/NikitaBoyarkin/python) · [sql-analytics-case-study](https://github.com/NikitaBoyarkin/sql-analytics-case-study)

**▶ [Схема: препроцессинг в scikit-learn без утечки данных](docs/diagrams/ml.html)** — открывается в браузере, ничего ставить и собирать не нужно

**На этой странице:** [Ситуация](#ситуация) · [Задача](#задача) · [Действия](#действия) · [Результат](#результат) · [Что делает](#что-делает) · [Структура](#структура) · [Запуск](#запуск) · [Тесты](#тесты) · [Заметки](#заметки)

## Ситуация

Пайплайн из курса ML A-Z легко испортить незаметно: импутеры и энкодеры
подглядывают в тестовую выборку, сплит зависит от случайного зерна, а API
scikit-learn уже устарели. Смысл проекта — **сам пайплайн**, сделанный
правильно и на современных инструментах.

## Задача

Собрать препроцессинг как надо: заполнить пропуски, закодировать категории,
разделить данные детерминированно и отмасштабировать признаки так, чтобы
информация из теста не попадала в обучение.

## Действия

- **Пропуски** — заполнение средним через `SimpleImputer` (`python/data_preprocessing.py`).
- **Категории** — `OneHotEncoder` для признаков, `LabelEncoder` для целевой переменной.
- **Сплит** — детерминированные 80/20: строки устойчиво сортируются по `(Country, Age)`, каждая пятая уходит в тест.
- **Масштабирование** — `StandardScaler` только на числовых столбцах.
- **Один `ColumnTransformer`** — `SimpleImputer` → `StandardScaler` для чисел, `OneHotEncoder` для категорий, обучение только на тренировочной части.
- **Вспомогательные скрипты** — `python/generate_dummy_data.py` (расширенные датасеты с зерном) и `python/eda_plots.py` (графики EDA в едином стиле `dark_background`).

## Результат

- **21 тест** — в `tests/` (7 в `test_data_preprocessing` + 9 в `test_eda_plots` + 5 в `test_generate_dummy_data`).
- **Train (8, 5) / Test (2, 5)** — детерминированный сплит 80/20, который печатает `python/data_preprocessing.py`.
- **400 / 300 / 500 / 800** — сгенерированных строк в `Data_large`, `Salary_Data_large`, `Housing_Data`, `Customer_Churn`.
- **300 dpi** — графики EDA в `output/`: по одной фигуре на датасет плюс общий `dashboard.png` (`python/eda_plots.py`).

Практическая реализация пайплайна из курса **Machine Learning A-Z** на Python
(scikit-learn). Начинается с препроцессинга (часть 1) — фундамента, на котором
стоят регрессия, классификация и кластеризация.

Смысл проекта — **сам пайплайн**, сделанный правильно на современных
инструментах: заполнение пропусков, кодирование категорий, разделение на
train/test и масштабирование признаков.

## Что делает

**`python/data_preprocessing.py`** — препроцессинг `Data.csv`:

- **Пропуски**: заполнение средним через `SimpleImputer`.
- **Категории**: `OneHotEncoder` для признаков, `LabelEncoder` для целевой
  переменной.
- **Сплит**: детерминированные 80/20 — строки устойчиво сортируются по
  `(Country, Age)`, каждая пятая уходит в тест. Случайное зерно не нужно.
- **Масштабирование**: `StandardScaler` только на числовых столбцах.

Всё собрано в один пайплайн `ColumnTransformer` (`SimpleImputer` →
`StandardScaler` для чисел, `OneHotEncoder` для категорий) и обучено **только
на тренировочной части**, поэтому тест не протекает в препроцессинг.

**`python/generate_dummy_data.py`** — детерминированный генератор (RNG с
зерном) для дополнительных датасетов в `data/`, по одному на этап пайплайна:

| Файл | Образец | Назначение |
|------|---------|------------|
| `Data_large.csv` | `Data.csv` | 400 строк, расширенный набор для препроцессинга |
| `Salary_Data_large.csv` | `Salary_Data.csv` | 300 строк, простая линейная регрессия |
| `Housing_Data.csv` / `.sql` | `50_Startups.csv` | 500 строк, множественная регрессия |
| `Customer_Churn.csv` / `.json` | `Social_Network_Ads.csv` | 800 строк, классификация |

**`python/eda_plots.py`** — графики EDA по каждому датасету в `output/`
(одна фигура на датасет плюс общий `dashboard.png`). Единый стиль
`dark_background`, объектный matplotlib (`fig, ax = plt.subplots`), PNG 300 dpi.

## Структура

```
ml/
├── data/              # 6 датасетов курса + сгенерированные крупные наборы
│   ├── Data.csv              # Часть 1 — препроцессинг (France/Spain/Germany)
│   ├── Salary_Data.csv       # Часть 2 — простая линейная регрессия
│   ├── 50_Startups.csv       # Часть 2 — множественная линейная регрессия
│   ├── Position_Salaries.csv # Часть 2 — полином / SVR / деревья
│   ├── Social_Network_Ads.csv# Часть 3 — классификация
│   ├── Mall_Customers.csv    # Часть 4 — кластеризация
│   └── Data_large.csv, Salary_Data_large.csv, Housing_Data.csv,
│       Housing_Data.sql, Customer_Churn.csv, Customer_Churn.json
├── python/           # data_preprocessing.py, generate_dummy_data.py, eda_plots.py
├── output/           # сгенерированные графики EDA (в gitignore)
└── tests/            # дымовые тесты на pytest
```

## Запуск

```bash
pip install -r requirements.txt

# препроцессинг (читает ../data/Data.csv относительно python/)
cd python
python3 data_preprocessing.py

# пересобрать синтетические датасеты (запускать из python/)
python3 generate_dummy_data.py

# отрисовать все графики EDA в output/ (из корня проекта)
python python/eda_plots.py
```

## Тесты

```bash
pip install pytest
pytest
```

## Заметки

- Переехали на современный scikit-learn: `sklearn.model_selection`,
  `sklearn.impute.SimpleImputer`, `ColumnTransformer`. Исходные
  `sklearn.cross_validation` / `Imputer` / `OneHotEncoder(categorical_features=)`
  из курса удалены в scikit-learn ≥ 0.22.
- Печатает `Train: (8, 5), Test: (2, 5)`.
