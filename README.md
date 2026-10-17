# ML Fundamentals

> Machine Learning A-Z pipeline in scikit-learn: leakage-free preprocessing (imputer → scaler → one-hot) fit on the train split, **21 tests**, Train (8, 5) / Test (2, 5).

## Navigation
- 🌐 [Portfolio site](https://nikitaboyarkin.github.io)
- 🏦 [Flagship case: Volta neobank](https://github.com/NikitaBoyarkin/volta-banking)
- 🧪 Related: [churn-prediction](https://github.com/NikitaBoyarkin/churn-prediction) · [python](https://github.com/NikitaBoyarkin/python) · [sql-analytics-case-study](https://github.com/NikitaBoyarkin/sql-analytics-case-study)

**▶ [Diagram: Leakage-free sklearn preprocessing pipeline](docs/diagrams/ml.html)** — opens in a browser, no install, no build

**On this page:** [Situation](#situation) · [Task](#task) · [Actions](#actions) · [Result](#result) · [What it does](#what-it-does) · [Structure](#structure) · [Run](#run) · [Test](#test) · [Notes](#notes)

## Situation

The ML A-Z course pipeline is easy to get subtly wrong: imputers and encoders
that see the test set, splits that depend on a random seed, or outdated
scikit-learn APIs. The point of this project is **the pipeline**, done correctly
with modern tooling.

## Task

Implement preprocessing the right way — missing-data imputation, categorical
encoding, a deterministic train/test split, and feature scaling — with no test
information leaking into the fit.

## Actions

- **Missing data** — mean imputation via `SimpleImputer` (`python/data_preprocessing.py`).
- **Categorical encoding** — `OneHotEncoder` for features, `LabelEncoder` for the target.
- **Train/test split** — deterministic 80/20: rows stable-sorted by `(Country, Age)`, every 5th row becomes the test set.
- **Feature scaling** — `StandardScaler` on numeric columns only.
- **One `ColumnTransformer`** — `SimpleImputer` → `StandardScaler` for numerics, `OneHotEncoder` for categoricals, fit on the training split only.
- **Support scripts** — `python/generate_dummy_data.py` (seeded extended datasets) and `python/eda_plots.py` (EDA charts in a unified `dark_background` style).

## Result

- **21 tests** — `tests/` (7 `test_data_preprocessing` + 9 `test_eda_plots` + 5 `test_generate_dummy_data`).
- **Train (8, 5) / Test (2, 5)** — deterministic 80/20 split, printed by `python/data_preprocessing.py`.
- **400 / 300 / 500 / 800** — generated rows in `Data_large`, `Salary_Data_large`, `Housing_Data`, `Customer_Churn`.
- **300 dpi** — EDA charts in `output/`, one figure per dataset plus a combined `dashboard.png` (`python/eda_plots.py`).

Hands-on implementation of the **Machine Learning A-Z** pipeline in Python
(scikit-learn). Starts with preprocessing (Part 1) — the foundation every later
section — regression, classification, clustering — builds on.

The point of this project is **the pipeline**, done correctly with modern
tooling: missing-data imputation, categorical encoding, train/test split and
feature scaling.

## What it does

**`python/data_preprocessing.py`** — preprocessing on `Data.csv`:

- **Missing data**: mean imputation via `SimpleImputer`.
- **Categorical encoding**: `OneHotEncoder` for features, `LabelEncoder` for
  the target.
- **Train/test split**: deterministic 80/20 — rows are stable-sorted by
  `(Country, Age)` and every 5th row becomes the test set (no reliance on a
  random seed).
- **Feature scaling**: `StandardScaler` on numeric columns only.

Everything is a single `ColumnTransformer` pipeline (`SimpleImputer` →
`StandardScaler` for numerics, `OneHotEncoder` for categoricals) **fit on the
training split only**, so no test information leaks into preprocessing.

**`python/generate_dummy_data.py`** — deterministic generator (seeded RNG) for
the extra datasets in `data/`, one per pipeline stage:

| File | Pattern | Purpose |
|------|---------|---------|
| `Data_large.csv` | `Data.csv` | 400 rows, extended preprocessing set |
| `Salary_Data_large.csv` | `Salary_Data.csv` | 300 rows, simple linear regression |
| `Housing_Data.csv` / `.sql` | `50_Startups.csv` | 500 rows, multiple regression |
| `Customer_Churn.csv` / `.json` | `Social_Network_Ads.csv` | 800 rows, classification |

**`python/eda_plots.py`** — EDA charts for every dataset in `output/` (one
figure per dataset + a combined `dashboard.png`), all in a unified
`dark_background` style, OO matplotlib (`fig, ax = plt.subplots`), 300 dpi PNG.

## Structure

```
ml/
├── data/              # 6 course datasets + generated large sets
│   ├── Data.csv              # Part 1 — preprocessing (France/Spain/Germany)
│   ├── Salary_Data.csv       # Part 2 — simple linear regression
│   ├── 50_Startups.csv       # Part 2 — multiple linear regression
│   ├── Position_Salaries.csv # Part 2 — polynomial / SVR / trees
│   ├── Social_Network_Ads.csv# Part 3 — classification
│   ├── Mall_Customers.csv    # Part 4 — clustering
│   └── Data_large.csv, Salary_Data_large.csv, Housing_Data.csv,
│       Housing_Data.sql, Customer_Churn.csv, Customer_Churn.json
├── python/           # data_preprocessing.py, generate_dummy_data.py, eda_plots.py
├── output/           # generated EDA charts (gitignored)
└── tests/            # pytest smoke tests
```

## Run

```bash
pip install -r requirements.txt

# preprocessing (reads ../data/Data.csv relative to python/)
cd python
python3 data_preprocessing.py

# regenerate the synthetic datasets (run from python/)
python3 generate_dummy_data.py

# render all EDA charts into output/ (run from project root)
python python/eda_plots.py
```

## Test

```bash
pip install pytest
pytest
```

## Notes

- Migrated to modern scikit-learn: `sklearn.model_selection`,
  `sklearn.impute.SimpleImputer`, `ColumnTransformer` (the course's original
  `sklearn.cross_validation` / `Imputer` / `OneHotEncoder(categorical_features=)`
  are removed in scikit-learn ≥ 0.22).
- Prints `Train: (8, 5), Test: (2, 5)`.