# ML Fundamentals

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