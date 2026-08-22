"""Data Preprocessing — ML A-Z Part 1 pipeline.

Imputation, categorical encoding, train/test split and feature scaling
done through a single scikit-learn pipeline fitted on the training split
only, so test information never leaks into the preprocessing step.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

DATA_PATH = "../data/Data.csv"
NUMERIC_COLUMNS = ["Age", "Salary"]
CATEGORICAL_COLUMNS = ["Country"]
TARGET_COLUMN = "Purchased"


def load_data(path=DATA_PATH):
    """Read the course dataset."""
    return pd.read_csv(path)


def deterministic_split(dataset):
    """80/20 split with no RNG: stable-sort by (Country, Age), every 5th row is test."""
    dataset = dataset.sort_values(
        by=["Country", "Age"], kind="stable"
    ).reset_index(drop=True)
    test_mask = (np.arange(len(dataset)) + 1) % 5 == 0
    return dataset[~test_mask], dataset[test_mask]


def build_preprocessor():
    """Impute + scale numerics and one-hot categoricals in one pipeline."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_COLUMNS),
            ("categorical", OneHotEncoder(), CATEGORICAL_COLUMNS),
        ]
    )


def preprocess(path=DATA_PATH):
    """Return (X_train, X_test, y_train, y_test)."""
    dataset = load_data(path)
    train, test = deterministic_split(dataset)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train[TARGET_COLUMN])
    y_test = label_encoder.transform(test[TARGET_COLUMN])

    features = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train[features])
    X_test = preprocessor.transform(test[features])
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")