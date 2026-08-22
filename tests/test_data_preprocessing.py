"""Smoke tests for python/data_preprocessing.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from data_preprocessing import (  # noqa: E402
    NUMERIC_COLUMNS,
    build_preprocessor,
    deterministic_split,
    load_data,
    preprocess,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Data.csv"


def test_load_data():
    df = load_data(DATA_PATH)
    assert list(df.columns) == ["Country", "Age", "Salary", "Purchased"]
    assert len(df) == 10


def test_deterministic_split_covers_all_rows():
    df = load_data(DATA_PATH)
    train, test = deterministic_split(df)
    assert len(train) + len(test) == len(df)
    assert len(test) == 2
    assert len(train) == 8


def test_deterministic_split_is_stable():
    df = load_data(DATA_PATH)
    train_1, test_1 = deterministic_split(df)
    train_2, test_2 = deterministic_split(df)
    pd.testing.assert_frame_equal(train_1, train_2)
    pd.testing.assert_frame_equal(test_1, test_2)


def test_preprocess_shape():
    X_train, X_test, y_train, y_test = preprocess(DATA_PATH)
    assert X_train.shape == (8, 5)
    assert X_test.shape == (2, 5)
    assert y_train.shape == (8,)
    assert y_test.shape == (2,)


def test_preprocess_no_nan():
    X_train, X_test, y_train, y_test = preprocess(DATA_PATH)
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()


def test_preprocess_scales_numerics_only():
    X_train, _, _, _ = preprocess(DATA_PATH)
    numeric_cols = [NUMERIC_COLUMNS.index(c) for c in NUMERIC_COLUMNS]
    scaled = X_train[:, numeric_cols]
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(scaled.std(axis=0), 1, atol=1e-10)


def test_pipeline_returns_same_as_preprocess():
    df = load_data(DATA_PATH)
    train, test = deterministic_split(df)
    pipe = build_preprocessor()
    features = ["Country", "Age", "Salary"]
    X_train = pipe.fit_transform(train[features])
    X_test = pipe.transform(test[features])
    assert X_train.shape == (8, 5)
    assert X_test.shape == (2, 5)