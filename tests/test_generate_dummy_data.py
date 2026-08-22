"""Smoke tests for python/generate_dummy_data.py."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from generate_dummy_data import (  # noqa: E402
    generate_churn,
    generate_housing,
    generate_preprocessing,
    generate_salary,
)


def df(generator, n):
    return pd.DataFrame.from_records(generator(n))


def test_preprocessing_schema_and_missing():
    d = df(generate_preprocessing, 100)
    assert list(d.columns) == ["Country", "Age", "Salary", "Purchased"]
    assert (d["Age"] == "").sum() > 0
    assert (d["Salary"] == "").sum() > 0
    assert set(d["Country"]) <= {"France", "Spain", "Germany"}
    assert set(d["Purchased"]) <= {"Yes", "No"}


def test_salary_schema_and_no_plateau():
    d = df(generate_salary, 300)
    assert list(d.columns) == ["YearsExperience", "Salary"]
    assert (d["Salary"] <= 8000).sum() == 0
    assert d["Salary"].min() > 8000
    assert d["Salary"].max() > 100_000
    assert (d["Salary"] == 8000).sum() == 0


def test_salary_is_monotonic_in_experience():
    d = df(generate_salary, 300)
    r = d["YearsExperience"].corr(d["Salary"])
    assert r > 0.95


def test_housing_schema_and_ids():
    d = df(generate_housing, 100)
    assert list(d.columns) == ["Id", "SquareFeet", "Bedrooms", "HouseAge", "State", "Price"]
    assert d["Id"].nunique() == 100
    assert d["Price"].min() > 50_000


def test_churn_schema_and_target_values():
    d = df(generate_churn, 100)
    assert list(d.columns) == [
        "CustomerID", "Gender", "Age", "TenureMonths", "Contract",
        "InternetService", "MonthlyCharges", "TotalCharges", "Churn",
    ]
    assert set(d["Churn"]) <= {0, 1}
    assert set(d["Contract"]) <= {"Month-to-month", "One year", "Two year"}