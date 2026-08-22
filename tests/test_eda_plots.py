"""Smoke tests for python/eda_plots.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import eda_plots

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(eda_plots, "OUT_DIR", tmp_path)
    return tmp_path


def test_plot_preprocessing_writes_png(out_dir):
    eda_plots.plot_preprocessing(DATA_DIR / "Data.csv", "pp.png")
    assert (out_dir / "pp.png").exists()
    assert (out_dir / "pp.png").stat().st_size > 0


def test_plot_salary_regression_writes_png(out_dir):
    eda_plots.plot_salary_regression(DATA_DIR / "Salary_Data.csv", "sal.png")
    assert (out_dir / "sal.png").exists()


def test_plot_startups_writes_png(out_dir):
    eda_plots.plot_startups(DATA_DIR / "50_Startups.csv", "startups.png")
    assert (out_dir / "startups.png").exists()


def test_plot_social_ads_writes_png(out_dir):
    eda_plots.plot_social_ads(DATA_DIR / "Social_Network_Ads.csv", "ads.png")
    assert (out_dir / "ads.png").exists()


def test_plot_mall_customers_writes_png(out_dir):
    eda_plots.plot_mall_customers(DATA_DIR / "Mall_Customers.csv", "mall.png")
    assert (out_dir / "mall.png").exists()


def test_plot_position_salaries_writes_png(out_dir):
    eda_plots.plot_position_salaries(DATA_DIR / "Position_Salaries.csv", "pos.png")
    assert (out_dir / "pos.png").exists()


def test_plot_housing_writes_png(out_dir):
    eda_plots.plot_housing(DATA_DIR / "Housing_Data.csv", "housing.png")
    assert (out_dir / "housing.png").exists()


def test_plot_churn_writes_png(out_dir):
    eda_plots.plot_churn(DATA_DIR / "Customer_Churn.csv", "churn.png")
    assert (out_dir / "churn.png").exists()


def test_plot_dashboard_writes_png(out_dir):
    eda_plots.plot_dashboard()
    assert (out_dir / "dashboard.png").exists()