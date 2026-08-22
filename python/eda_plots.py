"""EDA plots for every ML A-Z dataset.

Reads the course + generated datasets from data/ and renders one figure per
dataset plus a combined dashboard, in a single dark_background style.
Run from python/:  python3 eda_plots.py  (or: uv run python python/eda_plots.py)
Output lands in output/ next to data/.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
plt.style.use("dark_background")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C0, C1, C2, C3 = "#00d4ff", "#ff7f0e", "#2ca02c", "#d62728"


def save(fig, name):
    out_path = OUT_DIR / name
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


# ---------------------------------------------------------------------------
# 1. Preprocessing — Data.csv / Data_large.csv
#    Age vs Salary, colour = Purchased.
# ---------------------------------------------------------------------------
def plot_preprocessing(path, name):
    df = pd.read_csv(path).dropna()
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    for label, colour in [("No", C1), ("Yes", C0)]:
        sub = df[df["Purchased"] == label]
        ax.scatter(sub["Age"], sub["Salary"], s=45, alpha=0.85,
                   label=label, color=colour, edgecolor="none")
    ax.set(xlabel="Age", ylabel="Salary, $",
           title=f"Purchased by Age/Salary ({path.name})")
    ax.legend(frameon=False, title="Purchased")
    ax.grid(True, alpha=0.3)

    save(fig, name)


# ---------------------------------------------------------------------------
# 2. Simple linear regression — Salary_Data.csv / _large
#    scatter + OLS fit line.
# ---------------------------------------------------------------------------
def plot_salary_regression(path, name):
    df = pd.read_csv(path)
    x, y = df["YearsExperience"].to_numpy(), df["Salary"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    ax.scatter(x, y, s=45, color=C0, label="data")
    ax.plot(xs, slope * xs + intercept, color=C1, lw=2,
            label=f"OLS fit: Salary = {intercept:,.0f} + {slope:,.0f}·Years")
    ax.set(xlabel="Years of Experience", ylabel="Salary, $",
           title=f"Salary vs Experience ({path.name})")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    save(fig, name)


# ---------------------------------------------------------------------------
# 3. Multiple regression — 50_Startups.csv
#    Profit vs each spend, colour = State.
# ---------------------------------------------------------------------------
def plot_startups(path, name):
    df = pd.read_csv(path)
    spends = ["R&D Spend", "Administration", "Marketing Spend"]
    states = df["State"].unique()

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")
    for ax, spend in zip(axs, spends):
        for i, state in enumerate(states):
            sub = df[df["State"] == state]
            ax.scatter(sub[spend], sub["Profit"], s=40, alpha=0.85,
                       label=state, color=f"C{i}", edgecolor="none")
        ax.set(xlabel=spend, ylabel="Profit, $", title=f"Profit vs {spend}")
        ax.grid(True, alpha=0.3)
    axs[0].legend(frameon=False, title="State", loc="upper left")
    fig.suptitle(f"Startup Profit by Spend ({path.name})")

    save(fig, name)


# ---------------------------------------------------------------------------
# 4. Classification — Social_Network_Ads.csv
#    Age vs EstimatedSalary, colour = Purchased.
# ---------------------------------------------------------------------------
def plot_social_ads(path, name):
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    for label, colour in [(0, C1), (1, C0)]:
        sub = df[df["Purchased"] == label]
        ax.scatter(sub["Age"], sub["EstimatedSalary"], s=45, alpha=0.85,
                   label=f"Purchased={label}", color=colour, edgecolor="none")
    ax.set(xlabel="Age", ylabel="Estimated Salary, $",
           title=f"Purchase decision by Age/Salary ({path.name})")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, alpha=0.3)

    save(fig, name)


# ---------------------------------------------------------------------------
# 5. Clustering — Mall_Customers.csv
#    Annual Income vs Spending Score, colour = Genre.
# ---------------------------------------------------------------------------
def plot_mall_customers(path, name):
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    for i, genre in enumerate(df["Genre"].unique()):
        sub = df[df["Genre"] == genre]
        ax.scatter(sub["Annual Income (k$)"], sub["Spending Score (1-100)"],
                   s=45, alpha=0.85, label=genre, color=f"C{i}",
                   edgecolor="none")
    ax.set(xlabel="Annual Income, k$", ylabel="Spending Score (1-100)",
           title=f"Customer segments ({path.name})")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    save(fig, name)


# ---------------------------------------------------------------------------
# 6. Position_Salaries.csv — polynomial-style Level vs Salary curve.
# ---------------------------------------------------------------------------
def plot_position_salaries(path, name):
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    ax.plot(df["Level"], df["Salary"], marker="o", lw=2, color=C0)
    for _, row in df.iterrows():
        ax.annotate(row["Position"], (row["Level"], row["Salary"]),
                    fontsize=8, ha="center", xytext=(0, 10),
                    textcoords="offset points")
    ax.set(xlabel="Position Level", ylabel="Salary, $",
           title=f"Salary by position level ({path.name})")
    ax.grid(True, alpha=0.3)

    save(fig, name)


# ---------------------------------------------------------------------------
# 7. Housing_Data.csv — Price vs SquareFeet, colour = State.
# ---------------------------------------------------------------------------
def plot_housing(path, name):
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    for i, state in enumerate(df["State"].unique()):
        sub = df[df["State"] == state]
        ax.scatter(sub["SquareFeet"], sub["Price"], s=40, alpha=0.8,
                   label=state, color=f"C{i}", edgecolor="none")
    ax.set(xlabel="Square Feet", ylabel="Price, $",
           title=f"House price by size ({path.name})")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    save(fig, name)


# ---------------------------------------------------------------------------
# 8. Customer_Churn.csv — MonthlyCharges vs Tenure, colour = Churn.
# ---------------------------------------------------------------------------
def plot_churn(path, name):
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    for label, colour in [(0, C0), (1, C1)]:
        sub = df[df["Churn"] == label]
        ax.scatter(sub["TenureMonths"], sub["MonthlyCharges"], s=35,
                   alpha=0.7, label=f"Churn={label}", color=colour,
                   edgecolor="none")
    ax.set(xlabel="Tenure, months", ylabel="Monthly Charges, $",
           title=f"Churn by tenure/cost ({path.name})")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    save(fig, name)


# ---------------------------------------------------------------------------
# Dashboard — one combined figure for quick review.
# ---------------------------------------------------------------------------
def plot_dashboard():
    fig = plt.figure(figsize=(20, 14), layout="constrained")
    axd = fig.subplot_mosaic(
        [
            ["salary", "salary", "startups", "startups"],
            ["ads", "ads", "mall", "mall"],
            ["churn", "churn", "housing", "housing"],
        ],
    )

    salary = pd.read_csv(DATA_DIR / "Salary_Data.csv")
    x, y = salary["YearsExperience"], salary["Salary"]
    axd["salary"].scatter(x, y, s=40, color=C0)
    axd["salary"].set(title="Simple linear regression", xlabel="YearsExperience",
                      ylabel="Salary")

    startups = pd.read_csv(DATA_DIR / "50_Startups.csv")
    for i, state in enumerate(startups["State"].unique()):
        sub = startups[startups["State"] == state]
        axd["startups"].scatter(sub["R&D Spend"], sub["Profit"], s=40,
                                alpha=0.85, label=state, color=f"C{i}")
    axd["startups"].legend(frameon=False, fontsize=9)
    axd["startups"].set(title="Multiple regression", xlabel="R&D Spend",
                        ylabel="Profit")

    ads = pd.read_csv(DATA_DIR / "Social_Network_Ads.csv")
    for label, colour in [(0, C1), (1, C0)]:
        sub = ads[ads["Purchased"] == label]
        axd["ads"].scatter(sub["Age"], sub["EstimatedSalary"], s=35,
                           alpha=0.8, label=f"Purchased={label}",
                           color=colour)
    axd["ads"].legend(frameon=False, fontsize=9)
    axd["ads"].set(title="Classification", xlabel="Age",
                   ylabel="Estimated Salary")

    mall = pd.read_csv(DATA_DIR / "Mall_Customers.csv")
    for i, genre in enumerate(mall["Genre"].unique()):
        sub = mall[mall["Genre"] == genre]
        axd["mall"].scatter(sub["Annual Income (k$)"],
                            sub["Spending Score (1-100)"], s=35, alpha=0.8,
                            label=genre, color=f"C{i}")
    axd["mall"].legend(frameon=False, fontsize=9)
    axd["mall"].set(title="Clustering", xlabel="Annual Income, k$",
                    ylabel="Spending Score")

    churn = pd.read_csv(DATA_DIR / "Customer_Churn.csv")
    for label, colour in [(0, C0), (1, C1)]:
        sub = churn[churn["Churn"] == label]
        axd["churn"].scatter(sub["TenureMonths"], sub["MonthlyCharges"],
                             s=25, alpha=0.7, label=f"Churn={label}",
                             color=colour)
    axd["churn"].legend(frameon=False, fontsize=9)
    axd["churn"].set(title="Classification (churn)", xlabel="Tenure, months",
                     ylabel="Monthly Charges")

    housing = pd.read_csv(DATA_DIR / "Housing_Data.csv")
    for i, state in enumerate(housing["State"].unique()):
        sub = housing[housing["State"] == state]
        axd["housing"].scatter(sub["SquareFeet"], sub["Price"], s=25,
                               alpha=0.7, label=state, color=f"C{i}")
    axd["housing"].legend(frameon=False, fontsize=9)
    axd["housing"].set(title="Multiple regression (housing)",
                       xlabel="Square Feet", ylabel="Price")

    fig.suptitle("ML A-Z — dataset overview", fontsize=18)
    save(fig, "dashboard.png")


def main():
    print("Plotting:")
    plot_preprocessing(DATA_DIR / "Data.csv", "preprocessing_Data.png")
    plot_preprocessing(DATA_DIR / "Data_large.csv", "preprocessing_Data_large.png")
    plot_salary_regression(DATA_DIR / "Salary_Data.csv", "salary_regression.png")
    plot_salary_regression(DATA_DIR / "Salary_Data_large.csv", "salary_regression_large.png")
    plot_startups(DATA_DIR / "50_Startups.csv", "startups_profit.png")
    plot_social_ads(DATA_DIR / "Social_Network_Ads.csv", "social_ads_classification.png")
    plot_mall_customers(DATA_DIR / "Mall_Customers.csv", "mall_customers_clustering.png")
    plot_position_salaries(DATA_DIR / "Position_Salaries.csv", "position_salaries.png")
    plot_housing(DATA_DIR / "Housing_Data.csv", "housing_regression.png")
    plot_churn(DATA_DIR / "Customer_Churn.csv", "customer_churn.png")
    plot_dashboard()
    print("Done.")


if __name__ == "__main__":
    main()