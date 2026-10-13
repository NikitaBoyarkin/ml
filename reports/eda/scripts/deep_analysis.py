"""Deep EDA checks for the ml project (EDA fan-out, wave 2, slug=ml).

Reads the primary CSVs (Customer_Churn, Housing_Data, Social_Network_Ads) and the
remaining set, prints a structured text dump. Read-only: touches nothing but stdout.

Run:
  /Users/nikitaboarkin/Desktop/00 ide/00 portfolio/.eda-tools/.venv/bin/python \
      reports/eda/scripts/deep_analysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[3] / "data"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def sec(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def iqr_outliers(s):
    s = s.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < lo) | (s > hi)).sum()), lo, hi


# ---------------------------------------------------------------------------
sec("1. Customer_Churn.csv — classification set (n=800)")
ch = pd.read_csv(DATA / "Customer_Churn.csv")
print(
    "shape:",
    ch.shape,
    "| full-row dupes:",
    int(ch.duplicated().sum()),
    "| CustomerID dupes:",
    int(ch["CustomerID"].duplicated().sum()),
)
print(
    "\nChurn base rate:",
    ch["Churn"].mean().round(4),
    "| churners:",
    int(ch["Churn"].sum()),
)
print("\nChurn rate by Contract:")
g = ch.groupby("Contract")["Churn"].agg(["count", "mean", "sum"])
g["share_of_churners"] = g["sum"] / ch["Churn"].sum()
print(g.round(4))
print("\nChurn rate by InternetService:")
print(ch.groupby("InternetService")["Churn"].agg(["count", "mean", "sum"]).round(4))
print("\nChurn rate by Gender:")
print(ch.groupby("Gender")["Churn"].agg(["count", "mean", "sum"]).round(4))

# tenure distribution by contract (Simpson check for the contract gap)
print("\nTenureMonths by Contract (mean/median):")
print(
    ch.groupby("Contract")["TenureMonths"]
    .agg(["mean", "median", "min", "max"])
    .round(2)
)

# contract mix by gender (Simpson check for the gender gap)
print("\nContract mix by Gender (row %):")
print(pd.crosstab(ch["Gender"], ch["Contract"], normalize="index").round(4))

# churn by tenure bucket and monthly-charge bucket
ch["tenure_bucket"] = pd.cut(
    ch["TenureMonths"],
    [-1, 6, 12, 24, 48, 200],
    labels=["0-6", "7-12", "13-24", "25-48", "49+"],
)
print("\nChurn by tenure bucket:")
print(
    ch.groupby("tenure_bucket", observed=True)["Churn"].agg(["count", "mean"]).round(4)
)
ch["charge_bucket"] = pd.qcut(ch["MonthlyCharges"], 4, duplicates="drop")
print("\nChurn by MonthlyCharges quartile:")
print(
    ch.groupby("charge_bucket", observed=True)["Churn"].agg(["count", "mean"]).round(4)
)

# TotalCharges identity check: generator sets total = monthly * (tenure + 1)
print("\nTotalCharges vs MonthlyCharges*(TenureMonths+1):")
pred = ch["MonthlyCharges"] * (ch["TenureMonths"] + 1)
print("  exact match share:", round(float((ch["TotalCharges"] == pred).mean()), 4))
print("  corr(TotalCharges, pred):", round(float(ch["TotalCharges"].corr(pred)), 6))
print(
    "  corr(MonthlyCharges, TotalCharges):",
    round(float(ch["MonthlyCharges"].corr(ch["TotalCharges"])), 4),
)

# Age vs churn
print("\ncorr(Age, Churn):", round(float(ch["Age"].corr(ch["Churn"])), 4))
ch["age_bucket"] = pd.cut(
    ch["Age"], [17, 30, 45, 60, 75], labels=["18-30", "31-45", "46-60", "61-75"]
)
print(ch.groupby("age_bucket", observed=True)["Churn"].agg(["count", "mean"]).round(4))

for c in ["Age", "TenureMonths", "MonthlyCharges", "TotalCharges"]:
    n, lo, hi = iqr_outliers(ch[c])
    print(f"  IQR outliers {c}: {n} ({n / len(ch):.1%}), bounds [{lo:.1f}, {hi:.1f}]")

# ---------------------------------------------------------------------------
sec("2. Housing_Data.csv — multiple regression set (n=500)")
hs = pd.read_csv(DATA / "Housing_Data.csv")
print(
    "shape:",
    hs.shape,
    "| dupes:",
    int(hs.duplicated().sum()),
    "| Id dupes:",
    int(hs["Id"].duplicated().sum()),
)
print("Price range:", hs["Price"].min(), "->", hs["Price"].max())
print("rows at price floor (50000):", int((hs["Price"] == 50000).sum()))
print("\nPrice by State:")
print(hs.groupby("State")["Price"].agg(["count", "mean", "median", "std"]).round(2))
print("\ncorr matrix (numeric):")
print(hs[["SquareFeet", "Bedrooms", "HouseAge", "Price"]].corr().round(3))

# reproduce generator linear form: price = 56000 + sqft*180 - age*2200 + beds*14000 + state offset
offsets = {"New York": 38000, "California": 32000, "Florida": -14000}
base = 56000 + hs["SquareFeet"] * 180 - hs["HouseAge"] * 2200 + hs["Bedrooms"] * 14000
base = base + hs["State"].map(offsets)
resid = hs["Price"] - base
print("\nGenerator-form residuals (price - structural):")
print(
    "  mean:",
    round(resid.mean(), 2),
    "std:",
    round(resid.std(), 2),
    "(generator noise sigma=26000)",
)
print(
    "  share of variance explained by structural form:",
    round(1 - resid.var() / hs["Price"].var(), 4),
)
print("  corr(Price, structural):", round(float(hs["Price"].corr(base)), 4))

# ---------------------------------------------------------------------------
sec("3. Social_Network_Ads.csv — classification set (n=400)")
sn = pd.read_csv(DATA / "Social_Network_Ads.csv")
print(
    "shape:",
    sn.shape,
    "| dupes:",
    int(sn.duplicated().sum()),
    "| UserID dupes:",
    int(sn["User ID"].duplicated().sum()),
)
print(
    "Purchased rate:",
    round(float(sn["Purchased"].mean()), 4),
    "| purchasers:",
    int(sn["Purchased"].sum()),
)
print("\ncorr with Purchased:")
print(sn[["Age", "EstimatedSalary", "Purchased"]].corr()["Purchased"].round(4))
sn["age_bucket"] = pd.cut(
    sn["Age"], [17, 25, 35, 45, 60], labels=["18-25", "26-35", "36-45", "46-60"]
)
print("\nPurchased by age bucket:")
print(
    sn.groupby("age_bucket", observed=True)["Purchased"].agg(["count", "mean"]).round(4)
)
print("\nPurchased by Gender:")
print(sn.groupby("Gender")["Purchased"].agg(["count", "mean", "sum"]).round(4))
print("\nAge x Gender interaction (mean Purchased):")
print(
    sn.pivot_table(
        index="age_bucket",
        columns="Gender",
        values="Purchased",
        aggfunc="mean",
        observed=True,
    ).round(4)
)

# ---------------------------------------------------------------------------
sec("4. Remaining CSVs — noted as a set")
rest = [
    "Data.csv",
    "Data_large.csv",
    "Salary_Data.csv",
    "Salary_Data_large.csv",
    "50_Startups.csv",
    "Position_Salaries.csv",
    "Mall_Customers.csv",
]
for f in rest:
    d = pd.read_csv(DATA / f)
    miss = d.isna().sum().sum()
    print(
        f"{f:26s} rows={len(d):4d} cols={d.shape[1]:2d} dupes={int(d.duplicated().sum()):3d} "
        f"missing={miss:3d} | cols: {list(d.columns)}"
    )

print("\nData.csv (n=9) missing detail:")
d0 = pd.read_csv(DATA / "Data.csv")
print(d0.isna().sum()[d0.isna().sum() > 0].to_dict())
# reproduce deterministic_split: every 5th row (1-indexed) after stable sort by (Country, Age)
srt = d0.sort_values(["Country", "Age"], kind="stable").reset_index(drop=True)
mask = (np.arange(len(srt)) + 1) % 5 == 0
print(
    "deterministic split -> train rows:",
    int((~mask).sum()),
    "test rows:",
    int(mask.sum()),
    "(numeric 2 + one-hot Country 3 = 5 cols)",
)
print("test rows (Country,Age):")
print(srt[mask][["Country", "Age"]].to_string(index=False))

print(
    "\nSalary_Data_large floor check (generator floor=8000):",
    int((pd.read_csv(DATA / "Salary_Data_large.csv")["Salary"] <= 8000).sum()),
    "rows at/below floor",
)
print(
    "Salary_Data_large corr(YearsExperience, Salary):",
    round(float(pd.read_csv(DATA / "Salary_Data_large.csv").corr().iloc[0, 1]), 4),
)
print(
    "50_Startups corr(R&D Spend, Profit):",
    round(
        float(
            pd.read_csv(DATA / "50_Startups.csv")
            .corr(numeric_only=True)
            .loc["R&D Spend", "Profit"]
        ),
        4,
    ),
)
print(
    "Mall_Customers Income vs Spending corr:",
    round(
        float(
            pd.read_csv(DATA / "Mall_Customers.csv")
            .corr(numeric_only=True)
            .loc["Annual Income (k$)", "Spending Score (1-100)"]
        ),
        4,
    ),
)
print("\nDONE")
