"""Figures for the ml EDA report (EDA fan-out, wave 2, slug=ml).

Light theme, OO matplotlib, 300 dpi PNG. Reads project data read-only, writes only
into reports/eda/figures/.

Run:
  /Users/nikitaboarkin/Desktop/00 ide/00 portfolio/.eda-tools/.venv/bin/python \
      reports/eda/scripts/plot_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[3] / "data"
OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans", "sans-serif"],
        "axes.unicode_minus": False,
        "figure.dpi": 110,
    }
)

C = ["#2c6fbb", "#e07b39", "#4c9a5a", "#b04a4a", "#7a5aa6"]


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------------------
ch = pd.read_csv(DATA / "Customer_Churn.csv")
hs = pd.read_csv(DATA / "Housing_Data.csv")
sn = pd.read_csv(DATA / "Social_Network_Ads.csv")

# Fig 1 — churn rate by Contract and InternetService (lead finding)
contract = ch.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
internet = ch.groupby("InternetService")["Churn"].mean().sort_values(ascending=False)
base = ch["Churn"].mean()
fig, axs = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
for ax, s, title in [
    (axs[0], contract, "Churn rate by Contract"),
    (axs[1], internet, "Churn rate by InternetService"),
]:
    bars = ax.bar(s.index, s.values, color=C[0], edgecolor="black", linewidth=0.6)
    ax.axhline(
        base, color=C[3], linestyle="--", linewidth=1.5, label=f"overall = {base:.1%}"
    )
    ax.bar_label(bars, fmt="%.1f%%", labels=[f"{v:.1%}" for v in s.values], padding=2)
    ax.set(xlabel="", ylabel="Churn rate", title=title, ylim=(0, s.max() * 1.25))
    ax.legend(frameon=False, loc="upper right")
fig.suptitle(
    "Customer churn: contract type dominates, InternetService barely moves it",
    fontsize=13,
    fontweight="bold",
)
save(fig, "fig1_churn_by_segment.png")

# Fig 2 — churn rate by tenure bucket (behaviour), with volume on twin axis
ch["tenure_bucket"] = pd.cut(
    ch["TenureMonths"],
    [-1, 6, 12, 24, 48, 200],
    labels=["0-6", "7-12", "13-24", "25-48", "49+"],
)
tb = ch.groupby("tenure_bucket", observed=True)["Churn"].agg(["mean", "count"])
fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
ax.plot(
    tb.index.astype(str),
    tb["mean"],
    marker="o",
    color=C[0],
    linewidth=2,
    label="churn rate",
)
for x, (m, c) in enumerate(zip(tb["mean"], tb["count"])):
    ax.annotate(
        f"{m:.1%}\n(n={c})",
        (x, m),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=8,
    )
ax.set(
    xlabel="Tenure bucket (months)",
    ylabel="Churn rate",
    title="Churn falls monotonically with tenure (27.7% -> 1.1%)",
    ylim=(0, tb["mean"].max() * 1.35),
)
ax.legend(frameon=False)
save(fig, "fig2_churn_by_tenure.png")

# Fig 3 — Housing Price vs SquareFeet by State (regression signal)
fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
for i, state in enumerate(sorted(hs["State"].unique())):
    d = hs[hs["State"] == state]
    ax.scatter(d["SquareFeet"], d["Price"], s=18, alpha=0.65, color=C[i], label=state)
z = np.polyfit(hs["SquareFeet"], hs["Price"], 1)
xs = np.linspace(hs["SquareFeet"].min(), hs["SquareFeet"].max(), 50)
ax.plot(
    xs,
    np.polyval(z, xs),
    color="black",
    linewidth=1.8,
    linestyle="--",
    label=f"OLS (r={hs['SquareFeet'].corr(hs['Price']):.3f})",
)
ax.set(
    xlabel="Square feet",
    ylabel="Price ($)",
    title="Housing: Price is ~linear in SquareFeet; State shifts the level",
)
ax.legend(frameon=False)
save(fig, "fig3_price_vs_sqft.png")

# Fig 4 — Social Network Ads: Purchased rate by age bucket x gender
sn["age_bucket"] = pd.cut(
    sn["Age"], [17, 25, 35, 45, 60], labels=["18-25", "26-35", "36-45", "46-60"]
)
piv = sn.pivot_table(
    index="age_bucket",
    columns="Gender",
    values="Purchased",
    aggfunc="mean",
    observed=True,
)
fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
x = np.arange(len(piv))
w = 0.38
for i, g in enumerate(piv.columns):
    b = ax.bar(
        x + (i - 0.5) * w,
        piv[g],
        w,
        label=g,
        color=C[i],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar_label(
        b, fmt="%.0f%%", labels=[f"{v:.0%}" for v in piv[g]], padding=2, fontsize=8
    )
ax.set_xticks(x, piv.index.astype(str))
ax.set(
    xlabel="Age bucket",
    ylabel="Purchase rate",
    title="Social Network Ads: purchase is driven by Age, not Gender (0% -> 85%)",
)
ax.legend(frameon=False)
save(fig, "fig4_purchased_by_age.png")

# Fig 5 — numeric distributions (2x2), TotalCharges on log scale
fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
axs[0, 0].hist(ch["Age"], bins=25, color=C[0], edgecolor="black", linewidth=0.4)
axs[0, 0].set(xlabel="Age", ylabel="count", title="Customer Age (churn set)")
axs[0, 1].hist(
    ch["MonthlyCharges"], bins=25, color=C[1], edgecolor="black", linewidth=0.4
)
axs[0, 1].set(
    xlabel="MonthlyCharges", ylabel="count", title="MonthlyCharges (churn set)"
)
axs[1, 0].hist(
    ch["TotalCharges"], bins=30, color=C[2], edgecolor="black", linewidth=0.4
)
axs[1, 0].set(
    xlabel="TotalCharges (log scale)", ylabel="count", title="TotalCharges (churn set)"
)
axs[1, 0].set_xscale("log")
axs[1, 1].hist(hs["Price"], bins=30, color=C[4], edgecolor="black", linewidth=0.4)
axs[1, 1].set(xlabel="Price ($)", ylabel="count", title="Housing Price")
fig.suptitle(
    "Primary numeric distributions — near-symmetric; TotalCharges is right-skewed",
    fontsize=13,
    fontweight="bold",
)
save(fig, "fig5_distributions.png")

# Fig 6 — dataset overview: rows per CSV + missing cells
files = [
    "Customer_Churn.csv",
    "Housing_Data.csv",
    "Data_large.csv",
    "Social_Network_Ads.csv",
    "Salary_Data_large.csv",
    "Mall_Customers.csv",
    "50_Startups.csv",
    "Salary_Data.csv",
    "Position_Salaries.csv",
    "Data.csv",
]
rows, miss = [], []
for f in files:
    d = pd.read_csv(DATA / f)
    rows.append(len(d))
    miss.append(int(d.isna().sum().sum()))
fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
y = np.arange(len(files))
b = ax.barh(y, rows, color=C[0], edgecolor="black", linewidth=0.5)
ax.bar_label(
    b,
    labels=[f"{r}" + (f"  ({m} missing)" if m else "") for r, m in zip(rows, miss)],
    padding=3,
    fontsize=8,
)
ax.set_yticks(y, files)
ax.invert_yaxis()
ax.set(
    xlabel="Rows",
    ylabel="",
    title="Dataset inventory — 10 CSVs; only Data*.csv carry missing values",
)
save(fig, "fig6_dataset_inventory.png")

print("DONE")
