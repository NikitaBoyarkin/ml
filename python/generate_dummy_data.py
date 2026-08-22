"""Dummy dataset generator for the ML portfolio.

Produces realistic, deterministic sample datasets for every section of the
ML A-Z pipeline, matching the column conventions of the existing course data:

    data/Data_large.csv          extended preprocessing set (Data.csv pattern)
    data/Salary_Data_large.csv   extended simple-regression set
    data/Housing_Data.csv        multiple-regression set (50_Startups pattern)
    data/Housing_Data.sql        same data as INSERT statements
    data/Customer_Churn.csv      classification set (Social_Network_Ads pattern)
    data/Customer_Churn.json     same data as JSON

Run from python/:  python3 generate_dummy_data.py
"""

import csv
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ROWS = {
    "preprocessing": 400,
    "salary": 300,
    "housing": 500,
    "churn": 800,
}

rng = random.Random(42)

COUNTRIES = ["France", "Spain", "Germany"]
GENDERS = ["Male", "Female"]
STATES = ["New York", "California", "Florida"]
CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["DSL", "Fiber optic", "No"]
STREETS = [
    "Maple", "Oak", "Cedar", "Birch", "Elm", "Pine", "Spruce", "Willow",
    "Juniper", "Aspen",
]

FIRST_NAMES = [
    "James", "Maria", "Lucas", "Emma", "Noah", "Olivia", "Ethan", "Sofia",
    "Liam", "Ava", "Mateo", "Elena", "Diego", "Mia", "Hugo", "Luna",
]
LAST_NAMES = [
    "Smith", "Garcia", "Rossi", "Muller", "Lopez", "Martin", "Klein",
    "Schmidt", "Ferrari", "Wagner", "Sanz", "Bianchi", "Weber", "Silva",
]
EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "company.com"]

POSITIONS = [
    "Business Analyst", "Junior Consultant", "Senior Consultant",
    "Manager", "Principal Manager", "Partner", "Junior Executive",
    "Senior Executive", "CEO", "Junior Advisor", "Senior Advisor",
]


def weighted(pairs):
    """Return a random value according to (value, weight) pairs."""
    total = sum(w for _, w in pairs)
    roll = rng.random() * total
    for value, weight in pairs:
        roll -= weight
        if roll <= 0:
            return value
    return pairs[-1][0]


def first_last_name():
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def email_for(name):
    user = name.lower().replace(" ", ".").replace("'", "")
    return f"{user}{rng.randint(1, 99)}@{rng.choice(EMAIL_DOMAINS)}"


def write_csv(path, columns, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {len(rows)} rows -> {path}")


# ---------------------------------------------------------------------------
# 1. Preprocessing set (Data.csv pattern) — Country, Age, Salary, Purchased.
#    ~8% of Age/Salary left blank on purpose so imputation has work to do.
# ---------------------------------------------------------------------------
def generate_preprocessing(n):
    rows = []
    for i in range(1, n + 1):
        country = rng.choice(COUNTRIES)
        age = rng.randint(18, 65) if rng.random() > 0.08 else ""
        salary = int(rng.gauss(64000, 16000)) if rng.random() > 0.08 else ""
        purchased = "Yes" if rng.random() < 0.62 else "No"
        rows.append({
            "Country": country,
            "Age": age,
            "Salary": salary,
            "Purchased": purchased,
        })
    return rows


# ---------------------------------------------------------------------------
# 2. Simple linear regression set (Salary_Data.csv pattern).
#    salary ~ 20000 + 10000 * years, plus realistic noise. The floor is a
#    guard against negative outliers, not a clamp that flattens the curve.
# ---------------------------------------------------------------------------
def generate_salary(n):
    rows = []
    for _ in range(n):
        years = round(rng.uniform(0.5, 12.0), 1)
        salary = 20000 + 10000 * years + rng.gauss(0, 4000)
        rows.append({
            "YearsExperience": years,
            "Salary": round(max(8000, salary), 2),
        })
    rows.sort(key=lambda r: r["YearsExperience"])
    return rows


# ---------------------------------------------------------------------------
# 3. Multiple regression set (50_Startups pattern) — housing prices.
#    price = base + sqft * 180 - age * 2200 + beds * 14000, state offset.
# ---------------------------------------------------------------------------
def generate_housing(n):
    state_offsets = {"New York": 38000, "California": 32000, "Florida": -14000}
    rows = []
    for i in range(1, n + 1):
        sqft = int(rng.gauss(1900, 620))
        bedrooms = min(7, max(1, int(1 + (sqft - 600) / 420 + rng.gauss(0, 0.6))))
        age = max(0, int(rng.gauss(18, 14)))
        state = rng.choice(STATES)
        price = (56000 + sqft * 180 - age * 2200 + bedrooms * 14000
                 + state_offsets[state] + rng.gauss(0, 26000))
        rows.append({
            "Id": f"H{i:04d}",
            "SquareFeet": sqft,
            "Bedrooms": bedrooms,
            "HouseAge": age,
            "State": state,
            "Price": round(max(50000, price), 2),
        })
    return rows


# ---------------------------------------------------------------------------
# 4. Classification set (Social_Network_Ads pattern) — customer churn.
#    churn driven by tenure and contract type, no purchase-history leakage.
# ---------------------------------------------------------------------------
def generate_churn(n):
    rows = []
    for i in range(1, n + 1):
        gender = rng.choice(GENDERS)
        age = rng.randint(18, 75)
        tenure = int(max(0, rng.gauss(24, 20)))
        contract = weighted([
            ("Month-to-month", 0.52),
            ("One year", 0.28),
            ("Two year", 0.20),
        ])
        internet = weighted([("Fiber optic", 0.42), ("DSL", 0.33), ("No", 0.25)])
        monthly = int(rng.gauss(72, 18))
        total = monthly * (tenure + 1)

        base = 0.08
        if contract == "Month-to-month":
            base += 0.40
        elif contract == "One year":
            base -= 0.12
        else:
            base -= 0.30
        base -= tenure * 0.012
        base += max(0, (age - 45)) * 0.004
        churn = int(rng.random() < min(0.9, max(0.01, base)))
        rows.append({
            "CustomerID": i,
            "Gender": gender,
            "Age": age,
            "TenureMonths": tenure,
            "Contract": contract,
            "InternetService": internet,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": churn,
        })
    return rows


def save_sql(rows, path, table="housing"):
    with open(path, "w") as f:
        f.write(f"CREATE TABLE {table} (\n")
        f.write("    Id TEXT,\n")
        f.write("    SquareFeet INTEGER,\n")
        f.write("    Bedrooms INTEGER,\n")
        f.write("    HouseAge INTEGER,\n")
        f.write("    State TEXT,\n")
        f.write("    Price REAL\n);\n\n")
        for r in rows:
            f.write(f"INSERT INTO {table} VALUES ("
                    f"'{r['Id']}', {r['SquareFeet']}, {r['Bedrooms']}, "
                    f"{r['HouseAge']}, '{r['State']}', {r['Price']});\n")
    print(f"  wrote {len(rows)} INSERTs -> {path}")


def save_json(rows, path):
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  wrote {len(rows)} records -> {path}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Preprocessing (Data.csv pattern)")
    write_csv(DATA_DIR / "Data_large.csv", ["Country", "Age", "Salary", "Purchased"],
              generate_preprocessing(ROWS["preprocessing"]))

    print("2. Simple linear regression (Salary_Data.csv pattern)")
    write_csv(DATA_DIR / "Salary_Data_large.csv", ["YearsExperience", "Salary"],
              generate_salary(ROWS["salary"]))

    print("3. Multiple regression (housing)")
    housing = generate_housing(ROWS["housing"])
    write_csv(DATA_DIR / "Housing_Data.csv",
              ["Id", "SquareFeet", "Bedrooms", "HouseAge", "State", "Price"], housing)
    save_sql(housing, DATA_DIR / "Housing_Data.sql")

    print("4. Classification (customer churn)")
    churn = generate_churn(ROWS["churn"])
    write_csv(DATA_DIR / "Customer_Churn.csv", list(churn[0].keys()), churn)
    save_json(churn, DATA_DIR / "Customer_Churn.json")

    print("Done.")


if __name__ == "__main__":
    main()