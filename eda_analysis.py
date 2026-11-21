from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "eda_charts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "cleaned_hr_data.csv"

df = pd.read_csv(INPUT_FILE)

# --------------------------------------------------
# 2. VALIDATE REQUIRED COLUMNS
# --------------------------------------------------
required_columns = [
    "Attrition", "Department", "JobRole", "JobLevel",
    "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager",
    "TrainingTimesLastYear", "TotalWorkingYears"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

print("=" * 60)
print("CAREER PROGRESSION AND PROMOTION GAP ANALYSIS")
print("=" * 60)

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print(f"Missing cells: {df.isna().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# --------------------------------------------------
# 3. NORMALIZE ATTRITION LABELS
# --------------------------------------------------
# Accept either numeric 0/1 or text Yes/No.
if df["Attrition"].dtype == "object":
    attrition_map = {
        "yes": 1, "no": 0,
        "1": 1, "0": 0
    }
    df["Attrition"] = (
        df["Attrition"].astype(str).str.strip().str.lower()
        .map(attrition_map)
    )

if df["Attrition"].isna().any():
    raise ValueError("Attrition contains invalid or missing labels.")

if not df["Attrition"].isin([0, 1]).all():
    raise ValueError("Attrition must contain only 0 and 1.")

# 0 = stayed, 1 = left (verify this against your data source)
df["AttritionLabel"] = df["Attrition"].map({
    0: "Stayed",
    1: "Left"
})

# --------------------------------------------------
# 4. WORKFORCE OVERVIEW
# --------------------------------------------------
total_employees = len(df)
employees_left = int((df["Attrition"] == 1).sum())
employees_stayed = int((df["Attrition"] == 0).sum())
attrition_rate = df["Attrition"].mean() * 100

print("\nWORKFORCE OVERVIEW")
print("-" * 40)
print(f"Total records: {total_employees}")
print(f"Stayed: {employees_stayed}")
print(f"Left: {employees_left}")
print(f"Observed attrition rate: {attrition_rate:.2f}%")

# --------------------------------------------------
# 5. CAREER PROGRESSION SUMMARY
# --------------------------------------------------
career_columns = [
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "TrainingTimesLastYear",
    "JobLevel"
]

career_summary = df[career_columns].describe().T
career_summary.to_csv(
    OUTPUT_DIR / "career_feature_summary.csv"
)

print("\nCAREER FEATURE SUMMARY")
print(career_summary.round(2))

# --------------------------------------------------
# 6. COMPARE EMPLOYEES WHO STAYED VS LEFT
# --------------------------------------------------
group_summary = (
    df.groupby("AttritionLabel")[career_columns]
    .mean()
    .T
)

group_summary.to_csv(
    OUTPUT_DIR / "career_by_attrition.csv"
)

print("\nCAREER METRICS BY ATTRITION")
print(group_summary.round(2))

# --------------------------------------------------
# 7. DEPARTMENT ANALYSIS
# --------------------------------------------------
department_summary = (
    df.groupby("Department")
    .agg(
        EmployeeCount=("Attrition", "size"),
        EmployeesLeft=("Attrition", "sum"),
        AttritionRate=("Attrition", "mean"),
        AvgCompanyTenure=("YearsAtCompany", "mean"),
        AvgRoleTenure=("YearsInCurrentRole", "mean"),
        AvgPromotionGap=("YearsSinceLastPromotion", "mean"),
        AvgTraining=("TrainingTimesLastYear", "mean")
    )
    .reset_index()
)

department_summary["AttritionRate"] *= 100

department_summary.to_csv(
    OUTPUT_DIR / "department_summary.csv", index=False
)

print("\nDEPARTMENT SUMMARY")
print(department_summary.round(2).to_string(index=False))

# --------------------------------------------------
# 8. JOB ROLE ANALYSIS
# --------------------------------------------------
role_summary = (
    df.groupby("JobRole")
    .agg(
        EmployeeCount=("Attrition", "size"),
        EmployeesLeft=("Attrition", "sum"),
        AttritionRate=("Attrition", "mean"),
        AvgCompanyTenure=("YearsAtCompany", "mean"),
        AvgRoleTenure=("YearsInCurrentRole", "mean"),
        AvgPromotionGap=("YearsSinceLastPromotion", "mean"),
        AvgManagerTenure=("YearsWithCurrManager", "mean"),
        AvgTraining=("TrainingTimesLastYear", "mean")
    )
    .reset_index()
)

role_summary["AttritionRate"] *= 100

role_summary.to_csv(
    OUTPUT_DIR / "job_role_summary.csv", index=False
)

print("\nJOB ROLE SUMMARY")
print(
    role_summary.sort_values(
        "AttritionRate", ascending=False
    ).round(2).to_string(index=False)
)

# --------------------------------------------------
# 9. DEFINE A DESCRIPTIVE PROMOTION-GAP GROUP
# --------------------------------------------------
# This is an exploratory threshold, not a validated risk score.
df["PromotionGapGroup"] = pd.cut(
    df["YearsSinceLastPromotion"],
    bins=[-np.inf, 1, 4, np.inf],
    labels=["0-1 years", "2-4 years", "5+ years"]
)

promotion_summary = (
    df.groupby("PromotionGapGroup", observed=False)
    .agg(
        EmployeeCount=("Attrition", "size"),
        EmployeesLeft=("Attrition", "sum"),
        AttritionRate=("Attrition", "mean"),
        AvgRoleTenure=("YearsInCurrentRole", "mean"),
        AvgCompanyTenure=("YearsAtCompany", "mean")
    )
    .reset_index()
)

promotion_summary["AttritionRate"] *= 100

promotion_summary.to_csv(
    OUTPUT_DIR / "promotion_gap_summary.csv", index=False
)

print("\nPROMOTION GAP GROUPS")
print(promotion_summary.round(2).to_string(index=False))

# --------------------------------------------------
# 10. CHART 1: DEPARTMENT EMPLOYEE COUNTS
# --------------------------------------------------
sns.set_theme(style="whitegrid")

plt.figure(figsize=(9, 5))
sns.countplot(
    data=df, x="Department",
    order=df["Department"].value_counts().index
)
plt.title("Employee Records by Department")
plt.xlabel("Department")
plt.ylabel("Number of Records")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(
    CHART_DIR / "01_department_distribution.png",
    dpi=150
)
plt.close()

# --------------------------------------------------
# 11. CHART 2: YEARS SINCE LAST PROMOTION
# --------------------------------------------------
plt.figure(figsize=(9, 5))
sns.histplot(
    data=df,
    x="YearsSinceLastPromotion",
    bins=16,
    kde=False
)
plt.title("Distribution of Years Since Last Promotion")
plt.xlabel("Years Since Last Promotion")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig(
    CHART_DIR / "02_promotion_gap_distribution.png",
    dpi=150
)
plt.close()

# --------------------------------------------------
# 12. CHART 3: CAREER TENURE BY ATTRITION
# --------------------------------------------------
plot_data = df.melt(
    id_vars=["AttritionLabel"],
    value_vars=[
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsWithCurrManager"
    ],
    var_name="CareerMeasure",
    value_name="Years"
)

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=plot_data,
    x="CareerMeasure",
    y="Years",
    hue="AttritionLabel"
)
plt.title("Career Tenure Measures by Attrition Group")
plt.xlabel("Career Measure")
plt.ylabel("Years")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(
    CHART_DIR / "03_career_tenure_by_attrition.png",
    dpi=150
)
plt.close()

# --------------------------------------------------
# 13. CHART 4: ATTRITION RATE BY JOB ROLE
# --------------------------------------------------
role_plot = role_summary.sort_values("AttritionRate")

plt.figure(figsize=(10, 7))
sns.barplot(
    data=role_plot,
    x="AttritionRate",
    y="JobRole"
)
plt.title("Observed Attrition Rate by Job Role")
plt.xlabel("Employees Who Left (%)")
plt.ylabel("Job Role")
plt.tight_layout()
plt.savefig(
    CHART_DIR / "04_attrition_by_job_role.png",
    dpi=150
)
plt.close()

# --------------------------------------------------
# 14. FINAL OUTPUTS
# --------------------------------------------------
print("\nEDA COMPLETED SUCCESSFULLY")
print(f"Summary tables: {OUTPUT_DIR}")
print(f"Charts: {CHART_DIR}")
print("Review the outputs before drawing conclusions.")