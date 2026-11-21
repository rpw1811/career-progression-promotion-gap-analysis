
from pathlib import Path
import pandas as pd
import numpy as np

# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

CLUSTER_DIR = OUTPUT_DIR / "clustering"
PHASE5_DIR = OUTPUT_DIR / "retention_analysis"

PHASE5_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = (
    CLUSTER_DIR / "employee_cluster_assignments.csv"
)

df = pd.read_csv(INPUT_FILE)

print("=" * 65)
print("PHASE 5: RETENTION OPPORTUNITY ANALYSIS")
print("=" * 65)

# ==================================================
# 2. REQUIRED COLUMNS
# ==================================================

required = [
    "KMeansCluster",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "TrainingTimesLastYear"
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

numeric_cols = [
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "TrainingTimesLastYear"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ==================================================
# 3. VALIDATE CAREER DATA
# ==================================================

invalid = (
    (df["YearsAtCompany"] < 0)
    | (df["YearsInCurrentRole"] < 0)
    | (df["YearsSinceLastPromotion"] < 0)
    | (df["YearsWithCurrManager"] < 0)
    | (df["TrainingTimesLastYear"] < 0)
    | (df["YearsInCurrentRole"] > df["YearsAtCompany"])
    | (df["YearsSinceLastPromotion"] > df["YearsAtCompany"])
    | (df["YearsWithCurrManager"] > df["YearsAtCompany"])
)

invalid = invalid.fillna(True)

print("Records received:", len(df))
print("Invalid records flagged:", int(invalid.sum()))

# Do not silently score invalid records.
df["CareerDataValid"] = ~invalid

# ==================================================
# 4. PROMOTION GAP ANALYSIS
# ==================================================

df["PromotionGapRatio"] = (
    df["YearsSinceLastPromotion"]
    / df["YearsAtCompany"].replace(0, np.nan)
)

df["PromotionGapCategory"] = pd.cut(
    df["YearsSinceLastPromotion"],
    bins=[-np.inf, 1, 4, np.inf],
    labels=["Low", "Medium", "High"]
)

promotion_score_map = {
    "Low": 0,
    "Medium": 50,
    "High": 100
}

df["PromotionGapScore"] = (
    df["PromotionGapCategory"]
    .map(promotion_score_map)
    .astype(float)
)

# ==================================================
# 5. ROLE STAGNATION ANALYSIS
# ==================================================

df["RoleStagnationIndex"] = (
    df["YearsInCurrentRole"]
    / df["YearsAtCompany"].replace(0, np.nan)
)

def classify_stagnation(value):

    if pd.isna(value):
        return "Not Assessable"

    if value < 0.25:
        return "Low"

    elif value < 0.50:
        return "Moderate"

    return "High"

df["RoleStagnationCategory"] = (
    df["RoleStagnationIndex"]
    .apply(classify_stagnation)
)

df["RoleStagnationScore"] = (
    df["RoleStagnationCategory"]
    .map({
        "Low": 0,
        "Moderate": 50,
        "High": 100
    })
)

# ==================================================
# 6. TRAINING NEED ANALYSIS
# ==================================================

training_threshold = (
    df.loc[df["CareerDataValid"],
           "TrainingTimesLastYear"]
    .quantile(0.25)
)

df["TrainingNeedIndicator"] = np.where(
    df["TrainingTimesLastYear"].isna(),
    np.nan,
    np.where(
        df["TrainingTimesLastYear"] <= training_threshold,
        1,
        0
    )
)

df["TrainingNeedScore"] = (
    df["TrainingNeedIndicator"] * 100
)

print("\nTraining participation threshold:")
print(training_threshold)

# ==================================================
# 7. MANAGER STABILITY INSIGHTS
# ==================================================

df["ManagerTenureRatio"] = (
    df["YearsWithCurrManager"]
    / df["YearsAtCompany"].replace(0, np.nan)
)

def manager_tenure_category(value):

    if pd.isna(value):
        return "Not Assessable"

    if value < 1:
        return "Under 1 year"

    elif value < 5:
        return "1-4 years"

    return "5+ years"

df["ManagerTenureCategory"] = (
    df["YearsWithCurrManager"]
    .apply(manager_tenure_category)
)

# A high ratio means a large share of company tenure
# has been spent with the current manager.
# It does NOT establish manager effectiveness.

# ==================================================
# 8. RETENTION OPPORTUNITY INDEX
# ==================================================

weights = {
    "PromotionGapScore": 0.40,
    "RoleStagnationScore": 0.35,
    "TrainingNeedScore": 0.25
}

score_columns = list(weights.keys())

# Require all three component scores and valid career data.
score_complete = (
    df["CareerDataValid"]
    & df[score_columns].notna().all(axis=1)
    & df["YearsAtCompany"].gt(0)
)

df["RetentionOpportunityIndex"] = np.nan

df.loc[score_complete, "RetentionOpportunityIndex"] = (
    sum(
        df.loc[score_complete, col] * weight
        for col, weight in weights.items()
    )
)

# ==================================================
# 9. INTERVENTION PRIORITY
# ==================================================

def priority_category(score):

    if pd.isna(score):
        return "Not Assessed"

    if score < 35:
        return "Low"

    elif score < 65:
        return "Moderate"

    return "High"

df["InterventionPriority"] = (
    df["RetentionOpportunityIndex"]
    .apply(priority_category)
)

# ==================================================
# 10. RECOMMENDED DEVELOPMENT ACTIONS
# ==================================================

def recommend_actions(row):

    if not row["CareerDataValid"]:
        return "Review data quality before intervention"

    actions = []

    if row["PromotionGapCategory"] == "High":
        actions.append("Career and promotion eligibility review")

    if row["RoleStagnationCategory"] == "High":
        actions.append("Internal role rotation discussion")

    if row["TrainingNeedIndicator"] == 1:
        actions.append("Training needs assessment")

    if row["ManagerTenureCategory"] == "5+ years":
        actions.append("Career discussion with current manager")

    if not actions:
        actions.append("Routine career development discussion")

    return "; ".join(actions)

df["SuggestedDevelopmentActions"] = (
    df.apply(recommend_actions, axis=1)
)

# ==================================================
# 11. CLUSTER-LEVEL SUMMARY
# ==================================================

cluster_summary = (
    df[df["CareerDataValid"]]
    .groupby("KMeansCluster")
    .agg(
        EmployeeCount=("KMeansCluster", "size"),
        AvgPromotionGap=("YearsSinceLastPromotion", "mean"),
        AvgRoleTenure=("YearsInCurrentRole", "mean"),
        AvgCompanyTenure=("YearsAtCompany", "mean"),
        AvgManagerTenure=("YearsWithCurrManager", "mean"),
        AvgTraining=("TrainingTimesLastYear", "mean"),
        AvgOpportunityIndex=(
            "RetentionOpportunityIndex", "mean"
        ),
        HighPriorityCount=(
            "InterventionPriority",
            lambda x: (x == "High").sum()
        )
    )
    .reset_index()
)

cluster_summary.to_csv(
    PHASE5_DIR / "cluster_retention_summary.csv",
    index=False
)

# ==================================================
# 12. DEPARTMENT SUMMARY
# ==================================================

if "Department" in df.columns:

    department_summary = (
        df[df["CareerDataValid"]]
        .groupby("Department")
        .agg(
            EmployeeCount=("Department", "size"),
            AvgOpportunityIndex=(
                "RetentionOpportunityIndex", "mean"
            ),
            HighPriorityCount=(
                "InterventionPriority",
                lambda x: (x == "High").sum()
            )
        )
        .reset_index()
    )

    department_summary.to_csv(
        PHASE5_DIR / "department_retention_summary.csv",
        index=False
    )

# ==================================================
# 13. MANAGERIAL INSIGHT SUMMARY
# ==================================================

manager_summary = (
    df[df["CareerDataValid"]]
    .groupby("ManagerTenureCategory")
    .agg(
        EmployeeCount=("ManagerTenureCategory", "size"),
        AvgRoleTenure=("YearsInCurrentRole", "mean"),
        AvgPromotionGap=("YearsSinceLastPromotion", "mean"),
        AvgOpportunityIndex=(
            "RetentionOpportunityIndex", "mean"
        )
    )
    .reset_index()
)

manager_summary.to_csv(
    PHASE5_DIR / "manager_stability_summary.csv",
    index=False
)

# ==================================================
# 14. PRIORITY DISTRIBUTION
# ==================================================

priority_summary = (
    df.groupby("InterventionPriority", dropna=False)
    .size()
    .reset_index(name="EmployeeCount")
)

priority_summary["Percentage"] = (
    priority_summary["EmployeeCount"]
    / len(df) * 100
)

priority_summary.to_csv(
    PHASE5_DIR / "intervention_priority_summary.csv",
    index=False
)

# ==================================================
# 15. ATTRITION COMPARISON (POST-ANALYSIS ONLY)
# ==================================================

if "Attrition" in df.columns:

    df["Attrition"] = pd.to_numeric(
        df["Attrition"], errors="coerce"
    )

    attrition_summary = (
        df[df["CareerDataValid"]]
        .groupby("InterventionPriority")
        .agg(
            EmployeeCount=("Attrition", "size"),
            ObservedAttritionRate=("Attrition", "mean")
        )
        .reset_index()
    )

    attrition_summary["ObservedAttritionRate"] *= 100

    attrition_summary.to_csv(
        PHASE5_DIR / "priority_attrition_comparison.csv",
        index=False
    )

# ==================================================
# 16. SAVE EMPLOYEE-LEVEL RESULTS
# ==================================================

df.to_csv(
    PHASE5_DIR / "employee_retention_opportunities.csv",
    index=False
)

df.loc[~df["CareerDataValid"]].to_csv(
    PHASE5_DIR / "records_requiring_review.csv",
    index=False
)

# ==================================================
# 17. FINAL REPORT
# ==================================================

print("\nPHASE 5 COMPLETED")
print("-" * 50)

print("Training threshold:", training_threshold)

print("\nIntervention priority distribution:")
print(priority_summary.to_string(index=False))

print("\nCluster retention summary:")
print(cluster_summary.round(2).to_string(index=False))

print("\nOutput folder:", PHASE5_DIR)
