from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_FILE = DATA_DIR / "cleaned_hr_data.csv"
OUTPUT_FILE = DATA_DIR / "career_features.csv"
SUMMARY_FILE = OUTPUT_DIR / "career_feature_summary.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load cleaned data
df = pd.read_csv(INPUT_FILE)

required = [
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "TrainingTimesLastYear",
    "JobLevel",
    "TotalWorkingYears",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# 2. Ensure career measures are numeric
for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 3. Check invalid career-tenure records
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

print("Invalid career records flagged:", int(invalid.sum()))

# Do not silently delete records.
# Review invalid rows before using them in modelling.

# 4. Ratios; undefined when company tenure is zero
tenure = df["YearsAtCompany"].replace(0, np.nan)

df["PromotionGapRatio"] = (
    df["YearsSinceLastPromotion"] / tenure
)

df["RoleStagnationIndex"] = (
    df["YearsInCurrentRole"] / tenure
)

df["TrainingIntensityScore"] = (
    df["TrainingTimesLastYear"] / tenure
)

df["ManagerTenureRatio"] = (
    df["YearsWithCurrManager"] / tenure
)

# 5. Retain original measures as well as derived features.
# Ratios with zero company tenure remain NaN by design.

# 6. Descriptive promotion-gap category
df["PromotionGapCategory"] = pd.cut(
    df["YearsSinceLastPromotion"],
    bins=[-np.inf, 1, 4, np.inf],
    labels=["Low", "Medium", "High"]
)

# 7. Save feature-engineered data
df.to_csv(OUTPUT_FILE, index=False)

# 8. Summarize derived indicators
feature_cols = [
    "PromotionGapRatio",
    "RoleStagnationIndex",
    "TrainingIntensityScore",
    "ManagerTenureRatio",
]

summary = df[feature_cols].describe().T
summary.to_csv(SUMMARY_FILE)

print("\nFeature engineering completed.")
print("Output:", OUTPUT_FILE)
print("\nDerived feature summary:")
print(summary.round(3))
print("\nPromotion-gap category counts:")
print(df["PromotionGapCategory"].value_counts(dropna=False))
print("\nUndefined ratios due to zero tenure:")
print(df[feature_cols].isna().sum())