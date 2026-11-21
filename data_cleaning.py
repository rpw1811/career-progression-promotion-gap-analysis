from pathlib import Path
import pandas as pd
import numpy as np

#Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'outputs'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / 'Palo Alto Networks.csv'
OUTPUT_FILE = OUTPUT_DIR / 'cleanned_hr_data.csv'
REPORT_FILE = OUTPUT_DIR / 'data_quality_report.txt'

#1. Load the data
df = pd.read_csv(INPUT_FILE)
original_shape = df.shape

#2. Standardize column names
df.columns = (
  df.columns.str.strip()
  .str.replace(" ", "_", regex=False)
)

#3. validate required fields
required_columns = ["Age","Attrition","Department","JobRole","JobLevel","TrainingTimesLastYear","YearsAtCompany",
                    "YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager","TotalWorkingYears",
                    ]

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

#4. Check for missing values and duplicates record
missing_before = int(df.isna().sum().sum())
duplicates_before = int(df.duplicated().sum())

#5. validate numeric fields
numeric_columns = ["Age","Attrition","JobLevel","TrainingTimesLastYear","YearsAtCompany",
                    "YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager","TotalWorkingYears",
                      ]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

    # Report newly introduced missing values
    missing_after_conversion = int(df.isna().sum().sum())

    #6. Check key business rules before cleaning records
    check = {
        "invalid_age": ((df["Age"] < 18) | (df["Age"] > 100)
        ),
        "invalid_attrition": ~df["Attrition"].isin([0, 1]),
        "invalid_joblevel": ~df["JobLevel"].between(1, 5),
        "negative_company_tenure": (df["YearsAtCompany"] < 0),
        "role_tenure_gt_company": (
            df["YearsInCurrentRole"] > df["YearsAtCompany"]
        ),
        "promotion_gap_gt_company": (
            df["YearsSinceLastPromotion"] > df["YearsAtCompany"]
        ),
        "manager_tenure_gt_company": (
            df["YearsWithCurrManager"] > df["YearsAtCompany"]
        ),
        "role_tenure_gt_total_worke": (
            df["YearsInCurrentRole"] > df["TotalWorkingYears"]
        ),
        "negative_traning": df["TrainingTimesLastYear"] < 0,
    }
    
    # 7. Record issue counts; don't silently delete rows
issue_counts = {
    name: int(mask.fillna(False).sum())
    for name, mask in check.items()
}

# 8. Preserve all records pending review
# No blanket outlier removal or automatic row deletion.
clean_df = df.copy()

# 9. Save cleaned dataset
clean_df.to_csv(DATA_DIR / "cleaned_hr_data.csv", index=False)

# 10. Save audit report
report = [
    "CAREER PROGRESSION DATA QUALITY REPORT",
    "=" * 45,
    f"Input rows: {original_shape[0]}",
    f"Input columns: {original_shape[1]}",
    f"Missing cells before conversion: {missing_before}",
    f"Duplicate rows before cleaning: {duplicates_before}",
    f"Missing cells after numeric conversion: "
    f"{missing_after_conversion}",
    "",
    "BUSINESS RULE CHECKS",
]

for name, count in issue_counts.items():
    report.append(f"{name}: {count}")

report.extend([
    "",
    f"Output rows: {len(clean_df)}",
    f"Output columns: {len(clean_df.columns)}",
    f"Cleaned file: {OUTPUT_FILE}",
    "",
    "Review all flagged issues before modelling.",
    "No records were automatically removed."
])

REPORT_FILE.write_text("\n".join(report), encoding="utf-8")

print("\nData validation completed.")
print(f"Original shape: {original_shape}")
print(f"Cleaned shape: {clean_df.shape}")
print(f"Missing cells: {clean_df.isna().sum().sum()}")
print(f"Duplicate rows: {clean_df.duplicated().sum()}")
print("\nBusiness rule issues:")
for name, count in issue_counts.items():
    print(f"{name}: {count}")

print(f"\nSaved dataset: {OUTPUT_FILE}")
print(f"Saved report: {REPORT_FILE}")