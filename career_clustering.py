from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)
from sklearn.decomposition import PCA

# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
CLUSTER_DIR = OUTPUT_DIR / "clustering"

CLUSTER_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "career_features.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("CAREER PATH CLUSTERING")
print("=" * 60)

print("Dataset shape:", df.shape)

# ==================================================
# 2. FEATURE SELECTION
# ==================================================

features = [
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "TrainingTimesLastYear",
    "JobLevel"
]

missing = [c for c in features if c not in df.columns]

if missing:
    raise ValueError(f"Missing features: {missing}")

# Convert selected features to numeric
X = df[features].apply(
    pd.to_numeric, errors="coerce"
)

# ==================================================
# 3. DATA VALIDATION
# ==================================================

invalid = (
    (X["YearsAtCompany"] < 0)
    | (X["YearsInCurrentRole"] < 0)
    | (X["YearsSinceLastPromotion"] < 0)
    | (X["YearsWithCurrManager"] < 0)
    | (X["TrainingTimesLastYear"] < 0)
    | (X["JobLevel"] < 1)
    | (X["YearsInCurrentRole"] > X["YearsAtCompany"])
    | (X["YearsSinceLastPromotion"] > X["YearsAtCompany"])
    | (X["YearsWithCurrManager"] > X["YearsAtCompany"])
)

invalid = invalid.fillna(True)

print("Invalid rows flagged:", int(invalid.sum()))

# Keep only records passing career-data validation.
# Retain original row index to track excluded records.
valid_mask = ~invalid
valid_df = df.loc[valid_mask].copy()
X = X.loc[valid_mask].copy()

if len(X) < 10:
    raise ValueError("Not enough valid records for clustering.")

if X.isna().all().any():
    raise ValueError("At least one selected feature is entirely missing.")

# ==================================================
# 4. HANDLE MISSING VALUES
# ==================================================

imputer = SimpleImputer(strategy="median")

X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=features,
    index=X.index
)

# ==================================================
# 5. FEATURE SCALING
# ==================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_imputed)

print("Feature scaling completed.")
print("Valid employee records:", len(X_scaled))

# ==================================================
# 6. FIND OPTIMAL NUMBER OF CLUSTERS
# ==================================================

evaluation_results = []

max_k = min(8, len(X_scaled) - 1)

for k in range(2, max_k + 1):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X_scaled)

    silhouette = silhouette_score(
        X_scaled, labels
    )

    db_score = davies_bouldin_score(
        X_scaled, labels
    )

    ch_score = calinski_harabasz_score(
        X_scaled, labels
    )

    evaluation_results.append({
        "K": k,
        "Inertia": model.inertia_,
        "SilhouetteScore": silhouette,
        "DaviesBouldinScore": db_score,
        "CalinskiHarabaszScore": ch_score
    })

evaluation_df = pd.DataFrame(evaluation_results)

evaluation_df.to_csv(
    CLUSTER_DIR / "cluster_evaluation.csv",
    index=False
)

print("\nCLUSTER EVALUATION")
print(evaluation_df.round(4).to_string(index=False))

# ==================================================
# 7. ELBOW METHOD CHART
# ==================================================

plt.figure(figsize=(9, 5))

plt.plot(
    evaluation_df["K"],
    evaluation_df["Inertia"],
    marker="o"
)

plt.title("Elbow Method for K-Means")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(evaluation_df["K"])
plt.tight_layout()

plt.savefig(
    CLUSTER_DIR / "01_elbow_method.png",
    dpi=150
)

plt.close()

# ==================================================
# 8. SILHOUETTE SCORE CHART
# ==================================================

plt.figure(figsize=(9, 5))

plt.plot(
    evaluation_df["K"],
    evaluation_df["SilhouetteScore"],
    marker="o"
)

plt.title("Silhouette Score by Cluster Count")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(evaluation_df["K"])
plt.tight_layout()

plt.savefig(
    CLUSTER_DIR / "02_silhouette_scores.png",
    dpi=150
)

plt.close()

# ==================================================
# 9. SELECT K
# ==================================================

# Initial candidate: highest silhouette score.
# Inspect elbow, DB, CH and cluster sizes before finalizing.
best_row = evaluation_df.loc[
    evaluation_df["SilhouetteScore"].idxmax()
]

selected_k = int(best_row["K"])

print("\nInitial candidate K:", selected_k)

# ==================================================
# 10. FINAL K-MEANS MODEL
# ==================================================

kmeans = KMeans(
    n_clusters=selected_k,
    random_state=42,
    n_init=20
)

kmeans_labels = kmeans.fit_predict(X_scaled)

valid_df["KMeansCluster"] = kmeans_labels

# ==================================================
# 11. HIERARCHICAL CLUSTERING
# ==================================================

hierarchical = AgglomerativeClustering(
    n_clusters=selected_k,
    linkage="ward"
)

hierarchical_labels = hierarchical.fit_predict(
    X_scaled
)

valid_df["HierarchicalCluster"] = hierarchical_labels

# Compare agreement between methods
ari = adjusted_rand_score(
    kmeans_labels,
    hierarchical_labels
)

print("\nHIERARCHICAL VALIDATION")
print("Adjusted Rand Index:", round(ari, 4))

# ==================================================
# 12. CLUSTER DISTRIBUTION
# ==================================================

cluster_counts = (
    valid_df["KMeansCluster"]
    .value_counts()
    .sort_index()
    .rename_axis("Cluster")
    .reset_index(name="EmployeeCount")
)

cluster_counts["Percentage"] = (
    cluster_counts["EmployeeCount"]
    / len(valid_df) * 100
)

cluster_counts.to_csv(
    CLUSTER_DIR / "cluster_distribution.csv",
    index=False
)

print("\nCLUSTER DISTRIBUTION")
print(cluster_counts.round(2).to_string(index=False))

# ==================================================
# 13. CLUSTER PROFILES
# ==================================================

profile = (
    valid_df.groupby("KMeansCluster")[features]
    .mean()
    .round(2)
)

profile.to_csv(
    CLUSTER_DIR / "cluster_profiles.csv"
)

median_profile = (
    valid_df.groupby("KMeansCluster")[features]
    .median()
    .round(2)
)

median_profile.to_csv(
    CLUSTER_DIR / "cluster_median_profiles.csv"
)

print("\nCLUSTER PROFILES")
print(profile)

# ==================================================
# 14. ATTRITION ANALYSIS AFTER CLUSTERING
# ==================================================

# Attrition is not used as a clustering input.
# It is used only after clusters have been created.

if "Attrition" in valid_df.columns:

    attrition = pd.to_numeric(
        valid_df["Attrition"],
        errors="coerce"
    )

    valid_df["AttritionForAnalysis"] = attrition

    attrition_profile = (
        valid_df.groupby("KMeansCluster")
        .agg(
            EmployeeCount=("KMeansCluster", "size"),
            EmployeesLeft=("AttritionForAnalysis", "sum"),
            AttritionRate=("AttritionForAnalysis", "mean")
        )
        .reset_index()
    )

    attrition_profile["AttritionRate"] *= 100

    attrition_profile.to_csv(
        CLUSTER_DIR / "cluster_attrition_summary.csv",
        index=False
    )

    print("\nATTRITION BY CLUSTER")
    print(attrition_profile.round(2).to_string(index=False))

# ==================================================
# 15. PCA VISUALIZATION
# ==================================================

pca = PCA(n_components=2)

pca_values = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": pca_values[:, 0],
    "PC2": pca_values[:, 1],
    "Cluster": kmeans_labels.astype(str)
})

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="tab10",
    alpha=0.7
)

plt.title("Career Path Clusters - PCA Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.tight_layout()

plt.savefig(
    CLUSTER_DIR / "03_career_clusters_pca.png",
    dpi=150
)

plt.close()

# ==================================================
# 16. SAVE EMPLOYEE CLUSTER ASSIGNMENTS
# ==================================================

valid_df.to_csv(
    CLUSTER_DIR / "employee_cluster_assignments.csv",
    index=False
)

# Save invalid records separately for audit
df.loc[invalid].to_csv(
    CLUSTER_DIR / "excluded_records_for_review.csv",
    index=False
)

# Save model preprocessing parameters
scaler_info = pd.DataFrame({
    "Feature": features,
    "Mean": scaler.mean_,
    "Scale": scaler.scale_
})

scaler_info.to_csv(
    CLUSTER_DIR / "scaler_parameters.csv",
    index=False
)

print("\nCLUSTERING COMPLETED SUCCESSFULLY")
print("Selected candidate K:", selected_k)
print("Hierarchical ARI:", round(ari, 4))
print("Outputs saved in:", CLUSTER_DIR)
