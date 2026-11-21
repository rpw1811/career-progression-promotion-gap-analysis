from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Career Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
)

BASE = Path(__file__).resolve().parent
CL = BASE / "outputs" / "clustering"
RT = BASE / "outputs" / "retention_analysis"
EMP = RT / "employee_retention_opportunities.csv"


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def parse_bool(series: pd.Series) -> pd.Series:
    """Convert common CSV boolean representations to True/False."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float("nan"), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


st.title("Career Progression & Promotion Gap Analysis")
st.caption(
    "Career intelligence dashboard | Phase 4 clustering + Phase 5 retention analysis"
)

if not EMP.exists():
    st.error("Phase 5 employee results file not found. No substitute data will be generated.")
    st.code("outputs/retention_analysis/employee_retention_opportunities.csv")
    st.info(
        "Run: python 05_retention_opportunity_analysis.py\n"
        "Keep app.py beside your data/ and outputs/ folders."
    )
    st.stop()

try:
    df = load_csv(str(EMP))
except Exception as exc:
    st.error(f"Could not read employee results: {exc}")
    st.stop()

required = [
    "KMeansCluster",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "TrainingTimesLastYear",
    "RetentionOpportunityIndex",
    "InterventionPriority",
]
missing = [column for column in required if column not in df.columns]
if missing:
    st.error("The Phase 5 file is missing columns: " + ", ".join(missing))
    st.stop()

numeric_columns = [
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "TrainingTimesLastYear",
    "RetentionOpportunityIndex",
    "PromotionGapRatio",
    "RoleStagnationIndex",
    "PromotionGapScore",
    "RoleStagnationScore",
    "TrainingNeedScore",
    "TrainingNeedIndicator",
    "JobLevel",
]
for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

# Keep all rows visible so Not Assessed records are not silently hidden.
# Use assessed rows only for calculations that require valid career data.
if "CareerDataValid" in df.columns:
    df["_CareerDataValid"] = parse_bool(df["CareerDataValid"])
else:
    df["_CareerDataValid"] = True

st.sidebar.title("Career Intelligence")
page = st.sidebar.radio(
    "Dashboard pages",
    [
        "Career Path Clustering",
        "Promotion Gap Monitor",
        "Retention Opportunity",
        "Managerial Insights",
    ],
)
st.sidebar.divider()
st.sidebar.subheader("Filters")

departments = (
    sorted(df["Department"].dropna().astype(str).unique())
    if "Department" in df.columns
    else []
)
roles = (
    sorted(df["JobRole"].dropna().astype(str).unique())
    if "JobRole" in df.columns
    else []
)
selected_depts = st.sidebar.multiselect(
    "Department", departments, default=departments
)
selected_roles = st.sidebar.multiselect("Job role", roles, default=roles)
stage = st.sidebar.selectbox(
    "Career stage",
    [
        "All stages",
        "Early career (0–2 years)",
        "Developing (3–5 years)",
        "Established (6–10 years)",
        "Long tenure (11+ years)",
    ],
)
gap_threshold = st.sidebar.slider(
    "Promotion gap threshold (years)", 0, 20, 5
)

f = df.copy()
if departments:
    f = f[f["Department"].astype(str).isin(selected_depts)]
if roles:
    f = f[f["JobRole"].astype(str).isin(selected_roles)]

if stage != "All stages":
    bounds = {
        "Early career (0–2 years)": (0, 2),
        "Developing (3–5 years)": (3, 5),
        "Established (6–10 years)": (6, 10),
        "Long tenure (11+ years)": (11, float("inf")),
    }
    lower, upper = bounds[stage]
    tenure = numeric_column(f, "YearsAtCompany")
    if upper == float("inf"):
        f = f[tenure >= lower]
    else:
        f = f[tenure.between(lower, upper)]

st.caption(
    f"Showing {len(f):,} of {len(df):,} total records after sidebar filters. "
    "Unassessed records remain visible where relevant."
)
if f.empty:
    st.warning("No records match the selected filters. Adjust the filters.")
    st.stop()

# Career-data validity and opportunity-score assessment are separate concepts.
# Some records can pass career-data checks but still have no assessable
# RetentionOpportunityIndex (for example, zero company tenure).
assessed = f[
    f["_CareerDataValid"]
    & numeric_column(f, "RetentionOpportunityIndex").notna()
    & f["InterventionPriority"].fillna("Not Assessed").astype(str).ne("Not Assessed")
].copy()
valid_career_count = int(f["_CareerDataValid"].sum())
score_not_assessed_count = int(len(f) - len(assessed))


def assessed_mean(column: str) -> float:
    if column not in assessed.columns:
        return float("nan")
    return numeric_column(assessed, column).mean()


def show_kpis() -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Employees in view", f"{len(f):,}")
    c2.metric("Career data valid", f"{valid_career_count:,}")
    c3.metric("Opportunity scores not assessed", f"{score_not_assessed_count:,}")

    promotion_mean = numeric_column(f, "YearsSinceLastPromotion").mean()
    c4.metric(
        "Avg. promotion gap (assessed)",
        "N/A" if pd.isna(promotion_mean) else f"{promotion_mean:.1f} years",
    )

    opportunity_mean = numeric_column(f, "RetentionOpportunityIndex").mean()
    c5.metric(
        "Avg. opportunity index",
        "N/A" if pd.isna(opportunity_mean) else f"{opportunity_mean:.1f}/100",
    )


def optional_csv(path: Path):
    if not path.exists():
        return None
    try:
        return load_csv(str(path))
    except Exception as exc:
        st.warning(f"Could not read optional file {path.name}: {exc}")
        return None


if page == "Career Path Clustering":
    st.header("Career Path Clustering")
    st.write(
        "Explore K-Means career-pattern groups. Cluster IDs are identifiers, "
        "not preassigned employee types."
    )
    show_kpis()

    left, right = st.columns(2)
    cluster_counts = (
        f["KMeansCluster"].astype(str)
        .value_counts()
        .rename_axis("Cluster")
        .reset_index(name="Employees")
    )
    left.plotly_chart(
        px.bar(
            cluster_counts,
            x="Cluster",
            y="Employees",
            title="Employee distribution by cluster",
            text="Employees",
        ),
        use_container_width=True,
    )

    profiles = optional_csv(CL / "cluster_profiles.csv")
    if profiles is not None:
        if "KMeansCluster" not in profiles.columns:
            profiles = profiles.reset_index()
            first_col = profiles.columns[0]
            profiles = profiles.rename(columns={first_col: "KMeansCluster"})

        if "KMeansCluster" in profiles.columns:
            profiles["KMeansCluster"] = profiles["KMeansCluster"].astype(str)
            profile_columns = [
                column
                for column in [
                    "YearsAtCompany",
                    "YearsInCurrentRole",
                    "YearsSinceLastPromotion",
                    "YearsWithCurrManager",
                    "TrainingTimesLastYear",
                ]
                if column in profiles.columns
            ]
            if profile_columns:
                long_profiles = profiles.melt(
                    id_vars=["KMeansCluster"],
                    value_vars=profile_columns,
                    var_name="Career measure",
                    value_name="Mean",
                )
                right.plotly_chart(
                    px.bar(
                        long_profiles,
                        x="Career measure",
                        y="Mean",
                        color="KMeansCluster",
                        barmode="group",
                        title="Average career indicators by cluster",
                    ),
                    use_container_width=True,
                )
            else:
                right.info("Cluster profile file has no expected career columns.")
        else:
            right.info("Cluster profile file does not contain a cluster identifier.")
    else:
        right.info("Phase 4 cluster_profiles.csv not found.")

    cluster_options = sorted(f["KMeansCluster"].dropna().astype(str).unique())
    if cluster_options:
        selected_cluster = st.selectbox("Explore cluster", cluster_options)
        selected_rows = f[f["KMeansCluster"].astype(str) == selected_cluster]
        st.subheader(f"Cluster {selected_cluster}: employee records")
        display_columns = [
            column
            for column in [
                "Department",
                "JobRole",
                "JobLevel",
                "YearsAtCompany",
                "YearsInCurrentRole",
                "YearsSinceLastPromotion",
                "TrainingTimesLastYear",
                "InterventionPriority",
                "RetentionOpportunityIndex",
            ]
            if column in selected_rows.columns
        ]
        st.dataframe(
            selected_rows[display_columns],
            use_container_width=True,
            hide_index=True,
        )

    evaluation = optional_csv(CL / "cluster_evaluation.csv")
    if evaluation is not None:
        st.subheader("Clustering evaluation metrics")
        st.dataframe(evaluation, use_container_width=True, hide_index=True)
        st.caption(
            "Interpret inertia, silhouette, Davies-Bouldin and Calinski-Harabasz "
            "together; no single metric proves cluster validity."
        )


elif page == "Promotion Gap Monitor":
    st.header("Promotion Gap Monitor")
    st.write(
        "Explore time since last promotion and current-role tenure. "
        "The threshold is an exploratory filter, not an HR policy."
    )
    show_kpis()

    # Promotion-gap comparisons use assessed career records only.
    gap_source = assessed.copy()
    gap_source["YearsSinceLastPromotion"] = numeric_column(
        gap_source, "YearsSinceLastPromotion"
    )
    gap_source["YearsInCurrentRole"] = numeric_column(
        gap_source, "YearsInCurrentRole"
    )
    gap = gap_source[
        gap_source["YearsSinceLastPromotion"] >= gap_threshold
    ].copy()

    col1, col2, col3 = st.columns(3)
    col1.metric("Records at/above threshold", f"{len(gap):,}")
    avg_role = gap["YearsInCurrentRole"].mean()
    col2.metric(
        "Average role tenure in group",
        "N/A" if pd.isna(avg_role) else f"{avg_role:.1f} years",
    )
    col3.metric(
        "Share of assessed records",
        f"{len(gap) / len(assessed) * 100:.1f}%" if len(assessed) else "N/A",
    )

    left, right = st.columns(2)
    left.plotly_chart(
        px.histogram(
            gap_source,
            x="YearsSinceLastPromotion",
            nbins=20,
            title="Years since last promotion (assessed records)",
        ),
        use_container_width=True,
    )

    if "JobRole" in gap_source.columns:
        role_summary = (
            gap_source.groupby("JobRole")
            .agg(
                Employees=("JobRole", "size"),
                AverageGap=("YearsSinceLastPromotion", "mean"),
            )
            .reset_index()
            .sort_values("AverageGap")
        )
        right.plotly_chart(
            px.bar(
                role_summary,
                x="AverageGap",
                y="JobRole",
                orientation="h",
                hover_data=["Employees"],
                title="Average promotion gap by role",
            ),
            use_container_width=True,
        )

    if "Department" in gap_source.columns:
        department_summary = (
            gap_source.groupby("Department")
            .agg(
                Employees=("Department", "size"),
                AverageGap=("YearsSinceLastPromotion", "mean"),
            )
            .reset_index()
        )
        st.plotly_chart(
            px.bar(
                department_summary,
                x="Department",
                y="AverageGap",
                hover_data=["Employees"],
                title="Average promotion gap by department",
            ),
            use_container_width=True,
        )

    display_columns = [
        column
        for column in [
            "Department",
            "JobRole",
            "JobLevel",
            "YearsAtCompany",
            "YearsInCurrentRole",
            "YearsSinceLastPromotion",
            "PromotionGapCategory",
            "PromotionGapRatio",
            "InterventionPriority",
        ]
        if column in gap.columns
    ]
    st.subheader("Assessed employees meeting the selected threshold")
    st.dataframe(
        gap[display_columns].sort_values(
            "YearsSinceLastPromotion", ascending=False
        ),
        use_container_width=True,
        hide_index=True,
    )


elif page == "Retention Opportunity":
    st.header("Retention Opportunity Panel")
    st.write(
        "The index is a transparent development-intervention screening score, "
        "not a probability of resignation."
    )
    show_kpis()

    left, right = st.columns(2)
    priority_counts = (
        f["InterventionPriority"].fillna("Not Assessed").astype(str)
        .value_counts()
        .rename_axis("Priority")
        .reset_index(name="Employees")
    )
    left.plotly_chart(
        px.bar(
            priority_counts,
            x="Priority",
            y="Employees",
            title="Intervention priority distribution",
            text="Employees",
        ),
        use_container_width=True,
    )

    if "TrainingNeedIndicator" in f.columns:
        training_indicator = numeric_column(f, "TrainingNeedIndicator")
        training_status = training_indicator.map(
            {1: "Training review flag", 0: "No low-participation flag"}
        ).fillna("Not assessed")
        training_counts = (
            training_status.value_counts()
            .rename_axis("Training status")
            .reset_index(name="Employees")
        )
        right.plotly_chart(
            px.bar(
                training_counts,
                x="Training status",
                y="Employees",
                title="Training participation review flags",
            ),
            use_container_width=True,
        )
    else:
        right.info("TrainingNeedIndicator is not available in the Phase 5 file.")

    scores = f.dropna(subset=["RetentionOpportunityIndex"])
    if not scores.empty:
        st.plotly_chart(
            px.histogram(
                scores,
                x="RetentionOpportunityIndex",
                nbins=20,
                title="Retention Opportunity Index distribution",
            ),
            use_container_width=True,
        )

    priorities = st.multiselect(
        "Show priorities",
        ["High", "Moderate", "Low", "Not Assessed"],
        default=["High", "Moderate", "Not Assessed"],
    )
    rec = f[
        f["InterventionPriority"].fillna("Not Assessed").astype(str).isin(priorities)
    ].copy()

    search = st.text_input("Search department or job role")
    if search:
        search_mask = pd.Series(False, index=rec.index)
        for column in ["Department", "JobRole"]:
            if column in rec.columns:
                search_mask |= rec[column].astype(str).str.contains(
                    search, case=False, na=False
                )
        rec = rec[search_mask]

    display_columns = [
        column
        for column in [
            "Department",
            "JobRole",
            "JobLevel",
            "KMeansCluster",
            "PromotionGapCategory",
            "RoleStagnationCategory",
            "TrainingNeedIndicator",
            "ManagerTenureCategory",
            "RetentionOpportunityIndex",
            "InterventionPriority",
            "SuggestedDevelopmentActions",
        ]
        if column in rec.columns
    ]
    st.subheader("Career-development recommendations")
    sort_column = (
        "RetentionOpportunityIndex"
        if "RetentionOpportunityIndex" in rec.columns
        else display_columns[0]
    )
    st.dataframe(
        rec[display_columns].sort_values(
            sort_column, ascending=False, na_position="last"
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download filtered recommendations",
        rec.to_csv(index=False).encode("utf-8"),
        "career_development_recommendations.csv",
        "text/csv",
    )
    st.warning(
        "Use flags to offer supportive career conversations and development options. "
        "Do not use them alone for employment decisions."
    )


else:
    st.header("Managerial Insights")
    st.write(
        "Current-manager tenure is a proxy only. The source data does not include "
        "manager IDs or reporting-line history."
    )
    show_kpis()

    if "ManagerTenureCategory" in f.columns:
        left, right = st.columns(2)
        manager_counts = (
            f["ManagerTenureCategory"].fillna("Not available").astype(str)
            .value_counts()
            .rename_axis("Manager tenure")
            .reset_index(name="Employees")
        )
        left.plotly_chart(
            px.bar(
                manager_counts,
                x="Manager tenure",
                y="Employees",
                title="Employees by current-manager tenure",
            ),
            use_container_width=True,
        )

        manager_assessed = assessed.copy()
        manager_assessed["YearsSinceLastPromotion"] = numeric_column(
            manager_assessed, "YearsSinceLastPromotion"
        )
        manager_assessed["YearsInCurrentRole"] = numeric_column(
            manager_assessed, "YearsInCurrentRole"
        )
        manager_measures = (
            manager_assessed.groupby("ManagerTenureCategory", dropna=False)
            .agg(
                AssessedEmployees=("ManagerTenureCategory", "size"),
                AveragePromotionGap=("YearsSinceLastPromotion", "mean"),
                AverageRoleTenure=("YearsInCurrentRole", "mean"),
            )
            .reset_index()
        )
        right.plotly_chart(
            px.bar(
                manager_measures,
                x="ManagerTenureCategory",
                y=["AveragePromotionGap", "AverageRoleTenure"],
                barmode="group",
                title="Career measures by manager-tenure category",
            ),
            use_container_width=True,
        )
        st.caption(
            "Employee counts include all records. Career-measure averages use "
            "assessed records only."
        )

    if "Department" in f.columns:
        department_counts = (
            f.groupby("Department")
            .size()
            .rename("Employees")
            .reset_index()
        )
        if not assessed.empty:
            assessed_department = assessed.copy()
            for column in [
                "YearsWithCurrManager",
                "YearsSinceLastPromotion",
                "YearsInCurrentRole",
                "RetentionOpportunityIndex",
            ]:
                assessed_department[column] = numeric_column(
                    assessed_department, column
                )
            department_measures = (
                assessed_department.groupby("Department")
                .agg(
                    AssessedEmployees=("Department", "size"),
                    AverageManagerTenure=("YearsWithCurrManager", "mean"),
                    AveragePromotionGap=("YearsSinceLastPromotion", "mean"),
                    AverageRoleTenure=("YearsInCurrentRole", "mean"),
                    AverageOpportunityIndex=("RetentionOpportunityIndex", "mean"),
                )
                .reset_index()
            )
            department_summary = department_counts.merge(
                department_measures, on="Department", how="left"
            )
            st.plotly_chart(
                px.scatter(
                    department_summary,
                    x="AverageManagerTenure",
                    y="AveragePromotionGap",
                    size="Employees",
                    color="Department",
                    hover_data=[
                        "AssessedEmployees",
                        "AverageRoleTenure",
                        "AverageOpportunityIndex",
                    ],
                    title="Department manager tenure vs promotion gap",
                ),
                use_container_width=True,
            )
            st.dataframe(
                department_summary,
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Employee counts include all records; averages are calculated "
                "from assessed records only."
            )

st.divider()
st.caption(
    "Dataset provenance must be verified before attributing these records to "
    "Palo Alto Networks. The index is a rule-based screening aid, not a validated "
    "attrition prediction."
)
