"""
Hospital Bottleneck Analysis — Streamlit Lite
------------------------------------------------
A lightweight, interactive Streamlit port of the hospital_analysis notebook.

Run with:
    streamlit run streamlit_app.py

If you have the original CSVs (admissions.csv, billing.csv, diagnoses.csv,
hospitals.csv, patients.csv), upload them in the sidebar to use real data.
Otherwise the app generates synthetic data with the same schema and similar
statistical properties so the whole dashboard works out of the box.
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ────────────────────────────────────────────────────────────────────────────
# Page config & style
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Hospital Bottleneck Analysis — Lite", layout="wide", page_icon="🏥")

sns.set_theme(style="whitegrid", palette="muted", font_scale=0.95)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                      "figure.facecolor": "white"})
PALETTE = {"ICU": "#E24B4A", "NICU": "#c0392b", "HDU": "#EF9F27", "General": "#1D9E75"}

st.title("🏥 Hospital Bottleneck Analysis — Lite")
st.caption("EDA · Statistical Testing · Machine Learning — interactive Streamlit port of the analysis notebook")


# ────────────────────────────────────────────────────────────────────────────
# Data loading (real uploads if provided, else synthetic demo data)
# ────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def make_synthetic_data(n_admissions=12000, n_hospitals=33, seed=42):
    rng = np.random.default_rng(seed)

    states = ["Kerala", "Maharashtra", "MP", "Delhi", "Tamil Nadu", "Karnataka", "UP"]
    tiers = ["Tier-1 Private", "Tier-2 Private", "Tier-1 Govt", "Tier-2 Govt"]
    hosp = pd.DataFrame({
        "hospital_id": [f"H{i:03d}" for i in range(n_hospitals)],
        "hospital_name": [f"Hospital {i}" for i in range(n_hospitals)],
        "state": rng.choice(states, n_hospitals),
        "tier": rng.choice(tiers, n_hospitals, p=[0.25, 0.25, 0.2, 0.3]),
        "beds": rng.integers(40, 450, n_hospitals),
        "teaching": rng.choice([True, False], n_hospitals, p=[0.3, 0.7]),
    })

    n_patients = int(n_admissions * 0.72)
    pat = pd.DataFrame({
        "patient_id": [f"P{i:06d}" for i in range(n_patients)],
        "age": rng.integers(0, 95, n_patients),
        "sex": rng.choice(["M", "F"], n_patients),
        "bpl_status": rng.choice(["BPL", "Non-BPL"], n_patients, p=[0.35, 0.65]),
        "charlson_index": rng.poisson(2.2, n_patients).clip(0, 12),
        "comorbidity_count": rng.poisson(1.6, n_patients).clip(0, 8),
        "hba1c": np.round(rng.normal(6.2, 1.4, n_patients).clip(4, 14), 1),
        "creatinine": np.round(rng.normal(1.1, 0.5, n_patients).clip(0.3, 6), 2),
    })
    pat["haemoglobin"] = np.round(rng.normal(12.5, 2.1, n_patients).clip(5, 18), 1)

    ward_types = ["General", "ICU", "HDU", "NICU"]
    ward_p = [0.62, 0.18, 0.12, 0.08]
    admit_types = ["Elective", "Emergency", "OPD"]

    admit_dates = pd.to_datetime("2015-01-01") + pd.to_timedelta(
        rng.integers(0, 3650, n_admissions), unit="D"
    )

    ward = rng.choice(ward_types, n_admissions, p=ward_p)
    base_los = {"General": 3.5, "ICU": 8.5, "HDU": 5.5, "NICU": 10.5}
    los = np.array([max(1, int(rng.exponential(base_los[w]))) for w in ward])

    adm_hosp = rng.choice(hosp["hospital_id"], n_admissions)
    adm = pd.DataFrame({
        "admission_id": [f"A{i:07d}" for i in range(n_admissions)],
        "patient_id": rng.choice(pat["patient_id"], n_admissions),
        "admit_date": admit_dates,
        "los_days": los,
        "admit_type": rng.choice(admit_types, n_admissions, p=[0.4, 0.45, 0.15]),
        "ward_type": ward,
        "hospital_id": adm_hosp,
        "discharge_type": rng.choice(["Recovered", "Transferred", "LAMA", "Deceased"],
                                      n_admissions, p=[0.86, 0.06, 0.05, 0.03]),
        "num_procedures": rng.poisson(1.1, n_admissions),
    })
    adm["discharge_date"] = adm["admit_date"] + pd.to_timedelta(adm["los_days"], unit="D")

    # readmission likelihood driven by comorbidity/los (merged later for realism)
    merged_tmp = adm.merge(pat, on="patient_id", how="left")
    logit = (-2.4 + 0.18 * merged_tmp["comorbidity_count"] + 0.05 * merged_tmp["los_days"]
             + 0.12 * merged_tmp["charlson_index"])
    prob = 1 / (1 + np.exp(-logit))
    adm["readmitted_30d"] = (rng.random(n_admissions) < prob).astype(int)

    diag_categories = ["Cardiovascular", "Perinatal", "Respiratory", "Renal", "Orthopedic",
                        "Infectious", "Oncology", "Neurological"]
    n_diag = int(n_admissions * 1.9)
    diag = pd.DataFrame({
        "diagnosis_id": [f"D{i:07d}" for i in range(n_diag)],
        "admission_id": rng.choice(adm["admission_id"], n_diag),
        "category": rng.choice(diag_categories, n_diag),
        "is_primary": rng.choice([True, False], n_diag, p=[0.4, 0.6]),
    })

    cost_categories = ["Pharmacy", "Diagnostics", "Room Charges", "Procedures", "Consultation"]
    total_cost = (los * rng.integers(2500, 9000, n_admissions)
                  + merged_tmp["comorbidity_count"] * rng.integers(500, 3000, n_admissions))
    oop_frac = np.where(merged_tmp["bpl_status"] == "BPL",
                         rng.uniform(0.35, 0.75, n_admissions),
                         rng.uniform(0.55, 0.95, n_admissions))
    oop_cost = (total_cost * oop_frac).astype(int)
    bill = pd.DataFrame({
        "admission_id": adm["admission_id"],
        "total_cost": total_cost.astype(int),
        "oop_cost": oop_cost,
        "subsidy": (total_cost - oop_cost).astype(int),
        "top_cost_category": rng.choice(cost_categories, n_admissions,
                                         p=[0.32, 0.22, 0.24, 0.14, 0.08]),
    })

    return adm, bill, diag, hosp, pat


@st.cache_data(show_spinner=False)
def load_uploaded(files):
    adm = pd.read_csv(files["admissions"], parse_dates=["admit_date", "discharge_date"])
    bill = pd.read_csv(files["billing"])
    diag = pd.read_csv(files["diagnoses"])
    hosp = pd.read_csv(files["hospitals"])
    pat = pd.read_csv(files["patients"])
    return adm, bill, diag, hosp, pat


@st.cache_data(show_spinner=False)
def build_master(adm, bill, diag, hosp, pat):
    df = adm.merge(pat, on="patient_id", how="left")
    df = df.merge(hosp, on="hospital_id", how="left", suffixes=("", "_hosp"))
    df = df.merge(bill, on="admission_id", how="left")
    df["admit_year"] = df["admit_date"].dt.year
    df["admit_month"] = df["admit_date"].dt.month
    df["admit_dow"] = df["admit_date"].dt.dayofweek
    df["admit_dow_name"] = df["admit_date"].dt.day_name()
    df["age_band"] = pd.cut(df["age"], bins=[-1, 1, 18, 35, 50, 65, 120],
                             labels=["<1", "1-18", "19-35", "36-50", "51-65", "65+"])
    return df


with st.sidebar:
    st.header("Data source")
    use_upload = st.checkbox("Upload my own CSVs", value=False)
    data_files = None
    if use_upload:
        st.caption("Upload all 5 files (admissions, billing, diagnoses, hospitals, patients).")
        f_adm = st.file_uploader("admissions.csv", type="csv", key="adm")
        f_bill = st.file_uploader("billing.csv", type="csv", key="bill")
        f_diag = st.file_uploader("diagnoses.csv", type="csv", key="diag")
        f_hosp = st.file_uploader("hospitals.csv", type="csv", key="hosp")
        f_pat = st.file_uploader("patients.csv", type="csv", key="pat")
        if all([f_adm, f_bill, f_diag, f_hosp, f_pat]):
            data_files = {"admissions": f_adm, "billing": f_bill, "diagnoses": f_diag,
                          "hospitals": f_hosp, "patients": f_pat}
        else:
            st.info("Upload all 5 files to switch off demo data.")

    n_demo = st.slider("Synthetic admissions (demo mode)", 2000, 30000, 12000, step=1000,
                        disabled=use_upload and data_files is not None)

if data_files:
    adm, bill, diag, hosp, pat = load_uploaded(data_files)
    st.sidebar.success("Using your uploaded data ✓")
else:
    adm, bill, diag, hosp, pat = make_synthetic_data(n_admissions=n_demo)
    st.sidebar.warning("Using synthetic demo data (schema-matched, not real hospital records)")

df = build_master(adm, bill, diag, hosp, pat)

st.markdown(
    f"**Dataset:** {len(adm):,} admissions · {pat['patient_id'].nunique():,} patients · "
    f"{hosp.shape[0]} hospitals · date range {adm['admit_date'].min().date()} → {adm['admit_date'].max().date()}"
)

tabs = st.tabs([
    "1 · EDA", "2 · Capacity & LOS", "3 · Readmissions", "4 · Financials",
    "5 · Stat Tests", "6 · ML — LOS", "7 · ML — Readmission Risk",
    "8 · Clustering", "9 · Summary",
])

# ────────────────────────────────────────────────────────────────────────────
# 1. EDA
# ────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Exploratory Data Analysis")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total admissions", f"{len(adm):,}")
    c2.metric("Unique patients", f"{pat['patient_id'].nunique():,}")
    c3.metric("Avg LOS (days)", f"{adm['los_days'].mean():.1f}")
    c4.metric("30-day readmission", f"{adm['readmitted_30d'].mean()*100:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        sns.histplot(df["age"], bins=30, ax=ax, color="#378ADD")
        ax.set_title("Age distribution", fontweight="bold")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ward_counts = df["ward_type"].value_counts()
        ax.pie(ward_counts, labels=ward_counts.index, autopct="%1.0f%%",
               colors=[PALETTE.get(w, "#999") for w in ward_counts.index])
        ax.set_title("Admissions by ward type", fontweight="bold")
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(6, 4))
        yearly = df.groupby("admit_year").size()
        ax.plot(yearly.index, yearly.values, "o-", color="#1D9E75")
        ax.set_title("Admissions per year", fontweight="bold")
        ax.set_xlabel("Year"); ax.set_ylabel("Admissions")
        st.pyplot(fig)
    with col4:
        fig, ax = plt.subplots(figsize=(6, 4))
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_counts = df["admit_dow_name"].value_counts().reindex(dow_order)
        ax.bar(dow_counts.index, dow_counts.values, color="#EF9F27")
        ax.set_title("Admissions by day of week", fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig)

# ────────────────────────────────────────────────────────────────────────────
# 2. Capacity & LOS
# ────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Capacity & Length of Stay")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        order = df.groupby("ward_type")["los_days"].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x="ward_type", y="los_days", order=order, ax=ax,
                    hue="ward_type", palette=PALETTE, legend=False, showfliers=False)
        ax.set_title("LOS by ward type", fontweight="bold")
        st.pyplot(fig)
    with col2:
        long_stay = df.groupby("ward_type")["los_days"].apply(lambda s: (s > 14).mean() * 100)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        long_stay.sort_values(ascending=False).plot(kind="bar", ax=ax,
            color=[PALETTE.get(w, "#999") for w in long_stay.sort_values(ascending=False).index])
        ax.set_title("% of stays exceeding 14 days", fontweight="bold")
        ax.set_ylabel("% of stays")
        st.pyplot(fig)

    st.markdown("**Hospital-level occupancy proxy** (admissions ÷ beds, higher = more strained)")
    occ = df.groupby(["hospital_id", "state", "beds"]).size().reset_index(name="admissions")
    occ["occupancy_ratio"] = occ["admissions"] / occ["beds"]
    top_occ = occ.sort_values("occupancy_ratio", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(top_occ["hospital_id"] + " (" + top_occ["state"] + ")",
            top_occ["occupancy_ratio"], color="#E24B4A")
    ax.set_xlabel("Admissions per bed (proxy ratio)")
    ax.set_title("Top 15 most strained hospitals", fontweight="bold")
    ax.invert_yaxis()
    st.pyplot(fig)

# ────────────────────────────────────────────────────────────────────────────
# 3. Readmissions
# ────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Readmission Analysis")
    col1, col2 = st.columns(2)
    with col1:
        by_cat = diag[diag["is_primary"]].merge(
            adm[["admission_id", "readmitted_30d"]], on="admission_id", how="left"
        ).groupby("category")["readmitted_30d"].mean().sort_values(ascending=False) * 100
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        by_cat.plot(kind="bar", ax=ax, color="#c0392b")
        ax.set_ylabel("30-day readmission %")
        ax.set_title("Readmission rate by diagnosis category", fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig)
    with col2:
        by_tier = df.groupby("tier")["readmitted_30d"].mean().sort_values(ascending=False) * 100
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        by_tier.plot(kind="bar", ax=ax, color="#378ADD")
        ax.set_ylabel("30-day readmission %")
        ax.set_title("Readmission rate by hospital tier", fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        st.pyplot(fig)

# ────────────────────────────────────────────────────────────────────────────
# 4. Financials
# ────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Financial / Billing Analysis")
    col1, col2 = st.columns(2)
    with col1:
        cost_cat = bill.groupby("top_cost_category")["total_cost"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        (cost_cat / 1e7).plot(kind="bar", ax=ax, color="#1D9E75")
        ax.set_ylabel("Total cost (₹ Cr)")
        ax.set_title("Total cost by category", fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        st.pyplot(fig)
    with col2:
        bpl_cmp = df.groupby("bpl_status")[["oop_cost", "subsidy"]].mean()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        bpl_cmp.plot(kind="bar", ax=ax, color=["#E24B4A", "#1D9E75"])
        ax.set_ylabel("₹ (mean)")
        ax.set_title("Avg OOP cost & subsidy: BPL vs Non-BPL", fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=0)
        st.pyplot(fig)

    bpl_subsidy_cov = (df.loc[df["bpl_status"] == "BPL", "subsidy"] > 0).mean() * 100
    st.info(f"BPL subsidy coverage: **{bpl_subsidy_cov:.1f}%** of BPL admissions receive any subsidy.")

# ────────────────────────────────────────────────────────────────────────────
# 5. Statistical tests
# ────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Statistical Hypothesis Testing")

    st.markdown("**Kruskal–Wallis: does LOS differ across ward types?**")
    groups = [g["los_days"].values for _, g in df.groupby("ward_type")]
    h_stat, p_val = kruskal(*groups)
    st.write(f"H = {h_stat:.2f}, p-value = {p_val:.4g}",
             "→ significant difference" if p_val < 0.05 else "→ not significant")

    st.markdown("**Mann–Whitney U: OOP cost, BPL vs Non-BPL**")
    bpl_costs = df.loc[df["bpl_status"] == "BPL", "oop_cost"].dropna()
    nonbpl_costs = df.loc[df["bpl_status"] == "Non-BPL", "oop_cost"].dropna()
    u_stat, p_val2 = mannwhitneyu(bpl_costs, nonbpl_costs)
    st.write(f"U = {u_stat:.0f}, p-value = {p_val2:.4g}",
             "→ significant difference" if p_val2 < 0.05 else "→ not significant")
    st.write(f"Mean OOP — BPL: ₹{bpl_costs.mean():,.0f} · Non-BPL: ₹{nonbpl_costs.mean():,.0f}")

    st.markdown("**Chi-square: ward type vs discharge type**")
    ct = pd.crosstab(df["ward_type"], df["discharge_type"])
    chi2, p_val3, dof, _ = chi2_contingency(ct)
    st.dataframe(ct)
    st.write(f"χ² = {chi2:.2f}, dof = {dof}, p-value = {p_val3:.4g}",
             "→ significant association" if p_val3 < 0.05 else "→ not significant")

# ────────────────────────────────────────────────────────────────────────────
# 6. ML — LOS prediction
# ────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Machine Learning — LOS Prediction (Gradient Boosting)")

    with st.spinner("Training model..."):
        feat_cols_num = ["age", "charlson_index", "comorbidity_count", "num_procedures",
                          "hba1c", "creatinine", "haemoglobin", "beds"]
        feat_cols_cat = ["ward_type", "admit_type", "tier"]
        model_df = df[feat_cols_num + feat_cols_cat + ["los_days"]].dropna()

        X = pd.get_dummies(model_df[feat_cols_num + feat_cols_cat], columns=feat_cols_cat, drop_first=True)
        y = model_df["los_days"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        gbm = GradientBoostingRegressor(random_state=42, n_estimators=150, max_depth=3)
        gbm.fit(X_train, y_train)
        preds = gbm.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)

    c1, c2 = st.columns(2)
    c1.metric("R²", f"{r2:.3f}")
    c2.metric("MAE (days)", f"{mae:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(y_test, preds, alpha=0.15, s=10, color="#378ADD")
        lims = [0, max(y_test.max(), preds.max())]
        ax.plot(lims, lims, "--", color="#E24B4A")
        ax.set_xlabel("Actual LOS"); ax.set_ylabel("Predicted LOS")
        ax.set_title("Predicted vs Actual LOS", fontweight="bold")
        st.pyplot(fig)
    with col2:
        importances = pd.Series(gbm.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        importances.plot(kind="barh", ax=ax, color="#1D9E75")
        ax.invert_yaxis()
        ax.set_title("Top 10 feature importances", fontweight="bold")
        st.pyplot(fig)

# ────────────────────────────────────────────────────────────────────────────
# 7. ML — Readmission risk
# ────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("Machine Learning — Readmission Risk (Random Forest)")

    with st.spinner("Training model..."):
        feat_cols_num2 = ["age", "charlson_index", "comorbidity_count", "los_days",
                           "num_procedures", "hba1c", "creatinine", "haemoglobin"]
        feat_cols_cat2 = ["ward_type", "admit_type", "tier"]
        model_df2 = df[feat_cols_num2 + feat_cols_cat2 + ["readmitted_30d"]].dropna()

        X2 = pd.get_dummies(model_df2[feat_cols_num2 + feat_cols_cat2], columns=feat_cols_cat2, drop_first=True)
        y2 = model_df2["readmitted_30d"]
        X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.25,
                                                                  random_state=42, stratify=y2)

        rf = RandomForestClassifier(random_state=42, n_estimators=300, max_depth=8, class_weight="balanced")
        rf.fit(X2_train, y2_train)
        proba = rf.predict_proba(X2_test)[:, 1]
        auc = roc_auc_score(y2_test, proba)

    st.metric("ROC-AUC", f"{auc:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        importances2 = pd.Series(rf.feature_importances_, index=X2.columns).sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        importances2.plot(kind="barh", ax=ax, color="#E24B4A")
        ax.invert_yaxis()
        ax.set_title("Top 10 predictors of readmission", fontweight="bold")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.histplot(proba, bins=25, ax=ax, color="#EF9F27")
        ax.set_title("Predicted readmission risk distribution", fontweight="bold")
        ax.set_xlabel("Predicted probability")
        st.pyplot(fig)

# ────────────────────────────────────────────────────────────────────────────
# 8. Clustering
# ────────────────────────────────────────────────────────────────────────────
with tabs[7]:
    st.subheader("Patient Segmentation — K-Means Clustering")

    k = st.slider("Number of clusters (k)", 2, 8, 4)

    clust_cols = ["age", "charlson_index", "comorbidity_count", "los_days",
                  "num_procedures", "hba1c", "creatinine", "haemoglobin"]
    clust_df = df[clust_cols + ["readmitted_30d", "ward_type", "total_cost"]].dropna()
    X_cl = StandardScaler().fit_transform(clust_df[clust_cols])

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    clust_df = clust_df.copy()
    clust_df["cluster"] = km.fit_predict(X_cl)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_cl)
    cl_pal = ["#E24B4A", "#378ADD", "#1D9E75", "#EF9F27", "#9B59B6", "#16A085", "#F1C40F", "#7F8C8D"]

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        for i in range(k):
            mask = clust_df["cluster"] == i
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=cl_pal[i % len(cl_pal)], s=6, alpha=0.35,
                       label=f"Cluster {i}")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title(f"PCA — Patient clusters (k={k})", fontweight="bold")
        ax.legend(markerscale=3, fontsize=8)
        st.pyplot(fig)
    with col2:
        profile = clust_df.groupby("cluster")[clust_cols + ["readmitted_30d"]].mean()
        profile_norm = (profile - profile.min()) / (profile.max() - profile.min())
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(profile_norm.T, annot=profile.T.round(1), fmt="g", cmap="YlOrRd", ax=ax,
                    linewidths=0.5, cbar_kws={"label": "Normalized value"})
        ax.set_title("Cluster profiles (normalized)", fontweight="bold")
        st.pyplot(fig)

    st.markdown("**Cluster summary**")
    st.dataframe(
        clust_df.groupby("cluster")[["age", "charlson_index", "los_days", "readmitted_30d", "total_cost"]]
        .mean().round(2)
    )

# ────────────────────────────────────────────────────────────────────────────
# 9. Executive summary
# ────────────────────────────────────────────────────────────────────────────
with tabs[8]:
    st.subheader("Executive Summary")

    readm_rate = adm["readmitted_30d"].mean() * 100
    long_icu = (df.loc[df["ward_type"] == "ICU", "los_days"] > 14).mean() * 100
    long_nicu = (df.loc[df["ward_type"] == "NICU", "los_days"] > 14).mean() * 100

    st.markdown(f"""
**DATASET**
- Total admissions: **{len(adm):,}**
- Unique patients: **{pat['patient_id'].nunique():,}**
- Hospitals: **{hosp.shape[0]}**
- Date range: **{adm['admit_date'].min().date()} → {adm['admit_date'].max().date()}**

**BOTTLENECK 1 — Capacity Overload**
- ICU stays exceeding 14 days: **{long_icu:.1f}%**
- NICU stays exceeding 14 days: **{long_nicu:.1f}%**
- → Action: step-down pathways, patient diversion protocols

**BOTTLENECK 2 — Readmission Quality**
- System-wide 30-day readmission: **{readm_rate:.1f}%**
- ML readmission model AUC: see the "ML — Readmission Risk" tab
- → Action: discharge risk scoring, post-discharge follow-up

**BOTTLENECK 3 — Financial Inefficiency**
- BPL vs Non-BPL OOP cost gap: see the "Financials" tab / Mann–Whitney test
- → Action: audit subsidy pipeline, generic drug procurement

**Models built:** Gradient Boosting (LOS), Random Forest (Readmission), K-Means (Segmentation)
""")

    st.caption(
        "Note: when running on synthetic demo data, the exact figures above will differ from "
        "the original notebook's real-data results — upload your CSVs in the sidebar for the real numbers."
    )
