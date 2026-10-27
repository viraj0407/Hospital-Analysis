"""
Hospital Bottleneck Analysis — Streamlit Lite
------------------------------------------------
Interactive Streamlit port of hospital_analysis.ipynb, built against the real
5-table schema (admissions, billing, diagnoses, hospitals, patients).

Run with:
    streamlit run streamlit_app.py

By default it looks for admissions.csv, billing.csv, diagnoses.csv,
hospitals.csv, patients.csv in the same folder as this script. If they
aren't found, use the sidebar to upload them.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency

from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                               RandomForestClassifier, GradientBoostingClassifier)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              roc_auc_score, roc_curve, precision_recall_curve,
                              confusion_matrix, classification_report)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────────────
# Page config & style
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Hospital Bottleneck Analysis — Lite", layout="wide", page_icon="🏥")

sns.set_theme(style="whitegrid", palette="muted", font_scale=0.95)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                      "figure.facecolor": "white"})
PALETTE = {"ICU": "#E24B4A", "NICU": "#c0392b", "HDU": "#EF9F27", "General": "#1D9E75"}
WARD_ORDER = ["General", "HDU", "ICU", "NICU"]

st.title("🏥 Hospital Bottleneck Analysis — Lite")
st.caption("EDA · Statistical Testing · Machine Learning — interactive Streamlit port of the analysis notebook")

DATA_FOLDER = "indian-hospital-dataset"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_data_dir():
    """Look for the CSVs in indian-hospital-dataset/ (next to the script or in the
    current working directory), falling back to the script's own folder."""
    candidates = [
        os.path.join(_SCRIPT_DIR, DATA_FOLDER),
        os.path.join(os.getcwd(), DATA_FOLDER),
        _SCRIPT_DIR,
        os.getcwd(),
    ]
    for d in candidates:
        if os.path.exists(os.path.join(d, "admissions.csv")):
            return d
    return os.path.join(_SCRIPT_DIR, DATA_FOLDER)


DATA_DIR = _find_data_dir()
DEFAULT_FILES = {
    "admissions": os.path.join(DATA_DIR, "admissions.csv"),
    "billing": os.path.join(DATA_DIR, "billing.csv"),
    "diagnoses": os.path.join(DATA_DIR, "diagnoses.csv"),
    "hospitals": os.path.join(DATA_DIR, "hospitals.csv"),
    "patients": os.path.join(DATA_DIR, "patients.csv"),
}


# ────────────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def load_data(paths):
    adm = pd.read_csv(paths["admissions"], parse_dates=["admit_date", "discharge_date"])
    bill = pd.read_csv(paths["billing"])
    diag = pd.read_csv(paths["diagnoses"])
    hosp = pd.read_csv(paths["hospitals"])
    pat = pd.read_csv(paths["patients"])
    return adm, bill, diag, hosp, pat


@st.cache_data(show_spinner="Building master table...")
def build_master(adm, bill, diag, hosp, pat):
    bill_agg = (bill.groupby("admission_id")
                .agg(total_cost=("total_cost_inr", "sum"),
                     oop_cost=("out_of_pocket_inr", "sum"),
                     subsidy=("govt_subsidy_inr", "sum"))
                .reset_index())

    df = (adm.merge(pat, on="patient_id", how="left")
             .merge(hosp, on="hospital_id", how="left")
             .merge(bill_agg, on="admission_id", how="left"))

    df["admit_year"] = df["admit_date"].dt.year
    df["admit_month"] = df["admit_date"].dt.month
    df["admit_dow"] = df["admit_date"].dt.dayofweek
    df["admit_dow_name"] = df["admit_date"].dt.day_name()
    df["age_band"] = pd.cut(df["age"], bins=[0, 18, 35, 50, 65, 120],
                             labels=["0-18", "19-35", "36-50", "51-65", "65+"])
    return df


with st.sidebar:
    st.header("Data source")
    found_defaults = all(os.path.exists(p) for p in DEFAULT_FILES.values())
    paths = None

    if found_defaults:
        st.success(f"Loaded CSVs from {os.path.basename(DATA_DIR)}/ ✓")
        use_upload = st.checkbox("Upload different files instead", value=False)
    else:
        st.warning(f"CSVs not found in {DATA_FOLDER}/ next to the script.")
        use_upload = True

    if use_upload:
        st.caption("Upload all 5 files.")
        f_adm = st.file_uploader("admissions.csv", type="csv", key="adm")
        f_bill = st.file_uploader("billing.csv", type="csv", key="bill")
        f_diag = st.file_uploader("diagnoses.csv", type="csv", key="diag")
        f_hosp = st.file_uploader("hospitals.csv", type="csv", key="hosp")
        f_pat = st.file_uploader("patients.csv", type="csv", key="pat")
        if all([f_adm, f_bill, f_diag, f_hosp, f_pat]):
            paths = {"admissions": f_adm, "billing": f_bill, "diagnoses": f_diag,
                      "hospitals": f_hosp, "patients": f_pat}
        elif found_defaults:
            paths = DEFAULT_FILES
    else:
        paths = DEFAULT_FILES

if paths is None:
    st.info("⬅️ Upload all 5 CSVs in the sidebar to begin.")
    st.stop()

adm, bill, diag, hosp, pat = load_data(paths)
df = build_master(adm, bill, diag, hosp, pat)

st.markdown(
    f"**Dataset:** {len(adm):,} admissions · {pat['patient_id'].nunique():,} patients · "
    f"{hosp.shape[0]} hospitals · date range {adm['admit_date'].min().date()} → {adm['admit_date'].max().date()}"
)

tabs = st.tabs([
    "1 · EDA", "2 · Capacity", "3 · Length of Stay", "4 · Readmissions", "5 · Financials",
    "6 · Stat Tests", "7 · ML — LOS", "8 · ML — Readmission", "9 · Clustering", "10 · Summary",
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

    st.markdown("**Data quality**")
    miss = df.isnull().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        if len(miss) > 0:
            miss.plot(kind="barh", ax=ax, color="#E24B4A")
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
            ax.set_title("Missing value rate by column", fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No missing values!", ha="center", va="center",
                     fontsize=14, color="green", transform=ax.transAxes)
            ax.set_title("Missing values", fontweight="bold")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        df.dtypes.astype(str).value_counts().plot(kind="bar", ax=ax,
            color=["#378ADD", "#EF9F27", "#1D9E75", "#E24B4A"])
        ax.set_title("Column data types", fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=0)
        st.pyplot(fig)

    st.markdown("**Numeric distributions**")
    num_cols = ["los_days", "age", "charlson_index", "num_procedures",
                "comorbidity_count", "hba1c", "creatinine", "haemoglobin"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        data = df[col].dropna()
        axes[i].hist(data, bins=40, color="#378ADD", alpha=0.7, edgecolor="white")
        axes[i].axvline(data.mean(), color="#E24B4A", lw=1.5, linestyle="--", label=f"mean={data.mean():.1f}")
        axes[i].axvline(data.median(), color="#1D9E75", lw=1.5, linestyle=":", label=f"median={data.median():.1f}")
        axes[i].set_title(col.replace("_", " ").title(), fontweight="bold", fontsize=11)
        axes[i].legend(fontsize=7)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("**Categorical breakdowns**")
    cat_cols = ["admit_type", "ward_type", "discharge_type", "tier", "insurance_type"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for i, col in enumerate(cat_cols):
        col_data = df[col].value_counts()
        col_data.plot(kind="bar", ax=axes[i], color=sns.color_palette("muted", len(col_data)))
        axes[i].set_title(col.replace("_", " ").title(), fontweight="bold")
        axes[i].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("**Correlation matrix — clinical & financial variables**")
    corr_cols = ["los_days", "age", "charlson_index", "num_procedures", "comorbidity_count",
                 "hba1c", "creatinine", "haemoglobin", "systolic_bp", "readmitted_30d", "total_cost"]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                linewidths=0.5, ax=ax, annot_kws={"size": 8}, vmin=-0.6, vmax=0.6)
    ax.set_title("Correlation Matrix", fontweight="bold")
    st.pyplot(fig)

# ────────────────────────────────────────────────────────────────────────────
# 2. Capacity & Occupancy
# ────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Capacity & Occupancy Analysis")

    obs_days = (df["admit_date"].max() - df["admit_date"].min()).days
    hosp_load = (df.groupby("hospital_id")
                   .agg(total_los=("los_days", "sum"),
                        n_adm=("admission_id", "count"),
                        ra_rate=("readmitted_30d", "mean"),
                        avg_los=("los_days", "mean"),
                        avg_procedures=("num_procedures", "mean"))
                   .join(hosp.set_index("hospital_id")[["name", "beds", "tier", "state", "teaching"]]))
    hosp_load["avg_concurrent_pts"] = hosp_load["total_los"] / obs_days
    hosp_load["occupancy_rate"] = (hosp_load["avg_concurrent_pts"] / hosp_load["beds"] * 100).round(1)
    hosp_load["short_name"] = hosp_load["state"] + " (" + hosp_load["tier"] + ")"
    top_occ = hosp_load.sort_values("occupancy_rate", ascending=False).head(15)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        colors = ["#E24B4A" if x > 8 else "#EF9F27" if x > 4 else "#1D9E75" for x in top_occ["occupancy_rate"]]
        ax.barh(top_occ["short_name"], top_occ["occupancy_rate"], color=colors)
        ax.set_xlabel("Occupancy rate (avg concurrent pts / beds, %)")
        ax.set_title("Hospital Occupancy Rate (top 15)", fontweight="bold")
        ax.invert_yaxis()
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        tier_colors = {"tier1": "#378ADD", "tier2": "#EF9F27", "tier3": "#E24B4A"}
        for tier, grp in hosp_load.groupby("tier"):
            ax.scatter(grp["beds"], grp["n_adm"], c=tier_colors.get(tier, "gray"),
                       label=tier, alpha=0.75, s=80, edgecolors="white", linewidths=0.5)
        ax.set_xlabel("Bed capacity"); ax.set_ylabel("Total admissions")
        ax.set_title("Bed Capacity vs Admission Volume by Tier", fontweight="bold")
        ax.legend(title="Tier")
        st.pyplot(fig)

    st.markdown("**Top 5 overloaded hospitals**")
    st.dataframe(top_occ[["short_name", "beds", "n_adm", "occupancy_rate", "avg_los"]].head(5))

    col3, col4 = st.columns(2)
    with col3:
        monthly = df.groupby(df["admit_date"].dt.to_period("M")).size().reset_index()
        monthly.columns = ["period", "count"]
        monthly["date"] = monthly["period"].dt.to_timestamp()
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(monthly["date"], monthly["count"], color="#378ADD", lw=1.5)
        ax.fill_between(monthly["date"], monthly["count"], alpha=0.15, color="#378ADD")
        ax.set_title("Monthly Admission Volume", fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=30)
        st.pyplot(fig)
    with col4:
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_counts = df.groupby("admit_dow_name").size().reindex(dow_order)
        colors_dow = ["#E24B4A" if d in ["Saturday", "Sunday"] else "#378ADD" for d in dow_order]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        dow_counts.plot(kind="bar", ax=ax, color=colors_dow)
        ax.set_title("Admissions by Day of Week (weekends highlighted)", fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=30)
        st.pyplot(fig)

# ────────────────────────────────────────────────────────────────────────────
# 3. Length of Stay
# ────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Length of Stay (LOS) Deep Dive")

    col1, col2, col3 = st.columns(3)
    with col1:
        ward_data = [df[df["ward_type"] == w]["los_days"].dropna() for w in WARD_ORDER]
        fig, ax = plt.subplots(figsize=(5, 4.5))
        parts = ax.violinplot(ward_data, positions=range(len(WARD_ORDER)), showmedians=True)
        pal = ["#1D9E75", "#EF9F27", "#E24B4A", "#c0392b"]
        for pc, col in zip(parts["bodies"], pal):
            pc.set_facecolor(col); pc.set_alpha(0.7)
        ax.set_xticks(range(len(WARD_ORDER))); ax.set_xticklabels(WARD_ORDER)
        ax.set_ylabel("LOS (days)"); ax.set_title("LOS by Ward Type", fontweight="bold")
        st.pyplot(fig)
    with col2:
        admit_order = ["OPD", "Elective", "Emergency"]
        df_plot = df[df["admit_type"].isin(admit_order)]
        fig, ax = plt.subplots(figsize=(5, 4.5))
        sns.boxplot(x="admit_type", y="los_days", data=df_plot, order=admit_order, ax=ax,
                    hue="admit_type", palette="Blues", legend=False, showfliers=False)
        ax.set_title("LOS by Admission Type", fontweight="bold")
        st.pyplot(fig)
    with col3:
        long_stay = df.groupby("ward_type")["los_days"].apply(lambda x: (x > 14).mean() * 100).reindex(WARD_ORDER)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        bars = ax.bar(long_stay.index, long_stay.values, color=pal)
        for bar, val in zip(bars, long_stay.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.1f}%",
                     ha="center", fontsize=9, fontweight="bold")
        ax.set_ylabel("% of stays > 14 days")
        ax.set_title("Long-Stay Rate by Ward", fontweight="bold")
        st.pyplot(fig)

    col4, col5, col6 = st.columns(3)
    with col4:
        age_los = df.groupby("age_band")["los_days"].median()
        fig, ax = plt.subplots(figsize=(5, 4))
        age_los.plot(kind="bar", ax=ax, color="#378ADD", edgecolor="white")
        ax.set_title("Median LOS by Age Band", fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=0)
        st.pyplot(fig)
    with col5:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(data=df, x="gender", y="los_days", hue="gender", ax=ax,
                    palette="Set2", legend=False, showfliers=False)
        ax.set_title("LOS by Gender", fontweight="bold")
        st.pyplot(fig)
    with col6:
        sample = df[["charlson_index", "los_days"]].dropna().sample(min(5000, len(df)), random_state=42)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(sample["charlson_index"], sample["los_days"], alpha=0.15, s=10, color="#378ADD")
        m, b = np.polyfit(sample["charlson_index"], sample["los_days"], 1)
        x_line = np.linspace(sample["charlson_index"].min(), sample["charlson_index"].max(), 100)
        ax.plot(x_line, m * x_line + b, color="#E24B4A", lw=2, label=f"slope={m:.2f}")
        ax.set_xlabel("Charlson Index"); ax.set_ylabel("LOS (days)")
        ax.set_title("LOS vs Charlson Index", fontweight="bold")
        ax.legend()
        st.pyplot(fig)

# ────────────────────────────────────────────────────────────────────────────
# 4. Readmissions
# ────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Readmission Analysis")

    prim_diag = diag[diag["diag_rank"] == 1].merge(
        df[["admission_id", "readmitted_30d", "readmitted_7d", "los_days"]], on="admission_id")
    ra_by_diag = (prim_diag.groupby("diag_category")
                  .agg(ra30=("readmitted_30d", "mean"), ra7=("readmitted_7d", "mean"),
                       n=("readmitted_30d", "count"), avg_los=("los_days", "mean"))
                  .sort_values("ra30", ascending=False))

    col1, col2 = st.columns(2)
    with col1:
        colors_ra = ["#E24B4A" if v > 0.15 else "#EF9F27" if v > 0.10 else "#1D9E75" for v in ra_by_diag["ra30"]]
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.barh(ra_by_diag.index, ra_by_diag["ra30"] * 100, color=colors_ra)
        ax.axvline(ra_by_diag["ra30"].mean() * 100, color="black", lw=1.2, linestyle="--",
                   label=f"System avg: {ra_by_diag['ra30'].mean()*100:.1f}%")
        ax.set_xlabel("30-day readmission rate (%)")
        ax.set_title("Readmission by Primary Diagnosis Category", fontweight="bold")
        ax.legend()
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.scatter(ra_by_diag["n"], ra_by_diag["ra30"] * 100, s=ra_by_diag["avg_los"] * 30,
                   c=colors_ra, alpha=0.75, edgecolors="white", linewidths=0.5)
        for idx, row in ra_by_diag.iterrows():
            ax.annotate(idx, (row["n"], row["ra30"] * 100), fontsize=8, ha="center", va="bottom")
        ax.set_xlabel("Number of admissions"); ax.set_ylabel("30-day readmission rate (%)")
        ax.set_title("Volume vs Readmission Rate (bubble = avg LOS)", fontweight="bold")
        st.pyplot(fig)

    col3, col4, col5 = st.columns(3)
    with col3:
        tier_ra = df.groupby("tier")["readmitted_30d"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(5, 4))
        (tier_ra * 100).plot(kind="bar", ax=ax, color=["#E24B4A", "#EF9F27", "#1D9E75"][:len(tier_ra)])
        ax.set_ylabel("%"); ax.set_title("Readmission by Tier", fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=0)
        st.pyplot(fig)
    with col4:
        cc_ra = df.groupby("comorbidity_count")["readmitted_30d"].mean()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(cc_ra.index, cc_ra.values * 100, color="#E24B4A", alpha=0.7)
        ax.set_xlabel("Comorbidity count"); ax.set_ylabel("30-day RA rate (%)")
        ax.set_title("Readmission vs Comorbidity", fontweight="bold")
        st.pyplot(fig)
    with col5:
        ins_ra = df.groupby(["insurance_type", "ward_type"])["readmitted_30d"].mean().unstack()
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(ins_ra * 100, annot=True, fmt=".1f", ax=ax, cmap="YlOrRd",
                    linewidths=0.5, cbar_kws={"label": "RA %"})
        ax.set_title("Readmission: Insurance × Ward", fontweight="bold")
        st.pyplot(fig)

# ────────────────────────────────────────────────────────────────────────────
# 5. Financials
# ────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Financial / Billing Analysis")

    cost_cat = bill.groupby("cost_category")["total_cost_inr"].sum().sort_values(ascending=False)
    col1, col2, col3 = st.columns(3)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.pie(cost_cat.values, labels=cost_cat.index, autopct="%1.1f%%",
               colors=["#E24B4A", "#EF9F27", "#378ADD", "#1D9E75", "#9B59B6", "#16A085"][:len(cost_cat)],
               pctdistance=0.8, wedgeprops={"edgecolor": "white", "linewidth": 2})
        ax.set_title("Cost Distribution by Category", fontweight="bold")
        st.pyplot(fig)
    with col2:
        bpl_oop = df.loc[df["bpl_card"] == True, "oop_cost"].dropna()
        nonbpl_oop = df.loc[df["bpl_card"] == False, "oop_cost"].dropna()
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.hist(bpl_oop.clip(0, 200000), bins=50, alpha=0.6, color="#E24B4A", label="BPL")
        ax.hist(nonbpl_oop.clip(0, 200000), bins=50, alpha=0.5, color="#378ADD", label="Non-BPL")
        ax.axvline(bpl_oop.mean(), color="#E24B4A", lw=2, linestyle="--")
        ax.axvline(nonbpl_oop.mean(), color="#378ADD", lw=2, linestyle="--")
        ax.set_xlabel("Out-of-pocket cost (INR)"); ax.set_title("OOP: BPL vs Non-BPL", fontweight="bold")
        ax.legend()
        st.pyplot(fig)
    with col3:
        sample2 = df[["los_days", "total_cost", "ward_type"]].dropna().sample(min(3000, len(df)), random_state=42)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        for ward, grp in sample2.groupby("ward_type"):
            ax.scatter(grp["los_days"], grp["total_cost"], alpha=0.2, s=10, label=ward,
                       color=PALETTE.get(ward, "gray"))
        ax.set_xlabel("LOS (days)"); ax.set_ylabel("Total cost (INR)")
        ax.set_title("Total Cost vs LOS by Ward", fontweight="bold")
        ax.legend(markerscale=2)
        st.pyplot(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("BPL avg OOP", f"₹{bpl_oop.mean():,.0f}")
    c2.metric("Non-BPL avg OOP", f"₹{nonbpl_oop.mean():,.0f}")
    bpl_subsidy_cov = df.loc[df["bpl_card"], "subsidy"].sum() / df.loc[df["bpl_card"], "total_cost"].sum() * 100
    c3.metric("BPL subsidy coverage", f"{bpl_subsidy_cov:.1f}%")

# ────────────────────────────────────────────────────────────────────────────
# 6. Statistical tests
# ────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Statistical Hypothesis Testing")

    st.markdown("**Test 1 — Kruskal-Wallis: LOS across ward types**")
    groups_ward = [df[df["ward_type"] == w]["los_days"].dropna() for w in WARD_ORDER]
    h_stat, p_kw = kruskal(*groups_ward)
    st.write(f"H = {h_stat:.2f}, p = {p_kw:.2e} → "
             + ("**SIGNIFICANT**" if p_kw < 0.05 else "not significant") + " (α=0.05)")
    rows = []
    for i in range(len(WARD_ORDER)):
        for j in range(i + 1, len(WARD_ORDER)):
            g1 = df[df["ward_type"] == WARD_ORDER[i]]["los_days"].dropna()
            g2 = df[df["ward_type"] == WARD_ORDER[j]]["los_days"].dropna()
            u, p = mannwhitneyu(g1, g2)
            rows.append({"comparison": f"{WARD_ORDER[i]} vs {WARD_ORDER[j]}", "p_value": p,
                         "significant (Bonferroni α=0.0083)": p < 0.0083})
    st.dataframe(pd.DataFrame(rows))

    st.markdown("**Test 2 — Chi-squared: readmission × admit type**")
    ct = pd.crosstab(df["admit_type"], df["readmitted_30d"])
    chi2, p_chi2, dof, _ = chi2_contingency(ct)
    st.dataframe(ct)
    st.write(f"χ² = {chi2:.2f}, dof = {dof}, p = {p_chi2:.2e} → "
             + ("**SIGNIFICANT**" if p_chi2 < 0.05 else "not significant"))

    st.markdown("**Test 3 — Mann-Whitney: OOP cost, BPL vs Non-BPL**")
    bpl_oop = df.loc[df["bpl_card"] == True, "oop_cost"].dropna()
    nonbpl_oop = df.loc[df["bpl_card"] == False, "oop_cost"].dropna()
    u3, p_mw3 = mannwhitneyu(bpl_oop, nonbpl_oop)
    st.write(f"U = {u3:.0f}, p = {p_mw3:.2e} → BPL median ₹{bpl_oop.median():,.0f} vs "
             f"Non-BPL median ₹{nonbpl_oop.median():,.0f} → "
             + ("**SIGNIFICANT**" if p_mw3 < 0.05 else "not significant"))

    st.markdown("**Test 4 — Kruskal-Wallis: LOS across hospital tiers**")
    tiers = sorted(df["tier"].dropna().unique())
    groups_tier = [df[df["tier"] == t]["los_days"].dropna() for t in tiers]
    h4, p_kw4 = kruskal(*groups_tier)
    st.write(f"H = {h4:.2f}, p = {p_kw4:.2e}")
    st.dataframe(pd.DataFrame({"tier": tiers, "median_los": [df[df["tier"] == t]["los_days"].median() for t in tiers]}))

# ────────────────────────────────────────────────────────────────────────────
# 7. ML — LOS prediction
# ────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("Machine Learning — LOS Prediction")

    feat_cols = ["admit_type", "ward_type", "tier", "charlson_index", "num_procedures",
                 "age", "comorbidity_count", "prev_admissions", "hba1c",
                 "creatinine", "haemoglobin", "systolic_bp"]
    target = "los_days"

    with st.spinner("Training models (Gradient Boosting, Random Forest, Ridge)..."):
        ml = df[feat_cols + [target]].dropna().copy()
        for c in ["admit_type", "ward_type", "tier"]:
            ml[c] = LabelEncoder().fit_transform(ml[c].astype(str))
        X = ml[feat_cols]; y = ml[target]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                                            learning_rate=0.05, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=8, n_jobs=-1, random_state=42),
            "Ridge Regression": Ridge(alpha=1.0),
        }
        results = {}
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            results[name] = {"MAE": mean_absolute_error(y_te, preds),
                              "RMSE": np.sqrt(mean_squared_error(y_te, preds)),
                              "R2": r2_score(y_te, preds), "preds": preds}

    metrics_df = pd.DataFrame({k: {"MAE": v["MAE"], "RMSE": v["RMSE"], "R2": v["R2"]}
                                for k, v in results.items()}).T
    st.dataframe(metrics_df.round(3))

    best_model = models["Gradient Boosting"]
    fi = pd.Series(best_model.feature_importances_, index=feat_cols).sort_values(ascending=True)
    preds_gb = results["Gradient Boosting"]["preds"]

    col1, col2, col3 = st.columns(3)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        colors_fi = ["#E24B4A" if v > 0.05 else "#EF9F27" if v > 0.01 else "#73726c" for v in fi.values]
        (fi * 100).plot(kind="barh", ax=ax, color=colors_fi)
        ax.set_xlabel("Importance (%)"); ax.set_title("Feature Importance (GBM)", fontweight="bold")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.scatter(y_te, preds_gb, alpha=0.05, s=5, color="#378ADD")
        ax.plot([0, 30], [0, 30], "r--", lw=1.5, label="Perfect prediction")
        ax.set_xlabel("Actual LOS"); ax.set_ylabel("Predicted LOS")
        ax.set_title(f"Predicted vs Actual (R²={results['Gradient Boosting']['R2']:.3f})", fontweight="bold")
        ax.legend()
        st.pyplot(fig)
    with col3:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        metrics_df[["MAE", "RMSE"]].plot(kind="bar", ax=ax, color=["#E24B4A", "#EF9F27"], edgecolor="white")
        ax.set_title("Model Comparison (days)", fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=20)
        st.pyplot(fig)

# ────────────────────────────────────────────────────────────────────────────
# 8. ML — Readmission risk
# ────────────────────────────────────────────────────────────────────────────
with tabs[7]:
    st.subheader("Machine Learning — Readmission Risk Score")

    feat2 = ["admit_type", "ward_type", "tier", "charlson_index", "num_procedures",
             "age", "comorbidity_count", "prev_admissions", "los_days",
             "total_cost", "hba1c", "creatinine"]

    with st.spinner("Training models (Random Forest, Gradient Boosting, Logistic Regression)..."):
        ml2 = df[feat2 + ["readmitted_30d"]].dropna().copy()
        for c in ["admit_type", "ward_type", "tier"]:
            ml2[c] = LabelEncoder().fit_transform(ml2[c].astype(str))
        X2 = ml2[feat2]; y2 = ml2["readmitted_30d"]
        X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_tr2_sc = scaler.fit_transform(X_tr2)
        X_te2_sc = scaler.transform(X_te2)

        cls_models = {
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, n_jobs=-1, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                              learning_rate=0.05, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        }
        cls_results = {}
        for name, model in cls_models.items():
            X_tr_use = X_tr2_sc if name == "Logistic Regression" else X_tr2
            X_te_use = X_te2_sc if name == "Logistic Regression" else X_te2
            model.fit(X_tr_use, y_tr2)
            proba = model.predict_proba(X_te_use)[:, 1]
            cls_results[name] = {"proba": proba, "auc": roc_auc_score(y_te2, proba), "model": model}

    auc_df = pd.DataFrame({k: {"AUC": v["auc"]} for k, v in cls_results.items()}).T
    st.dataframe(auc_df.round(3))

    best_cls = cls_models["Random Forest"]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        colors_roc = ["#E24B4A", "#378ADD", "#1D9E75"]
        for (name, res), c in zip(cls_results.items(), colors_roc):
            fpr, tpr, _ = roc_curve(y_te2, res["proba"])
            ax.plot(fpr, tpr, lw=2, color=c, label=f"{name} ({res['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curves", fontweight="bold")
        ax.legend(fontsize=7)
        st.pyplot(fig)
    with col2:
        prec, rec, _ = precision_recall_curve(y_te2, cls_results["Random Forest"]["proba"])
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot(rec, prec, color="#E24B4A", lw=2)
        ax.axhline(y2.mean(), color="gray", lw=1, linestyle="--", label=f"Baseline ({y2.mean():.2f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("PR Curve (Random Forest)", fontweight="bold")
        ax.legend(fontsize=8)
        st.pyplot(fig)
    with col3:
        preds_rf = (cls_results["Random Forest"]["proba"] >= 0.5).astype(int)
        cm = confusion_matrix(y_te2, preds_rf)
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax,
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
        ax.set_title("Confusion Matrix (RF)", fontweight="bold")
        st.pyplot(fig)
    with col4:
        fi2 = pd.Series(best_cls.feature_importances_, index=feat2).sort_values(ascending=True)
        colors_fi2 = ["#E24B4A" if v > 0.10 else "#EF9F27" if v > 0.05 else "#73726c" for v in fi2.values]
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        (fi2 * 100).plot(kind="barh", ax=ax, color=colors_fi2)
        ax.set_xlabel("Importance (%)"); ax.set_title("Feature Importance (RF)", fontweight="bold")
        st.pyplot(fig)

    with st.expander("Classification report (Random Forest, threshold=0.5)"):
        st.text(classification_report(y_te2, preds_rf))

# ────────────────────────────────────────────────────────────────────────────
# 9. Clustering
# ────────────────────────────────────────────────────────────────────────────
with tabs[8]:
    st.subheader("Patient Segmentation — K-Means Clustering")

    clust_cols = ["age", "charlson_index", "comorbidity_count", "los_days",
                  "num_procedures", "hba1c", "creatinine", "haemoglobin"]
    clust_df_base = df[clust_cols + ["readmitted_30d", "ward_type", "total_cost"]].dropna()
    X_cl = StandardScaler().fit_transform(clust_df_base[clust_cols])

    col_a, col_b = st.columns([1, 2])
    with col_a:
        k = st.slider("Number of clusters (k)", 2, 8, 4)

    with st.spinner("Fitting K-Means..."):
        inertias = []
        K_range = range(2, 9)
        for kk in K_range:
            km_tmp = KMeans(n_clusters=kk, random_state=42, n_init=10)
            km_tmp.fit(X_cl)
            inertias.append(km_tmp.inertia_)

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        clust_df = clust_df_base.copy()
        clust_df["cluster"] = km.fit_predict(X_cl)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_cl)
    cl_pal = ["#E24B4A", "#378ADD", "#1D9E75", "#EF9F27", "#9B59B6", "#16A085", "#F1C40F", "#7F8C8D"]

    col1, col2, col3 = st.columns(3)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(list(K_range), inertias, "o-", color="#378ADD", lw=2)
        ax.axvline(k, color="#E24B4A", lw=1.5, linestyle="--", label=f"k={k} selected")
        ax.set_xlabel("k"); ax.set_ylabel("Inertia")
        ax.set_title("Elbow Plot", fontweight="bold")
        ax.legend()
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        for i in range(k):
            mask = clust_df["cluster"] == i
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=cl_pal[i % len(cl_pal)], s=5, alpha=0.3,
                       label=f"Cluster {i}")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title(f"PCA — Patient Clusters (k={k})", fontweight="bold")
        ax.legend(markerscale=3, fontsize=7)
        st.pyplot(fig)
    with col3:
        profile = clust_df.groupby("cluster")[clust_cols + ["readmitted_30d"]].mean()
        profile_norm = (profile - profile.min()) / (profile.max() - profile.min())
        fig, ax = plt.subplots(figsize=(5, 4.5))
        sns.heatmap(profile_norm.T, annot=profile.T.round(1), fmt="g", cmap="YlOrRd", ax=ax,
                    linewidths=0.5, cbar_kws={"label": "Normalized value"})
        ax.set_xticklabels([f"C{i}" for i in range(k)])
        ax.set_title("Cluster Profiles", fontweight="bold")
        st.pyplot(fig)

    st.markdown("**Cluster summary**")
    st.dataframe(clust_df.groupby("cluster")[["age", "charlson_index", "los_days", "readmitted_30d", "total_cost"]]
                 .mean().round(2))

# ────────────────────────────────────────────────────────────────────────────
# 10. Executive summary
# ────────────────────────────────────────────────────────────────────────────
with tabs[9]:
    st.subheader("Executive Summary")

    long_icu = (df.loc[df["ward_type"] == "ICU", "los_days"] > 14).mean() * 100
    avg_icu = df.loc[df["ward_type"] == "ICU", "los_days"].mean()
    long_nicu = (df.loc[df["ward_type"] == "NICU", "los_days"] > 14).mean() * 100
    avg_nicu = df.loc[df["ward_type"] == "NICU", "los_days"].mean()
    readm_rate = adm["readmitted_30d"].mean() * 100
    ra_by_diag_sorted = ra_by_diag.sort_values("ra30", ascending=False)
    top_diag = ra_by_diag_sorted.index[0] if len(ra_by_diag_sorted) else "N/A"
    top_diag_rate = ra_by_diag_sorted["ra30"].iloc[0] * 100 if len(ra_by_diag_sorted) else 0
    pharm_cost = bill.loc[bill["cost_category"] == "Pharmacy", "total_cost_inr"].sum() / 1e7
    bpl_oop_mean = df.loc[df["bpl_card"] == True, "oop_cost"].mean()
    nonbpl_oop_mean = df.loc[df["bpl_card"] == False, "oop_cost"].mean()
    bpl_subsidy_cov = df.loc[df["bpl_card"], "subsidy"].sum() / df.loc[df["bpl_card"], "total_cost"].sum() * 100
    gbm_r2 = results["Gradient Boosting"]["R2"]
    gbm_mae = results["Gradient Boosting"]["MAE"]
    rf_auc = cls_results["Random Forest"]["auc"]
    top_predictors = pd.Series(cls_models["Random Forest"].feature_importances_,
                                index=feat2).sort_values(ascending=False).head(3).index.tolist()


    # Extra numbers used by the "how to handle it" guidance
    icu_nicu = df[df["ward_type"].isin(["ICU", "NICU"])]
    excess_bed_days = (icu_nicu["los_days"] - 14).clip(lower=0).sum()
    n_long_critical = int((icu_nicu["los_days"] > 14).sum())
    top3_overloaded = ", ".join(top_occ["short_name"].head(3).tolist())
    weekend_share = df["admit_dow"].isin([5, 6]).mean() * 100
    emerg_share = (df["admit_type"] == "Emergency").mean() * 100
    pharm_share = (bill.loc[bill["cost_category"] == "Pharmacy", "total_cost_inr"].sum()
                   / bill["total_cost_inr"].sum() * 100)
    rf_proba = np.asarray(cls_results["Random Forest"]["proba"])
    thr90 = np.quantile(rf_proba, 0.90)
    top_decile_capture = (np.asarray(y_te2)[rf_proba >= thr90].sum() / max(np.asarray(y_te2).sum(), 1)) * 100

    st.markdown(f"""
**DATASET**
- Total admissions: **{len(adm):,}** · Unique patients: **{pat['patient_id'].nunique():,}**
- Hospitals: **{hosp.shape[0]}** · Date range: **{adm['admit_date'].min().date()} → {adm['admit_date'].max().date()}**

---

### BOTTLENECK 1 — Capacity Overload
**What the data shows**
- NICU: **{long_nicu:.1f}%** of stays exceed 14 days (avg {avg_nicu:.1f}d)
- ICU: **{long_icu:.1f}%** of stays exceed 14 days (avg {avg_icu:.1f}d)
- **{n_long_critical:,}** long ICU/NICU stays consume **{excess_bed_days:,.0f}** critical-care bed-days beyond day 14
- Most strained hospitals (occupancy proxy): {top3_overloaded}
- LOS model (GBM) R² = **{gbm_r2:.3f}** — ward type is a dominant driver (see ML tab)

**How to handle it**
1. **Step-down pathway** — move stable ICU/NICU patients to HDU/General once clinical criteria are met, so the long tail of stays stops blocking beds. Review every critical-care patient at day 7 and day 14.
2. **Predict stay length at admission** — use the LOS model to flag likely long stays on day 1 and plan bed turnover and discharge in advance.
3. **Divert and balance load** — route elective and lower-acuity admissions away from the most strained hospitals above to nearby sites with spare beds, especially in the same state.
4. **Match staffing to demand** — weekends take **{weekend_share:.0f}%** of admissions and emergencies **{emerg_share:.0f}%**; roster senior staff and keep a surge-bed buffer for emergency peaks.
5. **Track it** — monitor % of ICU/NICU stays over 14 days and occupancy per hospital monthly; set a target (e.g. cut long stays by a set percentage) and review progress.

---

### BOTTLENECK 2 — Readmission Quality
**What the data shows**
- System-wide 30-day readmission: **{readm_rate:.1f}%**
- Worst diagnosis category: **{top_diag}** at **{top_diag_rate:.1f}%**
- Readmission RF model AUC = **{rf_auc:.3f}**; top predictors: **{', '.join(top_predictors)}**
- The riskiest 10% of patients (by model score) account for **{top_decile_capture:.0f}%** of readmissions in the test set

**How to handle it**
1. **Risk-score every discharge** — run the readmission model at discharge and flag the top-risk decile for extra attention instead of spreading effort thinly.
2. **Targeted follow-up for flagged patients** — a call within 48-72 hours, a follow-up visit within 7 days, and a medication review before they leave.
3. **Diagnosis-specific discharge bundles** — start with **{top_diag}**: standard discharge checklist, patient education and a clear home-care plan.
4. **Don't discharge too early** — check that short stays for high-comorbidity patients are not driving readmissions (see the comorbidity chart on the Readmissions tab).
5. **Close the loop** — track 7-day and 30-day readmission by hospital and diagnosis, and compare against the system average.

---

### BOTTLENECK 3 — Financial Inefficiency
**What the data shows**
- Pharmacy total cost: **₹{pharm_cost:.0f} Cr** (**{pharm_share:.0f}%** of all billed cost)
- BPL subsidy coverage: **{bpl_subsidy_cov:.1f}%**
- BPL avg OOP **₹{bpl_oop_mean:,.0f}** vs Non-BPL **₹{nonbpl_oop_mean:,.0f}** (significance: Mann-Whitney test, Stat Tests tab)

**How to handle it**
1. **Audit the subsidy pipeline** — check that eligible BPL patients are enrolled at admission and that claims are filed and paid, so subsidy actually reaches the bill.
2. **Cut pharmacy spend** — move to generic and formulary drugs, use pooled procurement across hospitals, and review high-cost prescriptions.
3. **Shorten avoidable stays** — cost rises with length of stay, so the capacity actions above also reduce cost per admission.
4. **Protect low-income patients** — add financial counselling at admission and cap or flag high out-of-pocket bills for BPL patients.
5. **Track it** — monitor OOP per admission by BPL status, subsidy coverage and pharmacy cost share every quarter.

---

**Models built**
1. LOS Prediction — Gradient Boosting (R² = {gbm_r2:.3f}, MAE = {gbm_mae:.2f}d)
2. Readmission Risk — Random Forest (AUC = {rf_auc:.3f})
3. Patient Segmentation — K-Means (k={k}, PCA-verified)
""")

    st.caption(
        "The recommendations are suggested actions based on the patterns in this dataset. "
        "Pilot them and measure results before rolling out system-wide; they are not proven causal effects."
    )