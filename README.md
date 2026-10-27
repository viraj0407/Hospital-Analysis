# 🏥 Hospital Bottleneck Analysis

**EDA · Statistical Testing · Machine Learning**
*April 2026 | Confidential Healthcare Analytics Project*

🔗 Live Dashboard: https://hospital-bottleneck-analysis.streamlit.app/

## 📌 Overview

This project presents an end-to-end data science analysis of hospital system inefficiencies using **120,000 patient admissions across 33 hospitals (2015–2024)**. The goal is to identify critical operational bottlenecks and provide **data-driven, actionable solutions** to improve healthcare delivery.

The analysis integrates:

* Exploratory Data Analysis (EDA)
* Statistical hypothesis testing
* Machine Learning models
* Patient segmentation techniques
* An interactive **Streamlit dashboard** covering all of the above

---

## 📊 Key Objectives

* Identify **capacity bottlenecks** affecting hospital throughput
* Analyze **Length of Stay (LOS)** drivers
* Detect patterns in **30-day readmissions**
* Evaluate **financial inefficiencies and subsidy leakage**
* Build predictive models for **LOS and readmission risk**
* Segment patients for **personalized care pathways**

---

## 🚨 Key Findings

### 1. Capacity Bottlenecks (CRITICAL)

* ICU & NICU are **22.8% of admissions but ~46% of all bed-days** (ICU 32.7%, NICU 13.1%)
* NICU: **55% of stays exceed 14 days** (avg 16.6 days); ICU: **31.5%** (avg 12.9 days)
* ~10,100 long ICU/NICU stays use about **70,000 critical-care bed-days beyond day 14**
* Load per bed is highest in **small tier-3 (district) hospitals** (avg 121 beds vs 761 in tier-1): their average load per bed is ~6.7× the tier-1 average, and up to **~11× at the worst sites (Kerala, Madhya Pradesh, Gujarat)**

👉 **Impact:** Severe overcrowding, delayed admissions, compromised care quality

> Note: the occupancy figure is a proxy (average concurrent patients ÷ beds) computed on this dataset, so it is best read as *relative* load between hospitals, not as absolute "× over capacity".

---

### 2. Readmission Risk (HIGH)

* Overall 30-day readmission rate: **11.8%** (7-day: 0.8%)
* High-risk categories:

  * Perinatal: **18.6%**
  * Cardiovascular: **16.6%**
  * Respiratory: **16.2%**

👉 **Likely contributors (to validate):**

* Discharge planning gaps
* Limited follow-up care
* Medication non-adherence

*The data confirms who is readmitted (comorbidity, LOS, severity); the causes above are hypotheses to test, not findings from this dataset.*

---

### 3. Financial Leakage (HIGH)

* BPL patients OOP: **₹49,341 vs ₹47,500 (non-BPL)** — BPL patients pay slightly *more* out of pocket
* Subsidy covers **48.4% of BPL billed cost vs 50.5% for non-BPL** — subsidy is not reaching BPL patients preferentially
* Total billed: **₹1,149 Cr**, of which out-of-pocket **₹578 Cr** and subsidy **₹571 Cr** (2015–2024)
* Pharmacy is **~30% of billed cost (₹347 Cr)**

👉 **Impact:** Weak subsidy targeting and high drug spend; low-income patients bear the same cost burden as everyone else

---

## 🤖 Machine Learning Models

| Model             | Purpose              | Performance                       | Key Feature                     |
| ----------------- | -------------------- | --------------------------------- | ------------------------------- |
| Gradient Boosting | LOS Prediction       | R² = 0.532, MAE = 2.66 days       | Ward Type (84.6%)               |
| Random Forest     | Readmission Risk     | AUC = 0.736                       | Comorbidity Count (21%), LOS (20%) |
| K-Means           | Patient Segmentation | k = 4 clusters                    | Age + Severity (Charlson index) |

*Gradient Boosting and Random Forest give the same LOS accuracy; Ridge is lower (R² = 0.465). The three readmission models (RF, GBM, Logistic) all reach AUC ≈ 0.735, i.e. moderate discrimination.*

---

## 🧠 Insights

* **Ward assignment drives LOS** far more than clinical variables (84.6% of importance vs 6.2% for Charlson index)
* **Comorbidity count and length of stay are the strongest readmission predictors**
* **The highest-risk 35% of admissions (by model score) account for ~66% of readmissions**, and the top 10% account for ~31%, so targeting follow-up is worthwhile
* **Overload is concentrated in small tier-3 hospitals**, which points to a bed-capacity mismatch rather than a system-wide surge in demand

---

## 🛠️ Recommended Actions

### Capacity Optimization

* Introduce **step-down units (HDU expansion)** with day-7 / day-14 critical-care reviews
* Enable **patient diversion strategies** from the most strained tier-3 hospitals
* Use the LOS model to **flag likely long stays on day 1**
* Roster for weekend (29% of admissions) and emergency (55%) peaks

### Readmission Reduction

* Deploy **ML-based discharge risk scoring** and focus on the top-risk group
* Implement **48–72 hour follow-up programs** for flagged patients
* Standard **discharge bundles**, starting with Perinatal and Cardiovascular

### Financial Reform

* Audit **subsidy disbursement pipeline** (BPL coverage is below non-BPL)
* Promote **generic drug procurement** and pooled purchasing
* Add **financial counselling** for BPL patients at admission

*These are suggested actions based on patterns in the data; pilot and measure before system-wide rollout.*

---

## ▶️ Running the Dashboard

```
pip install -r requirements.txt
streamlit run hospital_analysis_streamlit.py
```

Place the five CSVs (`admissions`, `billing`, `diagnoses`, `hospitals`, `patients`) in an `indian-hospital-dataset/` folder next to the script, or upload them from the sidebar.

---

## 📈 Future Work

* Survival analysis for LOS prediction
* Time-series forecasting for admissions
* SHAP-based model explainability
* Federated learning across hospitals
* Real-time hospital monitoring dashboards

---

## 📂 Dataset Summary

* **Admissions:** 120,000
* **Patients:** 86,400
* **Hospitals:** 33
* **Period:** 2015–2024
* **Features:** Clinical, financial, demographic

---

## ⚠️ Disclaimer

This project is **confidential** and intended for internal analysis purposes only. All insights are based on retrospective data and should be validated before real-world deployment.
