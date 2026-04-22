# 🏥 Hospital Bottleneck Analysis

**EDA · Statistical Testing · Machine Learning**
*April 2026 | Confidential Healthcare Analytics Project*

## 📌 Overview

This project presents an end-to-end data science analysis of hospital system inefficiencies using **120,000 patient admissions across 33 hospitals**. The goal is to identify critical operational bottlenecks and provide **data-driven, actionable solutions** to improve healthcare delivery.

The analysis integrates:

* Exploratory Data Analysis (EDA)
* Statistical hypothesis testing
* Machine Learning models
* Patient segmentation techniques

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

* ICU & NICU dominate bed occupancy
* NICU: **55% long-stay patients (>14 days)**
* District hospitals operating at **10–11× capacity**

👉 **Impact:** Severe overcrowding, delayed admissions, compromised care quality

---

### 2. Readmission Risk (HIGH)

* Overall 30-day readmission rate: **10.8%**
* High-risk categories:

  * Perinatal: **18.6%**
  * Cardiovascular: **16.6%**

👉 **Root causes:**

* Poor discharge planning
* Lack of follow-up care
* Medication non-adherence

---

### 3. Financial Leakage (HIGH)

* BPL patients OOP: **₹49,341 vs ₹47,500 (non-BPL)**
* Subsidy coverage: **48.4% (vs 80% target)**

👉 **Impact:** ~₹160 Cr annual inefficiency

---

## 🤖 Machine Learning Models

| Model             | Purpose              | Performance    | Key Feature       |
| ----------------- | -------------------- | -------------- | ----------------- |
| Gradient Boosting | LOS Prediction       | R² = 0.87      | Ward Type (84%)   |
| Random Forest     | Readmission Risk     | AUC = 0.736    | Comorbidity Count |
| K-Means           | Patient Segmentation | k = 4 clusters | Age + Severity    |

---

## 🧠 Insights

* **Ward assignment drives LOS** more than clinical variables
* **Comorbidity count strongly predicts readmission risk**
* **35% of patients account for ~70% of readmissions**
* **Hospital overload is structural**, not demand-driven

---

## 🛠️ Recommended Actions

### Capacity Optimization

* Introduce **step-down units (HDU expansion)**
* Enable **patient diversion strategies**

### Readmission Reduction

* Deploy **ML-based discharge risk scoring**
* Implement **48-hour follow-up programs**

### Financial Reform

* Audit **subsidy disbursement pipeline**
* Promote **generic drug procurement**

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
* **Features:** Clinical, financial, demographic

---

## ⚠️ Disclaimer

This project is **confidential** and intended for internal analysis purposes only. All insights are based on retrospective data and should be validated before real-world deployment.

