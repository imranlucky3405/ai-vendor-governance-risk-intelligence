# 🚀 AI-Enabled Vendor Governance & Risk Intelligence Platform

![Banner](assets/banner.jpg)

> Transforming Telecom Vendor Governance from Reactive Escalation to Predictive Intelligence

![AI Enabled Intelligence Platform](assets/transformation.jpg) [![Python](https://img.shields.io/badge/Python-3.9-blue)]() [![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)]() [![Machine Learning](https://img.shields.io/badge/ML-RandomForest-green)]() [![Time Series](https://img.shields.io/badge/Forecasting-ARIMA-orange)]() [![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)]()

## 👋 About Me
I’m **Imran Sayyed**, a Senior Manager with **17+ years** in telecom vendor operations, compliance, and cost governance.  
I build AI-driven governance and risk intelligence systems to reduce leakage and improve vendor performance.

# 🚀 AI-Enabled Vendor Governance & Risk Intelligence System

An end-to-end AI-driven decision support system designed to transform telecom vendor governance from reactive escalation to proactive, data-driven risk intelligence.

---

## 🎯 Project Objective

Telecom vendor governance often suffers from:

- Delivery delays
- Cost overruns
- Compliance gaps
- Escalation-driven firefighting

This project builds an AI-powered intelligence layer that:

✔ Predicts vendor deterioration risk  
✔ Quantifies financial exposure  
✔ Forecasts delivery delays  
✔ Detects delay anomalies  
✔ Triggers automatic escalation  
✔ Provides role-based action recommendations  

---

## 🧠 Core Components

### 🔢 Vendor Performance Index (VPI)
Composite score based on:
- Delivery performance
- Quality metrics
- Compliance factors
- Commercial discipline

### 🤖 Predictive Risk Model
- Random Forest Classifier
- Class imbalance handled using `class_weight`
- Outputs deterioration probability
- Risk classification: Low / Moderate / High / Critical

### 💸 Financial Impact Simulation
Estimates:
- Working capital exposure
- Delivery delay cost impact
- Penalty exposure
- Estimated savings via mitigation

### 📈 Time-Series Intelligence
- Monthly delivery trend
- ARIMA forecasting (next 3 months)
- Rolling Z-score anomaly detection
- Automatic escalation logic

---

## 📸 Dashboard Preview

### 📉 Predictive Risk

![Predictive Risk](assets/predictive_risk.jpeg)

### 💸 Financial Impact
![Financial Impact](assets/financial_impact.jpeg)

### 📈 Time-Series & Escalation
![Time Series](assets/time_series_escalation.jpeg)

---
## ⚙️ Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py