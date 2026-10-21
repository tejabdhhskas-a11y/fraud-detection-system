# Real-Time Fraud Detection System

A production-grade machine learning system that detects fraudulent UPI transactions in real-time using XGBoost and FastAPI.

🔗 **Live API:** [https://fraud-detection-system-g7ig.onrender.com/docs](https://fraud-detection-system-g7ig.onrender.com/docs) *(Note: Free tier, takes ~50s to cold start)*

## 🎯 Project Overview
This system generates realistic synthetic financial data, trains a highly optimized XGBoost classifier, and serves predictions via a FastAPI endpoint. It addresses real-world ML challenges like class imbalance, temporal data leakage, and precision-recall trade-offs.

## 🛠️ Tech Stack
- **Data:** Pandas, NumPy, Faker
- **ML Model:** XGBoost, Scikit-learn
- **API:** FastAPI, Uvicorn, Pydantic (Strict Validation)
- **Deployment:** Docker, Render
- **Explainability:** SHAP

## 📊 Model Performance (Time-Based Split)
To simulate real-world deployment, the model is trained on the first 70% of chronological data and tested on the unseen future 30%. 

| Metric | Score (Threshold = 0.4) |
| :--- | :--- |
| **ROC AUC** | 0.8443 |
| **PR AUC** | 0.2158 |
| **Precision (Fraud)** | 0.27 |
| **Recall (Fraud)** | 0.40 |

*(Note: The model explicitly optimizes for the bank's cost of false negatives vs. false positives using a tuned decision threshold and balanced class weights).*

## 🚀 How to Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt