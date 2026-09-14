# Real-Time Fraud Detection System

A machine learning system that detects fraudulent UPI transactions in real-time using XGBoost and FastAPI.

## 🎯 Overview

This project generates 1 million synthetic UPI transactions with realistic fraud patterns, trains an XGBoost model with 0.9979 ROC AUC, and serves predictions through a FastAPI endpoint.

## 🛠️ Tech Stack

- **Python** — Core language
- **Pandas, NumPy** — Data manipulation
- **XGBoost** — Fraud detection model
- **FastAPI, Uvicorn** — Real-time API
- **Scikit-learn** — Preprocessing and evaluation
- **Faker** — Synthetic data generation

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt