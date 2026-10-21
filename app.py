from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import numpy as np
from pathlib import Path

# --- Load the trained model and scaler (Path security) ---
print("Loading model and scaler...")
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / 'fraud_model.pkl')
scaler = joblib.load(BASE_DIR / 'scaler.pkl')
print("Model loaded successfully!")

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection for UPI transactions",
    version="2.0.0"
)

# --- Strict Validation (Reviewer Fix: No default values) ---
class Transaction(BaseModel):
    amount: float = Field(gt=0, le=1_000_000)
    time_since_last_transaction: int = Field(ge=0, le=43200)
    transaction_hour: int = Field(ge=0, le=23)
    is_international: Literal[0, 1]
    previous_transaction_amount: float = Field(ge=0)
    is_home_city: Literal[0, 1]       # Must be provided
    is_preferred_device: Literal[0, 1] # Must be provided
    merchant_category: str = "unknown"

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running", "status": "healthy", "version": "2.0.0"}

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    try:
        # Prepare features
        features = np.array([[
            transaction.amount,
            transaction.time_since_last_transaction,
            transaction.transaction_hour,
            transaction.is_international,
            transaction.previous_transaction_amount,
            transaction.is_home_city,
            transaction.is_preferred_device
        ]])
        
        # Scale
        features_scaled = scaler.transform(features)
        
        # --- Explicit 0.4 Threshold (Reviewer Fix) ---
        THRESHOLD = 0.4
        probability = model.predict_proba(features_scaled)[0][1]
        prediction = 1 if probability >= THRESHOLD else 0
        
        return {
            "is_fraud": int(prediction),
            "fraud_probability": round(float(probability), 4),
            "risk_level": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW"
        }
    
    except Exception as e:
        # Don't leak internals to the user
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# --- Batch Prediction ---
class BatchRequest(BaseModel):
    transactions: list[Transaction]

@app.post("/predict_batch")
def predict_batch(batch: BatchRequest):
    try:
        results = []
        for transaction in batch.transactions:
            features = np.array([[transaction.amount, transaction.time_since_last_transaction,
                                  transaction.transaction_hour, transaction.is_international,
                                  transaction.previous_transaction_amount, transaction.is_home_city,
                                  transaction.is_preferred_device]])
            features_scaled = scaler.transform(features)
            
            # Explicit 0.4 Threshold
            THRESHOLD = 0.4
            probability = model.predict_proba(features_scaled)[0][1]
            prediction = 1 if probability >= THRESHOLD else 0
            
            results.append({
                "is_fraud": int(prediction),
                "fraud_probability": round(float(probability), 4)
            })
        return {"total": len(results), "predictions": results}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")