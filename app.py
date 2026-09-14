from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# --- Load the trained model and scaler ---
print("Loading model and scaler...")
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')
print("Model loaded successfully!")

# --- Initialize FastAPI app ---
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection for UPI transactions",
    version="1.0.0"
)

# --- Define the input schema ---
class Transaction(BaseModel):
    amount: float
    time_since_last_transaction: int
    transaction_hour: int
    is_international: int
    previous_transaction_amount: float
    is_home_city: int = 1
    is_preferred_device: int = 1

# --- Health check endpoint ---
@app.get("/")
def root():
    return {
        "message": "Fraud Detection API is running",
        "status": "healthy",
        "version": "1.0.0"
    }

# --- Prediction endpoint ---
@app.post("/predict")
def predict_fraud(transaction: Transaction):
    try:
        # Prepare features in the same order as training
        features = np.array([[
            transaction.amount,
            transaction.time_since_last_transaction,
            transaction.transaction_hour,
            transaction.is_international,
            transaction.previous_transaction_amount,
            transaction.is_home_city,
            transaction.is_preferred_device
        ]])
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        return {
            "is_fraud": int(prediction),
            "fraud_probability": round(float(probability), 4),
            "risk_level": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))