from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, List
import numpy as np
import xgboost as xgb
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
meta = json.load(open(BASE_DIR / "model_meta.json"))
FEATURES = meta["features"]
THRESHOLD = meta["threshold"]
VERSION = meta["version"]

model = xgb.XGBClassifier()
model.load_model(BASE_DIR / "fraud_model.json")

app = FastAPI(title="Fraud Detection API", version=VERSION)

class Transaction(BaseModel):
    amount: float = Field(gt=0, le=1_000_000)
    time_since_last_transaction: int = Field(ge=0, le=43200)
    transaction_hour: int = Field(ge=0, le=23)
    is_international: Literal[0, 1]
    previous_transaction_amount: float = Field(ge=0)
    is_home_city: Literal[0, 1]
    is_preferred_device: Literal[0, 1]

class BatchRequest(BaseModel):
    transactions: List[Transaction] = Field(min_length=1, max_length=1000) # FIX 3

def score(txns: List[Transaction]):
    # FIX 4: One score helper, vectorized
    X = np.array([[getattr(t, f) for f in FEATURES] for t in txns])
    probs = model.predict_proba(X)[:, 1]
    
    results = []
    for p in probs:
        is_fraud = int(p >= THRESHOLD)
        risk = "HIGH" if p >= 0.7 else "MEDIUM" if is_fraud else "LOW" # FIX 2
        results.append({
            "is_fraud": is_fraud,
            "fraud_probability": round(float(p), 4),
            "risk_level": risk,
            "model_version": VERSION
        })
    return results

@app.get("/")
def root(): return {"status": "healthy", "version": VERSION}

@app.post("/predict")
def predict(txn: Transaction): return score([txn])[0]

@app.post("/predict_batch")
def predict_batch(batch: BatchRequest): 
    return {"total": len(batch.transactions), "predictions": score(batch.transactions)}