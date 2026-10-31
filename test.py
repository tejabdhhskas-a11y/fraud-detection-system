from fastapi.testclient import TestClient
from app import app

client = TestClient(app)
good = dict(amount=500, time_since_last_transaction=60, transaction_hour=14, 
            is_international=0, previous_transaction_amount=400, 
            is_home_city=1, is_preferred_device=1)

def test_ok(): 
    assert client.post("/predict", json=good).status_code == 200

def test_missing_flag(): 
    bad = {k: v for k, v in good.items() if k != "is_home_city"}
    assert client.post("/predict", json=bad).status_code == 422

def test_bad_hour(): 
    assert client.post("/predict", json={**good, "transaction_hour": 25}).status_code == 422