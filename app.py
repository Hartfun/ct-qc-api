import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
# Copy specs, tolerances, LEAK_COLS, FEATURE_COLS, preprocess from your train.py here

app = FastAPI(title="CT QC Anomaly Detection API")

# Load model on startup
MODEL_PATH = 'model/ct_qc_production.pkl'
if not os.path.exists(MODEL_PATH):
    # Auto-train if missing
    df = pd.read_csv('data/CT-Test.csv')
    df = preprocess(df)
    # ... your full train() logic here ...
    os.makedirs('model', exist_ok=True)
    pickle.dump(model_bundle, open(MODEL_PATH, 'wb'))

MODEL = pickle.load(open(MODEL_PATH, 'rb'))

class ScanInput(BaseModel):
    serial_No: str
    # All 12 imaging + 4 leakage keys as float

@app.post("/predict")
def predict(scan: ScanInput) -> Dict[str, Any]:
    return predict(scan.dict())  # Your predict function

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}
