import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI(title="CT QC Anomaly Detection API")

# Paste your train.py globals here (specs, tolerances, FEATURE_COLS, preprocess, LEAK_COLS)
specs = {  # From your train.py
    'Slice thickness 1.5': 1.5,  # ... full dict
}
# ... tolerances, FEATURE_COLS, LEAK_COLS, LEAK_LIMIT, preprocess(df) function exactly from train.py

MODEL_PATH = 'model/ct_qc_production.pkl'

# Load model (fail if missing)
if not os.path.exists(MODEL_PATH):
    raise HTTPException(500, "Model not found—run train.py locally first")
MODEL = pickle.load(open(MODEL_PATH, 'rb'))

class ScanInput(BaseModel):
    serial_No: str
    # Add all param fields: Slice thickness 1.5: float, etc. (or use Dict[str, float])

def predict(input_data: dict) -> dict:
    # Your full predict function from the pasted code (exact copy)
    df = pd.DataFrame([input_data])
    df = preprocess(df)
    # ... rest unchanged

@app.get("/")
def root():
    return {"message": "CT QC API Live", "endpoints": ["/docs", "/predict", "/health"]}

@app.post("/predict")
def api_predict(scan: ScanInput):
    try:
        return predict(scan.dict())
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {str(e)}")

@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_PATH}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
