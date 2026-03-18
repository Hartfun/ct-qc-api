import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import threading  # For thread-safe lazy loading

app = FastAPI(title="CT QC Anomaly Detection API")

# Paste your train.py globals here EXACTLY
specs = {  # From your train.py - copy full dict
    'Slice thickness 1.5': 1.5,  # ... complete
}
# ... tolerances, FEATURE_COLS, LEAK_COLS, LEAK_LIMIT
def preprocess(df):  # Copy FULL function from train.py
    # Your exact preprocess logic
    pass

MODEL_PATH = 'model/ct_qc_production.pkl'
MODEL = None
MODEL_LOCK = threading.Lock()

def get_model():
    global MODEL
    if MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(500, f"Model missing: {MODEL_PATH}. Commit after running train.py")
        with MODEL_LOCK:
            if MODEL is None:  # Double-check
                MODEL = pickle.load(open(MODEL_PATH, 'rb'))
    return MODEL

class ScanInput(BaseModel):
    serial_No: str
    # Add ALL fields from FEATURE_COLS: Slice thickness 1.5: float, etc.
    # Or use Dict[str, Any] for flexibility

@app.get("/")
def root():
    return {"message": "CT QC API Live", "endpoints": ["/docs", "/predict", "/health"]}

@app.get("/health")
def health():
    try:
        model = get_model()  # Test load
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        return {"status": "unhealthy", "model_error": str(e)}

@app.post("/predict")
def api_predict(scan: ScanInput):
    try:
        model = get_model()
        df = pd.DataFrame([scan.dict()])
        df = preprocess(df)
        prediction = model.predict(df)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
