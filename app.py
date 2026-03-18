import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import threading
from fastapi.middleware.cors import CORSMiddleware

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
    """CT QC Record - ALL fields required"""
    serial_No: str
    slice_thickness_1_5: float = Field(..., alias="Slice thickness 1.5")
    slice_thickness_5: float = Field(..., alias="Slice thickness 5")
    slice_thickness_10: float = Field(..., alias="Slice thickness 10")
    kv_accuracy_80: float = Field(..., alias="KV accuracy 80")
    kv_accuracy_110: float = Field(..., alias="KV accuracy 110")
    kv_accuracy_130: float = Field(..., alias="KV accuracy 130")
    accuracy_timer_0_8: float = Field(..., alias="Accuracy Timer 0.8")
    accuracy_timer_1: float = Field(..., alias="Accuracy Timer 1")
    accuracy_timer_1_5: float = Field(..., alias="Accuracy Timer 1.5")
    radiation_dose_head: float = Field(..., alias="Radiation Dose Test (Head) 21.50")
    radiation_dose_body: float = Field(..., alias="Radiation Dose Test (Body) 10.60")
    low_contrast_resolution: float = Field(..., alias="Low Contrast Resolution 5.0")
    high_contrast_resolution: float = Field(..., alias="High Contrast Resolution 6.24")
    
    # Leakage
    leak_front: float = Field(..., alias="Radiation Leakage Levels (Front)")
    leak_back: float = Field(..., alias="Radiation Leakage Levels (Back)")
    leak_left: float = Field(..., alias="Radiation Leakage Levels (Left)")
    leak_right: float = Field(..., alias="Radiation Leakage Levels (Right)")

    class Config:
        populate_by_name = True 

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
