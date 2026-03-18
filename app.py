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
specs = {
    'Slice thickness 1.5': 1.5,
    'Slice thickness 5': 5,
    'Slice thickness 10': 10,
    'KV accuracy 80': 80,
    'KV accuracy 110': 110,
    'KV accuracy 130': 130,
    'Accuracy Timer 0.8': 0.8,
    'Accuracy Timer 1': 1.0,
    'Accuracy Timer 1.5': 1.5,
    'Radiation Dose Test (Head) 21.50': 21.50,
    'Radiation Dose Test (Body) 10.60': 10.60,
    'Low Contrast Resolution 5.0': 5.0,
    'High Contrast Resolution 6.24': 6.24,
}
tolerances = {
    'Slice thickness 1.5': 0.5,
    'Slice thickness 5': 2.5,
    'Slice thickness 10': 5.0,
    'KV accuracy 80': 2.0,
    'KV accuracy 110': 2.0,
    'KV accuracy 130': 2.0,
    'Accuracy Timer 0.8': 0.08,
    'Accuracy Timer 1': 0.1,
    'Accuracy Timer 1.5': 0.15,
    'Radiation Dose Test (Head) 21.50': 21.50 * 0.2,
    'Radiation Dose Test (Body) 10.60': 10.60 * 0.2,
    'High Contrast Resolution 6.24': 0.62,
}

LEAK_COLS = [
    'Radiation Leakage Levels (Front)',
    'Radiation Leakage Levels (Back)',
    'Radiation Leakage Levels (Left)',
    'Radiation Leakage Levels (Right)',
]
LEAK_LIMIT = 115.0

FEATURE_COLS = [
    'Slice thickness 1.5 %dev',
    'Slice thickness 5 %dev',
    'Slice thickness 10 %dev',
    'KV accuracy 80 %dev',
    'KV accuracy 110 %dev',
    'KV accuracy 130 %dev',
    'Accuracy Timer 0.8 %dev',
    'Accuracy Timer 1 %dev',
    'Accuracy Timer 1.5 %dev',
    'Radiation Dose Test (Head) 21.50 %dev',
    'Radiation Dose Test (Body) 10.60 %dev',
    'Low Contrast Resolution 5.0 %dev',
    'High Contrast Resolution 6.24 %dev',
    'Leakage_Max_Norm',
]

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values('Date')

    # % deviation features
    for col, spec in specs.items():
        df[f"{col} %dev"] = (df[col] - spec) / spec * 100

    # Pass/fail for imaging params
    pass_cols = []
    for col, spec in specs.items():
        tol = tolerances.get(col, 0.1)
        pass_col = f"{col} Pass"
        if col == 'Low Contrast Resolution 5.0':
            df[pass_col] = df[col] <= 5.0
        else:
            df[pass_col] = df[col].between(spec - tol, spec + tol)
        pass_cols.append(pass_col)

    # Leakage
    df['Leakage_Max_Norm'] = (500 * df[LEAK_COLS].max(axis=1)) / (60 * 8)
    df['Leakage Pass'] = df['Leakage_Max_Norm'] <= LEAK_LIMIT

    # Overall
    df['All_Imaging_Pass'] = df[pass_cols].all(axis=1)
    df['Overall_Acceptance_Pass'] = df['All_Imaging_Pass'] & df['Leakage Pass']

    return df
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
