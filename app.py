import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any
import threading
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="CT QC Anomaly Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === SPECS & TOLERANCES ===
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

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)

    for col, spec in specs.items():
        df[f"{col} %dev"] = (df[col] - spec) / spec * 100

    pass_cols = []
    for col, spec in specs.items():
        tol = tolerances.get(col, 0.1)
        pass_col = f"{col} Pass"
        if col == 'Low Contrast Resolution 5.0':
            df[pass_col] = df[col] <= 5.0
        else:
            df[pass_col] = df[col].between(spec - tol, spec + tol)
        pass_cols.append(pass_col)

    df['Leakage_Max_Norm'] = (500 * df[LEAK_COLS].max(axis=1)) / (60 * 8)
    df['Leakage Pass'] = df['Leakage_Max_Norm'] <= LEAK_LIMIT

    df['All_Imaging_Pass'] = df[pass_cols].all(axis=1)
    df['Overall_Acceptance_Pass'] = df['All_Imaging_Pass'] & df['Leakage Pass']

    return df
    # FIX: removed dangling `pass` after return statement


MODEL_PATH = os.environ.get("MODEL_PATH", "model/ct_qc_production.pkl")
MODEL = None
MODEL_LOCK = threading.Lock()


def get_model():
    global MODEL
    if MODEL is None:
        with MODEL_LOCK:
            if MODEL is None:
                if not os.path.exists(MODEL_PATH):
                    raise HTTPException(
                        status_code=500,
                        detail=f"Model file not found: {MODEL_PATH}. Run train.py and commit the .pkl file."
                    )
                with open(MODEL_PATH, "rb") as f:
                    MODEL = pickle.load(f)
    return MODEL


class ScanInput(BaseModel):
    """CT QC Record — all measurement fields required."""
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

    leak_front: float = Field(..., alias="Radiation Leakage Levels (Front)")
    leak_back: float = Field(..., alias="Radiation Leakage Levels (Back)")
    leak_left: float = Field(..., alias="Radiation Leakage Levels (Left)")
    leak_right: float = Field(..., alias="Radiation Leakage Levels (Right)")

    model_config = {"populate_by_name": True}
    # FIX: replaced deprecated inner `class Config` with model_config dict (Pydantic v2)


@app.get("/")
def root():
    return {"message": "CT QC API Live", "endpoints": ["/docs", "/predict", "/health"]}


@app.get("/health")
def health():
    try:
        get_model()
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        return {"status": "unhealthy", "model_error": str(e)}


@app.post("/predict")
def api_predict(scan: ScanInput):
    try:
        model = get_model()

        # FIX: use model_dump(by_alias=True) instead of deprecated .dict()
        raw = scan.model_dump(by_alias=True)
        df = pd.DataFrame([raw])
        df = preprocess(df)

        X = df[FEATURE_COLS].fillna(0)
        X_scaled = model["scaler"].transform(X)

        iso_score = float(model["iso"].decision_function(X_scaled)[0])
        lof_score = float(model["lof"].decision_function(X_scaled)[0])
        ensemble_score = 0.7 * iso_score + 0.3 * lof_score
        threshold = float(model["threshold"])
        is_anomaly = bool(ensemble_score < threshold)
        fail_prob = float(1 / (1 + np.exp(ensemble_score)))

        breakdown: dict[str, Any] = {}
        for col, spec_val in specs.items():
            tol = tolerances.get(col)
            val = raw.get(col)
            if val is None or tol is None:
                continue
            pass_flag = bool(df[f"{col} Pass"].iloc[0])
            pct_dev = (val - spec_val) / spec_val * 100 if spec_val != 0 else 0.0
            breakdown[col] = {
                "value": float(val),
                "spec": float(spec_val),
                "tolerance": float(tol),
                "pass": pass_flag,
                "pct_deviation": round(pct_dev, 3),
            }

        leak_vals = [raw.get(c, 0.0) for c in LEAK_COLS]
        breakdown["Leakage"] = {
            "max_raw": float(max(leak_vals)),
            "norm": float(df["Leakage_Max_Norm"].iloc[0]),
            "limit": float(LEAK_LIMIT),
            "pass": bool(df["Leakage Pass"].iloc[0]),
        }

        return {
            "serial_No": scan.serial_No,
            "iso_score": round(iso_score, 5),
            "lof_score": round(lof_score, 5),
            "ensemble_score": round(ensemble_score, 5),
            "threshold": round(threshold, 5),
            "anomaly_detected": is_anomaly,
            "fail_probability": round(fail_prob, 5),
            "overall_acceptance": bool(df["Overall_Acceptance_Pass"].iloc[0]),
            "parameter_breakdown": breakdown,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
