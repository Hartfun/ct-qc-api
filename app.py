import os
import math
import pickle
import threading
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator

app = FastAPI(title="CT QC Anomaly Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SPECS & TOLERANCES ────────────────────────────────────────────────────────
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
    # Low Contrast Resolution: one-sided upper limit (lower = better resolution).
    # AERB pass condition: measured <= 5.0 lp/cm.
    # Tolerance stored as 5.0 so the breakdown response shows the actual limit,
    # and pct_deviation = (measured - 5.0) / 5.0 * 100 (positive = worse than spec).
    'Low Contrast Resolution 5.0': 5.0,
}

LEAK_COLS = [
    'Radiation Leakage Levels (Front)',
    'Radiation Leakage Levels (Back)',
    'Radiation Leakage Levels (Left)',
    'Radiation Leakage Levels (Right)',
]
# Leakage_Max_Norm = (500 mA/hr × max_raw) / (60 min × 240 mA) — pass when <= 1.0
LEAK_LIMIT     = 1.0    # workload-normalised value — informational ML feature only
RAW_LEAK_LIMIT = 115.0  # AERB primary pass/fail gate: raw survey-meter reading <= 115 mR/hr

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

# Physical plausibility bounds — catches data-entry errors and unit confusions.
# These are wide enough that a legitimately failing scanner still passes them.
FIELD_BOUNDS: dict[str, tuple[float, float]] = {
    'Slice thickness 1.5':               (0.1,   10.0),
    'Slice thickness 5':                 (0.5,   20.0),
    'Slice thickness 10':                (1.0,   30.0),
    'KV accuracy 80':                    (40.0,  120.0),
    'KV accuracy 110':                   (60.0,  150.0),
    'KV accuracy 130':                   (80.0,  180.0),
    'Accuracy Timer 0.8':                (0.01,  5.0),
    'Accuracy Timer 1':                  (0.01,  5.0),
    'Accuracy Timer 1.5':                (0.01,  5.0),
    'Radiation Dose Test (Head) 21.50':  (1.0,   100.0),
    'Radiation Dose Test (Body) 10.60':  (0.5,   60.0),
    'Low Contrast Resolution 5.0':       (0.1,   20.0),
    'High Contrast Resolution 6.24':     (0.1,   30.0),
    'Radiation Leakage Levels (Front)':  (0.0,   500.0),
    'Radiation Leakage Levels (Back)':   (0.0,   500.0),
    'Radiation Leakage Levels (Left)':   (0.0,   500.0),
    'Radiation Leakage Levels (Right)':  (0.0,   500.0),
}


def _check_finite_positive(field_name: str, value: float, allow_zero: bool = False) -> float:
    """Raise ValueError if value is not finite or violates the positivity rule."""
    if value is None:
        raise ValueError(f"{field_name} is required.")
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number (got {value}).")
    lo, hi = FIELD_BOUNDS.get(field_name, (None, None))
    if lo is not None and value < lo:
        raise ValueError(
            f"{field_name} = {value} is below the physical minimum of {lo}. "
            "Check your reading or units."
        )
    if hi is not None and value > hi:
        raise ValueError(
            f"{field_name} = {value} exceeds the physical maximum of {hi}. "
            "Check your reading or units."
        )
    return value


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

    # Leakage_Max_Norm = (500 mA/hr × max_raw) / (60 min × 240 mA)
    # 500  = reference tube current (mA/hr, max rated)
    # 240  = actual tube current during leakage measurement (mA)
    # 60   = minutes in an hour
    # Pass when ratio <= 1.0
    df['Leakage_Max_Norm'] = ((500 * df[LEAK_COLS].max(axis=1)) / (60 * 240)*100)
    # AERB primary gate: raw max reading <= 115 mR/hr.
    # Leakage_Max_Norm is computed and stored as an informational ML feature only.
    df['Leakage Pass'] = df[LEAK_COLS].max(axis=1) <= RAW_LEAK_LIMIT

    df['All_Imaging_Pass']       = df[pass_cols].all(axis=1)
    df['Overall_Acceptance_Pass'] = df['All_Imaging_Pass'] & df['Leakage Pass']

    return df


# ── MODEL LOADING ─────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "model/ct_qc_production.pkl")
MODEL      = None
MODEL_LOCK = threading.Lock()


def get_model():
    global MODEL
    if MODEL is None:
        with MODEL_LOCK:
            if MODEL is None:
                if not os.path.exists(MODEL_PATH):
                    raise HTTPException(
                        status_code=500,
                        detail=f"Model file not found: {MODEL_PATH}. Run train.py and commit the .pkl file.",
                    )
                with open(MODEL_PATH, "rb") as f:
                    MODEL = pickle.load(f)
    return MODEL


# ── PYDANTIC INPUT MODEL ──────────────────────────────────────────────────────
class ScanInput(BaseModel):
    """CT QC Record — every measurement field is required and range-validated."""

    model_config = {"populate_by_name": True}

    # ── Identity ──────────────────────────────────────────────────────────────
    serial_No: str = Field(..., min_length=2, max_length=30)

    # ── Slice thickness (mm) ──────────────────────────────────────────────────
    slice_thickness_1_5:  float = Field(..., alias="Slice thickness 1.5")
    slice_thickness_5:    float = Field(..., alias="Slice thickness 5")
    slice_thickness_10:   float = Field(..., alias="Slice thickness 10")

    # ── KV accuracy ───────────────────────────────────────────────────────────
    kv_accuracy_80:  float = Field(..., alias="KV accuracy 80")
    kv_accuracy_110: float = Field(..., alias="KV accuracy 110")
    kv_accuracy_130: float = Field(..., alias="KV accuracy 130")

    # ── Timer accuracy (s) ────────────────────────────────────────────────────
    accuracy_timer_0_8: float = Field(..., alias="Accuracy Timer 0.8")
    accuracy_timer_1:   float = Field(..., alias="Accuracy Timer 1")
    accuracy_timer_1_5: float = Field(..., alias="Accuracy Timer 1.5")

    # ── Radiation dose (mGy) ──────────────────────────────────────────────────
    radiation_dose_head: float = Field(..., alias="Radiation Dose Test (Head) 21.50")
    radiation_dose_body: float = Field(..., alias="Radiation Dose Test (Body) 10.60")

    # ── Contrast resolution (lp/cm) ───────────────────────────────────────────
    low_contrast_resolution:  float = Field(..., alias="Low Contrast Resolution 5.0")
    high_contrast_resolution: float = Field(..., alias="High Contrast Resolution 6.24")

    # ── Leakage (mR/hr) ───────────────────────────────────────────────────────
    leak_front: float = Field(..., alias="Radiation Leakage Levels (Front)")
    leak_back:  float = Field(..., alias="Radiation Leakage Levels (Back)")
    leak_left:  float = Field(..., alias="Radiation Leakage Levels (Left)")
    leak_right: float = Field(..., alias="Radiation Leakage Levels (Right)")

    # ── serial_No validator ───────────────────────────────────────────────────
    @field_validator("serial_No")
    @classmethod
    def validate_serial(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("serial_No must not be blank.")
        import re
        if not re.match(r'^[A-Za-z0-9\-_/]{2,30}$', v):
            raise ValueError(
                "serial_No must be 2–30 characters and contain only "
                "letters, digits, hyphens, underscores, or /."
            )
        return v

    # ── Numeric field validators ───────────────────────────────────────────────
    # Using a single model_validator (after) to check all floats in one pass,
    # mapping each internal field name back to the aliased display name.
    @model_validator(mode="after")
    def validate_all_measurements(self) -> "ScanInput":
        alias_map = {
            "slice_thickness_1_5":  "Slice thickness 1.5",
            "slice_thickness_5":    "Slice thickness 5",
            "slice_thickness_10":   "Slice thickness 10",
            "kv_accuracy_80":       "KV accuracy 80",
            "kv_accuracy_110":      "KV accuracy 110",
            "kv_accuracy_130":      "KV accuracy 130",
            "accuracy_timer_0_8":   "Accuracy Timer 0.8",
            "accuracy_timer_1":     "Accuracy Timer 1",
            "accuracy_timer_1_5":   "Accuracy Timer 1.5",
            "radiation_dose_head":  "Radiation Dose Test (Head) 21.50",
            "radiation_dose_body":  "Radiation Dose Test (Body) 10.60",
            "low_contrast_resolution":  "Low Contrast Resolution 5.0",
            "high_contrast_resolution": "High Contrast Resolution 6.24",
            "leak_front": "Radiation Leakage Levels (Front)",
            "leak_back":  "Radiation Leakage Levels (Back)",
            "leak_left":  "Radiation Leakage Levels (Left)",
            "leak_right": "Radiation Leakage Levels (Right)",
        }
        errors = []
        for attr, display_name in alias_map.items():
            val = getattr(self, attr, None)
            try:
                _check_finite_positive(display_name, val)
            except ValueError as e:
                errors.append(str(e))

        if errors:
            raise ValueError(errors)   # Pydantic v2 will collect these into the 422 response
        return self


# ── ROUTES ────────────────────────────────────────────────────────────────────
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

        raw = scan.model_dump(by_alias=True)
        df  = pd.DataFrame([raw])
        df  = preprocess(df)

        X        = df[FEATURE_COLS].fillna(0)
        X_scaled = model["scaler"].transform(X)

        iso_score      = float(model["iso"].decision_function(X_scaled)[0])
        lof_score      = float(model["lof"].decision_function(X_scaled)[0])
        ensemble_score = 0.7 * iso_score + 0.3 * lof_score
        threshold      = float(model["threshold"])
        is_anomaly     = bool(ensemble_score < threshold)
        fail_prob      = float(1 / (1 + np.exp(ensemble_score)))

        breakdown: dict[str, Any] = {}
        for col, spec_val in specs.items():
            tol = tolerances.get(col)
            val = raw.get(col)
            if val is None or tol is None:
                continue
            pass_flag = bool(df[f"{col} Pass"].iloc[0])
            # Low Contrast Resolution is a one-sided upper-limit metric (lower = better).
            # pct_deviation = how far the reading is above/below the 5.0 lp/cm limit.
            # Positive value means worse than spec; negative means better than spec.
            if col == "Low Contrast Resolution 5.0":
                pct_dev = (val - spec_val) / spec_val * 100  # (measured - 5.0) / 5.0 * 100
            else:
                pct_dev = (val - spec_val) / spec_val * 100 if spec_val != 0 else 0.0
            breakdown[col] = {
                "value":         float(val),
                "spec":          float(spec_val),
                "tolerance":     float(tol),
                "one_sided":     col == "Low Contrast Resolution 5.0",
                "pass":          pass_flag,
                "pct_deviation": round(pct_dev, 3),
            }

        leak_vals = [raw.get(c, 0.0) for c in LEAK_COLS]
        breakdown["Leakage"] = {
            "max_raw":   float(max(leak_vals)),
            "norm":      float(df["Leakage_Max_Norm"].iloc[0]),  # informational ML feature
            "norm_limit": float(LEAK_LIMIT),                     # 1.0 — informational only
            "raw_limit": float(RAW_LEAK_LIMIT),                  # 115 mR/hr — AERB pass gate
            "pass":      bool(df["Leakage Pass"].iloc[0]),       # True when max_raw <= 115
        }

        return {
            "serial_No":        scan.serial_No,
            "iso_score":        round(iso_score,      5),
            "lof_score":        round(lof_score,      5),
            "ensemble_score":   round(ensemble_score, 5),
            "threshold":        round(threshold,      5),
            "anomaly_detected": is_anomaly,
            "fail_probability": round(fail_prob,      5),
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
