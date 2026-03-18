"""
predict.py — standalone inference module.

Loads the pre-trained model bundle and exposes a single `predict(input_data)` function.
Does NOT import from train.py to avoid a circular-import at deploy time when train.py
is not present in the container.  All shared constants are inlined here.
"""

import pickle
import numpy as np
import pandas as pd

# === SHARED CONSTANTS (kept in sync with train.py) ===
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
LEAK_LIMIT = 1.0   # normalized leakage limit: (500 mA/hr * max_raw) / (60 min * 240 mA) <= 1

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

MODEL_PATH = 'model/ct_qc_production.pkl'


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

    # Leakage_Max_Norm = workload-normalised leakage rate.
    # Formula: (500 mA/hr x max_raw_mR/hr) / (60 min * 240 mA)
    # 500  = reference tube current in mA/hr (max rated)
    # 240  = actual tube current used during leakage measurement (mA)
    # 60   = minutes in an hour
    # Result: dimensionless ratio; pass when <= 1.0
    # Formula: (500 mA × max_raw_reading) / (60 s × 8 min)
    # 500  = nominal tube current (mA)
    # 60*8 = 480 s = 8-minute scan workload
    # Pass when Leakage_Max_Norm <= 1.0
    df['Leakage_Max_Norm'] = (500 * df[LEAK_COLS].max(axis=1)) / (60 * 240)
    df['Leakage Pass'] = df['Leakage_Max_Norm'] <= LEAK_LIMIT
    df['All_Imaging_Pass'] = df[pass_cols].all(axis=1)
    df['Overall_Acceptance_Pass'] = df['All_Imaging_Pass'] & df['Leakage Pass']

    return df


def load_model(path: str = MODEL_PATH) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


# Loaded once at import time (same behaviour as before, but safe at module level)
MODEL = load_model()


def predict(input_data: dict) -> dict:
    """
    Accept a single CT QC record as a dict with keys matching CT-Test.csv column names
    and return anomaly scores, labels, and a per-parameter breakdown.
    """
    df = pd.DataFrame([input_data])
    df = preprocess(df)

    X = df[FEATURE_COLS].fillna(0)
    X_scaled = MODEL['scaler'].transform(X)

    iso_score = float(MODEL['iso'].decision_function(X_scaled)[0])
    lof_score = float(MODEL['lof'].decision_function(X_scaled)[0])
    ensemble_score = 0.7 * iso_score + 0.3 * lof_score

    threshold = float(MODEL['threshold'])
    is_anomaly = bool(ensemble_score < threshold)
    fail_prob = float(1 / (1 + np.exp(ensemble_score)))

    breakdown: dict = {}
    for col, spec_val in specs.items():
        val = input_data.get(col)
        tol = tolerances.get(col)
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

    leak_vals = [input_data.get(c, 0.0) for c in LEAK_COLS]
    breakdown["Leakage"] = {
        "max_raw": float(max(leak_vals)),
        "norm": float(df["Leakage_Max_Norm"].iloc[0]),
        "limit": float(LEAK_LIMIT),
        "pass": bool(df["Leakage Pass"].iloc[0]),
    }

    return {
        "iso_score": round(iso_score, 5),
        "lof_score": round(lof_score, 5),
        "ensemble_score": round(ensemble_score, 5),
        "threshold": round(threshold, 5),
        "anomaly_detected": is_anomaly,
        "fail_probability": round(fail_prob, 5),
        "overall_acceptance": bool(df["Overall_Acceptance_Pass"].iloc[0]),
        "parameter_breakdown": breakdown,
    }


if __name__ == "__main__":
    sample = {
        "Slice thickness 1.5": 1.5,
        "Slice thickness 5": 5.0,
        "Slice thickness 10": 10.0,
        "KV accuracy 80": 80.0,
        "KV accuracy 110": 110.0,
        "KV accuracy 130": 130.0,
        "Accuracy Timer 0.8": 0.8,
        "Accuracy Timer 1": 1.0,
        "Accuracy Timer 1.5": 1.5,
        "Radiation Dose Test (Head) 21.50": 21.50,
        "Radiation Dose Test (Body) 10.60": 10.60,
        "Low Contrast Resolution 5.0": 5.0,
        "High Contrast Resolution 6.24": 6.24,
        "Radiation Leakage Levels (Front)": 50.0,
        "Radiation Leakage Levels (Back)": 40.0,
        "Radiation Leakage Levels (Left)": 45.0,
        "Radiation Leakage Levels (Right)": 55.0,
        "Date": "2025-01-01",
        "serial No": "CT-DEMO-001",
    }
    import json
    print(json.dumps(predict(sample), indent=2))
