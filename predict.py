import pickle
import numpy as np
import pandas as pd

# === Reuse the same preprocessing logic ===
from train import preprocess, FEATURE_COLS

MODEL_PATH = 'model/ct_qc_production.pkl'

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

MODEL = load_model()  # Loaded once at startup


def predict(input_data: dict) -> dict:
    """
    Accept a single CT QC record as a dict,
    return risk score + label + parameter-level breakdown.
    """
    df = pd.DataFrame([input_data])
    df = preprocess(df)

    X = df[FEATURE_COLS].fillna(0)
    X_scaled = MODEL['scaler'].transform(X)

    iso_score = MODEL['iso'].decision_function(X_scaled)[0]
    lof_score = MODEL['lof'].decision_function(X_scaled)[0]
    ensemble_score = 0.7 * iso_score + 0.3 * lof_score

    is_anomaly = bool(ensemble_score < MODEL['threshold'])

    # Parameter-level pass/fail breakdown
    breakdown = {}
    for col, spec in MODEL['specs'].items():
        tol = MODEL['tolerances'][col]
        val = input_data.get(col)
        if val is not None:
            passed = (spec - tol) <= val <= (spec + tol)
            breakdown[col] = {
                'value': val,
                'spec': spec,
                'tolerance': tol,
                'pass': bool(passed),
                'pct_deviation': round((val - spec) / spec * 100, 3)
            }

    return {
        'iso_score': round(float(iso_score), 5),
        'lof_score': round(float(lof_score), 5),
        'ensemble_score': round(float(ensemble_score), 5),
        'threshold': round(float(MODEL['threshold']), 5),
        'anomaly_detected': is_anomaly,
        'overall_acceptance': bool(df['Overall_Acceptance_Pass'].iloc[0]),
        'leakage_max': float(df['Leakage Max'].iloc[0]),
        'leakage_pass': bool(df['Leakage Pass'].iloc[0]),
        'parameter_breakdown': breakdown,
    }
