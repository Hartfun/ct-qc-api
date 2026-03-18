import pickle
import numpy as np
import pandas as pd

from train import preprocess, FEATURE_COLS, specs, tolerances, LEAK_COLS, LEAK_LIMIT

MODEL_PATH = 'model/ct_qc_production.pkl'


def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


MODEL = load_model()  # Loaded once at import


def predict(input_data: dict) -> dict:
    """
    Accept a single CT QC record as a dict with keys matching CT-Test.csv
    (Slice thickness..., KV accuracy..., Accuracy Timer..., Dose tests, Contrast,
    Radiation Leakage Levels...), and return anomaly scores + labels + breakdown.
    """
    # Build one-row DataFrame and reuse training preprocess
    df = pd.DataFrame([input_data])
    df = preprocess(df)

    # Features exactly as in training
    X = df[FEATURE_COLS].fillna(0)
    X_scaled = MODEL['scaler'].transform(X)

    # Scores
    iso_score = float(MODEL['iso'].decision_function(X_scaled)[0])   # higher = more normal [web:28]
    lof_score = float(MODEL['lof'].decision_function(X_scaled)[0])   # higher = more normal [web:33]
    ensemble_score = 0.7 * iso_score + 0.3 * lof_score

    # Threshold from training
    threshold = float(MODEL['threshold'])
    is_anomaly = bool(ensemble_score < threshold)

    # Optional probability-like mapping for UI
    fail_prob = 1 / (1 + np.exp(ensemble_score))  # lower ensemble → closer to 1

    # Parameter-level breakdown
    breakdown = {}

    # Imaging parameters
    for col, spec in specs.items():
        val = input_data.get(col, None)
        tol = tolerances.get(col, None)
        if val is None or tol is None:
            continue

        pass_flag = bool(df[f"{col} Pass"].iloc[0])
        pct_dev = (val - spec) / spec * 100 if spec != 0 else 0.0

        breakdown[col] = {
            "value": float(val),
            "spec": float(spec),
            "tolerance": float(tol),
            "pass": pass_flag,
            "pct_deviation": round(pct_dev, 3),
        }

    # Leakage summary
    if LEAK_COLS:
        leak_vals = [input_data.get(c, 0.0) for c in LEAK_COLS]
        leak_max_raw = max(leak_vals)
    else:
        leak_max_raw = 0.0

    breakdown["Leakage"] = {
        "max_raw": float(leak_max_raw),
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
        "fail_probability": round(float(fail_prob), 5),
        "overall_acceptance": bool(df["Overall_Acceptance_Pass"].iloc[0]),
        "parameter_breakdown": breakdown,
    }


if __name__ == "__main__":
    # Quick smoke test with a near-spec record
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
        "Requester": "test",
        "Reviewer": "test",
        "Mode Name": "HEAD",
        "serial No": "CT-DEMO-001",
    }
    print(predict(sample))

