"""
train.py — run locally (NOT inside the Docker container).

  python train.py

Produces model/ct_qc_production.pkl which must be committed to the repo
before building the Docker image.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

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


def train():
    print("📂 Loading data from data/CT-Test.csv...")
    df = pd.read_csv('data/CT-Test.csv')
    print(f"✓ Loaded {len(df)} records")

    print("🔄 Preprocessing...")
    df = preprocess(df)

    print("📊 Preparing features...")
    X = df[FEATURE_COLS].fillna(0)

    print("🔧 Scaling features (RobustScaler)...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    print("🌲 Training Isolation Forest...")
    iso = IsolationForest(n_estimators=200, contamination=0.18, random_state=42)
    iso.fit(X_scaled)

    print("👥 Training Local Outlier Factor...")
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.18, novelty=True)
    lof.fit(X_scaled)

    print("📈 Computing ensemble scores & threshold...")
    iso_scores = iso.decision_function(X_scaled)
    lof_scores = lof.decision_function(X_scaled)
    ensemble_scores = 0.7 * iso_scores + 0.3 * lof_scores
    threshold = float(np.percentile(ensemble_scores, 10))
    print(f"✓ Threshold: {threshold:.6f}")

    model_bundle = {
        'scaler': scaler,
        'iso': iso,
        'lof': lof,
        'threshold': threshold,
        'feature_cols': FEATURE_COLS,
        'specs': specs,
        'tolerances': tolerances,
    }

    os.makedirs('model', exist_ok=True)
    with open('model/ct_qc_production.pkl', 'wb') as f:
        pickle.dump(model_bundle, f)

    print("\n✅ Training complete — model/ct_qc_production.pkl written.")
    print("   Commit this file before running `docker build`.")


if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
