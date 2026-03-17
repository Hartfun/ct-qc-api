import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import pickle
import os

# === SPECS & TOLERANCES ===
specs = {
    'Slice thickness 1.5': 1.5, 'Slice thickness 5': 5, 'Slice thickness 10': 10,
    'KV accuracy 80': 80, 'KV accuracy 110': 110, 'KV accuracy 130': 130,
    'Accuracy Timer 0.8': 0.8, 'Accuracy Timer 1': 1.0, 'Accuracy Timer 1.5': 1.5,
    'Radiation Dose Test (Head) 21.50': 21.50, 'Radiation Dose Test (Body) 10.60': 10.60,
}

tolerances = {
    'Slice thickness 1.5': 0.5, 'Slice thickness 5': 2.5, 'Slice thickness 10': 5.0,
    'KV accuracy 80': 2.0, 'KV accuracy 110': 2.0, 'KV accuracy 130': 2.0,
    'Accuracy Timer 0.8': 0.08, 'Accuracy Timer 1': 0.1, 'Accuracy Timer 1.5': 0.15,
    'Radiation Dose Test (Head) 21.50': 21.50 * 0.2,
    'Radiation Dose Test (Body) 10.60': 10.60 * 0.2,
}

LEAK_COLS = [
    'Radiation Leakage Levels (Front)', 'Radiation Leakage Levels (Back)',
    'Radiation Leakage Levels (Left)', 'Radiation Leakage Levels (Right)'
]
LEAK_LIMIT = 115.0

FEATURE_COLS = [
    'Slice thickness 1.5 %dev', 'Slice thickness 5 %dev', 'Slice thickness 10 %dev',
    'KV accuracy 80 %dev', 'KV accuracy 110 %dev', 'KV accuracy 130 %dev',
    'Accuracy Timer 0.8 %dev', 'Accuracy Timer 1 %dev', 'Accuracy Timer 1.5 %dev',
    'Radiation Dose Test (Head) 21.50 %dev', 'Radiation Dose Test (Body) 10.60 %dev',
    'Leakage Max'
]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline — reusable in inference."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date')

    # %dev features
    for col, spec in specs.items():
        df[f"{col} %dev"] = (df[col] - spec) / spec * 100

    # Pass/Fail per parameter
    pass_cols = []
    for col, spec in specs.items():
        tol = tolerances[col]
        pass_col = f"{col} Pass"
        df[pass_col] = df[col].between(spec - tol, spec + tol)
        pass_cols.append(pass_col)

    # Leakage
    df['Leakage Max'] = df[LEAK_COLS].max(axis=1)
    df['Leakage Pass'] = df['Leakage Max'] <= LEAK_LIMIT

    # Overall pass
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

    print("📈 Computing ensemble scores...")
    iso_scores = iso.decision_function(X_scaled)
    lof_scores = lof.decision_function(X_scaled)
    ensemble_score = 0.7 * iso_scores + 0.3 * lof_scores
    threshold = np.percentile(ensemble_score, 18)

    print(f"✓ Threshold computed: {threshold:.6f}")

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

    print("\n✅ Model training complete!")
    print("📦 Saved to: model/ct_qc_production.pkl")
    print(f"🎯 Ensemble threshold: {threshold:.6f}")


if __name__ == '__main__':
    try:
        train()
    except FileNotFoundError:
        print("❌ Error: data/CT-Test.csv not found!")
        print("   Please place CT-Test.csv in the data/ folder")
    except Exception as e:
        print(f"❌ Error: {e}")
