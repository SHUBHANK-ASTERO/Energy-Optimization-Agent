from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "energy_model.joblib"
FEATURES = ["hour", "temperature", "occupancy", "previous_kwh"]

def train_model(df):
    X = df[FEATURES]
    y = df["energy_kwh"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=180, max_depth=12, random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred))
    }
    return model, metrics

def ensure_model(df):
    # Train locally when the model is missing. This avoids cross-version
    # pickle warnings and keeps the project reproducible on the user's PC.
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass
    model, metrics = train_model(df)
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained. MAE={metrics['mae']:.3f}, R2={metrics['r2']:.3f}")
    return model

def predict_usage(model, row):
    x = pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)
    return max(0.0, float(model.predict(x)[0]))

def detect_anomalies(df):
    x = df["energy_kwh"]
    mean = x.rolling(24, min_periods=6).mean()
    std = x.rolling(24, min_periods=6).std().replace(0, np.nan)
    z = ((x - mean) / std).fillna(0)
    result = df[z.abs() > 2.2].copy()
    return result

def build_recommendations(predicted, latest, anomalies):
    tips = []
    if predicted > 5:
        tips.append("High consumption is predicted. Switch off non-essential appliances.")
    elif predicted > 3.5:
        tips.append("Moderate-high consumption is expected. Reduce unnecessary device usage.")
    else:
        tips.append("Consumption is currently within an efficient range.")

    if latest["temperature"] > 30:
        tips.append("Temperature is high: use AC efficiently and keep the set point around 24–26°C.")

    if latest["occupancy"] <= 1:
        tips.append("Low occupancy detected: turn off lights, fans and AC in unused areas.")

    if latest["ac_kwh"] > 2:
        tips.append("AC is a major load. Use timers, clean filters and avoid very low temperature settings.")

    if len(anomalies) > 0:
        tips.append("Unusual energy spikes were detected. Check high-power appliances and equipment.")

    return tips[:5]
