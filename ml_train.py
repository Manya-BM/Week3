# ml_train.py
# Trains a RandomForest pipeline for EV range prediction.
# If you have a dataset, pass its path to train_and_save(csv_path="data/ev_specs_clean.csv")
# Otherwise it synthesizes a plausible dataset so the app works immediately.

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

MODEL_PATH = "model.joblib"

def synth_dataset(n=350, seed=42):
    rng = np.random.RandomState(seed)
    battery = rng.uniform(30, 90, n)        # kWh
    weight = rng.uniform(1200, 2400, n)     # kg
    speed = rng.uniform(30, 100, n)         # km/h
    drag = rng.uniform(0.22, 0.36, n)
    efficiency = rng.uniform(4.2, 5.8, n)   # km per kWh base factor
    range_km = battery * (efficiency - speed / 320) - (weight / 1000) * 8
    range_km += rng.normal(0, 15, n)
    df = pd.DataFrame({
        "battery_kwh": battery,
        "curb_weight": weight,
        "avg_speed": speed,
        "drag_coefficient": drag,
        "real_world_range_km": range_km.clip(40, 700)
    })
    return df

def train_and_save(csv_path=None):
    if csv_path and os.path.exists(csv_path):
        print(f"Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path)
        # try to rename common alternative columns
        if "EPA_range" in df.columns and "real_world_range_km" not in df.columns:
            df = df.rename(columns={"EPA_range": "real_world_range_km"})
        required = ["battery_kwh","curb_weight","avg_speed","drag_coefficient","real_world_range_km"]
        if not all(c in df.columns for c in required):
            print("CSV present but missing expected columns. Falling back to synthetic dataset.")
            df = synth_dataset()
    else:
        print("No CSV provided — generating synthetic dataset.")
        df = synth_dataset()

    features = ["battery_kwh","curb_weight","avg_speed","drag_coefficient"]
    X = df[features].fillna(df[features].median())
    y = df["real_world_range_km"].fillna(df["real_world_range_km"].median())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=150, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Trained model saved to: {MODEL_PATH}")
    print(f"MAE: {mae:.2f} km   R2: {r2:.3f}")

if __name__ == "__main__":
    # Call with a real CSV path if you have one, for example:
    # train_and_save(csv_path="data/ev_specs_clean.csv")
    train_and_save()
