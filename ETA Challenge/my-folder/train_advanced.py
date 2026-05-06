import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

DATA_DIR = Path("c:/Users/Mehul/OneDrive/Desktop/Codes/Gobble AI Project/eta-challenge-starter/data")
MODEL_PATH = Path("c:/Users/Mehul/OneDrive/Desktop/Codes/Gobble AI Project/Submission/model.pkl")

def create_features(df, pair_means, global_mean):
    ts = pd.to_datetime(df["requested_at"])
    
    # Create the pair feature
    pairs = df["pickup_zone"].astype(str) + "_" + df["dropoff_zone"].astype(str)
    zone_pair_mean = pairs.map(pair_means).fillna(global_mean).astype(np.float32)
    
    X = pd.DataFrame({
        "pickup_zone":     df["pickup_zone"].astype("category"),
        "dropoff_zone":    df["dropoff_zone"].astype("category"),
        "hour":            ts.dt.hour.astype("category"),
        "dow":             ts.dt.dayofweek.astype("category"),
        "month":           ts.dt.month.astype("category"),
        "passenger_count": df["passenger_count"].astype("int8"),
        "zone_pair_mean":  zone_pair_mean,
        "is_weekend":      (ts.dt.dayofweek >= 5).astype("int8"),
    })
    return X

def main():
    train_path = DATA_DIR / "train.parquet"
    dev_path = DATA_DIR / "dev.parquet"

    print("Loading data...")
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)

    # 1. Target Encoding for zone pairs
    print("Computing target encoding...")
    pairs_train = train["pickup_zone"].astype(str) + "_" + train["dropoff_zone"].astype(str)
    pair_means = train.groupby(pairs_train)["duration_seconds"].mean().to_dict()
    global_mean = train["duration_seconds"].mean()

    print("Engineering features...")
    X_train = create_features(train, pair_means, global_mean)
    y_train = train["duration_seconds"].to_numpy()
    
    X_dev = create_features(dev, pair_means, global_mean)
    y_dev = dev["duration_seconds"].to_numpy()

    print("Training XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        n_jobs=-1,
        random_state=42,
    )
    
    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_dev, y_dev)], verbose=50)
    print(f"Trained in {time.time() - t0:.0f}s")

    preds = model.predict(X_dev)
    mae = float(np.mean(np.abs(preds - y_dev)))
    print(f"\nDev MAE: {mae:.1f} seconds")

    # Save the model AND the target encoding dictionaries
    artifacts = {
        "model": model,
        "pair_means": pair_means,
        "global_mean": global_mean
    }
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"Saved artifacts to {MODEL_PATH}")

if __name__ == "__main__":
    main()
