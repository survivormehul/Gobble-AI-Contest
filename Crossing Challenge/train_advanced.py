import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBClassifier, XGBRegressor

DATA = Path("c:/Users/Mehul/OneDrive/Desktop/Codes/Gobble AI Project/crossing-challenge-starter/data")
MODEL_PATH = Path("c:/Users/Mehul/OneDrive/Desktop/Codes/Gobble AI Project/Crossing Challenge/model.pkl")

REQUEST_FIELDS = [
    "ped_id", "frame_w", "frame_h",
    "time_of_day", "weather", "location", "ego_available",
    "bbox_history", "ego_speed_history", "ego_yaw_history",
    "requested_at_frame",
]

def _as_2d(x):
    return np.stack([np.asarray(r, dtype=np.float64) for r in x])

def row_to_request(row: pd.Series) -> dict:
    return {k: row[k] for k in REQUEST_FIELDS}

def _engineered_features(req: dict) -> np.ndarray:
    hist = _as_2d(req["bbox_history"])
    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    w = hist[:, 2] - hist[:, 0]
    h = hist[:, 3] - hist[:, 1]
    vx = np.diff(cx)
    vy = np.diff(cy)

    ego_s = np.asarray(req["ego_speed_history"], dtype=np.float64)
    ego_y = np.asarray(req["ego_yaw_history"], dtype=np.float64)

    fw = float(req["frame_w"])
    fh = float(req["frame_h"])
    
    # Adding a bit more history to features
    feats = [
        cx[-1] / fw, cy[-1] / fh, w[-1] / fw, h[-1] / fh,
        vx[-4:].mean() / fw, vy[-4:].mean() / fh,
        vx[-1] / fw, vy[-1] / fh,
        vx.std() / fw, vy.std() / fh,
        (h / (w + 1e-6)).mean(),
        float(req["ego_available"]),
        ego_s.mean(), ego_s[-1], ego_s.max(),
        ego_y.mean(), ego_y[-1], np.abs(ego_y).max(),
        1.0 if req.get("time_of_day") == "daytime" else 0.0,
        1.0 if req.get("time_of_day") == "nighttime" else 0.0,
        1.0 if req.get("weather") == "rain" else 0.0,
        1.0 if req.get("weather") == "snow" else 0.0,
        1.0 if req.get("location") == "street" else 0.0,
        1.0 if req.get("location") == "plaza" else 0.0,
    ]
    return np.asarray(feats, dtype=np.float32)

def featurize(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    sample = _engineered_features(row_to_request(df.iloc[0]))
    X = np.empty((n, len(sample)), dtype=np.float32)
    for i in range(n):
        X[i] = _engineered_features(row_to_request(df.iloc[i]))
    return X

def extract_trajectory_targets(df: pd.DataFrame) -> np.ndarray:
    targets = []
    horizons = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]
    for i in range(len(df)):
        row = df.iloc[i]
        fw, fh = float(row["frame_w"]), float(row["frame_h"])
        hist = _as_2d(row["bbox_history"])
        curr_box = hist[-1]
        
        # We will predict the normalized offset from the current box
        # [dx1, dy1, dx2, dy2] for each horizon
        y_row = []
        for h in horizons:
            fut_box = row[h]
            dx1 = (fut_box[0] - curr_box[0]) / fw
            dy1 = (fut_box[1] - curr_box[1]) / fh
            dx2 = (fut_box[2] - curr_box[2]) / fw
            dy2 = (fut_box[3] - curr_box[3]) / fh
            y_row.extend([dx1, dy1, dx2, dy2])
        targets.append(y_row)
    return np.array(targets, dtype=np.float32)

def main() -> None:
    print("Loading train + dev...")
    train = pd.read_parquet(DATA / "train.parquet")
    dev = pd.read_parquet(DATA / "dev.parquet")

    print("\nFeaturizing...")
    t0 = time.time()
    X_train = featurize(train)
    X_dev = featurize(dev)
    
    y_intent_train = train["will_cross_2s"].to_numpy(dtype=np.int32)
    y_intent_dev = dev["will_cross_2s"].to_numpy(dtype=np.int32)
    
    y_traj_train = extract_trajectory_targets(train)
    y_traj_dev = extract_trajectory_targets(dev)

    print("\nTraining Intent XGBClassifier...")
    intent_model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.03,
        tree_method="hist",
        n_jobs=-1,
        eval_metric="logloss",
    )
    intent_model.fit(X_train, y_intent_train, eval_set=[(X_dev, y_intent_dev)], verbose=False)
    
    dev_probs = intent_model.predict_proba(X_dev)[:, 1]
    ll = log_loss(y_intent_dev, np.clip(dev_probs, 1e-6, 1 - 1e-6))
    print(f"Dev log-loss: {ll:.4f}")

    print("\nTraining Trajectory MultiOutputRegressor...")
    base_reg = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        tree_method="hist",
        n_jobs=-1,
    )
    traj_model = MultiOutputRegressor(base_reg, n_jobs=1)
    traj_model.fit(X_train, y_traj_train)
    
    # Quick ADE evaluation on dev
    traj_preds_norm = traj_model.predict(X_dev)
    ades = []
    for i in range(len(dev)):
        row = dev.iloc[i]
        fw, fh = float(row["frame_w"]), float(row["frame_h"])
        curr_box = _as_2d(row["bbox_history"])[-1]
        
        pred = traj_preds_norm[i].reshape(4, 4)
        true_boxes = [row[h] for h in ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]]
        
        errs = []
        for h_idx in range(4):
            dx1, dy1, dx2, dy2 = pred[h_idx]
            px1 = curr_box[0] + dx1 * fw
            py1 = curr_box[1] + dy1 * fh
            px2 = curr_box[2] + dx2 * fw
            py2 = curr_box[3] + dy2 * fh
            p_cx, p_cy = (px1 + px2) / 2, (py1 + py2) / 2
            
            tx1, ty1, tx2, ty2 = true_boxes[h_idx]
            t_cx, t_cy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
            
            dist = np.hypot(p_cx - t_cx, p_cy - t_cy)
            errs.append(dist)
        ades.append(np.mean(errs))
        
    print(f"Dev ADE: {np.mean(ades):.2f} px")

    print(f"\nSaving models -> {MODEL_PATH}")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"intent": intent_model, "trajectory": traj_model}, f)

if __name__ == "__main__":
    main()
