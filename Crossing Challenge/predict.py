from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np

MODEL_PATH = Path(__file__).parent / "model.pkl"
HORIZON_KEYS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]

_cached_models = None

def _load_models():
    global _cached_models
    if _cached_models is None:
        with open(MODEL_PATH, "rb") as f:
            _cached_models = pickle.load(f)
    return _cached_models

def _as_2d(x) -> np.ndarray:
    return np.stack([np.asarray(r, dtype=np.float64) for r in x])

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

def predict(request: dict) -> dict:
    models = _load_models()
    intent_model = models["intent"]
    traj_model = models["trajectory"]
    
    feats = _engineered_features(request).reshape(1, -1)
    if not np.isfinite(feats).all():
        feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
        
    intent_prob = float(intent_model.predict_proba(feats)[0, 1])
    if not np.isfinite(intent_prob):
        intent_prob = 0.5

    traj_pred_norm = traj_model.predict(feats)[0].reshape(4, 4)
    
    hist = _as_2d(request["bbox_history"])
    curr_box = hist[-1]
    fw, fh = float(request["frame_w"]), float(request["frame_h"])

    out = {}
    for h_idx, key in enumerate(HORIZON_KEYS):
        dx1, dy1, dx2, dy2 = traj_pred_norm[h_idx]
        px1 = curr_box[0] + dx1 * fw
        py1 = curr_box[1] + dy1 * fh
        px2 = curr_box[2] + dx2 * fw
        py2 = curr_box[3] + dy2 * fh
        
        box = [float(px1), float(py1), float(px2), float(py2)]
        out[key] = [v if np.isfinite(v) else 0.0 for v in box]
        
    out["intent"] = intent_prob
    return out
