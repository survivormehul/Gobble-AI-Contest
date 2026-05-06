import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_MODEL_PATH = Path(__file__).parent / "model.pkl"

with open(_MODEL_PATH, "rb") as _f:
    _artifacts = pickle.load(_f)

_MODEL = _artifacts["model"]
_PAIR_MEANS = _artifacts["pair_means"]
_GLOBAL_MEAN = _artifacts["global_mean"]

def predict(request: dict) -> float:
    ts = datetime.fromisoformat(request["requested_at"])
    
    pu = request["pickup_zone"]
    do = request["dropoff_zone"]
    pair_str = f"{pu}_{do}"
    zone_pair_mean = _PAIR_MEANS.get(pair_str, _GLOBAL_MEAN)
    
    # We construct a pandas DataFrame directly because the model expects pandas Categories
    # It takes ~1ms to build this dictionary/DataFrame which easily fits in the 200ms limit.
    X = pd.DataFrame({
        "pickup_zone":     [pu],
        "dropoff_zone":    [do],
        "hour":            [ts.hour],
        "dow":             [ts.weekday()],
        "month":           [ts.month],
        "passenger_count": [int(request["passenger_count"])],
        "zone_pair_mean":  [np.float32(zone_pair_mean)],
        "is_weekend":      [int(ts.weekday() >= 5)],
    })
    
    # Cast categories so XGBoost recognizes them as categorical features
    X["pickup_zone"] = X["pickup_zone"].astype("category")
    X["dropoff_zone"] = X["dropoff_zone"].astype("category")
    X["hour"] = X["hour"].astype("category")
    X["dow"] = X["dow"].astype("category")
    X["month"] = X["month"].astype("category")
    X["passenger_count"] = X["passenger_count"].astype("int8")
    X["is_weekend"] = X["is_weekend"].astype("int8")
    
    return float(_MODEL.predict(X)[0])
