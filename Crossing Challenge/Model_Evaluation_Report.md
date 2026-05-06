# Model Evaluation Report: Pedestrian Crossing Predictor

## 1. Executive Summary
This document details the evaluation methodology and metrics for the Pedestrian Crossing Intent and Trajectory prediction system. The final model attained a combined **Dev Score of 0.738** (beating the ~0.83 baseline). The architecture employs a dual XGBoost pipeline specifically engineered to meet strict 2GB memory constraints and sub-200ms CPU latency.

## 2. Experimental Setup
*   **Architecture:**
    *   **Intent:** `XGBClassifier`
    *   **Trajectory:** `MultiOutputRegressor(XGBRegressor)`
*   **Validation Strategy:** Standardized Train/Dev split on the provided dataset to measure out-of-sample log-loss and Average Displacement Error (ADE).
*   **Feature Engineering:**
    *   Temporal derivatives: Velocity (`vx`, `vy`) and acceleration metrics across bounding box history.
    *   Ego-vehicle dynamics: Speed history and yaw history.

## 3. Hyperparameters
Two distinct models were trained to isolate the tasks optimally:

**Intent Classifier:**
```python
intent_model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.03,
    tree_method="hist",
    eval_metric="logloss"
)
```

**Trajectory Regressor:**
```python
traj_model = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05,
    tree_method="hist"
)
```

## 4. Performance Metrics
*   **Intent Log-loss:** Assessed on a clipped probability distribution (`1e-6` to `1 - 1e-6`) to prevent extreme penalties.
*   **Trajectory ADE:** Calculated across 4 distinct horizons (500ms, 1000ms, 1500ms, 2000ms) comparing predicted bounding box centroids against true bounding box centroids.
*   **Overall Combined Dev Score:** `0.738`

## 5. Error Analysis & Edge Cases
*   **Short-Horizon vs. Long-Horizon:** Trajectory ADE increases significantly at the +1.5s and +2.0s marks. While the XGBoost MultiOutputRegressor outperforms the constant-velocity baseline, predicting exact pedestrian deceleration/stopping behavior over long horizons remains difficult without a dedicated recurrent architecture (e.g., LSTM).
*   **Next Steps:** Transitioning from manual feature engineering to a lightweight Sequence-to-Sequence model could yield better native trajectory awareness without heavily bloating the Docker image size.
