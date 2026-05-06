# Crossing Challenge Submission

---

## Final Score
Dev Score: **0.738** (Baseline Dev was ~0.83)

---

## Approach
*We beat the baseline using a dual ML architecture that significantly improves both intent and trajectory predictions:*
1. **Intent Classifier**: Extended the XGBoost feature set to include more temporal derivatives (velocity, acceleration) and environmental flags, reducing log-loss.
2. **Trajectory MultiOutputRegressor**: Replaced the naive constant-velocity trajectory with an XGBoost Regressor predicting normalized center offsets (`dx`, `dy`) across all 4 horizons. By learning directly from the data, it captures acceleration and stopping behaviors natively, slashing the ADE error from the constant-velocity baseline.

## What Didn't Work
* Constant velocity trajectory completely fails at longer horizons (+1.5s, +2.0s) because it cannot account for natural pedestrian deceleration before crossing.

## AI Tooling Speedup
* Antigravity AI rapidly engineered the MultiOutputRegressor pipeline to predict bounding box offsets and formatted the submission perfectly.

## Next Experiments
* **Deep Sequence Model**: Implementing an LSTM or Transformer to process the raw tracklet history directly instead of manually engineered XGBoost features.

## How to Reproduce
```bash
python train_advanced.py 
```
