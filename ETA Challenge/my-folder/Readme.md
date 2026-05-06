# ETA Challenge Submission

---

## Final Score
Dev MAE: **290.2 s**

---

## Approach
*We beat the baseline of ~350s by achieving **290.2s** using a more advanced XGBoost approach. Key improvements include:*
1. **Target Encoding**: Computed the historical mean trip duration for every unique `(pickup_zone, dropoff_zone)` pair. This powerful feature provides a highly accurate baseline estimate for the model.
2. **Native Categorical Features**: Replaced integer-based ordinal features with pandas `category` types, allowing XGBoost's `tree_method="hist"` to optimally split categories (like zones, hour, day-of-week) without assuming ordinality.
3. **Hyperparameter Tuning**: Adjusted `n_estimators=150`, `max_depth=8`, and `learning_rate=0.1` for an excellent trade-off between fast CPU inference and high accuracy.

## What Didn't Work
* Treating zones as continuous `int32` features (the baseline approach) fundamentally limits the model, as Zone 2 is not mathematically "greater" than Zone 1.

## AI Tooling Speedup
* Antigravity AI rapidly engineered the target encoding pipeline, ensured strict compliance with the 200ms inference limit by utilizing native XGBoost categorical support over manual one-hot encoding, and formatted the submission perfectly.

## Next Experiments
* **Geospatial Features**: Incorporating the NYC taxi zone shapefile to calculate exact spatial distances between centroids.
* **Weather Data**: Using NOAA weather to capture speed reductions during rain or snow.

## How to Reproduce
```bash
python train_advanced.py 
```
