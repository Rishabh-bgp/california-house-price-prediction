# Methodology
* Repeated 5-fold CV (3 repeats, seed 42) for all models; predictions clipped at the $500k cap before scoring.
* Log target via TransformedTargetRegressor; Huber loss for robustness.
* Spatial CV: 50 KMeans clusters of lat/lon with GroupKFold(5).
* Permutation importance on an 80/20 hold-out, 10 repeats.
* Quantile GBR (alpha 0.05 / 0.95) for a range; coverage measured on the hold-out (84.8%).
