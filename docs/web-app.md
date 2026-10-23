# Web app
* `web/model.json` (~1 MB): point model (400 trees, depth 5) + two quantile models (150 trees, depth 4), exported by `src/train.py`.
* The size was budgeted up front by choosing smaller quantile models; no post-hoc tree downsampling was needed.
* `web/data.json`: slider ranges (0.5th–99.5th percentiles), histogram, 2,500 sampled map points, 5 presets (nearest real block group to each city).
* Inference: float32-cast features, `x <= threshold` traversal, `exp`, clipped at the cap. Parity with scikit-learn: max |diff| 2e-14 over 20,640 rows.
* Map: Leaflet 1.9.4 + OpenStreetMap tiles (attribution shown). Art is original inline SVG.
