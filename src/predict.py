"""Pure-Python inference from the exported JSON (same algorithm as web/app.js)."""
import math

import numpy as np


def _tree(t, x):
    # scikit-learn compares float32-cast features against float64 thresholds.
    n = 0
    while t["l"][n] != -1:
        n = t["l"][n] if x[t["f"][n]] <= t["t"][n] else t["r"][n]
    return t["v"][n]


def _gbr(m, x):
    return m["base"] + m["lr"] * sum(_tree(t, x) for t in m["trees"])


def predict_json(model: dict, raw: dict) -> dict:
    x = [float(np.float32(raw[f])) for f in model["features"]]
    return {k: min(math.exp(_gbr(model[k], x)), model["cap"]) for k in ("point", "lo", "hi")}
