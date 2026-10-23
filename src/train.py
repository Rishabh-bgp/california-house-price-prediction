"""Train the deployed California model and export it for the API (joblib) and the browser (JSON).

Deployed model: gradient-boosted trees (Huber loss) on log(value), plus two smaller quantile models (5% / 95%) on
log(value) for a rough range. Tree sizes are chosen so the exported JSON stays small enough for a browser
(see docs/web-app.md). Predictions are clipped to the $500k top-code, because the data cannot say anything above it.
"""
import json

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_validate

from src.features import CAP, FEATURES, ROOT, TARGET, load

SEED = 42
POINT = dict(n_estimators=400, learning_rate=0.1, max_depth=5, min_samples_leaf=20, subsample=0.8, random_state=SEED)
RANGE = dict(n_estimators=150, learning_rate=0.1, max_depth=4, min_samples_leaf=40, subsample=0.8, random_state=SEED)


def make(loss="huber", alpha=0.9):
    return GradientBoostingRegressor(loss=loss, alpha=alpha, **(POINT if loss != "quantile" else RANGE))


def export_gbr(m, X) -> dict:
    trees = []
    for est in m.estimators_[:, 0]:
        t = est.tree_
        trees.append({"f": t.feature.tolist(), "t": t.threshold.tolist(),
                      "l": t.children_left.tolist(), "r": t.children_right.tolist(),
                      "v": t.value[:, 0, 0].tolist()})
    lr = m.learning_rate
    tree_sum = sum(e.predict(X[:1])[0] for e in m.estimators_[:, 0])
    base = float(m.predict(X[:1])[0] - lr * tree_sum)
    return {"base": base, "lr": lr, "trees": trees}


def cv_scores(X, y):
    def rmse(est, Xv, yv):
        return -float(np.sqrt(np.mean((np.minimum(np.exp(est.predict(Xv)), CAP) - np.exp(yv)) ** 2)))
    r = cross_validate(make(), X, np.log(y), cv=KFold(5, shuffle=True, random_state=SEED), scoring={"rmse": rmse}, n_jobs=-1)
    return float(-r["test_rmse"].mean()), float(r["test_rmse"].std())


def train(cv: bool = True) -> dict:
    df = load()
    X, y = df[FEATURES].to_numpy(), df[TARGET].to_numpy()
    ly = np.log(y)
    point, lo, hi = make().fit(X, ly), make("quantile", 0.05).fit(X, ly), make("quantile", 0.95).fit(X, ly)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"point": point, "lo": lo, "hi": hi}, ROOT / "models" / "model.joblib")
    export = {"features": FEATURES, "target": "log(MedHouseVal)", "unit": "$100,000", "cap": CAP,
              "point": export_gbr(point, X), "lo": export_gbr(lo, X), "hi": export_gbr(hi, X)}
    (ROOT / "web" / "model.json").write_text(json.dumps(export, separators=(",", ":")))
    metrics = {"n": int(len(y))}
    if cv:
        m, s = cv_scores(X, y)
        metrics.update(cv5_rmse_mean=m, cv5_rmse_std=s)
    (ROOT / "models" / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=1))
