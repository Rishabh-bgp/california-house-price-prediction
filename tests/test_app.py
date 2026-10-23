import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.main import app, get_models
from src.features import CAP, FEATURES, load
from src.predict import predict_json

ROOT = Path(__file__).resolve().parent.parent
client = TestClient(app)
EX = {"MedInc": 3.87, "HouseAge": 29, "AveRooms": 5.43, "AveBedrms": 1.10, "Population": 1166, "AveOccup": 2.82,
      "Latitude": 34.26, "Longitude": -118.49}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict():
    b = client.post("/predict", json=EX).json()
    lo, hi = b["range_5_95_usd"]
    assert 15_000 < lo < hi <= 500_000 and 50_000 < b["median_value_usd"] < 500_000


def test_validation():
    assert client.post("/predict", json={**EX, "Latitude": 50}).status_code == 422


def test_higher_income_is_not_cheaper():
    lo = client.post("/predict", json={**EX, "MedInc": 2.0}).json()["median_value_usd"]
    hi = client.post("/predict", json={**EX, "MedInc": 8.0}).json()["median_value_usd"]
    assert hi > lo


def test_never_exceeds_top_code():
    b = client.post("/predict", json={**EX, "MedInc": 15, "Latitude": 37.45, "Longitude": -122.18}).json()
    assert b["median_value_usd"] <= CAP * 100_000 and max(b["range_5_95_usd"]) <= CAP * 100_000


def test_json_export_matches_sklearn():
    m = get_models()
    model = json.loads((ROOT / "web" / "model.json").read_text())
    df = load().sample(1500, random_state=0)
    ref = {k: np.minimum(np.exp(m[k].predict(df[FEATURES].to_numpy())), CAP) for k in ("point", "lo", "hi")}
    for i, row in enumerate(df[FEATURES].to_dict("records")):
        got = predict_json(model, row)
        for k in got:
            assert abs(got[k] - ref[k][i]) < 1e-6
