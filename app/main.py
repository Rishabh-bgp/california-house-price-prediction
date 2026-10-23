"""FastAPI service for the California house price model."""
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features import CAP, FEATURES

MODEL = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
app = FastAPI(title="California House Price API", version="1.0.0")
_models = None


def get_models():
    global _models
    if _models is None:
        if not MODEL.exists():
            from src.train import train
            train(cv=False)
        _models = joblib.load(MODEL)
    return _models


class BlockGroup(BaseModel):
    MedInc: float = Field(..., ge=0.4, le=15.5, examples=[3.87], description="Median household income, tens of thousands of USD")
    HouseAge: float = Field(..., ge=1, le=52, examples=[29], description="Median house age in years (top-coded at 52)")
    AveRooms: float = Field(..., ge=0.5, le=150, examples=[5.43], description="Average rooms per household")
    AveBedrms: float = Field(..., ge=0.3, le=35, examples=[1.10], description="Average bedrooms per household")
    Population: float = Field(..., ge=1, le=40000, examples=[1166], description="Block group population")
    AveOccup: float = Field(..., ge=0.5, le=1300, examples=[2.82], description="Average household members")
    Latitude: float = Field(..., ge=32.5, le=42.0, examples=[34.26], description="Block group latitude")
    Longitude: float = Field(..., ge=-124.4, le=-114.3, examples=[-118.49], description="Block group longitude")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(b: BlockGroup):
    d = b.model_dump()
    x = np.array([[d[f] for f in FEATURES]])
    m = get_models()
    point, lo, hi = (min(float(np.exp(m[k].predict(x)[0])), CAP) for k in ("point", "lo", "hi"))
    return {"median_value_usd": round(point * 100_000), "range_5_95_usd": [round(lo * 100_000), round(hi * 100_000)],
            "at_top_code": point >= CAP * 0.97,
            "unit": "median house value of a 1990 census block group, 1990 US dollars; capped at $500,000"}
