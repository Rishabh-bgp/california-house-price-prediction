"""Data loading for the California Housing data (1990 U.S. census block groups; Pace & Barry, 1997).

The target ``MedHouseVal`` is the median house value in units of $100,000. It is top-coded: every block group whose
median value was at or above $500,000 is recorded as 5.00001, so the highest values are unknown (see README).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "california_housing.csv"
TARGET = "MedHouseVal"
CAP = 5.0  # $500,000 top-code
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]


def load() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


def is_capped(y) -> "pd.Series":
    return y >= CAP
