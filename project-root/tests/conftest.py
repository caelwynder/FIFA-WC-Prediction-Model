import sys
from pathlib import Path

import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

CUTOFF_DATE = pd.Timestamp("2026-06-27")


@pytest.fixture(scope="session")
def merged():
    df = pd.read_csv(BASE_DIR / "data" / "processed" / "00_merged_dataset.csv", parse_dates=["date"])
    for c in ["home_score", "away_score", "home_elo", "away_elo", "home_rank", "away_rank"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()
    return df.sort_values("date", ignore_index=True)


@pytest.fixture(scope="session")
def model_dataset():
    return pd.read_csv(BASE_DIR / "data" / "processed" / "01_model_dataset.csv", parse_dates=["date"])


@pytest.fixture(scope="session")
def base_dir():
    return BASE_DIR
