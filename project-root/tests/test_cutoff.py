# The model is intentionally frozen at CUTOFF_DATE (end of the 2026 World Cup
# group stage) — no processed dataset may contain matches after it.

import pandas as pd

from conftest import CUTOFF_DATE


def test_merged_dataset_respects_cutoff(merged):
    assert merged["date"].max() <= CUTOFF_DATE


def test_model_dataset_respects_cutoff(model_dataset):
    assert model_dataset["date"].max() <= CUTOFF_DATE


def test_shootouts_respect_cutoff(base_dir):
    shootouts = pd.read_csv(base_dir / "data" / "raw" / "00_shootouts_final.csv", parse_dates=["date"])
    assert shootouts["date"].max() <= CUTOFF_DATE
