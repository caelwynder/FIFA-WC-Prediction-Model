# Train/inference parity: the features the training dataset (01) records for
# a match must equal what the shared snapshot builder (features.py) computes
# for the same team at the same date. A drift here means the model is trained
# on one definition of "form" and asked to predict with another — exactly the
# skew that existed when 01 derived rank_before by subtracting rank_change
# while inference carried the previous match's rank.

import numpy as np
import pandas as pd

from features import build_snapshot

CHECK_COLS = [
    ("home_elo_before", "elo_before"),
    ("home_rank_before", "rank_before"),
    ("home_matches_played_pre", "matches_played_pre"),
    ("home_form_points_sum_5", "form_points_sum_5"),
    ("home_win_rate_5", "win_rate_5"),
    ("home_gd_sum_5", "gd_sum_5"),
]


def sample_rows(model_dataset: pd.DataFrame, merged: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Recent rows where the home team plays only once that day (so the
    strict `date < as_of` filter in build_snapshot sees the same history
    as 01's sequential loop)."""
    recent = model_dataset[model_dataset["date"] >= "2024-01-01"]
    counts = merged.melt(
        id_vars="date", value_vars=["home_team", "away_team"], value_name="team"
    ).groupby(["date", "team"]).size()
    picked = []
    for _, row in recent.sample(min(80, len(recent)), random_state=7).iterrows():
        if counts.get((row["date"], row["home_team"]), 0) == 1:
            picked.append(row)
        if len(picked) == n:
            break
    return pd.DataFrame(picked)


def test_training_features_match_snapshot_builder(model_dataset, merged):
    rows = sample_rows(model_dataset, merged)
    assert len(rows) >= 10, "not enough single-match-day rows sampled"

    for _, row in rows.iterrows():
        snap = build_snapshot(merged, row["home_team"], as_of=row["date"])
        for ds_col, snap_key in CHECK_COLS:
            got, want = row[ds_col], snap[snap_key]
            if pd.isna(got) and pd.isna(want):
                continue
            assert np.isclose(got, want, atol=2e-3), (
                f"{row['home_team']} on {row['date'].date()}: dataset {ds_col}={got} "
                f"but snapshot builder computes {want}"
            )
