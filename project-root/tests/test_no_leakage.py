# Guards against the rank-leak bug: pre-match Elo/rank in the model dataset
# must equal the team's PREVIOUS match's post-match value (carried forward),
# never `rank - rank_change` — the eloratings rank_change is result-asymmetric
# (zero on ~93% of losses, nonzero on ~71% of wins), so subtracting it embeds
# the match result into a "pre-match" feature. Before the fix this test's
# match rate was ~40%; the carry method makes it exact.

import numpy as np
import pandas as pd


def carried_before_values(merged: pd.DataFrame) -> pd.DataFrame:
    """Walk chronologically, carrying each team's last non-null elo/rank."""
    last_elo, last_rank = {}, {}
    rows = []
    for _, r in merged.iterrows():
        rows.append({
            "date": r["date"], "home_team": r["home_team"], "away_team": r["away_team"],
            "exp_home_elo_before": last_elo.get(r["home_team"], np.nan),
            "exp_away_elo_before": last_elo.get(r["away_team"], np.nan),
            "exp_home_rank_before": last_rank.get(r["home_team"], np.nan),
            "exp_away_rank_before": last_rank.get(r["away_team"], np.nan),
        })
        for side, team in [("home", r["home_team"]), ("away", r["away_team"])]:
            if pd.notna(r[f"{side}_elo"]):
                last_elo[team] = float(r[f"{side}_elo"])
            if pd.notna(r[f"{side}_rank"]):
                last_rank[team] = float(r[f"{side}_rank"])
    return pd.DataFrame(rows)


def test_before_features_are_previous_match_carry(merged, model_dataset):
    expected = carried_before_values(merged)
    got = model_dataset.merge(expected, on=["date", "home_team", "away_team"], how="inner")
    assert len(got) > 20000, "join between model dataset and merged dataset unexpectedly small"

    for col in ["home_elo_before", "away_elo_before", "home_rank_before", "away_rank_before"]:
        both = got.dropna(subset=[col, f"exp_{col}"])
        match_rate = np.isclose(both[col], both[f"exp_{col}"], atol=0.51).mean()
        assert match_rate > 0.98, (
            f"{col}: only {match_rate:.1%} of rows equal the previous-match carry — "
            "pre-match values are being derived some other way (possible result leak)"
        )
