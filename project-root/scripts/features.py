# scripts/features.py
#
# Single source of truth for the outcome model's feature schema and the
# team-snapshot / feature-row builders. Every pipeline stage imports from
# here — do not redeclare FEATURE_COLS or tournament weights in a script.
#
# Leakage rule: all pre-match values (elo_before, rank_before, form stats)
# are carried from each team's PREVIOUS matches only. Never derive a
# "before" value by subtracting a recorded change column — the eloratings
# rank_change is result-asymmetric and leaks the match result.

import numpy as np
import pandas as pd

FORM_WINDOW = 5

TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 1.00,
    "Confederations Cup": 0.85,
    "UEFA Euro": 0.90,
    "Copa América": 0.90,
    "African Cup of Nations": 0.85,
    "AFC Asian Cup": 0.85,
    "CONCACAF Gold Cup": 0.80,
    "Oceania Nations Cup": 0.75,
    "FIFA World Cup qualification": 0.80,
    "UEFA Euro qualification": 0.75,
    "African Cup of Nations qualification": 0.70,
    "AFC Asian Cup qualification": 0.70,
    "CONCACAF Nations League": 0.65,
    "UEFA Nations League": 0.70,
    "Friendly": 0.30,
}

FEATURE_COLS = [
    "neutral",
    "home_elo_before",
    "away_elo_before",
    "home_rank_before",
    "away_rank_before",
    "elo_diff_before",
    "rank_diff_before",
    "tournament_weight",
    "home_matches_played_pre",
    "away_matches_played_pre",
    "home_form_points_avg_5",
    "away_form_points_avg_5",
    "home_form_points_sum_5",
    "away_form_points_sum_5",
    "home_gd_avg_5",
    "away_gd_avg_5",
    "home_gd_sum_5",
    "away_gd_sum_5",
    "home_gf_avg_5",
    "away_gf_avg_5",
    "home_ga_avg_5",
    "away_ga_avg_5",
    "home_win_rate_5",
    "away_win_rate_5",
    "home_last_match_gd",
    "away_last_match_gd",
    "home_last_match_points",
    "away_last_match_points",
    "home_days_since_last",
    "away_days_since_last",
    "form_points_avg_diff_5",
    "form_points_sum_diff_5",
    "gd_avg_diff_5",
    "gd_sum_diff_5",
    "gf_avg_diff_5",
    "ga_avg_diff_5",
    "win_rate_diff_5",
    "days_rest_diff",
]


def get_tournament_weight(name: str) -> float:
    if pd.isna(name):
        return 0.50

    name = str(name).strip()
    if name in TOURNAMENT_WEIGHTS:
        return TOURNAMENT_WEIGHTS[name]

    lower = name.lower()
    if "world cup" in lower and "qualification" in lower:
        return 0.80
    if "world cup" in lower:
        return 1.00
    if "qualification" in lower:
        return 0.70
    if "friendly" in lower:
        return 0.30
    if "nations league" in lower:
        return 0.65
    if "cup" in lower:
        return 0.75
    return 0.50


def build_snapshot(df: pd.DataFrame, team: str, as_of: pd.Timestamp | None = None) -> dict:
    """Pre-match feature snapshot for a team from matches strictly before
    `as_of` (or the team's full history in `df` if as_of is None).
    `df` must be the merged dataset, date-sorted, with numeric score/elo/rank."""
    home = df[df["home_team"] == team]
    away = df[df["away_team"] == team]

    rows = []
    for sub, gf, ga, elo, rank in [
        (home, "home_score", "away_score", "home_elo", "home_rank"),
        (away, "away_score", "home_score", "away_elo", "away_rank"),
    ]:
        for _, r in sub.iterrows():
            rows.append({
                "date": r["date"], "gf": r[gf], "ga": r[ga],
                "elo": r[elo], "rank": r[rank],
            })

    hist = pd.DataFrame(rows).sort_values("date", ignore_index=True)
    if as_of is not None and not hist.empty:
        hist = hist[hist["date"] < as_of]
    if hist.empty:
        raise ValueError(f"No match history found for team: {team}")

    recent = hist.tail(FORM_WINDOW)
    points = recent.apply(
        lambda r: 3 if r["gf"] > r["ga"] else (1 if r["gf"] == r["ga"] else 0), axis=1
    )
    gd = recent["gf"] - recent["ga"]

    # last non-null elo/rank, not just the last row's (a merge gap in the
    # final match must not blank the rating)
    elo_series = hist["elo"].dropna()
    rank_series = hist["rank"].dropna()

    return {
        "matches_played_pre": len(hist),
        "elo_before": float(elo_series.iloc[-1]) if len(elo_series) else np.nan,
        "rank_before": float(rank_series.iloc[-1]) if len(rank_series) else np.nan,
        "form_points_avg_5": float(points.mean()),
        "form_points_sum_5": float(points.sum()),
        "gd_avg_5": float(gd.mean()),
        "gd_sum_5": float(gd.sum()),
        "gf_avg_5": float(recent["gf"].mean()),
        "ga_avg_5": float(recent["ga"].mean()),
        "win_rate_5": float((points == 3).mean()),
        "last_match_gd": float(gd.iloc[-1]),
        "last_match_points": float(points.iloc[-1]),
        "last_date": hist["date"].iloc[-1],
    }


def make_feature_row(h: dict, a: dict, match_date: pd.Timestamp, neutral: int,
                     tournament_weight: float = 1.0) -> dict:
    """Model-input row from two team snapshots (h = home side, a = away side)."""
    h_days = float((match_date - h["last_date"]).days)
    a_days = float((match_date - a["last_date"]).days)
    return {
        "neutral": neutral,
        "home_elo_before": h["elo_before"],
        "away_elo_before": a["elo_before"],
        "home_rank_before": h["rank_before"],
        "away_rank_before": a["rank_before"],
        "elo_diff_before": h["elo_before"] - a["elo_before"],
        "rank_diff_before": a["rank_before"] - h["rank_before"],
        "tournament_weight": tournament_weight,
        "home_matches_played_pre": h["matches_played_pre"],
        "away_matches_played_pre": a["matches_played_pre"],
        "home_form_points_avg_5": h["form_points_avg_5"],
        "away_form_points_avg_5": a["form_points_avg_5"],
        "home_form_points_sum_5": h["form_points_sum_5"],
        "away_form_points_sum_5": a["form_points_sum_5"],
        "home_gd_avg_5": h["gd_avg_5"],
        "away_gd_avg_5": a["gd_avg_5"],
        "home_gd_sum_5": h["gd_sum_5"],
        "away_gd_sum_5": a["gd_sum_5"],
        "home_gf_avg_5": h["gf_avg_5"],
        "away_gf_avg_5": a["gf_avg_5"],
        "home_ga_avg_5": h["ga_avg_5"],
        "away_ga_avg_5": a["ga_avg_5"],
        "home_win_rate_5": h["win_rate_5"],
        "away_win_rate_5": a["win_rate_5"],
        "home_last_match_gd": h["last_match_gd"],
        "away_last_match_gd": a["last_match_gd"],
        "home_last_match_points": h["last_match_points"],
        "away_last_match_points": a["last_match_points"],
        "home_days_since_last": h_days,
        "away_days_since_last": a_days,
        "form_points_avg_diff_5": h["form_points_avg_5"] - a["form_points_avg_5"],
        "form_points_sum_diff_5": h["form_points_sum_5"] - a["form_points_sum_5"],
        "gd_avg_diff_5": h["gd_avg_5"] - a["gd_avg_5"],
        "gd_sum_diff_5": h["gd_sum_5"] - a["gd_sum_5"],
        "gf_avg_diff_5": h["gf_avg_5"] - a["gf_avg_5"],
        "ga_avg_diff_5": h["ga_avg_5"] - a["ga_avg_5"],
        "win_rate_diff_5": h["win_rate_5"] - a["win_rate_5"],
        "days_rest_diff": h_days - a_days,
    }
