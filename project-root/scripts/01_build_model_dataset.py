import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, deque


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # script directory
DATA_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"
SCRIPT_DIR = BASE_DIR / "scripts"

IN_FILE = DATA_DIR / "00_merged_dataset.csv"
OUT_FILE = DATA_DIR / "01_model_dataset.csv"


# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
# Use np.nan if you want missing values left blank
# Change to -99.0 if you want a sentinel instead
MISSING_VALUE = np.nan

FORM_WINDOW = 5


# Tournament weighting lives in features.py (shared across the pipeline)
from features import get_tournament_weight


def save_in_chunks(df, out_file, chunk_size=100000):
    """
    Save large DataFrame in smaller chunks to avoid memory overload.
    """
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        print(f"Saving chunk {i + 1}/{num_chunks}...")
        chunk = df[i * chunk_size: (i + 1) * chunk_size]
        chunk.to_csv(out_file, mode='a', header=(i == 0), index=False)
    print(f"✅ Finished saving dataset to {out_file}")

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def safe_mean(values):
    return float(np.mean(values)) if len(values) > 0 else MISSING_VALUE


def safe_sum(values):
    return float(np.sum(values)) if len(values) > 0 else MISSING_VALUE


def safe_last(values):
    return float(values[-1]) if len(values) > 0 else MISSING_VALUE


def build_team_snapshot(history, team):
    h = history[team]

    points = list(h["points"])
    gd = list(h["gd"])
    gf = list(h["gf"])
    ga = list(h["ga"])
    wins = list(h["wins"])

    snapshot = {
        "matches_played_pre": h["matches_played"],
        "form_points_avg_5": safe_mean(points),
        "form_points_sum_5": safe_sum(points),
        "gd_avg_5": safe_mean(gd),
        "gd_sum_5": safe_sum(gd),
        "gf_avg_5": safe_mean(gf),
        "ga_avg_5": safe_mean(ga),
        "win_rate_5": safe_mean(wins),
        "last_match_gd": safe_last(gd),
        "last_match_points": safe_last(points),
        "days_since_last": MISSING_VALUE,
        "elo_before": h["last_elo"] if h["last_elo"] is not None else MISSING_VALUE,
        "rank_before": h["last_rank"] if h["last_rank"] is not None else MISSING_VALUE,
    }

    if h["last_date"] is not None:
        snapshot["days_since_last"] = (current_date - h["last_date"]).days

    return snapshot


def get_match_outcome(home_score, away_score):
    if home_score > away_score:
        return "H"
    if away_score > home_score:
        return "A"
    return "D"


def get_home_points(home_score, away_score):
    if home_score > away_score:
        return 3
    if home_score == away_score:
        return 1
    return 0


def get_away_points(home_score, away_score):
    if away_score > home_score:
        return 3
    if home_score == away_score:
        return 1
    return 0


# -------------------------------------------------------------------
# Main feature builder
# -------------------------------------------------------------------
def build_model_dataset():
    df = pd.read_csv(IN_FILE)

    # ---------------------------------------------------------------
    # Basic cleaning
    # ---------------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    numeric_cols = [
        "home_score", "away_score",
        "home_rating_change", "away_rating_change",
        "home_elo", "away_elo",
        "home_rank_change", "away_rank_change",
        "home_rank", "away_rank",
        "neutral",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()

    # Sort by time first
    df = df.sort_values(
        by=["date", "home_team", "away_team", "home_score", "away_score"],
        ignore_index=True
    )

    # ---------------------------------------------------------------
    # Basic engineered features
    # ---------------------------------------------------------------
    # NOTE: pre-match Elo/rank (home_elo_before etc.) are computed inside the
    # rolling history loop below by carrying each team's PREVIOUS match's
    # post-match Elo/rank. Do NOT derive them by subtracting the recorded
    # rating/rank "change" columns: the eloratings rank_change is
    # result-asymmetric (zero on ~93% of losses, nonzero on ~71% of wins),
    # so rank - rank_change leaks the match result into a pre-match feature.
    df["score_diff"] = df["home_score"] - df["away_score"]
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["tournament_weight"] = df["tournament"].apply(get_tournament_weight)
    df["neutral"] = df["neutral"].fillna(0)

    # Target columns
    df["match_outcome"] = df.apply(lambda r: get_match_outcome(r["home_score"], r["away_score"]), axis=1)
    df["target_home_win"] = (df["match_outcome"] == "H").astype(int)
    df["target_draw"] = (df["match_outcome"] == "D").astype(int)
    df["target_away_win"] = (df["match_outcome"] == "A").astype(int)

    # ---------------------------------------------------------------
    # Rolling team features (NO LEAKAGE)
    # ---------------------------------------------------------------
    history = defaultdict(lambda: {
        "points": deque(maxlen=FORM_WINDOW),
        "gd": deque(maxlen=FORM_WINDOW),
        "gf": deque(maxlen=FORM_WINDOW),
        "ga": deque(maxlen=FORM_WINDOW),
        "wins": deque(maxlen=FORM_WINDOW),
        "matches_played": 0,
        "last_date": None,
        "last_elo": None,
        "last_rank": None,
    })

    feature_rows = []

    global current_date
    for _, row in df.iterrows():
        current_date = row["date"]

        home = row["home_team"]
        away = row["away_team"]

        home_snapshot = build_team_snapshot(history, home)
        away_snapshot = build_team_snapshot(history, away)

        feature_rows.append({
            "home_elo_before": home_snapshot["elo_before"],
            "away_elo_before": away_snapshot["elo_before"],

            "home_rank_before": home_snapshot["rank_before"],
            "away_rank_before": away_snapshot["rank_before"],

            "home_matches_played_pre": home_snapshot["matches_played_pre"],
            "away_matches_played_pre": away_snapshot["matches_played_pre"],

            "home_form_points_avg_5": home_snapshot["form_points_avg_5"],
            "away_form_points_avg_5": away_snapshot["form_points_avg_5"],

            "home_form_points_sum_5": home_snapshot["form_points_sum_5"],
            "away_form_points_sum_5": away_snapshot["form_points_sum_5"],

            "home_gd_avg_5": home_snapshot["gd_avg_5"],
            "away_gd_avg_5": away_snapshot["gd_avg_5"],

            "home_gd_sum_5": home_snapshot["gd_sum_5"],
            "away_gd_sum_5": away_snapshot["gd_sum_5"],

            "home_gf_avg_5": home_snapshot["gf_avg_5"],
            "away_gf_avg_5": away_snapshot["gf_avg_5"],

            "home_ga_avg_5": home_snapshot["ga_avg_5"],
            "away_ga_avg_5": away_snapshot["ga_avg_5"],

            "home_win_rate_5": home_snapshot["win_rate_5"],
            "away_win_rate_5": away_snapshot["win_rate_5"],

            "home_last_match_gd": home_snapshot["last_match_gd"],
            "away_last_match_gd": away_snapshot["last_match_gd"],

            "home_last_match_points": home_snapshot["last_match_points"],
            "away_last_match_points": away_snapshot["last_match_points"],

            "home_days_since_last": home_snapshot["days_since_last"],
            "away_days_since_last": away_snapshot["days_since_last"],
        })

        # update AFTER extracting features
        home_points = get_home_points(row["home_score"], row["away_score"])
        away_points = get_away_points(row["home_score"], row["away_score"])

        home_gd = row["home_score"] - row["away_score"]
        away_gd = row["away_score"] - row["home_score"]

        history[home]["points"].append(home_points)
        history[home]["gd"].append(home_gd)
        history[home]["gf"].append(row["home_score"])
        history[home]["ga"].append(row["away_score"])
        history[home]["wins"].append(1 if home_points == 3 else 0)
        history[home]["matches_played"] += 1
        history[home]["last_date"] = row["date"]
        if pd.notna(row["home_elo"]):
            history[home]["last_elo"] = float(row["home_elo"])
        if pd.notna(row["home_rank"]):
            history[home]["last_rank"] = float(row["home_rank"])

        history[away]["points"].append(away_points)
        history[away]["gd"].append(away_gd)
        history[away]["gf"].append(row["away_score"])
        history[away]["ga"].append(row["home_score"])
        history[away]["wins"].append(1 if away_points == 3 else 0)
        history[away]["matches_played"] += 1
        history[away]["last_date"] = row["date"]
        if pd.notna(row["away_elo"]):
            history[away]["last_elo"] = float(row["away_elo"])
        if pd.notna(row["away_rank"]):
            history[away]["last_rank"] = float(row["away_rank"])

    feature_df = pd.DataFrame(feature_rows)
    df = pd.concat([df, feature_df], axis=1)

    # ---------------------------------------------------------------
    # Difference features (usually useful for tree models)
    # ---------------------------------------------------------------
    df["elo_diff_before"] = df["home_elo_before"] - df["away_elo_before"]
    df["rank_diff_before"] = df["away_rank_before"] - df["home_rank_before"]  # positive favors home if away rank is worse
    df["form_points_avg_diff_5"] = df["home_form_points_avg_5"] - df["away_form_points_avg_5"]
    df["form_points_sum_diff_5"] = df["home_form_points_sum_5"] - df["away_form_points_sum_5"]
    df["gd_avg_diff_5"] = df["home_gd_avg_5"] - df["away_gd_avg_5"]
    df["gd_sum_diff_5"] = df["home_gd_sum_5"] - df["away_gd_sum_5"]
    df["gf_avg_diff_5"] = df["home_gf_avg_5"] - df["away_gf_avg_5"]
    df["ga_avg_diff_5"] = df["home_ga_avg_5"] - df["away_ga_avg_5"]
    df["win_rate_diff_5"] = df["home_win_rate_5"] - df["away_win_rate_5"]
    df["days_rest_diff"] = df["home_days_since_last"] - df["away_days_since_last"]

    df = df.map(lambda x: round(x, 3) if isinstance(x, (int, float)) else x)

    # ---------------------------------------------------------------
    # Replace NaN with sentinel if you prefer
    # ---------------------------------------------------------------
    if not pd.isna(MISSING_VALUE):
        df = df.fillna(MISSING_VALUE)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    print(f"✅ Saved model dataset to: {OUT_FILE}")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print(df.head())


if __name__ == "__main__":
    build_model_dataset()