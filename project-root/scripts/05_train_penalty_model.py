# scripts/05_train_penalty_model.py
#
# Penalty shootout model: P(home side wins the shootout), used in tandem with
# the outcome model — when the outcome model's top prediction is a draw, the
# match is sent here to decide a winner on penalties.
#
# Training data: international shootouts since 2000 (00_shootouts_final.csv,
# capped at the pipeline cutoff). Features are computed strictly from history
# BEFORE each shootout:
#   - each side's prior shootout record (count, win rate, recent win rate)
#   - in-game penalty-goal rate (00_penalty_goals_final.csv)
#   - World Cup shootout kick conversion (00_wc_penalty_kicks.csv, kick-by-kick)
#   - Elo / rank / recent form from the merged dataset
#
# Saves models/penalty_model.joblib with the model, feature columns, and a
# per-team feature profile frozen at the cutoff for inference.

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

MERGED_FILE = DATA_DIR / "00_merged_dataset.csv"
SHOOTOUTS_FILE = RAW_DIR / "00_shootouts_final.csv"
PEN_GOALS_FILE = RAW_DIR / "00_penalty_goals_final.csv"
WC_KICKS_FILE = RAW_DIR / "00_wc_penalty_kicks.csv"
OUT_MODEL = MODEL_DIR / "penalty_model.joblib"

CUTOFF_DATE = pd.Timestamp("2026-06-27")
MIN_DATE = pd.Timestamp("2000-01-01")  # merged dataset starts here
SPLIT_DATE = pd.Timestamp("2021-06-01")
RECENT_N = 5
FORM_WINDOW = 5

FEATURE_COLS = [
    "home_so_played", "away_so_played",
    "home_so_winrate", "away_so_winrate",
    "home_so_recent_winrate", "away_so_recent_winrate",
    "home_pen_goal_rate", "away_pen_goal_rate",
    "home_wc_kick_conv", "away_wc_kick_conv",
    "home_elo", "away_elo",
    "elo_diff", "rank_diff", "win_rate_5_diff",
    "so_winrate_diff", "so_experience_diff",
]


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
def load_all():
    merged = pd.read_csv(MERGED_FILE, parse_dates=["date"])
    merged["home_team"] = merged["home_team"].astype(str).str.strip()
    merged["away_team"] = merged["away_team"].astype(str).str.strip()
    for c in ["home_score", "away_score", "home_elo", "away_elo", "home_rank", "away_rank"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    merged = merged.sort_values("date", ignore_index=True)

    shootouts = pd.read_csv(SHOOTOUTS_FILE, parse_dates=["date"])
    pen_goals = pd.read_csv(PEN_GOALS_FILE, parse_dates=["date"])
    wc_kicks = pd.read_csv(WC_KICKS_FILE, parse_dates=["match_date"])
    return merged, shootouts, pen_goals, wc_kicks


# -------------------------------------------------------------------
# Per-team feature profile as of a date (history strictly before it)
# -------------------------------------------------------------------
def team_profile(team: str, as_of: pd.Timestamp, merged, shootouts, pen_goals, wc_kicks) -> dict:
    # shootout record
    so = shootouts[
        ((shootouts["home_team"] == team) | (shootouts["away_team"] == team))
        & (shootouts["date"] < as_of)
    ]
    so_played = len(so)
    wins = (so["winner"] == team).astype(int)
    so_winrate = float(wins.mean()) if so_played else 0.5
    so_recent = float(wins.tail(RECENT_N).mean()) if so_played else 0.5

    # matches + elo/rank/form from merged dataset
    home = merged[(merged["home_team"] == team) & (merged["date"] < as_of)]
    away = merged[(merged["away_team"] == team) & (merged["date"] < as_of)]
    n_matches = len(home) + len(away)

    gf = pd.concat([home["home_score"], away["away_score"]])
    ga = pd.concat([home["away_score"], away["home_score"]])
    dates = pd.concat([home["date"], away["date"]])
    elo = pd.concat([home["home_elo"], away["away_elo"]])
    rank = pd.concat([home["home_rank"], away["away_rank"]])
    order = dates.sort_values().index
    gf, ga, elo, rank = gf[order], ga[order], elo[order], rank[order]

    if n_matches:
        recent_wins = (gf.tail(FORM_WINDOW) > ga.tail(FORM_WINDOW)).astype(int)
        win_rate_5 = float(recent_wins.mean())
        last_elo = float(elo.iloc[-1]) if pd.notna(elo.iloc[-1]) else np.nan
        last_rank = float(rank.iloc[-1]) if pd.notna(rank.iloc[-1]) else np.nan
    else:
        win_rate_5, last_elo, last_rank = np.nan, np.nan, np.nan

    # in-game penalty goal rate
    pg = pen_goals[(pen_goals["team"] == team) & (pen_goals["date"] < as_of)]
    pen_goal_rate = len(pg) / n_matches if n_matches else np.nan

    # WC shootout kick conversion
    kicks = wc_kicks[(wc_kicks["team_name"] == team) & (wc_kicks["match_date"] < as_of)]
    wc_conv = float(kicks["converted"].mean()) if len(kicks) else np.nan

    return {
        "so_played": so_played,
        "so_winrate": so_winrate,
        "so_recent_winrate": so_recent,
        "pen_goal_rate": pen_goal_rate,
        "wc_kick_conv": wc_conv,
        "elo": last_elo,
        "rank": last_rank,
        "win_rate_5": win_rate_5,
    }


def make_row(hp: dict, ap: dict) -> dict:
    return {
        "home_so_played": hp["so_played"],
        "away_so_played": ap["so_played"],
        "home_so_winrate": hp["so_winrate"],
        "away_so_winrate": ap["so_winrate"],
        "home_so_recent_winrate": hp["so_recent_winrate"],
        "away_so_recent_winrate": ap["so_recent_winrate"],
        "home_pen_goal_rate": hp["pen_goal_rate"],
        "away_pen_goal_rate": ap["pen_goal_rate"],
        "home_wc_kick_conv": hp["wc_kick_conv"],
        "away_wc_kick_conv": ap["wc_kick_conv"],
        "home_elo": hp["elo"],
        "away_elo": ap["elo"],
        "elo_diff": hp["elo"] - ap["elo"] if pd.notna(hp["elo"]) and pd.notna(ap["elo"]) else np.nan,
        "rank_diff": ap["rank"] - hp["rank"] if pd.notna(hp["rank"]) and pd.notna(ap["rank"]) else np.nan,
        "win_rate_5_diff": hp["win_rate_5"] - ap["win_rate_5"] if pd.notna(hp["win_rate_5"]) and pd.notna(ap["win_rate_5"]) else np.nan,
        "so_winrate_diff": hp["so_winrate"] - ap["so_winrate"],
        "so_experience_diff": hp["so_played"] - ap["so_played"],
    }


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------
def build_training_frame(merged, shootouts, pen_goals, wc_kicks) -> pd.DataFrame:
    rows = []
    usable = shootouts[shootouts["date"] >= MIN_DATE]
    for _, s in usable.iterrows():
        hp = team_profile(s["home_team"], s["date"], merged, shootouts, pen_goals, wc_kicks)
        ap = team_profile(s["away_team"], s["date"], merged, shootouts, pen_goals, wc_kicks)
        row = make_row(hp, ap)
        row["date"] = s["date"]
        row["target"] = 1 if s["winner"] == s["home_team"] else 0
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    merged, shootouts, pen_goals, wc_kicks = load_all()
    assert shootouts["date"].max() <= CUTOFF_DATE, "shootouts file contains post-cutoff rows"

    print(f"Building features for {len(shootouts[shootouts['date'] >= MIN_DATE])} shootouts since {MIN_DATE.date()}...")
    df = build_training_frame(merged, shootouts, pen_goals, wc_kicks)

    train = df[df["date"] < SPLIT_DATE]
    test = df[df["date"] >= SPLIT_DATE]
    X_tr, y_tr = train[FEATURE_COLS].fillna(-99.0), train["target"]
    X_te, y_te = test[FEATURE_COLS].fillna(-99.0), test["target"]

    def fit(X, y):
        # Heavily regularized: shootouts are near coin flips and 400 rows
        # overfit fast. depth-2 trees + min_child_weight=20 was the only
        # variant that matched the base-rate log loss out of sample while
        # keeping probabilities in a sane band (~0.38-0.65).
        m = XGBClassifier(
            n_estimators=80, max_depth=2, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6, min_child_weight=20,
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
        m.fit(X, y)
        return m

    model = fit(X_tr, y_tr)
    prob = model.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, prob >= 0.5)
    ll = log_loss(y_te, prob)
    base_rate = y_te.mean()
    base_ll = log_loss(y_te, np.full(len(y_te), y_te.mean()))

    print(f"\nTrain shootouts: {len(train)} ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"Test shootouts:  {len(test)} ({test['date'].min().date()} to {test['date'].max().date()})")
    print(f"Test accuracy:   {acc:.4f}  (majority-class baseline {max(base_rate, 1-base_rate):.4f})")
    print(f"Test log loss:   {ll:.4f}  (base-rate baseline {base_ll:.4f})")

    # refit on everything, then freeze per-team profiles at the cutoff
    final_model = fit(df[FEATURE_COLS].fillna(-99.0), df["target"])

    wc_group = merged[(merged["tournament"] == "FIFA World Cup") & (merged["date"] >= "2026-06-11")]
    wc_teams = sorted(set(wc_group["home_team"]) | set(wc_group["away_team"]))
    print(f"\nFreezing cutoff profiles for {len(wc_teams)} World Cup teams...")
    profiles = {
        t: team_profile(t, CUTOFF_DATE, merged, shootouts, pen_goals, wc_kicks)
        for t in wc_teams
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "feature_cols": FEATURE_COLS,
            "profiles": profiles,
            "cutoff": str(CUTOFF_DATE.date()),
            "test_accuracy": acc,
            "test_log_loss": ll,
        },
        OUT_MODEL,
    )
    print(f"Saved penalty model -> {OUT_MODEL}")


if __name__ == "__main__":
    main()
