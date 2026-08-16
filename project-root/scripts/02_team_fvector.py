import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------------------------------------------------
# INPUTS
# -------------------------------------------------------------------
FIXTURES = [
    {
        "home_team": "Chile",
        "away_team": "Cape Verde",
        "match_date": "2026-03-27",
        "tournament": "Friendly",
        "neutral": 0,
    },
    {
        "home_team": "China",
        "away_team": "Curaçao",
        "match_date": "2026-03-27",
        "tournament": "Friendly",
        "neutral": 0,
    },
    {
        "home_team": "New Zealand",
        "away_team": "Finland",
        "match_date": "2026-03-27",
        "tournament": "Friendly",
        "neutral": 0,
    },
    {
        "home_team": "Solomon Islands",
        "away_team": "Bulgaria",
        "match_date": "2026-03-27",
        "tournament": "Friendly",
        "neutral": 0,
    },
    {
        "home_team": "Australia",
        "away_team": "Cameroon",
        "match_date": "2026-03-27",
        "tournament": "Friendly",
        "neutral": 0,
    },
    {
        "home_team": "Venezuela",
        "away_team": "Trinidad and Tobago",
        "match_date": "2026-03-27",
        "tournament": "Friendly",
        "neutral": 0,
    },
    {
        "home_team": "Iran",
        "away_team": "Nigeria",
        "match_date": "2026-03-27",
        "tournament": "Friendly",
        "neutral": 0,
    }, 
    {
        "home_team": "Indonesia",
        "away_team": "Saint Kitts and Nevis",
        "match_date": "2026-03-27",
        "tournament": "Friendly",
        "neutral": 0,
    }, 
    {
        "home_team": "Azerbaijan",
        "away_team": "Saint Lucia",
        "match_date": "2026-03-27",
        "tournament": "Friendly",
        "neutral": 0,
    }, 
    {
        "home_team": "Kenya",
        "away_team": "Estonia",
        "match_date": "2026-03-27",
        "tournament": "Friendly",
        "neutral": 0,
    }, 
    {
        "home_team": "Russia",
        "away_team": "Nicaragua",
        "match_date": "2026-03-27",
        "tournament": "Friendly",
        "neutral": 0,
    }, 
    {
        "home_team": "Austria",
        "away_team": "Ghana",
        "match_date": "2026-03-27",
        "tournament": "Friendly",
        "neutral": 0,
    }, 
    {
        "home_team": "Montenegro",
        "away_team": "Andorra",
        "match_date": "2026-03-27",
        "tournament": "Friendly",
        "neutral": 0,
    }, 
    {
        "home_team": "South Africa",
        "away_team": "Panama",
        "match_date": "2026-03-27",
        "tournament": "Friendly",
        "neutral": 0,
    }, 
    {
        "home_team": "Jordan",
        "away_team": "Costa Rica",
        "match_date": "2026-03-27",
        "tournament": "Friendly",
        "neutral": 0,
    }, 
    {
        "home_team": "Saudi Arabia",
        "away_team": "Egypt",
        "match_date": "2026-03-27",
        "tournament": "Friendly",
        "neutral": 0,
    }, 
    {
        "home_team": "Greece",
        "away_team": "Paraguay",
        "match_date": "2026-03-27",
        "tournament": "Friendly",
        "neutral": 0,
    }, 
    {
        "home_team": "Algeria",
        "away_team": "Guatemala",
        "match_date": "2026-03-27",
        "tournament": "Friendly",
        "neutral": 0,
    }, 
    {
        "home_team": "England",
        "away_team": "Uruguay",
        "match_date": "2026-03-27",
        "tournament": "Friendly",
        "neutral": 0,
    }, 
    {
        "home_team": "Netherlands",
        "away_team": "Norway",
        "match_date": "2026-03-27",
        "tournament": "Friendly",
        "neutral": 0,
    }, 
    {
        "home_team": "Switzerland",
        "away_team": "Germany",
        "match_date": "2026-03-27",
        "tournament": "Friendly",
        "neutral": 0,
    }, 
    {
        "home_team": "Spain",
        "away_team": "Serbia",
        "match_date": "2026-03-27",
        "tournament": "Friendly",
        "neutral": 0,
    }, 
    {
        "home_team": "Morocco",
        "away_team": "Ecuador",
        "match_date": "2026-03-27",
        "tournament": "Friendly",
        "neutral": 0,
    }, 
    {
        "home_team": "Argentina",
        "away_team": "Mauritania",
        "match_date": "2026-03-27",
        "tournament": "Friendly",
        "neutral": 0,
    }, 
    {
        "home_team": "Zambia",
        "away_team": "Malawi",
        "match_date": "2026-03-28",
        "tournament": "Friendly",
        "neutral": 0,
    },
    {
        "home_team": "Korea Republic",
        "away_team": "Ivory Coast",
        "match_date": "2026-03-28",
        "tournament": "Friendly",
        "neutral": 0,
    },
    {
        "home_team": "San Marino",
        "away_team": "Faroe Islands",
        "match_date": "2026-03-28",
        "tournament": "Friendly",
        "neutral": 0,
    },
    {
        "home_team": "Senegal",
        "away_team": "Peru",
        "match_date": "2026-03-28",
        "tournament": "Friendly",
        "neutral": 0,
    },
    {
        "home_team": "Canada",
        "away_team": "Iceland",
        "match_date": "2026-03-28",
        "tournament": "Friendly",
        "neutral": 0,
    },
    {
        "home_team": "Hungary",
        "away_team": "Slovenia",
        "match_date": "2026-03-28",
        "tournament": "Friendly",
        "neutral": 0,
    },
    {
        "home_team": "Scotland",
        "away_team": "Japan",
        "match_date": "2026-03-28",
        "tournament": "Friendly",
        "neutral": 0,
    },
    {
        "home_team": "United States",
        "away_team": "Belgium",
        "match_date": "2026-03-28",
        "tournament": "Friendly",
        "neutral": 0,
    },
    {
        "home_team": "Haiti",
        "away_team": "Tunisia",
        "match_date": "2026-03-29",
        "tournament": "Friendly",
        "neutral": 0,
    },
    {
        "home_team": "Mexico",
        "away_team": "Portugal",
        "match_date": "2026-03-29",
        "tournament": "Friendly",
        "neutral": 0,
    },
    {
        "home_team": "Lithuania",
        "away_team": "Georgia",
        "match_date": "2026-03-29",
        "tournament": "Friendly",
        "neutral": 0,
    },
    {
        "home_team": "Macau",
        "away_team": "Tanzania",
        "match_date": "2026-03-29",
        "tournament": "Friendly",
        "neutral": 0,
    },
    {
        "home_team": "Armenia",
        "away_team": "Belarus",
        "match_date": "2026-03-29",
        "tournament": "Friendly",
        "neutral": 0,
    },
    {
        "home_team": "Aruba",
        "away_team": "Liechtenstein",
        "match_date": "2026-03-29",
        "tournament": "Friendly",
        "neutral": 0,
    },
    {
        "home_team": "Colombia",
        "away_team": "France",
        "match_date": "2026-03-29",
        "tournament": "Friendly",
        "neutral": 0,
    },

]

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" 

MAIN_DATA_FILE = DATA_DIR / "processed" / "00_merged_dataset.csv"
OUT_DIR = DATA_DIR / "feature_vectors"
OUT_DIR_INDIVIDUAL = DATA_DIR / "feature_vectors" / "individual_matches"

# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
from features import FEATURE_COLS, FORM_WINDOW, get_tournament_weight


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def safe_mean(values):
    return float(np.mean(values)) if len(values) > 0 else np.nan


def safe_sum(values):
    return float(np.sum(values)) if len(values) > 0 else np.nan


def safe_last(values):
    return float(values[-1]) if len(values) > 0 else np.nan


def get_points(goals_for, goals_against):
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def clean_filename(text: str) -> str:
    return (
        str(text)
        .strip()
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "")
        .replace("*", "")
        .replace("?", "")
        .replace('"', "")
        .replace("<", "")
        .replace(">", "")
        .replace("|", "")
        .replace(" ", "_")
    )


def get_team_match_history(df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    home_matches = df[df["home_team"] == team_name].copy()
    away_matches = df[df["away_team"] == team_name].copy()

    if not home_matches.empty:
        home_matches["team_goals"] = home_matches["home_score"]
        home_matches["opp_goals"] = home_matches["away_score"]
        home_matches["team_elo_after"] = home_matches["home_elo"]
        home_matches["team_rank_after"] = home_matches["home_rank"]
        home_matches["opponent"] = home_matches["away_team"]

    if not away_matches.empty:
        away_matches["team_goals"] = away_matches["away_score"]
        away_matches["opp_goals"] = away_matches["home_score"]
        away_matches["team_elo_after"] = away_matches["away_elo"]
        away_matches["team_rank_after"] = away_matches["away_rank"]
        away_matches["opponent"] = away_matches["home_team"]

    team_matches = pd.concat([home_matches, away_matches], ignore_index=True)
    team_matches = team_matches.sort_values("date", ignore_index=True)

    if not team_matches.empty:
        team_matches["points"] = team_matches.apply(
            lambda r: get_points(r["team_goals"], r["opp_goals"]), axis=1
        )
        team_matches["gd"] = team_matches["team_goals"] - team_matches["opp_goals"]
        team_matches["win"] = (team_matches["points"] == 3).astype(int)

    return team_matches


def build_team_pre_match_features(
    df: pd.DataFrame,
    team_name: str,
    match_date: pd.Timestamp
) -> dict:
    team_matches = get_team_match_history(df, team_name)
    team_matches = team_matches[team_matches["date"] < match_date].copy()

    if team_matches.empty:
        return {
            "team_name": team_name,
            "matches_played_pre": 0,
            "elo_before": np.nan,
            "rank_before": np.nan,
            "form_points_avg_5": np.nan,
            "form_points_sum_5": np.nan,
            "gd_avg_5": np.nan,
            "gd_sum_5": np.nan,
            "gf_avg_5": np.nan,
            "ga_avg_5": np.nan,
            "win_rate_5": np.nan,
            "last_match_gd": np.nan,
            "last_match_points": np.nan,
            "days_since_last": np.nan,
        }

    last_match = team_matches.iloc[-1]
    recent = team_matches.tail(FORM_WINDOW)

    days_since_last = np.nan
    if pd.notna(last_match["date"]):
        days_since_last = float((match_date - pd.Timestamp(last_match["date"])).days)

    return {
        "team_name": team_name,
        "matches_played_pre": int(len(team_matches)),
        "elo_before": float(last_match["team_elo_after"]) if pd.notna(last_match["team_elo_after"]) else np.nan,
        "rank_before": float(last_match["team_rank_after"]) if pd.notna(last_match["team_rank_after"]) else np.nan,
        "form_points_avg_5": safe_mean(recent["points"].tolist()),
        "form_points_sum_5": safe_sum(recent["points"].tolist()),
        "gd_avg_5": safe_mean(recent["gd"].tolist()),
        "gd_sum_5": safe_sum(recent["gd"].tolist()),
        "gf_avg_5": safe_mean(recent["team_goals"].tolist()),
        "ga_avg_5": safe_mean(recent["opp_goals"].tolist()),
        "win_rate_5": safe_mean(recent["win"].tolist()),
        "last_match_gd": safe_last(recent["gd"].tolist()),
        "last_match_points": safe_last(recent["points"].tolist()),
        "days_since_last": days_since_last,
    }


def round_numeric_columns(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(decimals)
    return df


def load_main_data() -> pd.DataFrame:
    df = pd.read_csv(MAIN_DATA_FILE)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    numeric_cols = [
        "home_score",
        "away_score",
        "home_elo",
        "away_elo",
        "home_rank",
        "away_rank",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()

    return df


# -------------------------------------------------------------------
# Main vector builder for one match
# -------------------------------------------------------------------
def build_upcoming_match_feature_vector(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    match_date,
    tournament: str,
    neutral: int
) -> pd.DataFrame:
    match_date = pd.to_datetime(match_date)

    home_team = str(home_team).strip()
    away_team = str(away_team).strip()
    tournament = str(tournament).strip()
    neutral = int(neutral)

    out_file = OUT_DIR_INDIVIDUAL / f"02_{clean_filename(home_team)}_vs_{clean_filename(away_team)}_{match_date.strftime('%Y-%m-%d')}_vector.csv"

    home_features = build_team_pre_match_features(df, home_team, match_date)
    away_features = build_team_pre_match_features(df, away_team, match_date)

    feature_vector = {
        "date": match_date.strftime("%Y-%m-%d"),
        "home_team": home_team,
        "away_team": away_team,
        "tournament": tournament,
        "neutral": neutral,

        "home_elo_before": home_features["elo_before"],
        "away_elo_before": away_features["elo_before"],
        "home_rank_before": home_features["rank_before"],
        "away_rank_before": away_features["rank_before"],

        "elo_diff_before": (
            home_features["elo_before"] - away_features["elo_before"]
            if pd.notna(home_features["elo_before"]) and pd.notna(away_features["elo_before"])
            else np.nan
        ),
        "rank_diff_before": (
            away_features["rank_before"] - home_features["rank_before"]
            if pd.notna(home_features["rank_before"]) and pd.notna(away_features["rank_before"])
            else np.nan
        ),

        "tournament_weight": get_tournament_weight(tournament),

        "home_matches_played_pre": home_features["matches_played_pre"],
        "away_matches_played_pre": away_features["matches_played_pre"],

        "home_form_points_avg_5": home_features["form_points_avg_5"],
        "away_form_points_avg_5": away_features["form_points_avg_5"],

        "home_form_points_sum_5": home_features["form_points_sum_5"],
        "away_form_points_sum_5": away_features["form_points_sum_5"],

        "home_gd_avg_5": home_features["gd_avg_5"],
        "away_gd_avg_5": away_features["gd_avg_5"],

        "home_gd_sum_5": home_features["gd_sum_5"],
        "away_gd_sum_5": away_features["gd_sum_5"],

        "home_gf_avg_5": home_features["gf_avg_5"],
        "away_gf_avg_5": away_features["gf_avg_5"],

        "home_ga_avg_5": home_features["ga_avg_5"],
        "away_ga_avg_5": away_features["ga_avg_5"],

        "home_win_rate_5": home_features["win_rate_5"],
        "away_win_rate_5": away_features["win_rate_5"],

        "home_last_match_gd": home_features["last_match_gd"],
        "away_last_match_gd": away_features["last_match_gd"],

        "home_last_match_points": home_features["last_match_points"],
        "away_last_match_points": away_features["last_match_points"],

        "home_days_since_last": home_features["days_since_last"],
        "away_days_since_last": away_features["days_since_last"],

        "form_points_avg_diff_5": (
            home_features["form_points_avg_5"] - away_features["form_points_avg_5"]
            if pd.notna(home_features["form_points_avg_5"]) and pd.notna(away_features["form_points_avg_5"])
            else np.nan
        ),
        "form_points_sum_diff_5": (
            home_features["form_points_sum_5"] - away_features["form_points_sum_5"]
            if pd.notna(home_features["form_points_sum_5"]) and pd.notna(away_features["form_points_sum_5"])
            else np.nan
        ),
        "gd_avg_diff_5": (
            home_features["gd_avg_5"] - away_features["gd_avg_5"]
            if pd.notna(home_features["gd_avg_5"]) and pd.notna(away_features["gd_avg_5"])
            else np.nan
        ),
        "gd_sum_diff_5": (
            home_features["gd_sum_5"] - away_features["gd_sum_5"]
            if pd.notna(home_features["gd_sum_5"]) and pd.notna(away_features["gd_sum_5"])
            else np.nan
        ),
        "gf_avg_diff_5": (
            home_features["gf_avg_5"] - away_features["gf_avg_5"]
            if pd.notna(home_features["gf_avg_5"]) and pd.notna(away_features["gf_avg_5"])
            else np.nan
        ),
        "ga_avg_diff_5": (
            home_features["ga_avg_5"] - away_features["ga_avg_5"]
            if pd.notna(home_features["ga_avg_5"]) and pd.notna(away_features["ga_avg_5"])
            else np.nan
        ),
        "win_rate_diff_5": (
            home_features["win_rate_5"] - away_features["win_rate_5"]
            if pd.notna(home_features["win_rate_5"]) and pd.notna(away_features["win_rate_5"])
            else np.nan
        ),
        "days_rest_diff": (
            home_features["days_since_last"] - away_features["days_since_last"]
            if pd.notna(home_features["days_since_last"]) and pd.notna(away_features["days_since_last"])
            else np.nan
        ),
    }

    feature_df = pd.DataFrame([feature_vector])
    ordered_cols = ["date", "home_team", "away_team", "tournament"] + FEATURE_COLS
    feature_df = feature_df[ordered_cols]
    feature_df = round_numeric_columns(feature_df, decimals=3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(out_file, index=False)

    print(f"✅ Feature vector created: {home_team} vs {away_team}")
    print(f"Saved to: {out_file}")
    print()

    return feature_df


# -------------------------------------------------------------------
# Batch runner
# -------------------------------------------------------------------
def build_feature_vectors_for_fixtures(fixtures: list[dict]) -> pd.DataFrame:
    df = load_main_data()
    all_vectors = []

    for i, fixture in enumerate(fixtures, start=1):
        print(f"Processing match {i}/{len(fixtures)}...")

        feature_df = build_upcoming_match_feature_vector(
            df=df,
            home_team=fixture["home_team"],
            away_team=fixture["away_team"],
            match_date=fixture["match_date"],
            tournament=fixture["tournament"],
            neutral=fixture["neutral"],
        )

        all_vectors.append(feature_df)

    combined_df = pd.concat(all_vectors, ignore_index=True)

    combined_out_file = OUT_DIR / "02_all_upcoming_match_vectors.csv"
    combined_df.to_csv(combined_out_file, index=False)

    print("✅ All feature vectors created")
    print(f"Combined file saved to: {combined_out_file}")

    return combined_df


if __name__ == "__main__":
    build_feature_vectors_for_fixtures(FIXTURES)