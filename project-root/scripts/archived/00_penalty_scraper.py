# scripts/archived/00_penalty_scraper.py
#
# Downloads the freshest penalty-shootout data and caps it at the pipeline
# cutoff so the penalty model never sees post-cutoff results:
#
#   1. shootouts.csv  (martj42/international_results) — every international
#      shootout with its winner            -> data/raw/00_shootouts_final.csv
#   2. goalscorers.csv (same repo) — in-game goals with a penalty flag,
#      used for team penalty-scoring rates -> data/raw/00_penalty_goals_final.csv
#   3. penalty_kicks.csv (jfjelstul/worldcup) — kick-by-kick World Cup
#      shootout records with converted flag -> data/raw/00_wc_penalty_kicks.csv
#
# Team names are normalized to match the merged dataset (former names via
# former_names.csv, plus the South Korea fix).

import requests
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project-root
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Keep in sync with 00_run_preprocessing_pipeline.CUTOFF_DATE. The 2026
# knockout shootouts (June 29+) must not leak into training.
CUTOFF_DATE = "2026-06-27"

SOURCES = {
    "shootouts": "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv",
    "goalscorers": "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv",
    "former_names": "https://raw.githubusercontent.com/martj42/international_results/master/former_names.csv",
    "wc_penalty_kicks": "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/penalty_kicks.csv",
}

OUT_SHOOTOUTS = RAW_DIR / "00_shootouts_final.csv"
OUT_PEN_GOALS = RAW_DIR / "00_penalty_goals_final.csv"
OUT_WC_KICKS = RAW_DIR / "00_wc_penalty_kicks.csv"

NAME_FIXES = {"South Korea": "Korea Republic", "North Korea": "Korea DPR"}


def download_csv(url: str) -> pd.DataFrame:
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    resp.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(resp.text))


def build_name_map(former: pd.DataFrame) -> dict:
    mapping = dict(zip(former["former"].astype(str).str.strip(),
                       former["current"].astype(str).str.strip()))
    mapping.update(NAME_FIXES)
    return mapping


def normalize_teams(df: pd.DataFrame, cols: list[str], mapping: dict) -> pd.DataFrame:
    for col in cols:
        df[col] = df[col].astype(str).str.strip().replace(mapping)
    return df


def main():
    former = download_csv(SOURCES["former_names"])
    name_map = build_name_map(former)

    # --- 1. shootout outcomes ---
    shootouts = download_csv(SOURCES["shootouts"])
    shootouts = normalize_teams(shootouts, ["home_team", "away_team", "winner"], name_map)
    before = len(shootouts)
    shootouts = shootouts[shootouts["date"] <= CUTOFF_DATE]
    shootouts.to_csv(OUT_SHOOTOUTS, index=False)
    print(f"✅ shootouts: {len(shootouts)} rows (removed {before - len(shootouts)} after {CUTOFF_DATE}) -> {OUT_SHOOTOUTS}")

    # --- 2. in-game penalty goals ---
    goals = download_csv(SOURCES["goalscorers"])
    goals = normalize_teams(goals, ["home_team", "away_team", "team"], name_map)
    goals = goals[goals["date"] <= CUTOFF_DATE]
    pen_goals = goals[goals["penalty"] == True][["date", "home_team", "away_team", "team", "minute"]]
    pen_goals.to_csv(OUT_PEN_GOALS, index=False)
    print(f"✅ penalty goals: {len(pen_goals)} rows -> {OUT_PEN_GOALS}")

    # --- 3. kick-by-kick World Cup shootout data ---
    kicks = download_csv(SOURCES["wc_penalty_kicks"])
    kicks = normalize_teams(kicks, ["team_name"], name_map)
    kicks = kicks[kicks["match_date"] <= CUTOFF_DATE]
    keep = ["match_date", "stage_name", "team_name", "converted"]
    kicks[keep].to_csv(OUT_WC_KICKS, index=False)
    print(f"✅ WC shootout kicks: {len(kicks)} rows -> {OUT_WC_KICKS}")


if __name__ == "__main__":
    main()
