import pandas as pd
from pathlib import Path

# paths
BASE_DIR = Path(__file__).resolve().parent.parent
TSV_DIR = BASE_DIR / "data" / "tsv"
OUT_FILE = BASE_DIR / "data" / "raw" / "eloratings_all_matches_2000_2026.csv"


COLUMNS = [
    "year", "month", "day",
    "home_team", "away_team",
    "home_goals", "away_goals",
    "importance", "tournemnt_location",
    "home_rating_change",
    "home_elo", "away_elo",
    "home_rank_change", "away_rank_change",
    "home_rank", "away_rank"
]

dfs = []
for tsv in sorted(TSV_DIR.glob("*.tsv")):
    df = pd.read_csv(tsv, sep="\t", header=None)
    df.columns = COLUMNS

    insert_at = df.columns.get_loc("home_rating_change") + 1
    df.insert(insert_at, "away_rating_change", -df["home_rating_change"])

    dfs.append(df)
    print(f"Loaded {tsv.name}: {len(df)} rows")

final_df = pd.concat(dfs, ignore_index=True)

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
final_df.to_csv(OUT_FILE, index=False)

print("✅ Saved:", OUT_FILE)
print("Columns:", list(final_df.columns))
print("Rows:", len(final_df))