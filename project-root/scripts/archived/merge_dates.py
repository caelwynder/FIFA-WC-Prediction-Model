import pandas as pd
from pathlib import Path

# paths
BASE_DIR = Path(__file__).resolve().parent.parent
IN_FILE = BASE_DIR / "data" / "raw" / "eloratings_all_matches_2000_2026.csv"
OUT_FILE = BASE_DIR / "data" / "raw" / "eloratings_all_matches_2000_2026_merged_date.csv"

# load
df = pd.read_csv(IN_FILE)

# create date column (zero-padded month/day)
df["date"] = (
    df["year"].astype(int).astype(str).str.zfill(4) + "-" +
    df["month"].astype(int).astype(str).str.zfill(2) + "-" +
    df["day"].astype(int).astype(str).str.zfill(2)
)

# move date to first column
cols = ["date"] + [c for c in df.columns if c not in ["year", "month", "day", "date"]]
df = df[cols]

# save
df.to_csv(OUT_FILE, index=False)

print("✅ Saved:", OUT_FILE)

pd.to_datetime(df["date"], errors="raise")

print("Columns:", list(df.columns))
print(df.head())
