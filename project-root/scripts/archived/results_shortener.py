import pandas as pd
from pathlib import Path

# -----------------------
# CONFIG
# -----------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # project-root
IN_FILE = BASE_DIR / "data" / "raw" / "results.csv"
OUT_FILE = BASE_DIR / "data" / "raw" / "results_short.csv"
MIN_DATE = "2000-01-04"

# -----------------------
# LOAD
# -----------------------
df = pd.read_csv(IN_FILE)

# -----------------------
# ENSURE DATE COLUMN
# -----------------------
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
else:
    # try year/month/day fallback
    required = ["year", "month", "day"]
    if not all(c in df.columns for c in required):
        raise ValueError(
            "results.csv must contain either a 'date' column or year/month/day columns."
        )

    df["date"] = pd.to_datetime(
        {
            "year": pd.to_numeric(df["year"], errors="coerce"),
            "month": pd.to_numeric(df["month"], errors="coerce"),
            "day": pd.to_numeric(df["day"], errors="coerce"),
        },
        errors="coerce",
    )

# drop invalid dates
df = df.dropna(subset=["date"]).copy()

# -----------------------
# FILTER DATE RANGE
# -----------------------
cutoff = pd.to_datetime(MIN_DATE)
df = df[df["date"] >= cutoff].copy()

# normalize format
df["date"] = df["date"].dt.strftime("%Y-%m-%d")

# -----------------------
# SAVE
# -----------------------
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_FILE, index=False)

print(f"✅ Saved trimmed file: {OUT_FILE}")
print("Rows:", len(df))
print("Min date:", df["date"].min())
print("Max date:", df["date"].max())
print(df.head())
