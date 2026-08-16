import requests
import pandas as pd
from pathlib import Path

# -----------------------
# CONFIG
# -----------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project-root
RAW_DIR = BASE_DIR / "data" / "raw"

GITHUB_OWNER = "martj42"
GITHUB_REPO = "international_results"
GITHUB_BRANCH = "master"  # change to "main" if needed

RESULTS_CSV = RAW_DIR / "results.csv"
RESULTS_SHORT_CSV = RAW_DIR / "results_short.csv"
FORMER_NAMES_CSV = RAW_DIR / "former_names.csv"
RESULTS_UPDATED_NAMES_CSV = RAW_DIR / "00_results_final.csv"

MIN_DATE = "2000-01-04"
TEAM_COLS = ("home_team", "away_team")


# -----------------------
# 1) PULL RESULTS
# -----------------------
def download_results_csv(out_path: str | Path, branch: str = GITHUB_BRANCH) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{branch}/results.csv"
    ## urls = ["results.csv", "shootouts.csv", "goalscorers.csv", "former_names.csv"]
    headers = {"User-Agent": "Mozilla/5.0"}
    
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()

    out_path.write_bytes(resp.content)
    print(f"✅ Downloaded results.csv -> {out_path}")
    
    return out_path


# -----------------------
# 2) SHORTEN RESULTS
# -----------------------
def shorten_results_by_date(
    in_file: str | Path,
    out_file: str | Path,
    min_date: str = MIN_DATE,
) -> Path:
    in_file = Path(in_file)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_file)

    # Ensure 'date' column exists
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        required = ["year", "month", "day"]
        if not all(c in df.columns for c in required):
            raise ValueError("results.csv must contain either a 'date' column or year/month/day columns.")
        df["date"] = pd.to_datetime(
            {
                "year": pd.to_numeric(df["year"], errors="coerce"),
                "month": pd.to_numeric(df["month"], errors="coerce"),
                "day": pd.to_numeric(df["day"], errors="coerce"),
            },
            errors="coerce",
        )

    df = df.dropna(subset=["date"]).copy()

    cutoff = pd.to_datetime(min_date)
    df = df[df["date"] >= cutoff].copy()

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    df.to_csv(out_file, index=False)
    print(f"✅ Saved trimmed file: {out_file}")
    print("Rows:", len(df))
    print("Min date:", df["date"].min(), "Max date:", df["date"].max())
    return out_file


# -----------------------
# 3) UPDATE TEAM NAMES
# -----------------------
def apply_former_name_mapping(
    results_csv: str | Path,
    former_names_csv: str | Path,
    out_csv: str | Path,
    team_cols=TEAM_COLS,
) -> Path:
    results_csv = Path(results_csv)
    former_names_csv = Path(former_names_csv)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)
    names = pd.read_csv(former_names_csv)

    if not {"current", "former"}.issubset(names.columns):
        raise ValueError("former_names.csv must contain columns: 'former' and 'current'")

    names["former"] = names["former"].astype(str).str.strip()
    names["current"] = names["current"].astype(str).str.strip()
    mapping = dict(zip(names["former"], names["current"]))

    for col in team_cols:
        if col not in df.columns:
            raise ValueError(f"results file missing required column: {col}")

        before_unique = df[col].nunique(dropna=True)
        df[col] = df[col].astype(str).str.strip().replace(mapping)
        after_unique = df[col].nunique(dropna=True)
        print(f"{col}: unique teams {before_unique} -> {after_unique}")

    df.to_csv(out_csv, index=False)
    print(f"✅ Saved mapped file: {out_csv}")

    # optional quick report
    former_left = sorted(set(mapping.keys()) & (set(df[team_cols[0]].unique()) | set(df[team_cols[1]].unique())))
    if former_left:
        print("⚠️ Some former names still present (check spelling/casing). Sample:")
        print(former_left[:25], "..." if len(former_left) > 25 else "")

    return out_csv


# -----------------------
# PIPELINE
# -----------------------
def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Pull results.csv
    download_results_csv(RESULTS_CSV)

    # 2) Shorten to 2000-01-04+
    shorten_results_by_date(RESULTS_CSV, RESULTS_SHORT_CSV, MIN_DATE)

    # 3) Update names using former_names.csv
    if FORMER_NAMES_CSV.exists():
        apply_former_name_mapping(RESULTS_SHORT_CSV, FORMER_NAMES_CSV, RESULTS_UPDATED_NAMES_CSV, TEAM_COLS)
    else:
        print(f"⚠️ former_names.csv not found, skipping name mapping: {FORMER_NAMES_CSV}")
        RESULTS_SHORT_CSV.replace(RESULTS_UPDATED_NAMES_CSV)


    print("\n✅ Pipeline complete.")
    print("Final output:", RESULTS_UPDATED_NAMES_CSV)


if __name__ == "__main__":
    print({RAW_DIR})
    print({RESULTS_CSV})
    main()
