import time
import datetime
import requests
import pandas as pd
import sys
from pathlib import Path
from helper_eloratings_scraper import convert_eloratings_team_codes


BASE = "https://www.eloratings.net"
CURRENT_YEAR = datetime.date.today().year
YEARS = range(2000, CURRENT_YEAR + 1)

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project-root
TSV_DIR = BASE_DIR / "data" / "tsv"
RAW_DIR = BASE_DIR / "data" / "raw"

TSV_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

MERGED_OUT = RAW_DIR / f"eloratings_all_matches_2000_{CURRENT_YEAR}.csv"
FINAL_OUT = RAW_DIR / "eloratings.csv"

# NOTE: This matches your merge_tsv.py exactly (16 columns)
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

def download_yearly_tsvs():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/tab-separated-values",
        "Referer": "https://www.eloratings.net/"
    })

    for year in YEARS:
        url = f"{BASE}/{year}_results.tsv"
        out_path = TSV_DIR / f"{year}_results.tsv"

        # A year's TSV is only final if it was downloaded AFTER that year
        # ended; a file fetched mid-year is missing later matches and must
        # be re-downloaded to pick up new results.
        if out_path.exists() and out_path.stat().st_size > 0:
            downloaded_at = datetime.datetime.fromtimestamp(out_path.stat().st_mtime)
            if downloaded_at.year > year:
                print(f"Skipping {year} (already downloaded, year complete)")
                continue
            print(f"Refreshing {year} (file downloaded mid-year, may be stale)")

        print(f"Downloading {year} -> {out_path}")
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        time.sleep(0.3)

    print(f"\n✅ Done downloading TSVs into: {TSV_DIR}")

def merge_tsvs_to_csv():
    dfs = []
    tsv_files = sorted(TSV_DIR.glob("*.tsv"))
    if not tsv_files:
        raise FileNotFoundError(f"No TSV files found in {TSV_DIR}")

    for tsv in tsv_files:
        df = pd.read_csv(tsv, sep="\t", header=None)

        # if there are trailing empty columns, drop them
        df = df.dropna(axis=1, how="all")

        if df.shape[1] != len(COLUMNS):
            raise ValueError(
                f"{tsv.name} has {df.shape[1]} columns, expected {len(COLUMNS)}.\n"
                f"Fix your COLUMNS list or inspect the TSV."
            )

        df.columns = COLUMNS

        df["day"] = pd.to_numeric(df["day"], errors="coerce")
        df.loc[df["day"] == 0, "day"] = 1

        # ensure numeric so negation works
        df["home_rating_change"] = pd.to_numeric(df["home_rating_change"], errors="coerce")

        # insert away_rating_change right beside home_rating_change
        insert_at = df.columns.get_loc("home_rating_change") + 1
        df.insert(insert_at, "away_rating_change", -df["home_rating_change"])

        dfs.append(df)
        print(f"Loaded {tsv.name}: {len(df)} rows")

    final_df = pd.concat(dfs, ignore_index=True)

    MERGED_OUT.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(MERGED_OUT, index=False)

    print(f"\n✅ Saved merged CSV: {MERGED_OUT}")
    print("Rows:", len(final_df))
    print("Columns:", list(final_df.columns))

def merge_year_month_day_into_date():
    df = pd.read_csv(MERGED_OUT)

    # create YYYY-MM-DD date
    df["date"] = (
        df["year"].astype(int).astype(str).str.zfill(4) + "-" +
        df["month"].astype(int).astype(str).str.zfill(2) + "-" +
        df["day"].astype(int).astype(str).str.zfill(2)
    )

    # move date to first column + drop year/month/day
    cols = ["date"] + [c for c in df.columns if c not in ["year", "month", "day", "date"]]
    df = df[cols]

    # validate dates
    pd.to_datetime(df["date"], errors="raise")

    df.to_csv(FINAL_OUT, index=False)

    print(f"\n✅ Saved final CSV with merged date: {FINAL_OUT}")
    print("Columns:", list(df.columns))
    print(df.head())

if __name__ == "__main__":
    download_yearly_tsvs()
    merge_tsvs_to_csv()
    merge_year_month_day_into_date()

    # --- Name conversion step (elorating_namechange.py) ---
    results_final = RAW_DIR / "00_results_final.csv"
    out_full_names = RAW_DIR / "00_eloratings_final.csv"

    print({BASE_DIR})

    if results_final.exists():
        convert_eloratings_team_codes(FINAL_OUT, results_final, out_full_names)
    else:
        print(f"⚠️ Skipping name conversion: missing {results_final}")