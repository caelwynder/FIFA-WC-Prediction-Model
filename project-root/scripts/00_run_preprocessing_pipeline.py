# scripts/00_run_preprocessing_pipeline.py

import sys
import subprocess
import pandas as pd
from pathlib import Path



# Paths

BASE_DIR = Path(__file__).resolve().parent.parent  # script dir
DATA_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"
SCRIPT_DIR = BASE_DIR / "scripts" / "archived"

OUT_FILE = DATA_DIR / "00_merged_dataset.csv"
RAW_FILE = RAW_DIR / "00_merged_dataset_raw.csv"

# Hard cap: no matches after this date enter the dataset. The model is
# trained on everything up to the end of the 2026 World Cup group stage
# and used to predict every match after it.
CUTOFF_DATE = "2026-06-27"


# Helpers

def run_py(script_path: Path):
    print("\n" + "=" * 70)
    print(f"RUN: {script_path.name}")
    print("=" * 70)
    subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR), check=True)


def find_script(name: str) -> Path:
    """
    Finds scripts in SCRIPT_DIR.
    Supports either:
      - exact filename (with or without .py)
      - tries .py if missing
    """
    p = SCRIPT_DIR / name
    if p.exists():
        return p
    p2 = SCRIPT_DIR / f"{name}.py"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Could not find {name} or {name}.py in {SCRIPT_DIR}")


# Pipeline

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Results workflow (download -> trim -> update names)
    results_script = find_script("00_results_scraper")
    run_py(results_script)

    # Step 2: Extra Data Scraping workflow (download -> TBD)
    github_script = find_script("00_github_scraper")
    run_py(github_script)

    # Step 2b: Penalty shootout data (shootouts, penalty goals, WC kick-by-kick)
    penalty_script = find_script("00_penalty_scraper")
    run_py(penalty_script)

    # Step 3: EloRatings workflow (download -> merge -> merge dates -> namechange)
    eloratings_script = find_script("00_eloratings_scraper")
    run_py(eloratings_script)

    # Step 4: Merge datasets (outputs raw merged)
    merge_script = find_script("00_merge_datasets")
    run_py(merge_script)

    # Final: move/copy raw merged into processed location (standardize output)
    if RAW_FILE.exists():
        RAW_FILE.replace(OUT_FILE)
        print(f"\nFinal merged dataset saved to: {OUT_FILE}")
    elif OUT_FILE.exists():
        print(f"\nFinal merged dataset already exists: {OUT_FILE}")
    else:
        print("\nMerge step finished but expected output file not found.")
        print(f"Expected RAW_FILE: {RAW_FILE}")
        print(f"Expected OUT_FILE: {OUT_FILE}")

def cleanup_outputs(raw_dir: Path, processed_dir: Path):
    KEEP_FILES = {
        "00_merged_dataset.csv",
        "00_eloratings_final.csv",
        "00_results_final.csv",
        "00_merged_dataset_unmatched_eloratings.csv",
        "00_merged_dataset_unmatched_results.csv",
        "former_names.csv",
        "goalscorers.csv",
        "shootouts.csv",
        "00_shootouts_final.csv",
        "00_penalty_goals_final.csv",
        "00_wc_penalty_kicks.csv"
    }

    print("\n" + "=" * 70)
    print("CLEANUP: removing intermediate files")
    print("=" * 70)

    for directory in [raw_dir, processed_dir]:
        if not directory.exists():
            continue

        for file in directory.iterdir():
            if file.is_file() and file.name not in KEEP_FILES:
                file.unlink()
                print(f"Removed: {file}")


if __name__ == "__main__":

    main()

    if RAW_FILE.exists():   
        RAW_FILE.replace(OUT_FILE)
        print(f"\nFinal merged dataset saved to: {OUT_FILE}")
    elif OUT_FILE.exists():
        print(f"\nFinal merged dataset already exists: {OUT_FILE}")
    else:
        print("\nMerge step finished but expected output file not found.")

    cleanup_outputs(RAW_DIR, DATA_DIR)

    column_to_remove = ["importance", "tournemnt_location"]

    df = pd.read_csv(OUT_FILE)

    for column in range(len(column_to_remove)):
        if column_to_remove[column] in df.columns:
            df = df.drop(columns=[column_to_remove[column]])
            df.to_csv(OUT_FILE, index=False)
            print(f"Column '{column_to_remove[column]}' removed successfully.")
        else:
            print("Column not found in the CSV file.")

    # Enforce the cutoff: drop anything played after CUTOFF_DATE
    dates = pd.to_datetime(df["date"], errors="coerce")
    after_cutoff = dates > pd.Timestamp(CUTOFF_DATE)
    if after_cutoff.any():
        df = df[~after_cutoff]
        df.to_csv(OUT_FILE, index=False)
    print(f"Cutoff {CUTOFF_DATE}: removed {int(after_cutoff.sum())} matches after this date.")
    print(f"Dataset date range: {dates.min().date()} to {dates[~after_cutoff].max().date()}")


