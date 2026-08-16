import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# CONFIG: update these if your column names differ
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # script directory
DATA_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"

ELO_FILE = RAW_DIR / "00_eloratings_final.csv"
RES_FILE =  RAW_DIR / "00_results_final.csv"
OUT_FILE = DATA_DIR / "00_merged_dataset.csv"
RAW_FILE = RAW_DIR / "00_merged_dataset_raw.csv"

# join key columns (must exist in BOTH files)
# if your results file uses home_score/away_score, the code handles it below
KEY_DATE = "date"
KEY_HOME = "home_team"
KEY_AWAY = "away_team"

ELO_HG = "home_goals"
ELO_AG = "away_goals"

RES_HG_OPTIONS = ["home_goals", "home_score"]
RES_AG_OPTIONS = ["away_goals", "away_score"]


def _pick_col(df, options):
    for c in options:
        if c in df.columns:
            return c
    raise ValueError(f"None of these columns exist: {options}. Found: {list(df.columns)}")


def combine_and_report(elo_path: str | Path, res_path: str | Path, out_path: str | Path, raw_path: str | Path):
    elo_path = Path(elo_path)
    res_path = Path(res_path)
    out_path = Path(out_path)
    raw_path = Path(raw_path)

    elo = pd.read_csv(elo_path)
    res = pd.read_csv(res_path)

    # pick results score cols (home_score/away_score vs home_goals/away_goals)
    res_hg = _pick_col(res, RES_HG_OPTIONS)
    res_ag = _pick_col(res, RES_AG_OPTIONS)

    # basic checks
    for c in [KEY_DATE, KEY_HOME, KEY_AWAY, ELO_HG, ELO_AG]:
        if c not in elo.columns:
            raise ValueError(f"Elo file missing required column '{c}'. Columns: {list(elo.columns)}")
    for c in [KEY_DATE, KEY_HOME, KEY_AWAY, res_hg, res_ag]:
        if c not in res.columns:
            raise ValueError(f"Results file missing required column '{c}'. Columns: {list(res.columns)}")

    # normalize types + whitespace
    for df in (elo, res):
        df[KEY_DATE] = df[KEY_DATE].astype(str).str.strip()
        df[KEY_HOME] = df[KEY_HOME].astype(str).str.strip()
        df[KEY_AWAY] = df[KEY_AWAY].astype(str).str.strip()
    
    # Team name normalization - handle common variations
    team_mappings = {
        "China PR": "China",
        "China Pr": "China",
        "South Korea": "Korea Republic",
        "North Korea": "Korea DPR",
        "South Sudan": "Sudan",
        "Saint Kitts and Nevis": "Saint Kitts and Nevis",
        # Add more mappings as needed - check your unmatched data for variations
    }
    
    for df in (elo, res):
        df[KEY_HOME] = df[KEY_HOME].replace(team_mappings)
        df[KEY_AWAY] = df[KEY_AWAY].replace(team_mappings)

    elo[ELO_HG] = pd.to_numeric(elo[ELO_HG], errors="coerce")
    elo[ELO_AG] = pd.to_numeric(elo[ELO_AG], errors="coerce")
    res[res_hg] = pd.to_numeric(res[res_hg], errors="coerce")
    res[res_ag] = pd.to_numeric(res[res_ag], errors="coerce")

    # Rename elo goals to match results if they differ
    if res_hg != ELO_HG and ELO_HG in elo.columns:
        elo = elo.rename(columns={ELO_HG: res_hg})
    if res_ag != ELO_AG and ELO_AG in elo.columns:
        elo = elo.rename(columns={ELO_AG: res_ag})

    # build join keys (direct + swapped home/away, since datasets may disagree on orientation)
    elo_direct = elo.copy()
    elo_direct["_k_date"] = elo_direct[KEY_DATE]
    elo_direct["_k_home"] = elo_direct[KEY_HOME]
    elo_direct["_k_away"] = elo_direct[KEY_AWAY]
    elo_direct["_k_hg"] = elo_direct[res_hg]
    elo_direct["_k_ag"] = elo_direct[res_ag]
    elo_direct["_swapped"] = False

    # For the swapped orientation, flip EVERY home_*/away_* column (teams,
    # goals, elo, rank, rating/rank changes) — not just the join keys —
    # otherwise a swapped match attaches the home team's elo/rank data to
    # the away team and vice versa.
    swap_map = {}
    for col in elo.columns:
        if col.startswith("home_"):
            swap_map[col] = "away_" + col[len("home_"):]
        elif col.startswith("away_"):
            swap_map[col] = "home_" + col[len("away_"):]

    elo_swapped = elo.rename(columns=swap_map)
    elo_swapped["_k_date"] = elo_swapped[KEY_DATE]
    elo_swapped["_k_home"] = elo_swapped[KEY_HOME]
    elo_swapped["_k_away"] = elo_swapped[KEY_AWAY]
    elo_swapped["_k_hg"] = elo_swapped[res_hg]
    elo_swapped["_k_ag"] = elo_swapped[res_ag]
    elo_swapped["_swapped"] = True

    elo_keys = pd.concat([elo_direct, elo_swapped], ignore_index=True)

    res_keys = res.copy()
    res_keys["_k_date"] = res_keys[KEY_DATE]
    res_keys["_k_home"] = res_keys[KEY_HOME]
    res_keys["_k_away"] = res_keys[KEY_AWAY]
    res_keys["_k_hg"] = res_keys[res_hg]
    res_keys["_k_ag"] = res_keys[res_ag]

    join_cols = ["_k_date", "_k_home", "_k_away", "_k_hg", "_k_ag"]

    # if duplicates exist, keep the first match by adding row ids
    elo_keys["_elo_rowid"] = range(len(elo_keys))
    res_keys["_res_rowid"] = range(len(res_keys))

    # MATCH
    merged = res_keys.merge(
        elo_keys,
        on=join_cols,
        how="inner",
        suffixes=("", "_from_elo"),
    )

    # Deduplicate matches: one results row should match at most one elo row
    # keep first match for each results rowid (prioritize non-swapped matches)
    merged = merged.sort_values(["_res_rowid", "_swapped", "_elo_rowid"]).drop_duplicates(subset=["_res_rowid"], keep="first")

    # Identify unmatched rows in each file
    matched_res_ids = set(merged["_res_rowid"].unique())
    matched_elo_ids = set(merged["_elo_rowid"].unique())

    unmatched_res = res_keys[~res_keys["_res_rowid"].isin(matched_res_ids)].copy()
    unmatched_elo = elo_keys[~elo_keys["_elo_rowid"].isin(matched_elo_ids)].copy()

    # Clean final output: drop helper cols and duplicate columns
    helper_cols = [c for c in merged.columns if c.startswith("_k_")] + ["_elo_rowid", "_res_rowid", "_swapped"]
    final = merged.drop(columns=helper_cols, errors="ignore")
    
    # Remove columns that came from elo that are duplicates (they have _from_elo suffix)
    cols_to_drop = [c for c in final.columns if c.endswith("_from_elo")]
    final = final.drop(columns=cols_to_drop, errors="ignore")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_path, index=False)

    print(f"✅ Saved merged matches -> {out_path}")
    print(f"Matched rows: {len(final)}")
    print(f"Unmatched rows in results: {len(unmatched_res)}")
    print(f"Unmatched rows in eloratings: {len(unmatched_elo)}")
    print(f"\nFinal columns ({len(final.columns)}): {list(final.columns)}")

    # Save unmatched for inspection
    unmatched_res_path = raw_path.with_name(out_path.stem + "_unmatched_results.csv")
    unmatched_elo_path = raw_path.with_name(out_path.stem + "_unmatched_eloratings.csv")
    unmatched_res.drop(columns=[c for c in unmatched_res.columns if c.startswith("_k_")] + ["_res_rowid"], errors="ignore") \
                .to_csv(unmatched_res_path, index=False)
    unmatched_elo.drop(columns=[c for c in unmatched_elo.columns if c.startswith("_k_")] + ["_elo_rowid"], errors="ignore") \
                .to_csv(unmatched_elo_path, index=False)

    print(f"📋 Unmatched results saved -> {unmatched_res_path}")
    print(f"📋 Unmatched eloratings saved -> {unmatched_elo_path}")

    return final, unmatched_res, unmatched_elo


if __name__ == "__main__":
    combine_and_report(ELO_FILE, RES_FILE, OUT_FILE, RAW_FILE)