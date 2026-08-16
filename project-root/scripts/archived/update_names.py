import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # project-root

def apply_former_name_mapping(
    results_csv: str | Path,
    former_names_csv: str | Path,
    out_csv: str | Path,
    team_cols=("home_team", "away_team"),
):
    results_csv = Path(results_csv)
    former_names_csv = Path(former_names_csv)
    out_csv = Path(out_csv)

    df = pd.read_csv(results_csv)
    names = pd.read_csv(former_names_csv)

    # former_names.csv columns: current, former, start_date, end_date
    if not {"current", "former"}.issubset(names.columns):
        raise ValueError("former_names.csv must contain columns: 'former' and 'current'")

    # build mapping: former -> current
    names["former"] = names["former"].astype(str).str.strip()
    names["current"] = names["current"].astype(str).str.strip()
    mapping = dict(zip(names["former"], names["current"]))

    # apply mapping to each team column
    for col in team_cols:
        if col not in df.columns:
            raise ValueError(f"results.csv missing required column: {col}")

        before_unique = df[col].nunique(dropna=True)
        df[col] = df[col].astype(str).str.strip().replace(mapping)
        after_unique = df[col].nunique(dropna=True)

        print(f"{col}: unique teams {before_unique} -> {after_unique}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved mapped file to: {out_csv}")

    # quick report: which former names still appear (if any)
    still_present = sorted(set(mapping.keys()) & set(df[team_cols[0]].unique()) | set(mapping.keys()) & set(df[team_cols[1]].unique()))
    if still_present:
        print("\n⚠️ Some former names still appear (check spelling/casing):")
        print(still_present[:25], "..." if len(still_present) > 25 else "")

    return df

if __name__ == "__main__":
    # change these paths to wherever your files live
    apply_former_name_mapping(
        results_csv=BASE_DIR / "data" / "raw" / "results_short.csv",
        former_names_csv=BASE_DIR / "data" / "raw" / "former_names.csv",
        out_csv=BASE_DIR / "data" / "raw" / "results_names_updated.csv",
    )
