import pandas as pd
from pathlib import Path

MANUAL_OVERRIDES = {
    "AB": "Aruba",
    "AN": "Curaçao",
    "CC": "Cocos (Keeling) Islands",
    "CN": "China",
    "CW": "Curaçao",
    "CX": "Christmas Island",
    "EI": "Northern Ireland",
    "EN": "England",
    "EU": "Europe XI",
    "HG": "Hong Kong",
    "JS": "Jersey",
    "KD": "Cambodia",
    "KO": "Kosovo",
    "KT": "East Timor",
    "MO": "Macau",
    "NM": "Northern Mariana Islands",
    "NS": "Northern Cyprus",
    "RM": "Romania",
    "SQ": "Scotland",
    "SW": "Sweden",
    "SX": "Sint Maarten",
    "TE": "East Timor",
    "TI": "Tahiti",
    "TR": "Turkey",
    "VI": "United States Virgin Islands",
    "WA": "Wales",
    "YU": "Serbia",
    "ZN": "Zanzibar",
}


def build_alpha2_to_fullname(results_df: pd.DataFrame, team_cols=("home_team", "away_team")) -> dict:
    """
    Build mapping {ISO2: full_name} using results_names_updated.csv as the source of preferred names.
    Uses pycountry to convert full_name -> alpha_2, then inverts (alpha_2 -> most common full_name).
    """
    try:
        import pycountry
    except ImportError:
        raise ImportError(
            "Missing dependency: pycountry\n"
            "Install it with: pip install pycountry"
        )

    # Collect all full team names from results file
    names = pd.concat([results_df[team_cols[0]], results_df[team_cols[1]]], ignore_index=True)
    names = names.dropna().astype(str).str.strip()
    names = names[names != ""]
    # Prefer most frequent name if multiple variants map to same ISO2
    freq = names.value_counts()

    # Manual aliases for football naming quirks (add more if needed)
    alias_to_pycountry = {
        "USA": "United States",
        "United States of America": "United States",
        "IR Iran": "Iran, Islamic Republic of",
        "Iran": "Iran, Islamic Republic of",
        "Korea Republic": "Korea, Republic of",
        "Korea DPR": "Korea, Democratic People's Republic of",
        "Russia": "Russian Federation",
        "Syria": "Syrian Arab Republic",
        "Cape Verde": "Cabo Verde",
        "Curaçao": "Curacao",
        "Chinese Taipei": "Taiwan",
        "Brunei": "Brunei Darussalam",
        "Venezuela": "Venezuela, Bolivarian Republic of",
        "Bolivia": "Bolivia, Plurinational State of",
        "Tanzania": "Tanzania, United Republic of",
        "Moldova": "Moldova, Republic of",
        "Laos": "Lao People's Democratic Republic",
        "Vietnam": "Viet Nam",
        "Palestine": "Palestine, State of",
        "North Macedonia": "North Macedonia",
        "Czech Republic": "Czechia",
        "Ivory Coast": "Côte d'Ivoire",
        "DR Congo": "Congo, The Democratic Republic of the",
        "Republic of Ireland": "Ireland",
        "Saint Kitts and Nevis": "Saint Kitts and Nevis",
        "Saint Vincent / Grenadines": "Saint Vincent and the Grenadines",
        "Swaziland": "Eswatini",
    }

    MANUAL_OVERRIDES = {
    "AB": "Aruba",
    "AN": "Curaçao",
    "CC": "Cocos (Keeling) Islands",
    "CN": "China",
    "CW": "Curaçao",
    "CX": "Christmas Island",
    "EI": "Northern Ireland",
    "EN": "England",
    "EU": "Europe XI",
    "HG": "Hong Kong",
    "JS": "Jersey",
    "KD": "Cambodia",
    "KO": "Kosovo",
    "KT": "East Timor",
    "MO": "Macau",
    "NM": "Northern Mariana Islands",
    "NS": "Northern Cyprus",
    "RM": "Romania",
    "SQ": "Scotland",
    "SW": "Sweden",
    "SX": "Sint Maarten",
    "TE": "East Timor",
    "TI": "Tahiti",
    "TR": "Turkey",
    "VI": "United States Virgin Islands",
    "WA": "Wales",
    "YU": "Serbia",
    "ZN": "Zanzibar",
}
    # Add manual overrides to alias mapping

    def name_to_alpha2(n: str):
        n2 = alias_to_pycountry.get(n, n)

        # Try direct lookup first
        c = pycountry.countries.get(name=n2)
        if c and hasattr(c, "alpha_2"):
            return c.alpha_2

        # Try "common_name"/"official_name" fuzzy search
        try:
            matches = pycountry.countries.search_fuzzy(n2)
            if matches and hasattr(matches[0], "alpha_2"):
                return matches[0].alpha_2
        except Exception:
            pass

        return None

    # Convert full names -> alpha2, keeping the most frequent full name per alpha2
    alpha2_to_name = {}
    for full_name in freq.index.tolist():
        code = name_to_alpha2(full_name)
        if not code:
            continue
        # Keep the highest frequency name for that code (first one seen is highest due to freq ordering)
        if code not in alpha2_to_name:
            alpha2_to_name[code] = full_name

    return alpha2_to_name


def convert_eloratings_team_codes(
    eloratings_csv: str | Path,
    results_names_updated_csv: str | Path,
    out_csv: str | Path,
):
    eloratings_csv = Path(eloratings_csv)
    results_names_updated_csv = Path(results_names_updated_csv)
    out_csv = Path(out_csv)

    elo = pd.read_csv(eloratings_csv)
    res = pd.read_csv(results_names_updated_csv)

    # Detect team columns in eloratings file (supports both naming styles)
    possible_home = ["home_team", "hometeam", "HomeTeam", "home"]
    possible_away = ["away_team", "awayteam", "AwayTeam", "away"]

    home_col = next((c for c in possible_home if c in elo.columns), None)
    away_col = next((c for c in possible_away if c in elo.columns), None)

    if not home_col or not away_col:
        raise ValueError(
            f"Could not find home/away team columns in eloratings.csv.\n"
            f"Columns found: {list(elo.columns)}"
        )

    # Build mapping ISO2 -> preferred full name (as used in results_names_updated)
    mapping = build_alpha2_to_fullname(res, team_cols=("home_team", "away_team"))

    # Normalize eloratings codes (strip, uppercase)
    elo[home_col] = elo[home_col].astype(str).str.strip().str.upper()
    elo[away_col] = elo[away_col].astype(str).str.strip().str.upper()

    # Apply mapping
    elo[home_col] = elo[home_col].map(mapping).fillna(elo[home_col]).str.strip().replace(mapping)
    elo[away_col] = elo[away_col].map(mapping).fillna(elo[away_col]).str.strip().replace(mapping)
    
    # Fallback: if still a 2-letter code, try manual overrides
    for col in [home_col, away_col]:
        elo[col] = elo[col].astype(str).str.strip()
        elo[col] = elo[col].apply(lambda x: MANUAL_OVERRIDES.get(x.upper(), x) if isinstance(x, str) else x)


    # Report unmapped 2-letter codes still present
    def is_code(x: str) -> bool:
        x = str(x).strip()
        return len(x) == 2 and x.isalpha() and x.upper() == x

    unmapped_home = sorted({x for x in elo[home_col].unique() if is_code(x)})
    unmapped_away = sorted({x for x in elo[away_col].unique() if is_code(x)})
    unmapped = sorted(set(unmapped_home) | set(unmapped_away))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    elo.to_csv(out_csv, index=False)

    print(f"✅ Saved updated eloratings to: {out_csv}")
    print(f"Home column: {home_col} | Away column: {away_col}")
    print(f"ISO2 mapping size: {len(mapping)}")
    print(f"Unmapped 2-letter codes remaining: {len(unmapped)}")
    if unmapped:
        print("Sample unmapped codes:", unmapped[:30])

    return elo, unmapped


if __name__ == "__main__":
    # Adjust these paths to where the files live in your project
    BASE_DIR = Path(__file__).resolve().parent.parent  # project-root
    in_elo = BASE_DIR / "data" / "raw" / "eloratings.csv"
    in_res = BASE_DIR / "data" / "raw" / "results_names_updated.csv"
    out_elo = BASE_DIR / "data" / "raw" / "eloratings_full_names.csv"

    convert_eloratings_team_codes(in_elo, in_res, out_elo)
