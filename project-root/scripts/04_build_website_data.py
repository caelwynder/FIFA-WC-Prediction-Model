# scripts/04_build_website_data.py
#
# Generates website/data.json for the predictions site:
#   - the 16 real Round-of-32 fixtures (from scraped results), ordered as
#     bracket seeds, each with the model's prediction using the true venue
#   - a pairwise prediction table (outcome probabilities + estimated goals)
#     for every ordered pair of the 48 World Cup teams, on neutral ground,
#     so the browser can cascade winners through R16/QF/SF/Final and power
#     the head-to-head simulator
#
# Predictions use the trained XGBoost outcome classifier plus two Poisson
# XGBoost regressors (trained here on the same cutoff dataset) for goals.
# All features are computed from team history up to CUTOFF_DATE only.

import json
import importlib.util
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBRegressor

# -------------------------------------------------------------------
# Paths / constants
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
WEBSITE_DIR = BASE_DIR.parent / "website"

MERGED_FILE = DATA_DIR / "processed" / "00_merged_dataset.csv"
RESULTS_FILE = DATA_DIR / "raw" / "00_results_final.csv"
XGB_MODEL_PATH = MODEL_DIR / "xgb_outcome_model.joblib"
PENALTY_MODEL_PATH = MODEL_DIR / "penalty_model.joblib"
OUT_FILE = WEBSITE_DIR / "data.json"

from features import FEATURE_COLS, build_snapshot, make_feature_row

CUTOFF_DATE = pd.Timestamp("2026-06-27")
GROUP_STAGE_START = pd.Timestamp("2026-06-11")
# Nominal date for hypothetical later-round / head-to-head matchups
NOMINAL_DATE = pd.Timestamp("2026-07-04")

WC_TOURNAMENT_WEIGHT = 1.00  # every simulated match is a World Cup match

# results-file name -> merged-dataset name
NAME_FIXES = {"South Korea": "Korea Republic"}

# Round of 32 fixtures in bracket-seed order: consecutive pairs feed the
# same Round-of-16 match (slots 1+2 -> R16 match 1, 3+4 -> match 2, ...),
# and the lower slot's winner is the R16 home side. Ordered to match the
# official bracket as played: R16 was Paraguay-France, Canada-Morocco,
# Portugal-Spain, USA-Belgium, Brazil-Norway, Mexico-England,
# Argentina-Egypt, Switzerland-Colombia; QFs France-Morocco, Spain-Belgium,
# Norway-England, Argentina-Switzerland; SFs France-Spain,
# England-Argentina. Validated against RESULTS_FILE at runtime and by
# tests/test_bracket_structure.py.
R32_BRACKET_ORDER = [
    ("Germany", "Paraguay"),
    ("France", "Sweden"),
    ("South Africa", "Canada"),
    ("Netherlands", "Morocco"),
    ("Portugal", "Croatia"),
    ("Spain", "Austria"),
    ("United States", "Bosnia and Herzegovina"),
    ("Belgium", "Senegal"),
    ("Brazil", "Japan"),
    ("Ivory Coast", "Norway"),
    ("Mexico", "Ecuador"),
    ("England", "DR Congo"),
    ("Argentina", "Cape Verde"),
    ("Australia", "Egypt"),
    ("Switzerland", "Algeria"),
    ("Colombia", "Ghana"),
]

FLAGS = {
    "Algeria": "🇩🇿", "Argentina": "🇦🇷", "Australia": "🇦🇺", "Austria": "🇦🇹",
    "Belgium": "🇧🇪", "Bosnia and Herzegovina": "🇧🇦", "Brazil": "🇧🇷",
    "Canada": "🇨🇦", "Cape Verde": "🇨🇻", "Colombia": "🇨🇴", "Croatia": "🇭🇷",
    "Curaçao": "🇨🇼", "Czech Republic": "🇨🇿", "DR Congo": "🇨🇩",
    "Ecuador": "🇪🇨", "Egypt": "🇪🇬", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "France": "🇫🇷",
    "Germany": "🇩🇪", "Ghana": "🇬🇭", "Haiti": "🇭🇹", "Iran": "🇮🇷",
    "Iraq": "🇮🇶", "Ivory Coast": "🇨🇮", "Japan": "🇯🇵", "Jordan": "🇯🇴",
    "Korea Republic": "🇰🇷", "Mexico": "🇲🇽", "Morocco": "🇲🇦",
    "Netherlands": "🇳🇱", "New Zealand": "🇳🇿", "Norway": "🇳🇴",
    "Panama": "🇵🇦", "Paraguay": "🇵🇾", "Portugal": "🇵🇹", "Qatar": "🇶🇦",
    "Saudi Arabia": "🇸🇦", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Senegal": "🇸🇳",
    "South Africa": "🇿🇦", "Spain": "🇪🇸", "Sweden": "🇸🇪",
    "Switzerland": "🇨🇭", "Tunisia": "🇹🇳", "Turkey": "🇹🇷",
    "United States": "🇺🇸", "Uruguay": "🇺🇾", "Uzbekistan": "🇺🇿",
}

# -------------------------------------------------------------------
# Team snapshots as of the cutoff
# -------------------------------------------------------------------
def load_merged() -> pd.DataFrame:
    df = pd.read_csv(MERGED_FILE, parse_dates=["date"])
    df = df.dropna(subset=["date"])
    for col in ["home_score", "away_score", "home_elo", "away_elo", "home_rank", "away_rank"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()
    return df.sort_values("date", ignore_index=True)


# build_snapshot and make_feature_row come from features.py (shared with
# the whole pipeline). Every simulated match is a World Cup match, so the
# default tournament_weight=1.0 applies.


# -------------------------------------------------------------------
# Goals regressors (Poisson) trained on the same cutoff dataset
# -------------------------------------------------------------------
def train_goals_models():
    df = pd.read_csv(DATA_DIR / "processed" / "01_model_dataset.csv", parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    X = df[FEATURE_COLS].fillna(-99.0)

    models = {}
    for target in ["home_score", "away_score"]:
        m = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="count:poisson",
            random_state=42,
            n_jobs=-1,
        )
        m.fit(X, df[target])
        models[target] = m
    return models["home_score"], models["away_score"]


# -------------------------------------------------------------------
# Penalty shootout model (tandem: outcome model draw -> shootout winner)
# -------------------------------------------------------------------
def load_penalty_predictor():
    """Returns pen_probs(pairs_of_names) -> {"A|B": P(A wins shootout)}.

    Probabilities are symmetrized across the two orientations so the
    arbitrary home/away labeling of a neutral knockout doesn't matter:
    P(A beats B) = (p(A,B) + 1 - p(B,A)) / 2.
    """
    spec = importlib.util.spec_from_file_location(
        "penalty_train", Path(__file__).parent / "05_train_penalty_model.py"
    )
    pen_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pen_mod)

    payload = joblib.load(PENALTY_MODEL_PATH)
    model = payload["model"]
    feature_cols = payload["feature_cols"]
    profiles = payload["profiles"]

    def pen_probs(name_pairs: list[tuple[str, str]]) -> dict:
        rows = [pen_mod.make_row(profiles[a], profiles[b]) for a, b in name_pairs]
        X = pd.DataFrame(rows)[feature_cols].fillna(-99.0)
        raw = dict(zip(
            (f"{a}|{b}" for a, b in name_pairs),
            model.predict_proba(X)[:, 1],
        ))
        out = {}
        for a, b in name_pairs:
            p_fwd = raw[f"{a}|{b}"]
            p_rev = raw.get(f"{b}|{a}")
            out[f"{a}|{b}"] = round(float((p_fwd + (1 - p_rev)) / 2 if p_rev is not None else p_fwd), 4)
        return out

    return pen_probs


# -------------------------------------------------------------------
# Round-of-32 fixtures from scraped results
# -------------------------------------------------------------------
def load_r32_fixtures() -> list[dict]:
    df = pd.read_csv(RESULTS_FILE, parse_dates=["date"])
    df["home_team"] = df["home_team"].astype(str).str.strip().replace(NAME_FIXES)
    df["away_team"] = df["away_team"].astype(str).str.strip().replace(NAME_FIXES)

    ko = df[
        (df["tournament"] == "FIFA World Cup")
        & (df["date"] > CUTOFF_DATE)
        & (df["date"] <= pd.Timestamp("2026-07-03"))
    ]
    if len(ko) != 16:
        raise ValueError(f"Expected 16 R32 fixtures in {RESULTS_FILE}, found {len(ko)}")

    by_pair = {(r["home_team"], r["away_team"]): r for _, r in ko.iterrows()}
    fixtures = []
    for home, away in R32_BRACKET_ORDER:
        if (home, away) not in by_pair:
            raise ValueError(f"R32 fixture not found in results file: {home} vs {away}")
        r = by_pair[(home, away)]
        fixtures.append({
            "home": home,
            "away": away,
            "date": r["date"].strftime("%Y-%m-%d"),
            "city": r["city"],
            "country": r["country"],
            "neutral": 1 if bool(r["neutral"]) else 0,
        })
    return fixtures


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    merged = load_merged()

    wc_group = merged[
        (merged["tournament"] == "FIFA World Cup") & (merged["date"] >= GROUP_STAGE_START)
    ]
    teams = sorted(set(wc_group["home_team"]) | set(wc_group["away_team"]))
    assert len(teams) == 48, f"Expected 48 WC teams, got {len(teams)}"
    missing_flags = [t for t in teams if t not in FLAGS]
    assert not missing_flags, f"No flag emoji for: {missing_flags}"

    print(f"Building snapshots for {len(teams)} teams (history through {CUTOFF_DATE.date()})...")
    snapshots = {t: build_snapshot(merged, t) for t in teams}

    payload = joblib.load(XGB_MODEL_PATH)
    clf = payload["model"]
    temperature = payload.get("temperature", 1.0)
    inv_label_map = payload["inverse_label_map"]  # {0:'H',1:'D',2:'A'}
    h_idx = [k for k, v in inv_label_map.items() if v == "H"][0]
    d_idx = [k for k, v in inv_label_map.items() if v == "D"][0]
    a_idx = [k for k, v in inv_label_map.items() if v == "A"][0]

    print(f"Training Poisson goals regressors... (outcome temperature {temperature:.3f})")
    goals_home, goals_away = train_goals_models()

    def predict_batch(rows: list[dict]) -> list[list[float]]:
        X = pd.DataFrame(rows)[FEATURE_COLS].fillna(-99.0)
        margins = clf.predict(X, output_margin=True) / temperature
        e = np.exp(margins - margins.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        gh = goals_home.predict(X)
        ga = goals_away.predict(X)
        return [
            [
                round(float(p[h_idx]), 4),
                round(float(p[d_idx]), 4),
                round(float(p[a_idx]), 4),
                round(float(max(hg, 0.0)), 2),
                round(float(max(ag, 0.0)), 2),
            ]
            for p, hg, ag in zip(probs, gh, ga)
        ]

    print("Loading penalty shootout model...")
    pen_probs = load_penalty_predictor()

    # --- Round of 32 (true dates/venues) ---
    fixtures = load_r32_fixtures()
    r32_rows = [
        make_feature_row(
            snapshots[f["home"]], snapshots[f["away"]],
            pd.Timestamp(f["date"]), f["neutral"],
        )
        for f in fixtures
    ]
    r32_pens = pen_probs([(f["home"], f["away"]) for f in fixtures])
    for f, pred in zip(fixtures, predict_batch(r32_rows)):
        f["pred"] = pred + [r32_pens[f"{f['home']}|{f['away']}"]]

    # --- Pairwise table: every ordered pair, neutral ground ---
    print(f"Predicting all {len(teams) * (len(teams) - 1)} ordered team pairs...")
    name_pairs, pair_keys, pair_rows = [], [], []
    for t1 in teams:
        for t2 in teams:
            if t1 == t2:
                continue
            name_pairs.append((t1, t2))
            pair_keys.append(f"{t1}|{t2}")
            pair_rows.append(make_feature_row(snapshots[t1], snapshots[t2], NOMINAL_DATE, 1))
    all_pens = pen_probs(name_pairs)
    pairs = {
        key: pred + [all_pens[key]]
        for key, pred in zip(pair_keys, predict_batch(pair_rows))
    }

    data = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "cutoff": CUTOFF_DATE.strftime("%Y-%m-%d"),
        "model": "XGBoost outcome model + Poisson goal regressors + penalty shootout model",
        "teams": [{"name": t, "flag": FLAGS[t]} for t in teams],
        "r32": fixtures,
        "pairs": pairs,
    }

    WEBSITE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    size_kb = OUT_FILE.stat().st_size / 1024
    print(f"✅ Wrote {OUT_FILE} ({size_kb:.0f} KB)")
    print("Sample R32 prediction:", fixtures[0]["home"], "vs", fixtures[0]["away"], fixtures[0]["pred"])


if __name__ == "__main__":
    main()
