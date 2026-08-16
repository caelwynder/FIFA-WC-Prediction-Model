import pandas as pd
from pathlib import Path
import joblib


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PRED_DIR = BASE_DIR / "data" / "predictions"
MODEL_DIR = BASE_DIR / "models"

FEATURE_VECTOR_FILE = DATA_DIR / "feature_vectors" / "02_all_upcoming_match_vectors.csv"

RF_MODEL_PATH = MODEL_DIR / "rf_outcome_model.joblib"
XGB_MODEL_PATH = MODEL_DIR / "xgb_outcome_model.joblib"

OUT_FILE = PRED_DIR / "03_all_upcoming_match_predictions.csv"


from features import FEATURE_COLS

OUTCOME_LABELS = {
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win",
    0: "Home Win",
    1: "Draw",
    2: "Away Win",
}


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def clean_filename(text: str) -> str:
    return (
        str(text)
        .strip()
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "")
        .replace("*", "")
        .replace("?", "")
        .replace('"', "")
        .replace("<", "")
        .replace(">", "")
        .replace("|", "")
        .replace(" ", "_")
    )


def load_feature_vectors() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(FEATURE_VECTOR_FILE)

    if df.empty:
        raise ValueError("Feature vector file is empty.")

    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in feature vector file: {missing}")

    X = df[FEATURE_COLS].copy()
    X = X.fillna(-99.0)

    return df, X


def format_probs(class_labels, probs):
    out = {
        "P(Home Win)": None,
        "P(Draw)": None,
        "P(Away Win)": None,
    }

    for label, prob in zip(class_labels, probs):
        if label in ("H", 0):
            out["P(Home Win)"] = float(prob)
        elif label in ("D", 1):
            out["P(Draw)"] = float(prob)
        elif label in ("A", 2):
            out["P(Away Win)"] = float(prob)

    return out


def round_prediction_row(row: dict) -> dict:
    rounded = row.copy()
    for key in ["home_win_prob", "draw_prob", "away_win_prob"]:
        if rounded.get(key) is not None and pd.notna(rounded.get(key)):
            rounded[key] = round(float(rounded[key]), 5)
    return rounded


def predict_with_random_forest_all(base_df: pd.DataFrame, X: pd.DataFrame) -> list[dict]:
    model = joblib.load(RF_MODEL_PATH)

    preds = model.predict(X)
    probs = model.predict_proba(X)
    class_labels = list(model.classes_)

    rows = []

    for i in range(len(base_df)):
        prob_map = format_probs(class_labels, probs[i])
        pred = preds[i]

        row = {
            "date": base_df.loc[i, "date"] if "date" in base_df.columns else None,
            "home_team": base_df.loc[i, "home_team"] if "home_team" in base_df.columns else None,
            "away_team": base_df.loc[i, "away_team"] if "away_team" in base_df.columns else None,
            "tournament": base_df.loc[i, "tournament"] if "tournament" in base_df.columns else None,
            "model": "Random Forest",
            "predicted_class": pred,
            "home_win_prob": prob_map["P(Home Win)"],
            "draw_prob": prob_map["P(Draw)"],
            "away_win_prob": prob_map["P(Away Win)"],
            "predicted_label": OUTCOME_LABELS.get(pred, str(pred)),
        }

        rows.append(round_prediction_row(row))

    return rows


def predict_with_xgboost_all(base_df: pd.DataFrame, X: pd.DataFrame) -> list[dict]:
    payload = joblib.load(XGB_MODEL_PATH)

    model = payload["model"]
    inverse_label_map = payload["inverse_label_map"]

    pred_nums = model.predict(X)
    probs = model.predict_proba(X)
    class_labels = list(range(len(probs[0])))

    rows = []

    for i in range(len(base_df)):
        pred_num = int(pred_nums[i])
        pred_label = inverse_label_map[pred_num]
        prob_map = format_probs(class_labels, probs[i])

        row = {
            "date": base_df.loc[i, "date"] if "date" in base_df.columns else None,
            "home_team": base_df.loc[i, "home_team"] if "home_team" in base_df.columns else None,
            "away_team": base_df.loc[i, "away_team"] if "away_team" in base_df.columns else None,
            "tournament": base_df.loc[i, "tournament"] if "tournament" in base_df.columns else None,
            "model": "XGBoost",
            "predicted_class": pred_label,
            "predicted_label": OUTCOME_LABELS.get(pred_num, pred_label),
            "home_win_prob": prob_map["P(Home Win)"],
            "draw_prob": prob_map["P(Draw)"],
            "away_win_prob": prob_map["P(Away Win)"],
        }

        rows.append(round_prediction_row(row))

    return rows


def save_individual_match_prediction_files(out_df: pd.DataFrame):
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    grouped = out_df.groupby(["date", "home_team", "away_team"], dropna=False)

    for (date, home_team, away_team), group_df in grouped:
        safe_home = clean_filename(home_team)
        safe_away = clean_filename(away_team)
        safe_date = str(date)

        out_path = PRED_DIR / f"03_{safe_home}_vs_{safe_away}_{safe_date}_prediction.csv"
        group_df.to_csv(out_path, index=False)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    base_df, X = load_feature_vectors()

    rf_rows = predict_with_random_forest_all(base_df, X)
    xgb_rows = predict_with_xgboost_all(base_df, X)

    output_rows = rf_rows + xgb_rows
    out_df = pd.DataFrame(output_rows)

    sort_cols = [col for col in ["date", "home_team", "away_team", "model"] if col in out_df.columns]
    if sort_cols:
        out_df = out_df.sort_values(sort_cols).reset_index(drop=True)

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_FILE, index=False)

    save_individual_match_prediction_files(out_df)

    print("-" * 60)
    print("Batch prediction complete")
    print(f"Input feature file: {FEATURE_VECTOR_FILE}")
    print(f"Combined predictions saved to: {OUT_FILE}")
    print(f"Total matches processed: {len(base_df)}")
    print(f"Total prediction rows written: {len(out_df)}")
    print("-" * 60)

    for _, row in out_df.iterrows():
        print(f"Model: {row['model']}")
        print(f"Match: {row['home_team']} vs {row['away_team']}")
        print(f"Date: {row['date']}")
        print(f"Tournament: {row['tournament']}")
        print(f"Predicted Outcome: {row['predicted_label']}")
        print(f"P(Home Win): {row['home_win_prob']}")
        print(f"P(Draw): {row['draw_prob']}")
        print(f"P(Away Win): {row['away_win_prob']}")
        print("-" * 60)


if __name__ == "__main__":
    main()