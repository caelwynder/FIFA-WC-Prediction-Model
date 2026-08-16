# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pipeline that scrapes international football (soccer) results and Elo ratings, engineers
time-aware form/rating features, trains classifiers to predict match outcomes (Home/Draw/Away),
and produces per-fixture predictions — currently tuned around 2026 FIFA World Cup fixtures.

All real code lives under `project-root/`. The repo root only holds `README.md`.

## Environment

`project-root/.venv` (Python 3.14, not committed to git) has everything the live pipeline needs —
both scraping deps (requests, pandas, pycountry) and the ML stack (numpy, scikit-learn, xgboost,
scipy, joblib). There is no `requirements.txt`; install missing packages straight into `.venv`.
`project-root/scripts/pred.yml` is a separate conda env (`prediction`) that predates `.venv` and
pulls in bs4/lxml/playwright — none of the currently-live scripts import those, so treat `pred.yml`
as legacy unless you're reviving one of the dead scripts in `scripts/archived/` that needs it.

To run any script directly, activate `.venv` from `project-root`:
```bash
source project-root/.venv/bin/activate
python project-root/scripts/02_train_outcome_model.py
```

**xgboost import quirk on macOS without Homebrew**: the PyPI xgboost wheel links `libomp.dylib` via
an `@rpath` that resolves to `/opt/homebrew/opt/libomp`. On a machine without Homebrew this import
fails (`Library not loaded: @rpath/libomp.dylib`) even though a working copy already ships inside
`sklearn`'s bundled `.dylibs`. `run_full_pipeline.py` works around this by setting
`DYLD_LIBRARY_PATH` to `.venv/lib/python3.14/site-packages/sklearn/.dylibs` before invoking any
step — if you run `02_train_outcome_model.py` directly and hit this error, export that same
`DYLD_LIBRARY_PATH` yourself, or `brew install libomp`.

## Full refresh — one command

`project-root/scripts/run_full_pipeline.py` chains scrape → merge → rebuild features → retrain
outcome models → retrain penalty model into a single run, so "get fresh data into up-to-date
models" is one command:
```bash
project-root/.venv/bin/python project-root/scripts/run_full_pipeline.py
```
It does **not** run step 3/5 (upcoming-fixture feature vectors + predictions) — rerun
`02_team_fvector.py` / `03_predict_matches.py` separately if you also want refreshed predictions
for the fixtures in `FIXTURES`. Each step still overwrites the tracked CSVs/joblib artifacts in
place, same as running them individually.

## Pipeline (run in order)

Each stage reads the previous stage's output from `data/` and is a standalone script — there is no
CLI entrypoint or task runner, just `python scripts/<name>.py` from `project-root/`.

1. **`00_run_preprocessing_pipeline.py`** — orchestrator. Shells out (via `subprocess`) to the
   scrapers in `scripts/archived/` in sequence: results scraper → GitHub data scraper (goalscorers/
   shootouts/former_names from `martj42/international_results`) → EloRatings.net scraper → dataset
   merge. Moves the raw merge output to `data/processed/00_merged_dataset.csv`, then strips a couple
   of always-unwanted columns in place and **drops all matches after `CUTOFF_DATE`** (currently
   `2026-06-27`, end of the 2026 World Cup group stage) — this cap is intentional, so the model is
   frozen at that date and predicts every match after it; don't "fix" it as stale data. Despite
   living in `archived/`, these scraper scripts are the live implementation this orchestrator
   depends on — don't remove them as dead code.
2. **`01_build_model_dataset.py`** — turns `00_merged_dataset.csv` into
   `data/processed/01_model_dataset.csv`. Computes pre-match Elo/rank by **carrying each team's
   previous match's post-match value** inside the history loop — NEVER by subtracting the recorded
   rating/rank "change" columns: eloratings' `rank_change` is result-asymmetric (zero on ~93% of
   losses) so `rank − rank_change` leaks the match result (this bug once inflated accuracy to a
   fake 87.9%; honest is ~60%; guarded by `tests/test_no_leakage.py`). Also computes tournament
   weight and rolling 5-match form features via a chronological single pass with `deque(maxlen=5)`
   history — **all pre-match features are extracted before that row's own result updates the
   history**. Derives the `match_outcome`/`target_*` labels.
3. **`02_team_fvector.py`** — builds feature vectors for *upcoming* (unplayed) fixtures, using the
   same feature schema as step 1 but computed on demand from `FIXTURES`, a hardcoded list of dicts
   at the top of the file (home/away team, date, tournament, neutral flag). Update `FIXTURES` to
   change which matches get predicted. Writes one CSV per fixture to
   `data/feature_vectors/individual_matches/` plus a combined
   `data/feature_vectors/02_all_upcoming_match_vectors.csv`.
4. **`02_train_outcome_model.py`** — trains both a `RandomForestClassifier` and an `XGBClassifier`
   on `01_model_dataset.csv` using a **time-based split** (`SPLIT_DATE = "2021-06-01"`, not a random
   split — respect this when evaluating changes, since shuffled splits leak future form into
   training). Missing feature values are filled with `-99.0` (a sentinel, not imputed) at this
   stage. Also fits a **temperature-scaling calibrator** on `[SPLIT_DATE, CALIB_END)` (draw probs
   were overconfident) and reports calibrated log loss on the post-`CALIB_END` holdout. Saves both
   models under `models/` via `joblib`; the XGBoost payload bundles the model, label maps, feature
   columns, and `temperature` — consumers must apply `softmax(margins / temperature)`.
5. **`03_predict_matches.py`** — loads the combined feature vector CSV from step 3, scores it with
   both saved models, and writes combined + per-fixture prediction CSVs to `data/predictions/`.
6. **`04_build_website_data.py`** — generates `website/data.json` for the predictions website:
   the 16 real Round-of-32 fixtures (bracket-seed ordered, predicted with true venues) plus a
   pairwise prediction table (probabilities + Poisson-regressed goals + shootout win prob) for all
   48 WC teams on neutral ground. The website (`website/index.html`, served per WEBSITE.md)
   cascades bracket rounds client-side from that table, supports manual winner overrides
   (localStorage), and has a head-to-head simulator. Rerun this script after retraining to refresh
   the site.
7. **`05_train_penalty_model.py`** — penalty shootout model, used **in tandem** with the outcome
   model: when the outcome model's top prediction is a draw, the match winner comes from this
   model instead (each prediction array's last element = P(home wins shootout), symmetrized across
   orientations). Trained on international shootouts since 2000 (`00_shootouts_final.csv`, from
   `scripts/archived/00_penalty_scraper.py`, which also pulls in-game penalty goals and
   kick-by-kick WC shootout data from jfjelstul/worldcup, all capped at the cutoff). Shootouts are
   near coin flips: the model is deliberately tiny/heavily regularized (depth-2 XGB) and only
   matches the base-rate log loss — expect probabilities in ~0.38–0.65, never confident calls.
   The saved payload includes per-team feature profiles frozen at the cutoff for inference.

8. **`06_simulate_tournament.py`** — exact per-team round-reach/champion odds via dynamic
   programming over the bracket (with a 10k Monte Carlo cross-check), written into
   `website/data.json` as `odds`. The website re-implements the same DP in JS so manual winner
   overrides condition the Trophy Odds leaderboard live. Run after `04`.

`scripts/features.py` is the **single source** of `FEATURE_COLS`, `TOURNAMENT_WEIGHTS`,
`get_tournament_weight`, and the snapshot/feature-row builders — scripts 1–5 import it; never
redeclare the schema in a script. `tests/` (pytest, run with
`./.venv/bin/python -m pytest tests/ -q` from `project-root/`) guards merge orientation, leakage,
cutoff, and train/inference parity — run it after any pipeline change. `requirements.txt` pins the
verified dependency versions.

## Key modeling conventions

- **No shuffled train/test split** — always split by `date` to avoid leakage; see `SPLIT_DATE` in
  step 4.
- **Rolling features must stay leakage-free**: any change to `01_build_model_dataset.py`'s history
  loop must preserve "extract features first, update history after" ordering per row.
- Outcome labels: `"H"`/`"D"`/`"A"` in dataset form; `0`/`1`/`2` (`LABEL_MAP`/`INV_LABEL_MAP`) for
  XGBoost, which needs numeric class labels.
- Missing numeric values are `np.nan` through feature engineering, then filled with `-99.0` only at
  the point of feeding a model (not persisted into the feature CSVs that way).

## Data layout (`project-root/data/`, `models/`, all git-tracked)

- `raw/` — scraped/downloaded source CSVs.
- `tsv/` — per-year Elo ratings TSVs scraped from eloratings.net. The scraper treats a year's TSV
  as final only if its file mtime is after that year ended; a file fetched mid-year is re-downloaded
  on the next run to pick up new matches (the merged dataset is capped by the Elo side, so a stale
  current-year TSV silently caps the whole dataset).
- `processed/` — `00_merged_dataset.csv` (raw merge) and `01_model_dataset.csv` (features + labels).
- `feature_vectors/` — upcoming-fixture feature vectors from step 3.
- `predictions/` — model output from step 4.
- `models/` — `rf_outcome_model.joblib`, `xgb_outcome_model.joblib` (large binary artifacts, tracked
  in git). All `joblib.dump` calls use `compress=3` — uncompressed the random forest is ~207 MB,
  over GitHub's 100 MB per-file hard limit, which blocks the push; compressed it is ~37 MB. Keep
  `compress=3` on any new or changed model save.

## Scripts not part of the pipeline

`project-root/scripts/archived/` also contains superseded one-off scripts (`test.py`,
`merge_tsv.py`, `merge_dates.py`, `results_shortener.py`, `update_names.py`, `pull_results.py`,
`eloratings_scraper.py`) that predate the current `00_*` scraper/merge scripts and are not called
by the orchestrator — don't assume everything in `archived/` is live; check whether
`00_run_preprocessing_pipeline.py`'s `find_script()` calls actually reference it.
