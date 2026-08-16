# ROADMAP — Improving the Soccer Prediction Model

Where the project stands after the July 2026 rebuild, and a prioritized list of
improvements with concrete steps. Priorities 1–2 are about *trusting* the numbers,
3–5 about making the predictions better, 6+ about engineering health and product.

> **Status update (July 3, 2026): Priorities 1, 2, 3, and 6 are DONE — see
> `WORKFLOW.md` for the evidence and the rerun checklist.** The leakage audit found and
> fixed a real leak (eloratings' rank_change is result-asymmetric, so `rank −
> rank_change` embedded the result; honest accuracy is 60.1%, not 87.9%), temperature
> scaling (T=1.141) is embedded in the model payload, exact bracket odds (DP,
> Monte-Carlo cross-checked) power a Trophy Odds leaderboard on the site, and
> `scripts/features.py` + `tests/` + `requirements.txt` exist. The sections below are
> kept for the reasoning; the remaining open items are P4, P5, P7, P8.

## Current state (July 2026, post-audit)

- **Data**: 23,955 internationals (2000 → June 27, 2026), scraped fresh each run
  (results + eloratings.net + penalty/shootout data), hard-capped at `CUTOFF_DATE`.
- **Models**: XGBoost 3-class outcome model (60.1% holdout accuracy, 0.875 log loss,
  temperature-calibrated — at the strong end of published international benchmarks),
  Poisson goal regressors, and a regularized penalty-shootout model used in tandem
  when the outcome model's top call is a draw.
- **Product**: `website/` bracket + Trophy Odds leaderboard + head-to-head simulator.
- **Known weaknesses**: ~1,800 results rows still unmatched in the Elo merge; form
  features currently add nothing over an Elo-only baseline (P5 is the accuracy lever).

---

## Priority 1 — Audit for residual leakage (credibility)

**Why first:** 87.9% three-class accuracy is far above published benchmarks for
international football (typically 52–60%). That gap appeared after fixing the
swapped-Elo merge bug, which is plausible — but the same audit that caught that bug
should be finished before trusting or publishing any numbers. The main suspect is
the `*_rating_change` / `*_rank_change` semantics: `elo_before = home_elo −
home_rating_change` only recovers the true pre-match rating if the change column is
*this match's* change.

**Steps:**
1. Pick ~20 matches across eras and manually compare the computed `home_elo_before`
   against the pre-match rating shown on eloratings.net for that team/date.
2. Print a per-class confusion matrix on the test set (`H/D/A`), and accuracy split
   by tournament type. If draws are being "predicted" at far above the ~55% ceiling
   research suggests, something is leaking.
3. Train an elo-only baseline (single feature: `elo_diff_before`) on the same split.
   If the full model beats it by more than ~5–8 points of accuracy, be suspicious.
4. Re-derive `elo_before` a second, independent way — from the team's *previous*
   match's post-match Elo (as `02_team_fvector.py` does) — and diff the two columns.
   Any large disagreement pinpoints rows with wrong `rating_change` semantics.

## Priority 2 — Probability calibration

**Why:** several knockout predictions put 85–94% on a draw, which is implausible
(historical draw base rate in internationals is ~22–25%). Even if accuracy is fine,
downstream consumers (penalty tandem, bracket advancement) use the *probabilities*.

**Steps:**
1. Plot a reliability diagram per class on the holdout (predicted prob vs observed
   frequency, 10 bins). `sklearn.calibration.calibration_curve` does this directly.
2. If miscalibrated, fit isotonic or temperature scaling on a validation slice
   (e.g. 2021–2023) and evaluate on 2024+ only — never calibrate on the test years.
3. Wrap the saved model payload with the calibrator so `03_predict_matches.py` and
   `04_build_website_data.py` get calibrated probabilities transparently.

## Priority 3 — Monte Carlo tournament simulation

**Why:** the bracket currently advances a single argmax path, so "Argentina 44% to
win the final" understates uncertainty and there is no real "chance to win the
trophy" number. Sampling fixes that cheaply, since the pairwise table already exists.

**Steps:**
1. In `04_build_website_data.py` (or a new `06_simulate_tournament.py`), simulate
   the bracket 10,000×: at each match, sample the outcome from `[pH, pD, pA]`; on a
   draw, sample the shootout from the penalty prob.
2. Record per-team counts of reaching R16/QF/SF/Final/Champion → per-team advancement
   percentages.
3. Add to `data.json` and show on the site: champion-odds leaderboard, and per-tile
   "% chance to reach this round". Manual overrides can condition the simulation by
   fixing that match's winner.

## Priority 4 — Recover the unmatched merge rows

**Why:** ~1,833 results rows (about 7%) fail to join with Elo data and are dropped.
Smaller nations are hit hardest, and those are exactly the teams the model knows
least about.

**Steps:**
1. Read `00_merged_dataset_unmatched_results.csv` and tabulate failure causes:
   date off-by-one (different timezone conventions), team-name variants, score
   disagreements between the sources.
2. Add a ±1-day date-tolerance pass to `00_merge_datasets.py` for rows that match
   on teams + score but not date.
3. Extend the name-normalization map for the recurring offenders the tabulation
   reveals; rerun and confirm matched-row count rises and no duplicates appear.

## Priority 5 — Feature engineering

Candidates, roughly in expected-value order:

1. **Explicit Elo expectancy**: `1/(1+10^(−elo_diff/400))` — gives the tree model
   the standard nonlinearity for free.
2. **Head-to-head record** vs this specific opponent (last N meetings: win rate, GD).
3. **Opponent-adjusted form**: average Elo of the opponents in each team's last 5,
   so "5 wins vs minnows" stops looking like "5 wins vs France".
4. **Longer/decayed form windows**: 10-match window and/or exponential decay
   alongside the current 5-match stats.
5. **Context flags**: knockout vs group stage, confederation of each side,
   same-confederation matchup.

**Steps:** add each feature in `01_build_model_dataset.py` (inside the leak-free
history loop) *and* the duplicated `FEATURE_COLS` consumers — or do Priority 6
first so there is only one place to change. Evaluate one feature group at a time on
the fixed time split; keep only what moves test log loss.

## Priority 6 — Single shared feature module + tests

**Why:** `FEATURE_COLS`, `TOURNAMENT_WEIGHTS`, `get_tournament_weight`, and the
snapshot builders are duplicated across `01`, `02_team_fvector`, `02_train`, `03`,
`04`, and `05`. Every feature change risks silent train/inference skew — this was
already nearly a bug during the website build.

**Steps:**
1. Create `scripts/features.py` exporting `FEATURE_COLS`, tournament weighting,
   and the team-snapshot / feature-row builders; import it everywhere.
2. Add `pytest` with a `tests/` folder covering: cutoff enforcement (no rows after
   `CUTOFF_DATE`), no swapped-Elo corruption (assert a few known matches' Elo values),
   leak-freedom of the rolling loop (a team's features on match N don't change if
   match N's result changes), and train/inference feature-vector parity for one team.
3. Add `requirements.txt` (pandas, numpy, scikit-learn, xgboost, requests, pycountry,
   joblib) and pin versions — the pandas-3 breakages this month are the argument.
4. Optional: GitHub Actions workflow running the tests on push.

## Priority 7 — Model improvements

1. **Walk-forward hyperparameter tuning**: instead of one 2021 split, use expanding
   windows (train→2018 test 2019, train→2019 test 2020, …) and tune XGBoost params
   (depth, learning rate, estimators with early stopping) on average log loss.
2. **Coherent goals model**: the independent Poisson regressors can disagree with
   the classifier (a predicted winner with fewer estimated goals). A Dixon-Coles /
   bivariate Poisson model produces outcome probabilities *and* scorelines from one
   set of attack/defense ratings — replacing both heads consistently.
3. **Use or drop the Random Forest**: it is trained and saved but nothing downstream
   uses it. Either ensemble it (average probabilities with XGBoost, check holdout)
   or remove it and halve `models/` size (it is 130 MB in git).
4. **Penalty model upgrades**: player-level penalty conversion from the
   Transfermarkt penalty dataset (needs a scraper + squad mapping), goalkeeper
   save rates, or accept the coin-flip reality and shrink further toward 0.5.

## Priority 8 — Product & operations

1. **Model-vs-reality tracker**: knockout results are arriving daily; add a page
   section comparing predictions to actual outcomes (the tandem already called
   Paraguay over Germany on penalties correctly — surface that).
2. **Deploy the site**: `website/` is fully static — GitHub Pages or Netlify works
   as-is. Keep `data.json` regeneration manual (retrain → rerun `04` → push).
3. **Scheduled refresh**: a cron/launchd job (or `claude /schedule`) running
   `run_full_pipeline.py` + `04_build_website_data.py` after each matchday — only
   worthwhile once the cutoff freeze is lifted after the tournament.
4. **Track metrics over time**: append each training run's accuracy/log-loss to
   `reports/metrics_history.csv` so improvements (and regressions) are visible.

---

## Suggested order of attack

| # | Item | Effort | Payoff |
|---|------|--------|--------|
| 1 | Leakage audit (P1) | ~1 session | Trust in every other number |
| 2 | Calibration check + fix (P2) | ~1 session | Sane draw probabilities → better tandem |
| 3 | Shared feature module + tests (P6) | ~1 session | Makes everything after safe |
| 4 | Monte Carlo simulation (P3) | ~1 session | Best product improvement per hour |
| 5 | Merge recovery (P4) | ~1 session | More data, esp. small nations |
| 6 | Feature engineering rounds (P5) | ongoing | Incremental log-loss gains |
| 7 | Walk-forward tuning + goals model (P7) | 2–3 sessions | Squeezes the remaining edge |
| 8 | Deploy + results tracker (P8) | ~1 session | Shareable, self-updating product |
