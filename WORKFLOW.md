# WORKFLOW — Researched Execution Report (ROADMAP Priorities 1, 2, 6, 3)

Status report and workflow for the top four ROADMAP items, produced July 3, 2026.
Everything in the **Findings** section was measured against this repo's data and models;
every number is reproducible with the commands listed. All four phases below are
**implemented and verified** — this document records what was done, the evidence, and
the acceptance criteria a future rerun must still meet.

---

## Findings

### 1. The 87.9% accuracy was a data leak (now fixed)

The audit found a second real data bug (after the swapped-Elo merge fixed on July 2):
`01_build_model_dataset.py` derived pre-match rank as `rank − rank_change`, but the
eloratings **rank_change column is result-asymmetric** — it is zero on 93.4% of home
losses yet nonzero on 71.3% of home wins (mean |change| 0.07 vs 2.31). Subtracting it
embedded the match result into a "pre-match" feature. Inference code already used the
clean previous-match carry, so training and inference also disagreed (train/inference skew).

Evidence trail (all on the 2021-06+ test split):

| Metric | Leaky (before) | Clean (after) |
|---|---|---|
| Accuracy | 87.9% | 60.1% |
| Log loss | 0.301 | 0.875 |
| Draw recall | 71.9% (impossible) | 7.3% (typical) |
| Top feature | `home_rank_before` (0.229) | `elo_diff_before` (0.173) |
| Elo-only baseline | 61.1% | 61.1% |

- The Elo-derivation check was clean: subtract-method vs previous-match carry agreed
  within 1 point on 96.2% of 23,840 rows. The rank columns agreed on only 40.6%.
- Fix: both `elo_before` and `rank_before` are now carried from each team's previous
  match inside the leak-free history loop (`01_build_model_dataset.py`), matching
  inference exactly. Guarded by `tests/test_no_leakage.py`.
- Context from the literature: the Soccer Prediction Challenge best entries sit around
  **51.5% accuracy** on international matches ([survey](https://arxiv.org/pdf/2403.07669),
  [alternative ranking measures](https://arxiv.org/pdf/2405.10247)). The clean 60.1% —
  driven almost entirely by Elo — is at the strong end of published results.
  Note the full model currently *ties* the Elo-only baseline (60.1% vs 61.1% acc); the
  form features add nothing yet, which is the real motivation for ROADMAP P5.

### 2. Draw probabilities were overconfident → temperature scaling embedded

Reliability curves on 2021-06→2023-12 (n=2,769): home/away classes were well calibrated,
but predicted draw probabilities of 0.31–0.39 corresponded to observed draw rates of only
0.27–0.28. Per the calibration literature for tree ensembles on this data size,
temperature scaling beats isotonic (which needs ≫1000 calibration samples per class and
can overfit) — see [Johansson et al.](https://proceedings.mlr.press/v152/johansson21a/johansson21a.pdf),
[scikit-learn CalibratedClassifierCV docs](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html).

Measured on the untouched 2024+ holdout (n=2,491):

| | Log loss | ECE (top label) |
|---|---|---|
| Uncalibrated | 0.8818 | 0.0408 |
| **Temperature T=1.141** | **0.8754** | **0.0183** |
| Isotonic per class | 0.8763 | 0.0241 |

Implemented in `02_train_outcome_model.py` (fits T on `SPLIT_DATE..CALIB_END`, stores it
in the model payload); `04_build_website_data.py` applies it via softmax(margins/T).

### 3. Shared feature module + regression tests now exist

- `project-root/scripts/features.py` is the single source of `FEATURE_COLS`,
  `TOURNAMENT_WEIGHTS`/`get_tournament_weight`, `build_snapshot`, `make_feature_row`.
  Scripts `01`–`05` import it; no duplicated schema remains.
- `project-root/tests/` (pytest, 7 tests, ~1s) locks in the invariants that would have
  caught both of this week's bugs:
  - `test_merge_orientation.py` — known matches carry Elo/rank on the correct side
    (catches the swapped-Elo merge bug),
  - `test_no_leakage.py` — every `*_before` value equals the previous-match carry
    (catches the rank-change leak; match rate was ~40% under the leaky derivation),
  - `test_cutoff.py` — no post-cutoff rows in any processed output,
  - `test_train_inference_parity.py` — training rows equal the shared snapshot builder's
    output for sampled 2024+ matches.
- `project-root/requirements.txt` pins the verified versions (pandas is pinned hard —
  the 2.x→3.0 upgrade broke this codebase twice).

### 4. Trophy odds: exact bracket probabilities (better than Monte Carlo)

Following [Brandes et al. 2025](https://arxiv.org/pdf/2307.10411) ("Stop Simulating!"),
round-reach probabilities are computed **exactly** by dynamic programming over the fixed
bracket instead of sampling. Per-match advance probability folds the tandem models:
`P(A advances) = P(A wins) + P(draw) × P(A wins shootout)`.

- `scripts/06_simulate_tournament.py`: exact DP + a 10,000-run Monte Carlo cross-check
  (measured max deviation 0.0049 — inside sampling noise), champion odds sum to 1,
  writes `odds` into `website/data.json`.
- The website computes the same DP in JS so **manual winner overrides condition the
  odds live**, and shows a Trophy Odds leaderboard (R16/QF/SF/Final/Champion per team).
- Current model says: Argentina 21.3%, France 17.4%, Spain 15.3%, Colombia 11.6%,
  Brazil 7.5%, England 6.9% — a far more honest picture than the old single-path
  "champion at 44%" display.

---

## Workflow (rerun after any data/model change)

Each step's acceptance criteria must hold before moving on.

1. **Refresh + retrain** — `./​.venv/bin/python scripts/run_full_pipeline.py`
   (from `project-root/`, with `DYLD_LIBRARY_PATH=.venv/lib/python3.14/site-packages/sklearn/.dylibs`).
   ✓ Cutoff message shows `2026-06-27`; XGB accuracy lands ~58–63%; temperature ~1.0–1.3;
   penalty model log loss ≈ base rate.
2. **Tests** — `./​.venv/bin/python -m pytest tests/ -q`.
   ✓ 7/7 pass. If `test_no_leakage` fails, someone reintroduced a change-subtraction.
3. **Sanity diagnostics** (catch new leaks early):
   confusion matrix per class (draw recall must stay < 40%), feature importances
   (`elo_diff_before` should lead; any single feature > 0.2 that isn't Elo-related is
   suspect), Elo-only baseline within ~3 points of the full model until P5 lands.
4. **Website data** — `04_build_website_data.py` then `06_simulate_tournament.py`.
   ✓ 06's assertions pass (odds sum to 1; MC deviation < 0.015).
5. **Site check** — `node serve.mjs` + `node screenshot.mjs http://localhost:3000/website/`.
   ✓ Bracket fits, Trophy Odds table populated, no JS console errors, header accuracy
   claim matches the current model.

## Deferred (next up, in order of expected value)

- **P5 feature engineering** — now the top accuracy lever: the model ties an Elo-only
  baseline, so everything beyond Elo is headroom. Start with the explicit Elo win
  expectancy `1/(1+10^(−diff/400))`, head-to-head records, opponent-adjusted form.
- **P7 goals model** — replace the independent Poisson regressors with Dixon-Coles
  (rho-corrected low-score dependency, time-weighted MLE) so outcome and scoreline agree:
  [original approach explained](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/),
  [python walkthrough](https://pena.lt/y/2021/06/24/predicting-football-results-using-python-and-dixon-and-coles/).
- **P4 merge recovery** — ~1,833 unmatched results rows (±1-day tolerance + name map).
- **P8 ops** — model-vs-reality tracker page; static deployment; metrics history file.
