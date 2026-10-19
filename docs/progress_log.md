# Progress log

Chronology: what was done each session, and what was verified. Decisions go in `decisions_log.md`; results with their method go in reports.

## 2026-10-09 · Plan v2 and the data layer (zip 1)

**Plan.** Plan of record revised to version 2 after review (`plan_of_record.md`, Part D lists every change). Hypotheses H1–H5 kept; stage 2 made a forecasting task; fly reduced to an optional reservoir row.

**Repository.** Study 1 (conflict) is archived in git history before the deletion commit (tag it `conflict-final`). New layout: `config/` (all analytic choices), `R/` (ERT and conversion only), `src/flypredictor/` (package), numbered notebook phases, `tests/`, `prereg/`, `docs/`.

**Built.**
* `fetch.py`: pinned download of ERT v16 + V-Dem v16 from the ERT repository at commit `7fd51ce`, SHA-256 manifest, Parquet conversion (pyreadr or R).
* `data/episodes.py`: 101 democratic episodes, confirmation/breakdown years, severity, country-level events.
* `data/ert_jobs.py` + `R/ert_variants.R`: `get_eps` re-run on V-Dem truncated at every year 1900–2025 (real-time status; ~5 s per year).
* `data/states.py`: stage 0/1/2 per country-year (sources: `ert_truncated`, `rule`, `hindsight`).
* `data/labels.py`: stage-1 (declaration / confirmation / onset) and stage-2 (breakdown) labels, observability and maturity; optional stage-1 breakdown target.
* `data/panel.py`: country-year panel with the raw indicators from `config/features.yaml`.
* `audit/`: coverage, missingness by decade, era-proxy flags, episode recount, maturity lags, state-source comparison.

**Verified.**
* 33 tests pass: 30 on synthetic fixtures with hand-computed answers (including a 200-case brute-force check of the window labels), 3 on the real outputs.
* Notebooks 01–03 run end to end from an empty data folder (about 5 minutes, mostly the R re-runs).
* `get_eps` truncated at 2025 reproduces the shipped ERT coding exactly (all 179 countries).
* R and Python Parquet conversions give identical values.
* Every variable in `features.yaml` exists in V-Dem v16.

**Audit findings (outcome-only or feature-only).**
* 101 democratic episodes: 71 breakdown, 13 averted, 1 preempted, 1 diminished, 15 censored. ERT's codebook says "diminished" does not occur with default parameters, yet 1 does: check that episode.
* Start → confirmation lag: median 2 years, 90th percentile 7. Breakdown → genuine: 62% immediate, 90th percentile 5.
* 51 of 71 breakdowns occur at or before real-time confirmation (23 in the first episode year), so real-time stage 2 has only ~148 country-years (decision D-08).
* Real-time vs final coding in 2010: Hungary, India, Serbia and Bolivia not yet in a known episode; Bosnia, Fiji, Sri Lanka and Palestine in one that the final coding later closed or dropped.
* Coverage: e_gdppc ends 2019; e_pt_coup starts 1950; LIED ends 2024; BMR ends 2020; v2xpas covers 16% of democracy-years. All other indicators ≥ 98% complete in democracy rows. Coups and GDP flagged as era proxies.

**Next.** Mark decides D-07 and D-08 → zip 2: splits, metrics, bootstrap, lockbox, features.py, power simulation → fill and freeze the pre-registration.

## 2026-10-09 · First run on Mark's machine and review of zip 1

**Run.** Python 3.14.3 venv, R 4.5.3 + Rtools, `renv` library. Notebooks 01–03 ran end to end; `pytest` 33/33. `conflict-final` tagged locally on `ec30f50`.

**Verified independently.**
* Real-time re-run cut at 2025 = shipped ERT coding for all 179 countries (`aut_ep`, `aut_ep_id`, `aut_ep_prch`, `aut_ep_pbr`, `aut_ep_outcome`).
* Each of the 101 episodes is declared exactly once in the re-runs, with the same start year; no declaration is ever withdrawn.

**Review findings (severity order).**
1. HIGH · The stage-1 `confirmation` event is mostly breakdown in the development years (7 of 36 k = 1 positives are declarations while still a democracy; 40 of 53 in the locked test). Decision D-07a.
2. HIGH · RoW vs ERT regime-type mismatch put 45 ERT-autocratic rows in the risk set, labelling coups such as Argentina 1976 and Fiji 2006 as 0. Fixed (D-15); `row_breakdown` sensitivity added.
3. MEDIUM · Shift between development and locked test: k = 1 declaration base rate 1.1% → 3.1%; before 2010 declarations ≈ breakdowns, in the 2010s 22 vs 7. Expect Brier skill to suffer from miscalibration; PR-AUC ÷ base rate is the more robust primary metric. To state in the pre-registration.
4. MEDIUM · "Real-time" states remove back-dated starts and known end years, not revisions to V-Dem values (all re-runs read v16). Only E8 (vintage) tests data-revision hindsight; consider making it required. To state in the paper.
5. MEDIUM · Era-proxy audit missed `v2xpas_*`; GDP's carry-forward feature is out of distribution in 2020–2024. Audit fixed (D-16); GDP decision pending (D-17).
6. LOW · `requirements-lock.txt` was written by PowerShell as UTF-16 and pins the package to commit `b10ed54` with `-e git+…`. Regenerate with `python -m pip freeze --exclude-editable | Set-Content -Encoding utf8 requirements-lock.txt`. Lock shows pandas 3.0.6.
7. LOW · Zip 1 not yet committed; `conflict-final` push not confirmed.
8. LOW · BIH_2004_2007 (the "diminished democracy" episode) has rule confirmation 2009, after its end year 2007 (the rule allows end + tolerance). Affects only `state_source: rule`; documented, not changed.

**Built.**
* `states.py`: `democracy.require_ert_democracy` (default true); new column `row_democracy`.
* `episodes.country_events(eps, rt, ert)`: new event `row_breakdown`; `labels.py`: optional target `stage1_row_breakdown_target` (fails loudly if the events are missing).
* `audit/coverage.py`: `era_proxy_flags` gains the `ends_early` rule and a `reason` column; new `window_coverage`.
* `audit/definitions.py`: `risk_set_vs_ert`, `row_transitions`, `declaration_timing`.
* Notebook 02 passes `ert` to `country_events`; notebook 03 gains A7b, A8, A9 and a filled-in findings cell.
* `tests/test_definitions.py`: 7 new fixture tests (37 fixture tests in total, all passing); fixtures carry `reg_type`.

**Effect on the data (in-memory run, before re-running the notebooks).** Stage 1 5,076 → 5,031 rows; stage 2 unchanged (148); event counts unchanged. k = 1 locked-window events: declarations 53, ERT breakdowns 13, RoW breakdowns 26, stage-2 breakdowns 7. Development stage-1 rows 3,271 → 3,255.

**Next.** Re-run notebooks 02 and 03, then `pytest`. Regenerate the lock file, commit zip 1, push `conflict-final`. Decide D-07a, D-08 and D-17 (D-07a after the power simulation) → zip 2.

## 2026-10-09 (evening) · Decisions and zip 2: features, evaluation harness, power simulation

**Decisions (Mark).** D-07b (H1 = any real-time declaration; new H1b on gradual declarations; annual refits in the locked window), D-08b (H5 = real-time breakdown on all democracies; conditional stage 2 exploratory), D-12b (party arm cut), D-17b (GDP dropped). Implementation choices open to override: D-18 (intercept updating), D-19 (power-simulation design).

**Discipline.** No model has been fit to real outcomes. The plan puts development runs after the freeze, so the power simulation uses synthetic labels and reads only the `observable` column and the per-period event rates from the audits.

**Built.**
* `data/episodes.split_declarations` + labels `declaration_dem` / `declaration_abrupt` (H1b); notebook 02 updated.
* `data/features.py`: levels, 1- and 3-year changes (missing across coding gaps), EDI decline from the 10-year peak, liberal flag, coup coverage rule (3-year count, pre-1950 indicator); `FeaturePreprocessor` (median imputation, group missingness flags, standardisation; training rows only). Clean 59 columns, full 115.
* `eval/splits.py` (training cutoff t − 1 − k − lag; 7 development blocks 1970–2004; annual or frozen locked schedules; locked folds refused outside the lockbox), `eval/metrics.py` (tie-aware weighted AP matching scikit-learn, Brier skill, log loss, alerts, event-level hit rate), `eval/bootstrap.py` (country-clustered, paired), `eval/recalibration.py` (D-18), `eval/lockbox.py` (tag `prereg-v1`, unchanged `config/` and `prereg/` or a logged deviation, once-only ledger), `eval/power.py`.
* `models/baselines.py` (persistence bins, EDI momentum), `models/linear.py` (elastic net, ridge).
* `notebooks/0x_data/04_power.ipynb`.
* Tests: 26 new (features, harness, lockbox on a temporary git repo, power helpers) plus 2 real-data checks (feature truncation on the real panel; declaration split).

**Verified.**
* 63 tests pass; the new real-data split check fails until notebook 02 is re-run (stale labels), as intended.
* Feature truncation on the real panel (T = 1975, 2004, 2015): identical rows, so no look-ahead.
* Notebook 04 runs end to end on a reduced grid (1 rep per scenario).

**Findings while building the simulation.**
* Under a planted effect of β = 1 (log-odds per SD of a leading-sign index), even the true probabilities reach only AUC ≈ 0.65–0.73 at *k* = 1; β = 2 gives ≈ 0.82–0.85.
* The leading-sign index correlates 0.72 with the negative 3-year EDI change in the real panel.
* Models trained on mostly low-rate history under-predict the locked window (mean forecast 1.5–2% vs 3%) → D-18.
* Stand-in check (1 rep, β = 2): elastic net Brier skill ≥ ridge at *k* = 1–3, AP ratios similar → simulated power is conservative for the elastic net.

**Next.** Mark: re-run 02 → 03 → `pytest` (all pass) → 04 full run (~30–60 min on 8 cores) → commit. Then fill δs, the H5 horizon and the H1b / H3 / H4 status in the pre-registration; check the `verify: true` items in `features.yaml` against the v16 codebook; tag `prereg-v1`. After the tag: the model zip (tuning on development folds, gradient boosting, GRU, kitchen-sink, stacking, drop-group refits).


## 2026-10-10 · Power simulation speed-up

* First full run of notebook 04 ran over an hour with no output: the run cell printed nothing until the end, and half of each simulated dataset's time went to pandas `fillna` in `FeaturePreprocessor`.
* `FeaturePreprocessor` rewritten in numpy: identical output (max difference 4e-13 on the real panel), ~4x faster; a full simulated dataset now takes ~6 s on one core (was > 20 s).
* `power.run` now runs one scenario at a time, prints its time and an ETA, and checkpoints each scenario to `results/dev/power/sim_parts/`; re-running resumes (checked: identical results). The run cell first times one dataset and prints an estimate.
* 64 tests pass (including the real-data declaration-split check after the notebook 02 re-run).

## 2026-10-10 · Power simulation results (notebook 04, 200 reps × 7 scenarios)

Files: `results/dev/power/mde.csv`, `power_table.csv`, `stand_in_check.csv`. β = log-odds per SD of the planted leading-sign index; oracle AUC = AUC of the true probabilities at *k* = 1 (β = 1 → 0.68, 2 → 0.82, 3 → 0.91).

* **Type-I error is controlled.** Pass rate with no signal: 0–1% for H1, H1b, H2a; 0–3.5% for H5 (within Monte Carlo error of 2.5%, SE ≈ 1.1%).
* **H1, every k (as drafted):** 32% power at β = 2, 69% at β = 3. Never reaches 80%; *k* = 5 is the bottleneck (36% / 69%).
* **H1 by horizon set (power at β = 2 / 3):** *k* = 1: 0.85 / 1.00; *k* = 2: 0.85 / 1.00; {1, 2}: 0.76 / 1.00; {1, 2, 3}: 0.67 / 0.97.
* **H1b:** *k* = 1: 0.64 / 0.95; {1, 2}: 0.46 / 0.92.
* **H2a:** *k* = 3: 0.62 / 0.98; *k* = 5: 0.22 / 0.46; both: 0.22 / 0.46.
* **H5:** at best 0.26 / 0.47 (*k* = 1, about 8–11 locked positive rows); falls with *k*.
* **Stand-in check (10 datasets, β = 2, frozen fit):** the elastic net with fixed settings (C = 0.1, l1_ratio = 0.5) passed less often than the ridge stand-in: in 28 paired cases ridge passed and the elastic net failed, never the reverse. **This corrects the earlier note** (from a single dataset) that the simulation is conservative for the elastic net: it is not, at these settings.
* Proposed consequences: decision D-20 (pending).

## 2026-10-19 · Pre-freeze decisions; Check-in 2

**Decisions (Mark).** D-20b (ridge primary; H1 confirmatory at *k* ∈ {1, 2}; every other hypothesis estimation only; δ = minimum detectable effect, not a threshold), D-18b (intercept updating kept), D-21 (tuning budget 40 per class per horizon; GRU 5 seeds), D-23 (living docs back in `docs/`).

**Still open before the freeze.** D-22: check the `verify: true` items in `features.yaml` against the V-Dem v16 codebook (Mark).

**Done.** Pre-registration blanks filled (sections 2, 4, 7, 8); `config/project.yaml` gains a `confirmatory` and a `tuning` block; model docstrings updated to ridge primary; Check-in 2 drafted in `docs/Check in 2/` (Quarto, Jupyter engine, `.venv` kernel; reads `data/processed/` and the audit CSVs; no predictor-outcome relationships shown).

**Codebook check (D-22b).** Passed against the V-Dem v16 codebook; `verify` flags removed from `features.yaml`.

**Next.** Mark: commit → tag `prereg-v1` → push. Then the model zip.

