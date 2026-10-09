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

