# Decisions log

One line of rationale per decision; full reasoning in `plan_of_record.md` (cited by section) or in the progress log. A decision is changed by adding a new entry, never by editing an old one.

Status: **D** decided · **P** pending (needs Mark's call before the pre-registration freeze) · **I** implementation choice (made while coding; open to override)

| ID | Date | Status | Decision | Why / where |
|---|---|---|---|---|
| D-01 | 2026-10-09 | D | Democratic episodes = ERT `aut_ep_prch == 1`; regime read in the pre-episode year | ERT's own convention; 101 episodes, not 110 (plan 3.2) |
| D-02 | 2026-10-09 | D | Risk sets use only what was knowable at *t* (real-time states) | Final ERT coding backdates starts and knows end years (plan 3.3) |
| D-03 | 2026-10-09 | D | `v2x_regime` and all EDI-derived variables only in the full set | RoW is built from EDI thresholds (plan 4.1) |
| D-04 | 2026-10-09 | D | H2a = incremental skill via drop-group refits; H3 = relative mechanical share | Original H2/H3 true under persistence alone (plan 2.1) |
| D-05 | 2026-10-09 | D | Common window for *k* ≤ 5; *k* = 10 exploratory | *k* = 10 shrank the locked window to 2005–2013 (plan 3.6) |
| D-06 | 2026-10-09 | D | Training population (P1–P3) chosen on development folds | V-Dem already exceeds 10,000 rows (plan 3.7) |
| D-07 | 2026-10-09 | **P** | **Stage-1 event: `confirmation` (recommended) or `onset`** | See note D-07 below |
| D-08 | 2026-10-09 | **P** | **Stage 2 / H5: real-time stage 2 is too small; choose a re-scope** | See note D-08 below |
| D-09 | 2026-10-09 | I | V-Dem v16 taken from the ERT v16 package data (`vdem.RData`), pinned to ERT commit `7fd51ce` | Exactly the data ERT v16 was built from; byte-reproducible download from GitHub; no web form |
| D-10 | 2026-10-09 | I | `state_source: ert_truncated` (ERT `get_eps` re-run on data truncated at every year 1900–2025) | Exact real-time status, about 5 s per year; the Python rule agrees on 99.6% of democracy-years and is the fallback |
| D-11 | 2026-10-09 | I | R writes Parquet with `nanoparquet`, not `arrow` | CRAN binaries for Windows, no C++ toolchain; output verified identical to the Python path |
| D-12 | 2026-10-09 | I | Party arm uses V-Dem's own `v2xpas_*_government` indices instead of a V-Party merge | No governing-party matching needed; same 1970–2019 coverage. Covers only 16% of democracy-years: candidate to cut (see note) |
| D-13 | 2026-10-09 | I | Label maturity buffer applies to positives and negatives alike | Dropping only negatives near the data end would inflate the base rate |
| D-14 | 2026-10-09 | I | Coups (`e_pt_coup*`) and GDP (`e_gdppc`) get coverage rules, not missingness flags | Both flagged by the era-proxy audit (coups start 1950; GDP ends 2019) |
| D-15 | 2026-10-09 | I | Risk sets = RoW democracy **and** ERT `reg_type == 1` (`democracy.require_ert_democracy: true`); add a `row_breakdown` sensitivity outcome (any RoW democracy → autocracy change) | 45 RoW-democracy rows that ERT treats as autocracies could never carry a positive, so their coups were labelled 0. See note D-15 |
| D-16 | 2026-10-09 | I | Era-proxy audit also flags series that **end early** (≥ 95% missing in the last 6 years); new development-vs-locked coverage audit (`window_coverage`) | The correlation rule missed `v2xpas_*` (corr −0.16, 100% missing 2020–2025). See note D-16 |
| D-17 | 2026-10-09 | **P** | **GDP after 2019: drop `e_gdppc`, or replace it with a series that runs to 2024** | "Years since observed" is ~0 in training and 1–5 in the locked window. See note D-16 |
| D-07a | 2026-10-09 | **P** | **Addendum to D-07: the `confirmation` event is mostly breakdown in the development years** | Changes the D-07 recommendation. See note D-07a |
| D-07b | 2026-10-09 | D | **Resolves D-07 / D-07a.** H1 event = any real-time declaration (`confirmation`). New **H1b**: the H1 models scored on gradual declarations only (declared while still a democracy); pass/fail or estimation-only set by the power simulation. Locked test refit **annually** with hyperparameters frozen from development; the frozen-at-2004 fit is a sensitivity | Gradual backsliding has 7 events before 2005 and 40 after: too few to train on, enough to test transfer. See note D-07b |
| D-08b | 2026-10-09 | D | **Resolves D-08.** H5 = real-time breakdown forecasting on all stage-1 democracies (option A), *k* chosen by the power simulation; conditional stage 2 with hindsight membership is exploratory (option B, E1–E3); real-time stage 2 is descriptive, and the 51/71 timing result is reported | Mark agreed with the recommendation in note D-08 |
| D-12b | 2026-10-09 | D | **Party arm cut** (`v2xpas_*` removed from `features.yaml`; E12 dropped) | 16% coverage, ends 2019 (notes D-12, D-16) |
| D-17b | 2026-10-09 | D | **`e_gdppc` dropped** from the predictors | Ends 2019; carry-forward is out of distribution in the locked window (note D-16) |
| D-18 | 2026-10-09 | I | **Intercept updating** before each annual forecast: every model's predictions (persistence included) are shifted on the logit scale to match the event rate of the last 10 years of labels known at *t* (`evaluation.recalibration_window: 10`) | The rate triples between development and the locked window; models trained on mostly low-rate history under-predict (simulated mean forecast 1.5–2% vs true 3%). Uses only known labels. See note D-18 |
| D-20 | 2026-10-10 | P → D-20b | **Confirmatory family after the power simulation**: (a) primary model = ridge logit (elastic net and lasso stay in the lineup); (b) H1 at *k* ∈ {1, 2} (intersection), *k* = 3 and 5 reported as the decay curve with intervals; (c) H1b, H2a, H3, H4, H5 estimation only; (d) δ₁ = Brier skill 0.06, the design effect (β = 2, oracle AUC 0.82) | See progress log 2026-10-10 |
| D-19 | 2026-10-09 | I | **Power-simulation design**: planted signal = mean of 6 clean leading features; mechanical signal (−3-year EDI change) fixed at β = 0.5; event rates per period from the audits; annual refits + intercept updating as in the real design; ridge logit as a fast stand-in for the elastic net (checked in notebook 04) | See note D-19 |
| D-20b | 2026-10-19 | D | **Resolves D-20.** (a) Primary confirmatory model = **ridge logit**. Elastic net, lasso and the rest of the lineup are still fit, tuned and reported (Mark wants as many models tested as possible; H4 compares classes). (b) **H1 confirmatory at *k* ∈ {1, 2}** (intersection); *k* = 3 and 5 reported as the decay curve with intervals. (c) **H1 is the only confirmatory hypothesis**; H1b, H2a, H2b, H3, H4 and H5 are estimation only (intervals, no pass/fail), so Holm is not needed. (d) **δ is the minimum detectable effect, not a pass threshold**: the pass rule stays "95% interval excludes the baseline"; δ₁ = Brier skill ≈ 0.06 (β = 2, oracle AUC ≈ 0.82) | Power simulation 2026-10-10: ridge passed where the fixed elastic net failed in 28 paired cases, never the reverse, so simulated power applies to ridge. H1 power at β = 2: all *k* 32%, {1, 2} 76%, *k* = 1 85%. H2a *k* = 3 62%; H5 ≤ 47% at β = 3. Naming several primary models would split α and lower power |
| D-18b | 2026-10-19 | D | **Intercept updating (D-18) kept** (`recalibration_window: 10`) | Without it, models under-predict the locked window (1.5–2% vs 3%); persistence gets the same shift |
| D-21 | 2026-10-19 | D | **Tuning budget:** 40 configurations per model class per horizon (a 40-point grid for the penalised logits, random search otherwise); GRU searched with one seed, best configuration re-run with 5 seeds; bootstrap B = 2,000 | Equal budgets keep the bake-off fair (plan 6.5); 40 matches Study 1 |
| D-22 | 2026-10-19 | **P** | **Codebook check before the freeze:** the `verify: true` items in `config/features.yaml` (legislative-constraint codes; `v2regsupgroups_2/_3/_5` category numbering) against the V-Dem v16 codebook. **Must be done before tagging `prereg-v1`** | Mark to check; a wrong code would be frozen into the pre-registration |
| D-20c | 2026-10-19 | D | **Clarifies D-20b (d):** δ₁ ≈ 0.06 Brier skill is the *design effect* (β = 2), where the confirmatory H1 rule has 76.5% power; it is not strictly the 80%-power MDE, which lies between β = 2 and 3 on the simulation grid. Still not a pass threshold | Found when notebook 04 computed the joint *k* = 1 and 2 rule (`h1_confirmatory.csv`) |
| D-22b | 2026-10-19 | D | **Resolves D-22: codebook check passed, no changes needed.** Legislative codes are exactly the four inputs of `v2xlg_legcon` (codebook 2.2.11); `v2regsupgroups_2/_3/_5` = party elites / business elites / military (3.5.1.7). No clean-set variable is a source of `v2x_polyarchy` or of its five components (`v2x_freexp_altinf`, `v2x_frassoc_thick`, `v2xel_frefair`, `v2x_suffr`, `v2x_elecoff`); courts, legislature and `v2clkill` feed `v2x_liberal` / `v2xcl_rol` only. Direction: higher `v2cacamps` = more polarized, higher `v2caviol` = more violence | Mark supplied the V-Dem v16 codebook |
| D-23 | 2026-10-19 | D | Living docs (plan of record, decisions log, progress log) live in `docs/`; `docs/Check in N/` folders hold only check-in material | Fixes README links; the freeze checks only `config/` and `prereg/` |

## Note D-07 · What does stage 1 predict? (pending)

The plan v2 text says stage 1 predicts **onset**: an episode *starts* in *t*+1 … *t*+*k*. Building the labels exposed a problem with that.
- An episode's start year is only knowable once the decline reaches 0.10. The audit shows this takes a median of 2 years and up to 10.
- So the rows between start and confirmation are either labelled 0 (`onset_unconfirmed_rows: zero`), which penalises a model for correctly flagging a slide that is under way, or dropped (`drop`), which brings back hindsight.
- And the onset label needs a 7-year maturity buffer (90th percentile of start→confirmation lags), which removes most of the recent locked-test origins.

**Recommendation: `stage1_event: confirmation`.** The event is "ERT first codes the country as being in a democratic episode within *k* years" (a real-time declaration from the truncated re-runs).
- It is observable the year it happens, so no buffer is needed.
- Every at-risk row is labelled consistently.
- The research question becomes "will this democracy be in a recognised backsliding episode within *k* years?", which is also what a user of a warning system would act on.

Both options are implemented. Switching is one line in `config/project.yaml`.

## Note D-08 · Stage 2 and H5 (pending)

The audit (`results/dev/audits/breakdown_timing.csv`) shows:
- **51 of the 71 breakdowns happen at or before the point where the episode could be confirmed in real time**, and 23 happen in the episode's first year (coups, self-coups).
- Real-time stage 2 (democracies inside a *known* episode) therefore has about 148 country-years in total.
  - Development: 21 observable rows, 6 events at *k* = 1.
  - Locked test: 53 rows, 7 events.
- That is too small for confirmatory ML.

Options (not exclusive):
- **A. Breakdown forecasting on all democracies (real-time).** Label every stage-1 row with "breakdown within *k* years" (`stage1_breakdown_target: true`, already built).
  - Development: 35 events at *k* = 1, 181 positive rows at *k* = 5.
  - Locked test: 13 events at *k* = 1, 56 positive rows at *k* = 5.
  - This keeps a fully ML breakdown task with the whole lineup, close to V-Forecast's design.
- **B. Conditional stage 2 ("given a slide has begun").** Use hindsight membership (`state_source: hindsight` for stage 2 only): about 393 country-years from 65 episodes. This is honest as a conditional analysis but not a real-time forecast. The survival models and the Boese et al. baseline live here; it fits as exploratory.
- **C. Keep real-time stage 2** as a descriptive case study (20 forecastable breakdowns).

Recommendation: **A as H5 (at *k* = 3 or 5, chosen by the power simulation), B as exploratory ML (E1–E3), and report the 51/71 timing result itself.** That breakdowns mostly come before a slide is recognisable is a substantive finding for the paper and the talk.

## Note D-07a · The stage-1 "confirmation" event is mostly breakdown before 2005 (pending)

Found in the zip-1 review (audit A9, `results/dev/audits/declaration_timing.csv`).
- 54 of the 101 real-time declarations come in a year when the country is **already not a democracy**: the slide is first recognised at or after breakdown. 38 of the 71 breakdowns are declared in the breakdown year itself (mostly coups).
- By period: development 35 of 42 declarations come at or after breakdown; later years 19 of 59.
- In the labels (stage 1, k = 1): only **7 of 36** development positives are declarations made while the country is still a democracy, against **40 of 53** in the locked test.
- Overlap with the stage-1 breakdown target: 36 of 66 declaration positives are also breakdown positives at k = 1; 213 of 309 at k = 5.

Consequences:
- A stage-1 model trained on 1900–2004 learns mostly to predict coups and sudden breakdowns, then is tested on gradual backsliding. H1 would partly measure that change in the event, not forecasting skill.
- H1 and H5 are close to the same test; E1 (do onset and breakdown drivers differ?) is weakened.

Options:
- **A. "Declared while still a democracy"** as the stage-1 event, with breakdown as a competing outcome (H5). Cleanest match to the Paxton framing. Only 7 development events at k = 1 and 34 at k = 5, so it probably forces k = 5 as the primary horizon or a different training window. Needs the power simulation.
- **B. Keep `confirmation`, reframe** stage 1 as "any recognised autocratization event (gradual or abrupt)", and make breakdown (H5) the main breakdown question. State the composition shift in the pre-registration.

Recommendation: decide after the power simulation. Adopt A if 34 development events at k = 5 support a detectable δ; otherwise B. The original D-07 recommendation (`confirmation`) stands only in form B. Implementing A is a new event type in `episodes.country_events` (a declaration whose year is stage 2).

## Note D-07b · Stage-1 event, H1b and annual refits (decided)

- **H1 stays on `confirmation`**: "ERT recognises an autocratization episode in this democracy within *k* years". Observable the year it happens (buffer 0), enough events (42 development declarations, 53 locked positives at *k* = 1), and what a warning system acts on. The pre-registration states that development events are mostly abrupt (35 of 42 declared at or after breakdown).
- **H1b**: same models, same rows, scored on gradual events only. Positives = rows with a `declaration_dem` event in the window; rows whose only event is abrupt are removed from the H1b scoring set. Locked-window *k* = 1: 40 gradual vs 13 abrupt. It asks whether warning signs learned mostly from coups transfer to executive aggrandizement (an out-of-era test inside the locked window). Pass/fail only if the power simulation supports it; otherwise estimation only.
- **Annual refits**: for each locked origin *t*, train on every label known at *t* (origins ≤ *t* − 1 − *k* − lag), hyperparameters frozen from development. No leakage; mirrors how a warning system would run; lets models see modern cases as they accumulate (about 14 gradual events by 2015). Cost: about 20 refits per model × horizon × seed. Frozen-at-2004 is the sensitivity; the gap between them is reported.
- **Rejected**: `onset` (back-dated starts, 7-year buffer, unlabelable rows); strict gradual event as the primary (7 training events).

## Note D-18 · Intercept updating (implementation choice, open to override)

- Found while building the power simulation. With annual refits, a model at origin 2010 is still trained mostly on 1900–2003, when declarations ran at about 1.1% per stage-1 year; the locked window runs at about 3.0%. Simulated forecasts averaged 1.5–2% against a true 3%. Under-prediction costs Brier skill even when the ranking is good.
- Fix: before scoring origin *t*, compute the logit shift that makes each fitted model's mean prediction on its own most recent 10 years of training rows equal the observed rate there, and apply it to the forecast. Same treatment for persistence, so the H1/H5 comparison stays fair.
- Effect in a pilot (β = 2, *k* = 1): Brier skill vs persistence 0.059 → 0.078 and 0.071 → 0.073; mean forecast moves from ~1.5% to ~2.3%. Small cost in pooled AUC (yearly shifts change cross-year ranking).
- Alternatives: no updating (plan v2); recency-weighted training. Override by setting `recalibration_window: 0`.

## Note D-19 · Power simulation design

- **Why synthetic labels:** the plan orders the pre-registration before any development run, so the simulation must not fit models to real outcomes. It reads the labels table only for its `observable` column, plus event rates per period (already in the audits).
- **Signal:** s = standardised mean of `v2cacamps`, `v2caviol__d3`, and minus the 3-year changes of `v2x_jucon`, `v2xlg_legcon`, `v2exrescon`, `v2regsupgroupssize`. In the real data corr(s, −ΔEDI₃) = 0.72: the leading signs move with the slide. This is why the clean model's edge over persistence is small unless β is large, and it is itself relevant to H2a.
- **Scale of β:** at *k* = 1, β = 1 gives an oracle AUC of only about 0.65–0.73 and oracle Brier skill about 0.02–0.05 over persistence. β = 2 gives about 0.82–0.84 and 0.19. Notebook 04 reports the oracle AUC at each MDE so the δs can be read in AUC terms.
- **Stand-in:** ridge logit (lbfgs, C = 0.01), about 300× faster than saga elastic net. Notebook 04 §4 compares both on the same datasets.
- **Not simulated:** H3 (needs a decay-by-horizon process) and H4 (needs boosting and the GRU). Recommendation: report H3 and H4 as estimation only unless the model zip adds them to the simulation.

## Note D-15 · Risk-set definition: RoW vs ERT regime type

Found in the zip-1 review (audit A8, `results/dev/audits/row_transitions.csv`).
- Stages used RoW (`v2x_regime ≥ 2`). ERT decides whether an episode starts in a democracy with `reg_type`, and codes democratic breakdowns only inside such episodes.
- 45 RoW-democracy rows have ERT `reg_type == 0`. They include the years before Argentina 1976, Ghana 1981, Fiji 2006, Madagascar 2009, Libya 2014 and Mali 2018; ERT files those coups as "regressed autocracy" (outcome 5), so they were labelled 0 on every target.
- **Fix (implemented):** `democracy.require_ert_democracy: true` removes those rows (stage 1: 5,076 → 5,031; stage 2 unchanged at 148). Event counts are unchanged; only mislabelled negatives go. `states.row_democracy` keeps the RoW-only flag for diagnostics.
- RoW counts 104 democracy → autocracy changes since 1900; ERT counts 71 as democratic breakdowns. Of the other 33, 11 revert within 2 years (near-threshold flicker); 22 last. Some of the lasting ones are in ERT democracies but never reach an ERT episode (Kenya 2017, Nigeria 2021, Georgia 2008, Montenegro 2013, Sierra Leone 2023).
- **Sensitivity (implemented):** `labels.stage1_row_breakdown_target: true` adds the event `row_breakdown` (any RoW change, from `row_regch_event == −1`) on stage-1 rows, buffer 0. k = 1: development 42 positives, locked 26 (vs 35 and 13 for ERT breakdowns; the locked counts are not on the same origin range because the ERT target has a 5-year buffer).
- For a fully RoW-based sensitivity, run with `require_ert_democracy: false` as well (the `row_breakdown` target on the reduced risk set cannot see the reg_type-0 coups).
- To pre-register: the primary breakdown outcome is ERT's; `row_breakdown` is a sensitivity analysis; near-threshold RoW drops are excluded by ERT's definition, and the paper says so.

## Note D-16 · Coverage that ends before the locked window

Found in the zip-1 review (audits A7 and A7b, `era_proxy_flags.csv`, `window_coverage.csv`).
- `era_proxy_flags` now has a second rule, `ends_early`, and a `reason` column. Flagged: `e_pt_coup*` (corr), `e_gdppc` (corr + ends_early), `v2xpas_*` (ends_early; previously missed).
- `window_coverage` compares missing shares in development vs locked democracy rows and flags any series ending before the last locked origin (2024): `e_gdppc` (missing 1.5% in development, 24.5% in the locked window) and `v2xpas_*` (end 2019).
- **D-17 (pending):** the `carry_forward_with_age` rule for GDP produces a feature that is out of distribution in the locked window. Options: drop `e_gdppc` from the clean set; or replace it with a series that runs to 2024 (would need a new pinned source). Recommendation: drop it unless a pinned replacement is easy, and keep GDP as a context variable in descriptive tables only.
- `v2xpas_*`: strengthens the case in note D-12 for cutting the party arm.

## Note D-12 · Party arm coverage

`v2xpas_*` cover 1970–2019 and only about 16% of democracy country-years. A party arm fit on that subset will be small and will end before the locked window's last 5 years. It is already near the top of the cut order; consider cutting it outright once the power simulation is done.
