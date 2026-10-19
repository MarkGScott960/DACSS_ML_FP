# FlyPredictor: Forecasting Democratic Backsliding
### DACSS 756 Final Project: plan of record, version 2 (revised October 9, 2026)

Mark Scott, UMass Amherst DACSS. This file is the plan of record for the backsliding study. It supersedes version 1 (morning of October 9). The live companion doc ([FlyPredictor Project Plan](https://claude.ai/code/artifact/347c360f-aa81-4e8f-b84d-999293f76c3f)) is behind this file until synced.

**How to read this file.**
- Part A is the plan: what will be built and why.
- Part B lists what is still open, ordered by severity.
- Part C holds the data facts and the coverage checks the plan rests on.
- Part D records what changed from version 1 and why.

**Data-blindness note.** The counts in Part C come from ERT v16 outcome variables only. No relationship between any predictor and any outcome has been looked at. The audits in section 7 (coverage, missingness, label maturity, power) also use no predictor-outcome relationship.

---

# PART A: THE PLAN

## 1. Summary

The project forecasts three things for the world's democracies:
- **Onset (stage 1):** will an autocratization episode start within *k* years?
- **Completion (stage 2):** for a democracy already inside an episode, will it break down within *k* years?
- **Severity:** how far will the Electoral Democracy Index (EDI) fall over an episode?

The framing is Robert Paxton's account of fascism as a process with warning signs.

The project is **hypothesis-driven and ML-first**.
- The substantive claim: leading warning signs (polarization, political violence, attacks on courts, weakened legislative constraints, party anti-pluralism) carry forecasting information *beyond* what the trajectory of democracy itself already tells you.
- The ML machinery that tests it:
  - a multi-model bake-off on two forecasting tasks;
  - a clean-vs-full feature audit for circularity;
  - drop-group refits;
  - a horizon-decay curve;
  - calibration;
  - a locked test set scored once.

**Model comparison is the robustness machinery.** A finding that holds across penalized regression, gradient boosting and a recurrent net is not an artifact of one algorithm. The fly connectome appears only as an optional exploratory reservoir row (section 6.4).

## 2. Research questions and hypotheses (for pre-registration)

**RQ1.** Can V-Dem warning signs forecast the onset of autocratization in democracies, and how fast does that skill decay with horizon?

**RQ2.** Do Paxton's *leading* signs add forecasting information beyond the outcome's own trajectory, especially at long horizons, while the outcome's own components mainly add short-horizon "momentum"?

**RQ3.** Is breakdown predictable once an episode has begun, and do its drivers differ from the drivers of onset?

### 2.1 Confirmatory hypotheses

These are scored once on the locked test set.
- **Primary model:** ~~elastic net~~ **ridge logistic regression** (D-20b, 2026-10-19). **Only H1 (at *k* ∈ {1, 2}) is confirmatory**; H1b, H2a, H2b, H3, H4 and H5 are estimation only. The pre-registration (section 2) supersedes the table below.
- **Multiple comparisons:** Holm correction across H1 to H5.
- **Intervals:** country-clustered bootstrap (resample countries).
- **Thresholds:** the minimum effect worth reporting (δ) for each hypothesis is set from the power simulation (section 7.2) before the pre-registration is frozen.

| ID | Hypothesis | Test / decision rule |
|---|---|---|
| **H1** | Stage-1 **clean** models beat the persistence baseline at every *k* ∈ {1, 2, 3, 5}. | Brier skill score > 0 **and** PR-AUC ÷ base rate above the baseline's, each with a 95% interval excluding the baseline, at every *k*. This is an intersection test: all horizons must pass, so no correction is needed within H1. |
| **H2a** *(primary RQ2 test)* | Leading-sign groups add skill to the **full** model at long horizons. | Drop-group refit: Δ log loss from removing **all leading groups** from the full model at *k* = 3 and *k* = 5. The interval must exclude 0 in the direction of "leading groups help", by at least δ₂. |
| **H2b** *(secondary pattern)* | Lagging (EDI-component) groups gain relative importance as *k* shrinks. | Drop-group Δ for the lagging groups, divided by the total of all group Δs, at *k* = 1 vs *k* = 5. **Caveat stated in advance:** plain persistence predicts this pattern too, so H2b alone is not evidence for Paxton. It is reported to describe the decay, and H2a carries the claim. |
| **H3** | The *relative* mechanical share falls as *k* grows. | Relative share = (full − clean) ÷ (full − baseline), in Brier skill. The share at *k* = 1 must be greater than the share at *k* = 5, with an interval. The relative form is used because absolute gaps shrink automatically as all models approach the baseline. |
| **H4** | The H2a result holds across model classes. | The sign of the H2a contrast agrees in elastic net, gradient boosting and the GRU, and the interval excludes 0 in at least two of the three. |
| **H5** | Stage-2 **clean** models forecast breakdown inside episodes better than persistence at *k* = 1. | Brier skill score > 0 and PR-AUC ÷ base rate above baseline, with the 95% interval excluding the baseline. Power is low; the hypothesis is stated anyway so that the stage-2 result is confirmatory, not post hoc. |

### 2.2 Exploratory analyses (reported, not confirmatory)

- **E1.** Onset vs completion drivers: compare drop-group profiles across stages 1 and 2.
- **E2.** Severity regression (EDI decline per episode).
- **E3.** Survival-model family for stage 2 (random survival forest, gradient-boosted Cox).
- **E4.** Kitchen-sink comparator: what does theory-driven selection cost or buy?
- **E5.** Stacked ensemble.
- **E6.** *k* = 10.
- **E7.** Out-of-era test.
- **E8.** Vintage (hindsight) test.
- **E9.** External outcomes.
- **E10.** V-Forecast reproduction.
- **E11.** Error disparities.
- **E12.** Party-augmented arm.
- **E13.** Optional reservoir row, including the fly.

### 2.3 Stated threat to the theory test

The leading-sign emphasis in the modern literature (Bermeo 2016; Levitsky and Ziblatt 2018) was built largely on cases that fall inside the locked test window: Hungary, Poland, Turkey, Venezuela. Paxton built his account from the interwar cases in the out-of-era panel. Neither window is fully out of sample *for the theory*. The development period (1970 to 2004) is the cleanest, so development-fold results are reported alongside the locked test.

## 3. Data, unit and sample

### 3.1 Unit and sources

**Unit.** Country-year, 1900 to 2025 (set by ERT coverage).

| Dataset | Role | Coverage notes (Part C) |
|---|---|---|
| [V-Dem Country-Year v16, Full+Others](https://www.v-dem.net/data/the-v-dem-dataset/) | Features; structural covariates; external outcomes | Use measurement-model point estimates, not `_ord` or `_osp`. |
| [ERT v16](https://github.com/vdeminstitute/ERT) | Outcome labels: onset, breakdown year, censoring, severity | R package; the CSV is also shipped. Default thresholds: start −0.01, cumulative −0.10, tolerance 5 years. |
| [V-Party v2](https://www.v-dem.net/data/v-party-dataset/) | Party-arm features only (E12) | **1970 to 2019 only.** |
| [V-Dem archive](https://www.v-dem.net/dsarchive.html) | Vintage test (v10) | Needs a v10 ↔ v16 crosswalk. |

### 3.2 Defining democratic episodes (revised)

- A **democratic episode** is an ERT autocratization episode with `aut_ep_prch == 1`: one that begins in a democracy and could produce a breakdown.
- Regime type is read in the **pre-episode year**, which is ERT's own convention.
- Episodes with `aut_ep_outcome == 5` ("regressed autocracy") started from autocracy in the pre-episode year. They are not democratic onsets. Version 1's 9 "anomalies" were this.
- **Onset year** = ERT episode start year.
- **Breakdown year** = first year with `aut_ep_pbr == 1` inside the episode. Unit-test it against `aut_ep_outcome == 1`.
- **Severity** = total EDI decline from the pre-episode year to episode end.
- Part C counts must be recomputed under these definitions. The expected number of democratic onsets is about 101, not 110.

### 3.3 Real-time at-risk definition (revised)

Version 1 dropped every row "inside an episode". That uses hindsight: an episode is confirmed only once the cumulative EDI decline reaches −0.10, often years after its start.

- **Primary rule (real-time):**
  - A row at origin *t* is at risk for stage 1 if the country is a democracy at *t* and is **not in an episode that had already crossed the −0.10 confirmation threshold by year *t***.
  - Rows between an episode's start and its confirmation stay in the at-risk set with label 0. Their episode started at or before *t*, so no onset falls in *t*+1 … *t*+*k*.
  - A "decline since last EDI peak" feature lets the model see an ongoing slide. It is EDI-derived, so it goes in the full set only.
- **Gold-standard check:** re-run ERT `get_eps` on V-Dem truncated at a sample of origin years. This verifies that the primary rule reproduces real-time episode status.
- **Sensitivity:** version 1's hindsight rule, reported as the "hindsight-filtered" variant.
- **Stage 2 at-risk:** origin *t* is a year inside a democratic episode, before any breakdown, with the country still democratic at *t*.

### 3.4 Labels

- **Stage 1:** `y₁(t, k) = 1` if a democratic-episode onset falls in *t*+1 … *t*+*k*.
- **Stage 2:** `y₂(t, k) = 1` if the episode's breakdown year falls in *t*+1 … *t*+*k*.
  - Ongoing (censored) episodes contribute every episode-year whose label window is fully observed.
- **Severity:** one row per completed democratic episode, with features at the pre-episode year.

### 3.5 Label maturity (data-derived)

ERT confirms an onset only after the −0.10 decline. It confirms a breakdown only after a founding autocratic election or 5 years of autocracy. Recent windows therefore hold false negatives.

- **Stage 1 buffer:** default 2 years. Final value set by the audit: the 90th percentile of the time from episode start to confirmation, over historical episodes.
- **Stage 2 buffer:** default 5 years, set the same way from historical time from breakdown to confirmation.
- **Rule:** drop origins where *t* + *k* > 2025 − buffer.
- **Sensitivity:** buffers of 0 and 4 years for stage 1; 3 and 7 years for stage 2.

### 3.6 Horizons

- **Stage 1:** *k* ∈ {1, 2, 3, 5} confirmatory. *k* = 10 is exploratory (E6), reported mainly from development folds, because it leaves almost no locked-test origins.
- **Stage 2:** *k* ∈ {1, 2, 3}. Episodes are short.
- **Search and selection:** hyperparameters are searched within each horizon; *k* is never chosen to maximize a score.
- **Two versions of every decay curve:**
  1. **Common window:** all confirmatory *k* scored on the same forecast origins. This is the headline.
  2. **Maximal window:** each *k* scored on every origin it can use.

### 3.7 Training population is a hyperparameter

The course's 10,000-row minimum is met by V-Dem itself (about 28,000 country-years), so the choice of training population is made on learning grounds. Three candidates:

| ID | Training population |
|---|---|
| **P1** | All regimes (rows not in a confirmed episode). Positives include autocracies sliding further. |
| **P2** | Democracies plus electoral autocracies. |
| **P3** | Democracies only. |

Each model class picks among P1 to P3 on development folds, by log loss **on democracy rows only**, before the lock. Scoring is always on democracy at-risk rows. The losing populations are reported as robustness.

## 4. Features

### 4.1 Paxton groups and the clean/full split

ERT episodes are defined from the EDI (`v2x_polyarchy`). Predictors built into the EDI, or derived from it, are mechanically tied to the outcome.

| Group | Indicators | Inside EDI? | Set |
|---|---|---|---|
| Polarization | `v2cacamps` | No | Clean (leading) |
| Political violence | `v2caviol`, `v2clkill` | No | Clean (leading) |
| Courts / constitution | `v2x_jucon`, `v2juhcind`, `v2juncind`, `v2juhccomp`, `v2jureview`, `v2exrescon` | No (liberal component) | Clean (leading) |
| Legislative constraints | `v2xlg_legcon` and components | No | Clean (leading) |
| Elite alignment (proxy) | Regime support groups (`v2regsupgroups` family; **verify names in v16**) | No | Clean (leading) |
| Party anti-pluralism (Eco) | V-Party anti-pluralism, populism, demonization, encouragement of violence, `v2paviol` | No | **Party arm only (E12)** |
| Structural | GDP per capita level and growth; coups and coup attempts | No | Clean (context) |
| Media | `v2mecenefm`, `v2meharjrn`, `v2meslfcen`, `v2mebias`, `v2mecrit`, `v2merange` | **Yes** | Full only (lagging) |
| Association / opposition | `v2csreprss`, `v2psoppaut`, party bans and barriers, CSO entry/exit | **Yes** | Full only (lagging) |
| Elections | Election management body autonomy and capacity, vote buying, irregularities, intimidation, free and fair | **Yes** | Full only (lagging) |
| Outcome trajectory | EDI level; 1- and 3-year EDI change; decline since last peak | Is the EDI | Full only |
| Regime subtype | Electoral vs liberal democracy flag | Derived from EDI thresholds | Full only |

**Two pre-registered feature sets:**
- **Clean:** every "Clean" row above. It contains no EDI-derived variable. `v2x_regime` is **removed** from the clean set, because Regimes of the World is built partly from EDI thresholds.
- **Full:** clean plus every "Full only" row. This is the best-possible forecast.

**Stated limitation (coder halo).** V-Dem's experts rate courts, media and elections in the same survey, and the measurement model re-estimates history with each release. "Clean" features can therefore carry some of the coders' overall sense of decline, and some hindsight. The vintage test (E8) measures the hindsight part.

**Before freezing:** check each code's EDI membership against the v16 "Structure of Indices" document.

### 4.2 Construction

- **Timing.** Every feature at origin *t* uses only years ≤ *t*. Note that V-Dem year-*t* data are released around March of *t*+1. This matters for the present-day map, not for backtests.
- **Levels plus changes.** Each indicator enters as its level, its 1-year change and its 3-year change.
- **Coups (`e_pt_coup`, `e_pt_coup_attempts`).** These are structurally missing before 1950, if the audit confirms coverage starts then. They enter with a "pre-coverage" indicator and are excluded from the out-of-era model.
- **GDP (`e_gdppc`).** If the series ends before 2025, carry the last value forward with a "years since observed" feature. Median imputation with a missingness flag is ruled out here, because the flag would act as a proxy for recent years, which also have the highest base rate.
- **Other missingness.** Median imputation fit inside each training fold, plus a missingness indicator per feature group. The audit (section 7.2) checks every flag for era-proxy behavior: if a flag mostly marks a period, it is replaced by a coverage rule.
- **Selection, not reduction.** Inputs are the original variables; no PCA on inputs. Lasso and elastic net select within folds, and selection stability is reported. PCA and clustering are used descriptively (section 11).

## 5. Tasks and models

### 5.1 Stage 1: onset (primary task)

- Binary forecast on real-time at-risk democracy rows, for each *k*.
- The full model lineup (section 6) runs here.

### 5.2 Stage 2: breakdown within an episode (second forecasting task; revised)

- **Unit:** episode-year.
- **Size:** several hundred rows. About 71 rows are breakdown events.
- **Pipeline:** the same lineup, harness, clean/full split, drop-group refits and calibration as stage 1.
- **Natural fit for the GRU:** each episode is a short sequence, and the trajectory inside the episode is the signal.
- **Clean/full for stage 2:** inside an episode the EDI is falling by definition, so the full model looks strong for mechanical reasons. The clean-vs-full gap here is itself a finding: how much of breakdown is predictable beyond the slide itself?
- **Survival family (E3):**
  - random survival forest and gradient-boosted Cox (`scikit-survival`; **check Python 3.14 wheels**, fall back to `lifelines`);
  - a classic Cox model as the linear reference;
  - scored by concordance index and integrated Brier score.
- **Theory model:** the stage-2 specification of Boese et al. (2021) enters the bake-off as a published political-science baseline.

### 5.3 Severity regression (E2)

- **Target:** total EDI decline per completed democratic episode.
- **Models:** elastic net and gradient boosting.
- **Metrics:** RMSE and MAE.

### 5.4 Combined risk

P(breakdown within window) = P(onset) × P(breakdown | onset). Stage-2 features are measured inside the episode and stage-1 features at origin, so this product is approximate. It is shown in the paper with that label and is **not** a headline number in the demo.

### 5.5 Robustness formulations (cut-able)

- **Ordinal ladder** (0 no episode / 1 episode, democracy survives / 2 breakdown): ordinal logit plus multiclass gradient boosting, with a proportional-odds test.
- **Three-class** (autocratization / democratization / stable): one run.

## 6. Model lineup

Every model emits probabilities on the same scoring rows and is scored by the same metrics.

### 6.1 Baselines

1. **Persistence baseline:** the base rate within pre-registered bins of regime subtype × recent EDI trend. Bins: 3-year EDI change in terciles fixed on development data.
2. **EDI-momentum logit:** EDI level plus 1- and 3-year EDI changes. This is the skeptic's baseline: "the slide predicts the slide."

### 6.2 Main lineup (stages 1 and 2)

3. **Penalized logistic regression:** lasso, ridge and elastic net, with regime × feature interactions when the chosen training population is P1 or P2. **Ridge is the confirmatory primary model (D-20b); the elastic net and lasso stay in the lineup.**
4. **Gradient boosting** (LightGBM or XGBoost; small grid around sensible defaults).
5. **GRU** over each country's history (stage 1) or each episode's history (stage 2). Use a small hidden size and strong regularization. Sequences are truncated at each fold's training cutoff, and the model is unidirectional only.
6. **Kitchen-sink gradient boosting (E4):** all V-Dem indicators, with the same lag rules. This tests the theory-driven feature set against "throw everything in".
7. **Stacked ensemble (E5):** a logistic meta-learner over the base models' out-of-fold predictions, fit on development folds only. One more contestant, not the primary model.

### 6.3 Stage-2 additions

8. Survival family (section 5.2).
9. Boese et al. theory model.

### 6.4 Optional reservoir row (E13)

- **Contents:** fly connectome reservoir, a matched random reservoir and a no-recurrence leaky bank, fed current-year values only.
- **Condition:** runs **only if** the network-layer repair (gate G1) is finished by code freeze.
- **Expected result:** a tie, as in Study 1 Tier 1. The task supplies engineered history features, and short annual series need little reservoir memory.
- **Framing:** "does this task need reservoir memory?", not a test of the fly's wiring.

### 6.5 Shared training rules

- **Tuning objective:** log loss. PR-AUC is too noisy to tune on.
- **No class reweighting.** If any model needs it to train, it is recalibrated afterwards.
- **Equal tuning budget per model class.** Random search, with small grids.
- **One seed set for every model class.** Neural models use 5 seeds and report the mean.

## 7. Validation design and order of operations

### 7.1 Splits

**Development period:** forecast origins up to 2004.
- Rolling-origin, expanding-window folds.
- 5-year validation blocks from 1970 to 2004.
- A gap of *k* years between the last training label and the first validation origin.

**Locked test (scored once):** forecast origins 2005 to 2024, subject to the horizon and maturity cut-offs.
- Training for the locked run uses origins up to 2004 − *k*. Stage 2 uses the same origin-year split.
- Expected size: about 40 to 55 stage-1 democratic onsets after cut-offs. Development has about 30.

**Distribution shift is expected and reported.** Development onsets are mostly post-communist, Latin American and African transitions. The test window is dominated by executive aggrandizement.

### 7.2 Pre-freeze audits (no predictor-outcome looks)

1. **Coverage:** confirm the actual end and start years of `e_gdppc`, `e_pt_coup`, the LIED and BMR copies, and every feature.
2. **Missingness:** tabulate missingness by indicator × decade; check whether any missingness flag acts as an era proxy.
3. **Label maturity:** measure the time from episode start to confirmation, and from breakdown to confirmation, to set the buffers.
4. **Episode recount:** recount onsets, breakdowns and censoring under the `aut_ep_prch` definition.
5. **Real-time at-risk check:** confirm the threshold rule against ERT re-runs on truncated data.
6. **Power simulation:**
   - plant known effects of several sizes in synthetic labels on the real panel structure;
   - run the full pipeline;
   - read off the minimum detectable effect for H1 to H5;
   - set each δ from it.

### 7.3 Order of operations (strict)

1. Repository scaffold, environment, data download with a hash manifest.
2. Panel build, ERT labels (stages 1 and 2, severity) and the real-time at-risk set, **with unit tests**.
3. Pre-freeze audits (section 7.2).
4. **Pre-registration frozen:** hypotheses, δs, feature lists, horizons, buffers, locked window, tuning budget, primary model. Commit and tag.
5. Development runs:
   - baselines → elastic net → gradient boosting → kitchen-sink → GRU → stacked ensemble;
   - stage 2, including the survival family and the theory model;
   - severity;
   - ladder;
   - training-population selection.
6. **Leakage gate:** global label-permutation placebo plus a planted-leak test. Then **code freeze**.
7. Optional reservoir row (frozen pipeline; only if G1 is done).
8. **Locked test scored once** for every model.
9. **After the locked test only:**
   - out-of-era test;
   - vintage test;
   - external outcomes;
   - V-Forecast reproduction;
   - hindsight-filtered sensitivity;
   - ERT threshold sensitivity;
   - error disparities;
   - present-day map.
   
   Several of these train on, or score, locked-window years.

## 8. Metrics and inference

| Purpose | Metric |
|---|---|
| Discrimination | PR-AUC ÷ base rate (primary); ROC-AUC (secondary) |
| Probabilistic accuracy | Brier skill score against the persistence baseline; log loss |
| Calibration | Reliability plots on democracy rows; Platt recalibration fit on democracy development rows |
| Operational (course classification metrics) | Precision, recall and F1 at a fixed alert budget (top 10% of risk scores) |
| Event-level | Share of onsets (or breakdowns) ranked in the top 10% in **any** of their *k* lead years. This corrects for overlapping labels, where one onset makes up to *k* positive rows. |
| Survival (E3) | Concordance index; integrated Brier score |
| Regression (E2) | RMSE; MAE |
| Uncertainty | Country-clustered bootstrap for every interval and every paired model-vs-model difference |
| Multiple comparisons | Holm across H1 to H5 |

## 9. Interpretation

- **Primary (confirmatory for H2 and H4): drop-group refits.**
  - Remove a group, refit with frozen hyperparameters, and measure the change in log loss on the locked test.
  - Do this for each Paxton group and for the "all leading" and "all lagging" bundles.
  - Groups differ in size, so each Δ is also reported per feature.
- **Display only: grouped permutation importance.**
  - It understates groups that are correlated with other groups, because the model substitutes one for the other.
  - For the GRU, whole country sequences are permuted, not single rows.
- **Demo: SHAP values from gradient boosting**, showing per-country which warning sign fired.
- **Selection stability:** how often each variable and group is selected across folds. Instability is reported.

## 10. Robustness suite (all after the locked test)

### 10.1 Out-of-era test (E7)

- **Setup:** train stage 1 on 1945 to 2025 with frozen hyperparameters, clean V-Dem features only. Coups are excluded (no coverage before 1950); V-Party is excluded (starts 1970). Score every interwar democracy-year, 1919 to 1939.
- **Framing:** stage 1 scores only *pre-onset* years, so each case's risk path runs up to its onset year.
  - Germany's onset is 1923, so its path covers 1919 to 1922; Weimar 1930 to 1933 is in-episode.
  - **Stage 2 scores the in-episode years.** That is where the Weimar slide appears, through the breakdown model.
- **Panel:** Germany, Austria, Estonia and Spain, plus **every** interwar democracy that had no onset, as the negative-control set. A rule, not a hand-picked pair.
- **Interpretation:**
  - Hit: the process that took the interwar democracies is recognizably today's.
  - Miss: backsliding changed form after 1945 (Bermeo's executive aggrandizement and "promissory coups"), which limits how far the present-day map can be trusted.
  - Either result is reportable.

### 10.2 Vintage test (E8)

- **Setup:** the same frozen code, run twice.
  - **A:** V-Dem v10 (data to 2019) with ERT labels recomputed from v10.
  - **B:** V-Dem v16 cut off at 2019.
- Both forecast onsets from 2020 on and are scored against ERT v16.
- **Precise claim:** B minus A measures the forecast value of revisions made after 2019. It is a lower bound on hindsight, because v10 was itself coded with hindsight for earlier years.
- **Reporting:** descriptive, with intervals, because only about 13 democratic onsets start in the 2020s.

### 10.3 External outcomes (E9)

- **Question:** does high predicted risk precede independently coded breakdowns?
- **Scoring:** event-level, because LIED and BMR code breakdown events, not onset of decline. This validates stage 2 and the combined risk, not stage 1 directly.
- **Sources:** LIED primary. BMR ends in 2020, so it covers the locked window only through 2020.

### 10.4 V-Forecast reproduction (E10)

- **Setup:** run the clean, full and kitchen-sink gradient-boosting models once on V-Forecast's outcome (a year-to-year drop in Regimes of the World), with a 2-year window, 2010 to 2019 test forecasts.
- **Published reference:** AUC-PR 0.45 and AUC-ROC 0.85.
- **Purpose:** pipeline validation against a published benchmark. Run after the lock, because its test years overlap the locked window.

### 10.5 Other robustness checks

- **Training population:** report the two populations not chosen in section 3.7.
- **ERT thresholds:** regenerate episodes with stricter and looser thresholds using the ERT `get_eps` function.
- **Hindsight-filtered at-risk set:** version 1's rule, as a sensitivity check.
- **Ladder and three-class versions:** section 5.5.
- **Measurement uncertainty (optional):** one rerun with features shifted to V-Dem's `_codelow`/`_codehigh` bounds.

## 11. Error disparities and unsupervised analyses (ethics section)

**Error disparities (E11).** Does skill differ by region (`e_regionpol`), income tercile, or era (pre-1945, Cold War, post-1990)?
- V-Dem coder coverage and data quality are uneven across regions.
- A model that only sees backsliding coming in Europe has a fairness problem: it would under-warn exactly where outside attention is lowest.
- Report PR-AUC ÷ base rate and recall at the alert budget per group, with intervals.

**Unsupervised work (course techniques, descriptive only):**
- **PCA** on countries' pre-onset trajectories of the leading-sign groups: do onsets share a common shape?
- **Clustering** of countries by error profile: where and when do models miss, and do the misses cluster by region or era?

Neither feeds back into the forecasting inputs.

## 12. The talk and demo

1. **Hook: the out-of-era panel.** *"A model that has only seen the post-war world: would it have flagged Weimar before 1923, and seen the breakdown coming once the slide began?"* Germany, Austria, Estonia and Spain are shown beside every interwar democracy that held.
2. **Methods in under a minute:** warning signs → onset → breakdown, and why clean vs full matters.
3. **Forecast-decay curve** (common window): how early can you see it coming?
4. **Present-day map, all democracies:**
   - stage-1 onset risk for countries not in a confirmed episode;
   - stage-2 breakdown risk for countries inside one. ERT v16 lists ongoing episodes including the U.S. (2024), Hungary, India, Italy, the U.K., Slovakia and Argentina.
   - Every number is shown with its interval and labelled as a model output.
   - Episode status comes from ERT's coding, not this model.
   - Calibration evidence is shown beside the map.
   - No ranking language unless the intervals separate.
5. **Which warning signs fired:** SHAP for two or three countries.
6. **ML vs theory slide:** the kitchen-sink and Boese et al. baselines against the theory-driven lineup.

## 13. Prior work and what this adds

**Forecasting prior work:**
- [V-Forecast / PART](https://v-dem.net/shiny/VForecast/): adverse regime transitions (Regimes of the World drops) two years ahead; about a 4% base rate per two years; ensemble over V-Dem plus external data; test AUC-PR 0.45 and AUC-ROC 0.85 (2010 to 2019). Working paper: [Varieties of Forecasts](https://gupea.ub.gu.se/handle/2077/60276).
- Djuve, Knutsen and Wig, ["Forecasting Regime Breakdown"](https://coms.events/epsa2025/data/abstracts/en/abstract_0962_0943_0369.html) (EPSA 2025): ML models on a theory-restricted predictor set compared with broader sets.

**Two-stage design:** Boese, Edgell, Hellmeier, Maerz and Lindberg, ["How democracies prevail: democratic resilience as a two-stage process"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8336576/) (*Democratization*, 2021).
- They split resilience into onset resilience and breakdown resilience, on ERT.
- Once autocratization begins, about one in five democracies avert breakdown.
- This project's stage split follows their template; their stage-2 model is a baseline here.

**Theory and context:** Paxton, *The Anatomy of Fascism* (2004); Eco, "Ur-Fascism" (1995); Bermeo, "On Democratic Backsliding" (2016); Lührmann and Lindberg, "A third wave of autocratization is here" (2019); Levitsky and Ziblatt, *How Democracies Die* (2018).

**This project adds:**
1. an **incremental-skill test** of Paxton's leading signs over the outcome's own trajectory (H2a), rather than an importance ranking;
2. a **circularity audit** (clean vs full) on both stages;
3. **real-time labels**: an at-risk set that does not use ERT's backdating;
4. the **horizon-decay curve** for onset and for breakdown;
5. **stage 2 as a forecasting task** on episode-years, with ML and survival models against a published political-science baseline;
6. a **kitchen-sink comparator** and a **published-benchmark reproduction**;
7. **out-of-era and vintage tests**;
8. **error disparities**.

## 14. Schedule

Course weeks are from the Final Project Guide; confirm dates on Canvas. **Check-in 1 (Proposal) has not been submitted yet; Dr. Adler gave verbal approval for the topic.**

| Week | Course | Project work |
|---|---|---|
| 6 (now, to ~Oct 16) | Check-in 1 (Proposal) due soon; confirm what else is due | Submit Check-in 1 on this plan. Repository scaffold. Data download and hash manifest. Panel, labels and real-time at-risk set with unit tests. |
| 7 (~Oct 16–23) | Check-in 2 or 3 (confirm) | Pre-freeze audits and power simulation; literature review (section 13); **freeze the pre-registration**; splits and harness. |
| 8 (~Oct 23–30) | | Baselines, elastic net, gradient boosting, kitchen-sink on development folds. Lighter week: the Team 5 project ends around Halloween. |
| 9 (~Oct 30–Nov 6) | Check-in 4: Model Selection | GRU (stages 1 and 2); stage 2 and survival family; severity; training-population selection; stacked ensemble. |
| 10 (~Nov 6–13) | | Ladder; leakage gate; **code freeze**; optional reservoir row. |
| 11 (~Nov 13) | Check-in 5: Results | Development results; **locked test scored once** (~Nov 20). |
| 12 (~Nov 20–27) | | Robustness suite, V-Forecast reproduction, error disparities, PCA and clustering (Thanksgiving Nov 26). |
| 13–14 (to ~Dec 11) | Workshops; presentations | Paper, figures, reproducibility package, talk rehearsal. |

**Cut order if time runs short** (first cut first):
1. three-class version
2. optional reservoir row
3. ERT threshold sensitivity
4. party-augmented arm
5. ordinal ladder
6. vintage test (cut only to a descriptive minimum)
7. survival family (stage 2 keeps the classification lineup)

**The core is never cut:**
- stage 1 and stage 2 with the clean/full sets;
- the full baseline-to-GRU lineup;
- drop-group refits;
- the locked test;
- the out-of-era panel.

---

# PART B: OPEN ITEMS (severity-ordered)

## Critical

**B1. Confirm the course calendar with Dr. Adler.**
- Which check-in is due now?
- Whether Check-in 1 goes in on this plan.
- Whether the topic change needs written approval beyond the verbal one.

**B2. Does Study 1 (conflict) stay in the final report?**
- Option 1: a short "first result" section with the bounded null and its mechanism.
- Option 2: an appendix.
- Option 3: drop it.

This was not decided in the version 2 review. It affects the paper outline only, not the code.

**B3. R/Python boundary.** ERT and `vdemdata` are R packages, and `get_eps` is needed for:
- the real-time at-risk check;
- threshold sensitivity;
- v10 relabelling for the vintage test.

*Proposed:* Python 3.14 is the main stack (notebooks in VS Code, `src/` modules). A small set of R scripts generates ERT outputs and is called from Python (via `subprocess` and `Rscript`), writing Parquet or CSV files that Python reads. Main labels come from the shipped ERT v16 file. Confirm that R is installed on the working machine.

## High

**B4. Coverage facts still to verify in the audit (section 7.2):**
- the `e_gdppc` end year;
- the `e_pt_coup` start year (expected 1950);
- the end year of V-Dem's LIED copy;
- names and coverage of the regime-support-group variables;
- `v2cacamps` and `v2caviol` coverage before 1945.

**B5. Package availability on Python 3.14:** `scikit-survival`, `lightgbm` or `xgboost`, `shap`, PyTorch (the CUDA build for the RTX 4070). Fall back to `lifelines` and pure-PyTorch implementations where wheels are missing.

**B6. Power is low by construction.** The δs come from the simulation. If the minimum detectable effects are larger than any plausible effect, the pre-registration says so, and H2a or H4 are reported as estimation exercises (intervals, no pass/fail).

## Medium

**B7. Network-layer repair (G1)** gates the optional reservoir row only. It is a side thread and not on the critical path.

**B8. The live companion doc** ([FlyPredictor Project Plan](https://claude.ai/code/artifact/347c360f-aa81-4e8f-b84d-999293f76c3f)) and the older `Final Project Guide.md` and `Project Plan.md` still describe the blind screen and the fly-first design. Mark them superseded by this file.

---

# PART C: DATA FACTS

## C.1 ERT v16 counts (version 1 definitions; recompute under section 3.2)

| Quantity | Value |
|---|---|
| Country-years, all regimes | 19,857 |
| … not inside an autocratization episode (hindsight rule) | 18,549 |
| Democracies plus electoral autocracies (RoW ≥ 1) | 9,968 |
| Democracy country-years (RoW ≥ 2) | 5,224 |
| … at risk (hindsight rule) | 4,829 |
| … at risk, 2005 onward (before horizon and maturity cuts) | 1,637 |
| Autocratization episodes, all regimes | 262 |
| Episodes with democracy in the **start** year (version 1 definition) | 110 |
| … of which `aut_ep_outcome == 5` (autocratic in the pre-episode year) | 9 |
| Expected democratic episodes under `aut_ep_prch` | about 101 (verify) |
| … starting 1990 or later | 80 (version 1 definition) |
| … starting before 1940 | 12 (version 1 definition) |
| Censored, all regimes | 31 |
| Censored democratic episodes (outcome 6) | 15 |

Training-pool rows by period (hindsight rule): before 1970, 9,946; before 1990, 12,964; before 2005, 15,462.

**Outcomes of the 110 version-1 episodes** (`aut_ep_outcome`; meanings confirmed against the v16 codebook):

| Code | Meaning | Count |
|---|---|---|
| 1 | Democratic breakdown | 71 |
| 2 | Preempted breakdown | 1 |
| 3 | Diminished democracy | 1 *(the codebook says this outcome does not occur with default parameters; check this row)* |
| 4 | Averted regression | 13 |
| 5 | Regressed autocracy (autocratic in the pre-episode year, so not a democratic onset) | 9 |
| 6 | Outcome censored | 15 |

**Onsets by decade** (version 1 definition): 1910s 1 · 1920s 3 · 1930s 8 · 1940s 5 · 1950s 2 · 1960s 2 · 1970s 5 · 1980s 4 · 1990s 14 · 2000s 21 · 2010s 32 · 2020s 13.

**Interwar onsets in democracies:** Belgium 1914, Germany 1923, Lithuania 1926, Poland 1926, Austria 1930, Czechoslovakia 1930, Estonia 1932, Uruguay 1933, Latvia 1934, Spain 1936, Finland 1939 (averted), France 1939.

**Ongoing episodes at data end** (selected): Hungary 2006, India 2009, Serbia 2010, Philippines 2016, Italy 2019, United Kingdom 2019, Greece 2020, Mexico 2020, Slovakia 2023, Slovenia 2023, Argentina 2024, United States 2024.

## C.2 ERT definitions (v16 codebook)

**Default thresholds:**
- start inclusion: −0.01 EDI;
- cumulative inclusion: −0.10;
- tolerance: 5 years. The tolerance also sets how long a country must stay autocratic for a breakdown to count.

**Breakdown (`aut_ep_outcome == 1`):** the country becomes a closed autocracy, or an electoral autocracy that stays so for at least one election or for the tolerance period.

**`aut_ep_prch`:** the episode originates in a democracy and could produce a breakdown. The codebook recommends it for studying democratic regression.

**`aut_ep_pbr`:** marks years inside an episode after a democratic breakdown has occurred.

## C.3 Coverage facts

| Source | Coverage | Status |
|---|---|---|
| V-Party | 1970 to 2019; newest release v2 (Feb 2022) | Verified |
| BMR (`e_boix_regime`) | 1800 to 2020 | Verified |
| LIED | v6.9 exists; end year of V-Dem's copy unknown | Check in audit |
| `e_pt_coup` | Expected 1950 onward (Powell & Thyne) | Check in audit |
| `e_gdppc` | Possibly ends around 2019 | Check in audit |
| V-Forecast | Outcome: year-to-year Regimes of the World drop; about a 4% two-year base rate; test AUC-PR 0.45, AUC-ROC 0.85 (2010 to 2019) | Verified |

---

# PART D: CHANGES FROM VERSION 1

| Area | Version 1 | Version 2 | Why |
|---|---|---|---|
| Democratic episode | Democracy in the start year | `aut_ep_prch == 1`; regime read in the pre-episode year | ERT's own convention; the 9 "anomalies" were autocratic in the pre-episode year |
| At-risk set | Not inside any episode | Not inside a *confirmed* episode as of *t* | The version 1 rule used hindsight |
| Clean set | Included `v2x_regime` | `v2x_regime` and all EDI-derived variables in the full set only | Regime type is built from EDI thresholds |
| H2 | Leading out-rank lagging (permutation) | H2a: incremental skill via drop-group refits; H2b: the lagging pattern, with a caveat | The original pattern is predicted by persistence alone |
| H3 | Absolute full-minus-clean gap | Relative mechanical share | Absolute gaps shrink automatically |
| H4 | Importance ordering across classes | Sign of the H2a contrast across classes | Follows H2a |
| H5 | Exploratory (onset vs completion drivers) | Confirmatory stage-2 skill; the driver comparison is now E1 | Stage 2 is now a full ML task |
| Primary model | Unspecified | Elastic net | Avoids choosing the best model on the test set |
| Stage 2 | One row per episode, about 15 minority cases | Episode-year forecasting task; survival family; Boese baseline | Gives the models enough data to learn from and keeps stage 2 fully ML |
| Horizons | *k* = 1, 2, 3, 5, 10 in a common window | Common window for *k* ≤ 5; *k* = 10 exploratory | *k* = 10 shrank the window to 2005–2013 |
| Training population | Pooled, justified by row count | Hyperparameter chosen on development folds | V-Dem already exceeds 10,000 rows |
| Importance | Grouped permutation | Drop-group refits (permutation for display only) | Correlated groups and unequal group sizes bias permutation |
| Maturity buffer | 2 years, assumed | Data-derived; separate stage-2 buffer | Breakdown confirmation takes up to 5 years |
| New models | n/a | EDI-momentum logit, kitchen-sink gradient boosting, stacked ensemble, survival family, severity regression | More ML; stronger baselines |
| New checks | n/a | Power simulation, event-level metric, V-Forecast reproduction, PCA and clustering | Power, overlapping labels, benchmark, course techniques |
| Party arm | Post-1970 arm | 1970 to 2019 arm, exploratory | V-Party ends in 2019 |
| Coups and GDP | Assumed full coverage | Coverage rules (pre-1950 coups; GDP carry-forward) | Coverage gaps; risk of era-proxy leakage |
| Out-of-era hook | Weimar risk path from stage 1 | Pre-onset path from stage 1 plus the in-episode path from stage 2; rule-based negative controls | Stage 1 does not score in-episode years |
| Fly | Exploratory lineup member, cut last | Optional reservoir row with matched controls, cut early | The task does not need reservoir memory; a null would say little about the fly |
| Prior work | Novelty claims for the stage split and theory-driven selection | Boese et al. and Djuve et al. cited; novelty list revised | Both are published |

---

### Sources
- [ERT GitHub repository and v16 codebook](https://github.com/vdeminstitute/ERT)
- [The V-Dem Dataset](https://www.v-dem.net/data/the-v-dem-dataset/)
- [V-Dem data archive](https://www.v-dem.net/dsarchive.html)
- [V-Party dataset (QoG Data Finder)](https://datafinder.qog.gu.se/dataset/vparty)
- [Boix-Miller-Rosato, 1800–2020 (QoG Data Finder)](https://datafinder.qog.gu.se/dataset/bmr)
- [Boese et al. 2021, How democracies prevail](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8336576/)
- [V-Forecast / PART](https://v-dem.net/shiny/VForecast/)
- [Varieties of Forecasts: Predicting Adverse Regime Transitions](https://gupea.ub.gu.se/handle/2077/60276)
- [Djuve, Knutsen and Wig, Forecasting Regime Breakdown (EPSA 2025)](https://coms.events/epsa2025/data/abstracts/en/abstract_0962_0943_0369.html)
- [Lott & Croissant, Conceptualizing and measuring autocratization episodes](https://heibib.ub.uni-heidelberg.de/search/Record/1760693472)
