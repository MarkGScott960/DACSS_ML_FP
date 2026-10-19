# Pre-registration · FlyPredictor (DACSS 756)

**Status: DRAFT, not frozen.** All design decisions are made (D-20b, D-18b, D-21, 2026-10-19). The codebook check (D-22) passed with no changes (D-22b); nothing remains before the freeze. Freeze = commit this file together with `config/project.yaml` and `config/features.yaml`, then:

```bash
git tag -a prereg-v1 -m "Pre-registration frozen"
git push origin prereg-v1
```

After the tag exists, any change to this file or to `config/` is a **deviation** and is logged in section 10 with a reason. The lockbox (zip 2) refuses to score the locked test unless `prereg-v1` exists.

All former blanks (⟨…⟩) are filled from `notebooks/0x_data/03_audits.ipynb`, the power simulation (`notebooks/0x_data/04_power.ipynb`) and the decisions in `docs/decisions_log.md` (D-07b, D-08b, D-12b, D-15 to D-23).

---

## 1 · Research questions

- **RQ1.** Can V-Dem warning signs forecast democratic backsliding in democracies, and how fast does that skill decay with horizon? Do warning signs learned mostly from abrupt breakdowns transfer to gradual executive aggrandizement?
- **RQ2.** Do Paxton's leading signs add forecasting information beyond the outcome's own trajectory, especially at long horizons?
- **RQ3.** Is democratic breakdown predictable, and do its drivers differ from those of backsliding?

## 2 · Confirmatory hypotheses

**General rules:**
- **Locked test:** forecast origins 2005–2024, scored once through the lockbox. Models are **refit every year** on all labels known at the origin (origins ≤ *t* − 1 − *k* − lag), with hyperparameters frozen from development, and predictions get **intercept updating** on the last 10 known years (D-07b, D-18).
- **Primary model:** ridge logistic regression (D-20b). The power simulation used ridge, and a fixed elastic net passed less often than ridge on the same simulated data, so the simulated power applies to ridge. The elastic net, lasso and every other model in the lineup (section 7) are still tuned, fit and reported; H4 compares model classes.
- **Intervals:** country-clustered bootstrap with B = 2,000 resamples (paired across models).
- **Confirmatory family:** H1 only, at α = 0.05 with 95% intervals (D-20b). With one confirmatory hypothesis no multiplicity correction is needed. Every other hypothesis is **estimation only**: its effect is reported with a 95% interval and no pass/fail verdict.

| ID | Hypothesis | How it is tested | Status and power (β = 2 → oracle AUC ≈ 0.82) |
|---|---|---|---|
| **H1** | The stage-1 **clean** model beats persistence at *k* ∈ {1, 2}. Event: ERT first declares a democratic episode within *k* years (real time). | Locked test, annual refits. Pass if, at **both** *k* = 1 and *k* = 2, Brier skill vs persistence > 0 **and** PR-AUC ÷ base rate is above persistence's, each with its 95% interval excluding the baseline (intersection test). *k* = 3 and 5 are scored the same way and reported as the horizon-decay curve, with intervals and no verdict. | **Confirmatory.** Power 76.5% at β = 2, 99.5% at β = 3. Design effect δ₁ ≈ 0.06 Brier skill (β = 2, 76.5% power); the strict 80% MDE lies between β = 2 and 3 on the simulation grid. |
| **H1b** | The H1 models also beat persistence on **gradual** events (declared while the country is still a democracy). | Same models and rows; rows whose only event is abrupt are removed. Brier skill and PR-AUC ÷ base rate with 95% intervals at *k* ∈ {1, 2, 3, 5}. | Estimation only (power 64% at β = 2, *k* = 1). |
| **H2a** | Leading groups add skill to the **full** model at long horizons. | Drop-group refit: remove every leading group from the full model, refit with frozen hyperparameters, and report Δ log loss on the locked test with a paired bootstrap interval, at *k* = 3 and 5 (also shown at *k* = 1, 2). Positive Δ = leading groups help. | Estimation only (power 62% at *k* = 3, 22% at *k* = 5). |
| **H2b** | *(secondary)* Lagging groups gain relative importance as *k* shrinks. | Lagging drop-group Δ ÷ sum of all group Δs, *k* = 1 vs *k* = 5. Pre-stated caveat: persistence alone predicts this pattern. | Descriptive. |
| **H3** | The *relative* mechanical share falls as *k* grows. | (full − clean) ÷ (full − persistence) in Brier skill; difference between *k* = 1 and *k* = 5 with a bootstrap interval. | Estimation only (not simulated). |
| **H4** | The H2a contrast holds across model classes. | Sign of the H2a Δ in ridge, gradient boosting and the GRU (elastic net and lasso also shown), each with its interval. | Estimation only (not simulated). |
| **H5** | Real-time **breakdown** forecasting on all stage-1 democracies (D-08b): the clean model beats persistence. | Brier skill and PR-AUC ÷ base rate with 95% intervals; headline estimate at *k* = 1, also *k* = 2, 3, 5. | Estimation only (power ≤ 47% even at β = 3). |

**What δ means (D-20b, D-20c).** δ is the *design effect* read from the power simulation (200 simulated datasets per effect size): the effect size the test is powered for, **not** a pass threshold: H1 passes when its intervals exclude the baseline. A failed H1 therefore says that a true effect of about 0.06 Brier skill (oracle AUC ≈ 0.82) or more would have been detected about three times in four; it does not rule out smaller effects. The simulation covered H1, H1b, H2a and H5; H3 and H4 were not simulated.

## 3 · Data facts the design rests on (from the audits)

- ERT v16 / V-Dem v16, pinned to ERT commit `7fd51ce856a054582820bb0ac0ccae85371ff9be`. Manifests: `data/manifests/*.json`.
- Democratic episodes (`aut_ep_prch == 1`): **101**. Outcomes: 71 breakdown, 13 averted, 1 preempted, 1 diminished, 15 censored. Every episode is declared exactly once in the real-time re-runs; no declaration is withdrawn.
- Risk set: RoW democracy **and** ERT `reg_type == 1` (D-15): 5,031 stage-1 country-years (45 RoW-democracy rows that ERT treats as autocracies removed).
- Declarations made while the country is still a democracy (gradual): 7 of 42 up to 2005, 40 of 59 after. Development events are therefore mostly abrupt (at or after breakdown).
- Start → confirmation lag: median **2**, 90th percentile **7** years.
- Breakdown → genuine lag: 90th percentile **5** years.
- Breakdowns at or before real-time confirmation: **51** of 71 (23 in the episode's first year).
- Locked-window events at *k* = 1:
  - stage-1 declarations **53** (gradual 40, abrupt 13);
  - stage-1 ERT breakdowns **13** (origins ≤ 2019 because of the 5-year buffer); RoW breakdowns (sensitivity) **26**;
  - stage-2 breakdowns **7** (descriptive only, D-08b).
- Development-window events (origins ≤ 2004), stage-1 declarations at *k* = 1: **36** positive rows.
- Event rate per stage-1 country-year: declarations 1.1% (development) vs 3.0% (locked); ERT breakdowns 1.1% vs 0.9%.

## 4 · Outcomes and risk sets

- **State source:** `ert_truncated`: ERT `get_eps` re-run on V-Dem truncated at each year (real-time status). Sensitivity: `hindsight`.
- **Risk set:** RoW democracy (`v2x_regime ≥ 2`) and ERT `reg_type == 1` at *t* (D-15).
- **Stage-1 event (H1):** `confirmation`: ERT first declares a democratic episode in *t*+1 … *t*+*k* (D-07b). **H1b** splits it: gradual (`declaration_dem`, still a democracy in the declaration year) vs abrupt (`declaration_abrupt`).
- **Breakdown event (H5):** democratic breakdown (`row_regch_event == −1` inside an ERT breakdown episode), forecast from every stage-1 row. Sensitivity: any RoW democracy → autocracy change (`row_breakdown`), also with the RoW-only risk set.
- **Stage 2:** real-time stage 2 (148 country-years) is descriptive only; a conditional stage 2 with hindsight membership is exploratory (E1–E3).
- **Horizons:** stage 1 *k* ∈ {1, 2, 3, 5} (+ 10 exploratory), with *k* ∈ {1, 2} confirmatory for H1; breakdown *k* ∈ {1, 2, 3} for stage 2 and {1, 2, 3, 5} for the stage-1 breakdown target (H5).
- **Maturity buffers:** confirmation 0; breakdown 5 (90th percentile of the breakdown → genuine lag); onset (sensitivity only) 7. Applied symmetrically to positives and negatives.

## 5 · Features

- Feature sets as defined in `config/features.yaml` at the `prereg-v1` commit, built by `src/flypredictor/data/features.py`: **clean** (59 columns; no EDI component or EDI-derived variable) and **full** (115 columns).
- **Transformations:** level, 1-year change, 3-year change. EDI also gets "decline from the 10-year peak". Regime subtype enters as a liberal-democracy flag (full set only).
- **Coverage rules:** coups structurally missing before 1950 (zero plus a pre-coverage indicator; 3-year coup count). GDP dropped (D-17b); party arm cut (D-12b).
- **Imputation:** median imputation fit on each training set, plus a missingness indicator per feature group (none for coverage-rule groups), then standardisation. Fit on training rows only.

## 6 · Validation design

- **Development folds:** rolling origin, expanding window. 5-year validation blocks over 1970–2004. Gap of *k* years between the last training label and the first validation origin.
- **Locked test:** annual refits: for each origin *t* in 2005–2024, train on origins ≤ *t* − 1 − *k* − lag (lag = 0 for declarations, 5 for ERT breakdowns), hyperparameters frozen from development, intercept updating on the last 10 known years. Scored once, after code freeze and the leakage gate. **Sensitivity:** a single frozen fit on origins ≤ 2004 − *k* − lag.
- **Training population (D-06, D-24):** P1 / P2 / P3, chosen per model class on development folds by log loss on democracy rows. Scoring is always on stage-1 democracy rows.
  - **P3** = democracy rows (stage 1). **P2** adds electoral autocracies; **P1** adds closed autocracies too. A non-democracy row is eligible if it is not inside an autocratization episode of any kind that ERT has already recognised in real time (`states.pop_group`); RoW-democracy rows that ERT codes as autocracies (D-15) count as electoral autocracies.
  - **Labels:** democracy rows keep the H1 `declaration` label in every population. Added non-democracy rows are labelled `declaration_any`: ERT, re-run on data up to each year, first codes an autocratization episode of any kind (identified by country and start year) within *k* years. All 262 ERT episodes are recovered this way.
  - **Regime group in the models:** linear models add electoral- and closed-autocracy indicators, each interacted with every feature; boosting and the GRU take the group as an input. Both are 0 on every scored row, so no EDI-derived information reaches a democracy's forecast.
  - **Breakdown models (H5)** train on P3 only.
  - **Sizes (development, *k* = 1):** P3 3,255 rows / 36 events; P2 6,573 / 141; P1 15,264 / 171 (`results/dev/audits/population_counts.csv`).

## 7 · Models and tuning

- **Lineup:** persistence, EDI-momentum logit, elastic net, gradient boosting, kitchen-sink gradient boosting, GRU, stacked ensemble. Optional reservoir row.
- **Tuning objective:** log loss. No class reweighting.
- **Budget (D-21):** 40 configurations per model class per horizon (a 40-point grid for the penalised logits, random search for the others); the GRU is searched with one seed and its best configuration re-run with 5 seeds (mean reported). Training population P1–P3 is chosen inside this budget.

## 8 · Metrics and inference

- **Primary:** PR-AUC ÷ base rate; Brier skill vs persistence.
- **Secondary:** ROC-AUC; log loss; reliability (after Platt recalibration on democracy development rows); precision, recall and F1 at the top-10% alert budget; event-level hit rate.
- **Intervals:** country-clustered bootstrap (2,000 resamples, paired across models). One confirmatory hypothesis (H1), so no multiplicity correction.

## 9 · Exploratory (not confirmatory)

E1 onset vs breakdown drivers · E2 severity regression · E3 survival family (conditional stage 2, hindsight membership) · E4 kitchen-sink comparator · E5 stacked ensemble · E6 *k* = 10 · E7 out-of-era · E8 vintage · E9 external outcomes · E10 V-Forecast reproduction · E11 error disparities · ~~E12 party arm~~ (cut, D-12b) · E13 reservoir row.

## 9a · Statements made before the locked test

1. **Event composition shifts.** Before 2005 most recognised episodes are abrupt (coups, self-coups): 35 of 42 declarations come at or after breakdown. After 2005 most are gradual (40 of 59). H1 trains mostly on the first kind and is tested mostly on the second; H1b isolates the second.
2. **Base rate shifts.** Declarations run at about 1.1% per stage-1 year before 2005 and 3.0% after. Brier-based criteria are sensitive to this; intercept updating (D-18) is the pre-specified response, and PR-AUC ÷ base rate is the more robust primary metric.
3. **"Real time" removes back-dated episode starts and known end years, not revisions to V-Dem values.** All re-runs read V-Dem v16. Data-revision hindsight is tested only by E8 (vintage).
4. **Leading signs move with the slide.** In the real panel the planted leading-sign index correlates 0.72 with the negative 3-year EDI change, which limits how much a clean model can beat persistence and is relevant to H2a.

## 10 · Deviations after the freeze

| Date | Section | Change | Reason |
|---|---|---|---|
| | | | |
