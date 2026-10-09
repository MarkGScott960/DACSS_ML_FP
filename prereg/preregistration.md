# Pre-registration · FlyPredictor (DACSS 756)

**Status: DRAFT, not frozen.** Freeze = commit this file together with `config/project.yaml` and `config/features.yaml`, then:

```bash
git tag -a prereg-v1 -m "Pre-registration frozen"
git push origin prereg-v1
```

After the tag exists, any change to this file or to `config/` is a **deviation** and is logged in section 10 with a reason. The lockbox (zip 2) refuses to score the locked test unless `prereg-v1` exists.

Blanks marked ⟨…⟩ are filled from `notebooks/0x_data/03_audits.ipynb` and the power simulation (`04_power`, zip 2). Items marked **PENDING** depend on decisions D-07 and D-08 (`docs/decisions_log.md`).

---

## 1 · Research questions

- **RQ1.** Can V-Dem warning signs forecast democratic backsliding in democracies, and how fast does that skill decay with horizon?
- **RQ2.** Do Paxton's leading signs add forecasting information beyond the outcome's own trajectory, especially at long horizons?
- **RQ3.** Is democratic breakdown predictable, and do its drivers differ from those of backsliding?

## 2 · Confirmatory hypotheses

**General rules:**
- **Locked test:** forecast origins 2005–2024, scored once.
- **Primary model:** elastic-net logistic regression (H1, H2, H3, H5).
- **Intervals:** country-clustered bootstrap with ⟨B = 2,000⟩ resamples.
- **Multiple comparisons:** Holm across H1–H5 at family-wise α = 0.05.

| ID | Hypothesis | Decision rule | Minimum effect δ |
|---|---|---|---|
| H1 | Stage-1 **clean** model beats persistence at every *k* ∈ {1, 2, 3, 5} | Brier skill > 0 **and** PR-AUC ÷ base rate above persistence's, 95% CI excluding the baseline, at every *k* (intersection test) | δ₁ = ⟨…⟩ Brier skill |
| H2a | Leading groups add skill to the **full** model at *k* = 3 and *k* = 5 | Drop-group refit, Δ log loss (all leading groups removed) > δ₂ with the CI excluding 0, at both horizons | δ₂ = ⟨…⟩ |
| H2b | *(secondary)* Lagging groups gain relative importance as *k* shrinks | Lagging drop-group Δ ÷ sum of group Δs, *k* = 1 vs *k* = 5. Pre-stated caveat: persistence alone predicts this | descriptive |
| H3 | The *relative* mechanical share falls with *k* | (full − clean) ÷ (full − baseline) Brier skill: share at *k* = 1 > share at *k* = 5, CI on the difference excluding 0 | ⟨…⟩ |
| H4 | H2a holds across model classes | Sign of the H2a contrast agrees in elastic net, gradient boosting and GRU; CI excludes 0 in ≥ 2 of 3 | n/a |
| H5 | **PENDING (D-08).** Recommended: real-time **breakdown** forecasting on all democracies, clean model beats persistence at *k* = ⟨3 or 5⟩ | Brier skill > 0 and PR-AUC ÷ base rate above persistence's, CI excluding the baseline | ⟨…⟩ |

If the power simulation shows that a δ cannot be detected with the locked-test event count, that hypothesis is reported as **estimation only** (interval, no pass/fail) and this is stated here before the freeze.

## 3 · Data facts the design rests on (from the audits)

- ERT v16 / V-Dem v16, pinned to ERT commit `7fd51ce856a054582820bb0ac0ccae85371ff9be`. Manifests: `data/manifests/*.json`.
- Democratic episodes (`aut_ep_prch == 1`): **101**. Outcomes: 71 breakdown, 13 averted, 1 preempted, 1 diminished, 15 censored.
- Start → confirmation lag: median ⟨2⟩, 90th percentile ⟨7⟩ years.
- Breakdown → genuine lag: 90th percentile ⟨5⟩ years.
- Breakdowns at or before real-time confirmation: ⟨51⟩ of 71.
- Locked-window events at *k* = 1:
  - stage-1 declarations ⟨53⟩;
  - stage-1 breakdowns ⟨13⟩;
  - stage-2 breakdowns ⟨7⟩.
- Development-window events (validation blocks 1970–2004), stage-1 declarations at *k* = 1: ⟨…⟩.

## 4 · Outcomes and risk sets

- **State source:** `ert_truncated`: ERT `get_eps` re-run on V-Dem truncated at each year (real-time status). Sensitivity: `hindsight`.
- **Stage-1 event — PENDING (D-07).** Recommended: `confirmation`, meaning ERT first codes a democratic episode in *t*+1 … *t*+*k*.
- **Stage-2 event:** democratic breakdown (`row_regch_event == −1` inside a breakdown episode).
- **Horizons:** stage 1 *k* ∈ {1, 2, 3, 5} (+ 10 exploratory); breakdown *k* ∈ {1, 2, 3} for stage 2 and ⟨1, 2, 3, 5⟩ for the stage-1 breakdown target.
- **Maturity buffers:** confirmation 0; breakdown ⟨5⟩; onset (if chosen) ⟨7⟩. Applied symmetrically to positives and negatives.

## 5 · Features

- Feature sets as defined in `config/features.yaml` at the `prereg-v1` commit: **clean** (no EDI component or EDI-derived variable) and **full**.
- **Transformations:** level, 1-year change, 3-year change.
- **Coverage rules:** GDP carried forward with a "years since observed" feature; coups structurally missing before 1950.
- **Imputation:** median imputation fit inside each training fold, plus a missingness indicator per group, except for variables flagged as era proxies.

## 6 · Validation design

- **Development folds:** rolling origin, expanding window. 5-year validation blocks over 1970–2004. Gap of *k* years between the last training label and the first validation origin.
- **Locked test:** training on origins ≤ 2004 − *k*; scored once, after code freeze and the leakage gate.
- **Training population:** P1 / P2 / P3, chosen per model class on development folds by log loss on democracy rows.

## 7 · Models and tuning

- **Lineup:** persistence, EDI-momentum logit, elastic net, gradient boosting, kitchen-sink gradient boosting, GRU, stacked ensemble. Optional reservoir row.
- **Tuning objective:** log loss. No class reweighting.
- **Budget:** ⟨…⟩ random-search configurations per model class per horizon; GRU ⟨…⟩ configurations × 5 seeds.

## 8 · Metrics and inference

- **Primary:** PR-AUC ÷ base rate; Brier skill vs persistence.
- **Secondary:** ROC-AUC; log loss; reliability (after Platt recalibration on democracy development rows); precision, recall and F1 at the top-10% alert budget; event-level hit rate.
- **Intervals:** country-clustered bootstrap. Holm across H1–H5.

## 9 · Exploratory (not confirmatory)

E1 onset vs breakdown drivers · E2 severity regression · E3 survival family (conditional stage 2) · E4 kitchen-sink comparator · E5 stacked ensemble · E6 *k* = 10 · E7 out-of-era · E8 vintage · E9 external outcomes · E10 V-Forecast reproduction · E11 error disparities · E12 party arm · E13 reservoir row.

## 10 · Deviations after the freeze

| Date | Section | Change | Reason |
|---|---|---|---|
| | | | |
