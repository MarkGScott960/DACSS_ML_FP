---
aliases:
  - FP Progress Log
  - ML Final Project Log
---
# ML FP Progress Log
Session-by-session record for the fly connectome conflict forecasting project. Plan and settled decisions live in [[Final Project Guide]]; this file holds diagnostics, verification results, and code state.

---

## 2026-09-19 — Project scoping

### Decisions
- **Dropped:** Epstein redacted-email sender matching. Not personally interesting, and a partner from the original network analysis already did it.
- **Dropped:** fly brain trading stocks (original Idea 1). Markets are close to efficient, so noise would swamp any architectural effect, and the framing is not social science.
- **Adopted:** fly connectome as a reservoir forecasting armed conflict, with three tiers of learning.
- **Split settled:** fly + conflict is the class project; RL crisis-bargaining agents are the self-study follow-on, kept in mind throughout so the class work builds toward it.

### Reasoning worth keeping
The original sketch had a fatal tension: freeze the weights and nothing learns; train every weight freely and the fly-ness stops mattering, since any network of that size would do. The three-tier design resolves it by making *how much the network is allowed to learn* the experimental variable rather than a fixed choice.

Conflict was chosen over immigration on data grounds. A country-month panel gives ~190 countries × decades = tens of thousands of training examples; a single migration series is ~300 monthly points and would badly overfit a trainable network. ViEWS also provides published benchmark forecasts, so results can be compared against real models rather than only self-built baselines.

---

## 2026-10-01 — Connectome loaded and verified

### Order-of-work change
Loaded the connectome **before** acquiring conflict data, deviating from the original Week 4–6 plan. Rationale: the mushroom body could have turned out to be feedforward and unusable as a reservoir, which would have invalidated the whole approach. Better to find that out before investing in data acquisition.

Outcome: the approach is viable (see below). Cost: Check-in 2 is Data Collection and the conflict pipeline has not started. **The persistence baseline — the number everything else gets compared against — does not exist yet. This is now the critical path.**

### Dataset selection
Chose **FAFB v783 with Princeton synapses** over BANC, MCNS, MANC, MAOL. Full reasoning in [[Final Project Guide#Dataset Decision SETTLED]]. Headline factors: signed edges available, published simulation precedent, both hemispheres, direct CSV download, and the nerve cord being irrelevant to time-series forecasting.

Downloaded three files from `storage.googleapis.com/flywire-data/codex/data/fafb/783/`:
`connections_princeton.csv.gz`, `classification.csv.gz`, `consolidated_cell_types.csv.gz`.

### Verification 1 — is the connections file thresholded?
**Initial reading:** `min syn_count: 1` suggested the no-threshold file, implying a manual 5-synapse filter was needed.

**Checked by sweeping thresholds on pair totals:**
```
file rows:           5,342,446
unique pairs:        3,732,460
pairs with >=5 syn:  3,732,460
synapses retained:   50,666,648
  threshold  1:  3,732,460 pairs
  threshold  2:  3,732,460 pairs
  threshold  3:  3,732,460 pairs
  threshold  5:  3,732,460 pairs
  threshold 10:  1,554,065 pairs
```

**Result:** identical counts at 1 through 5 means **no pair total below 5 exists**. The file is already thresholded at the pair level, as Codex defines a connection. The `min syn_count = 1` was a per-neuropil row fragment, not a pair total — the multi-row-per-pair issue in reverse.

**Validation against published figures:** 3,732,460 pairs and 50,666,648 synapses match the Codex headline figures exactly. The aggregation is correct and the headline numbers can be cited without hedging.

**Consequence:** threshold robustness can only be tested upward. Downward requires the separate "Connections Princeton No Threshold" file. Optional extension.

### Verification 2 — neurotransmitter consensus
**Initial approach:** synapse-weighted majority vote over each neuron's outgoing edges, flagging neurons below 70% agreement.

**Result:** 0 of 137,518 below 70%.

**Interpretation:** not a clean bill of health — the signature of a column that was already per-neuron. The prediction is assigned to the neuron and joined onto every connection row, so the vote is unanimous by construction. **Do not write "100% NT agreement" in methods as if it validated anything.** Deduplicating by `pre_root_id` is sufficient.

**Distribution (as expected for the fly brain):**
```
ACH     94,160   (68%)
GLUT    22,657   (16%)
GABA    18,374   (13%)
SER      1,435
DA         786
OCT        106
```

### Verification 3 — is the mushroom body actually recurrent?
**Concern raised:** the canonical MB architecture is PN → KC → MBON, with Kenyon cells described as making few connections to each other. If the KC layer were acyclic, the matrix would be nilpotent — all eigenvalues exactly 0 — and could not be scaled to a useful spectral radius. A leaky integrator would still give memory, but only exponential low-pass filtering that a *random* feedforward network would match equally. The controls would tie for an uninteresting reason.

**Checked by computing strongly connected components.**

**Result:**
```
strongly connected components: 71
largest SCC: 5,535 neurons (98.7% of subgraph)
spectral radius (raw weights): 1522.683
```

**Conclusion: the concern does not hold.** 98.7% in one SCC is a genuine recurrent reservoir. The textbook "KCs don't interconnect" picture is largely hemibrain-era; the whole-brain reconstruction at 5+ synapses shows otherwise.

**Three independent sources of recurrence, from the flow table:**

| Flow | Edges | Synapses |
|---|---|---|
| KCab → KCab | 11,590 | 74,385 |
| KCg → MBON | 10,414 | 82,528 |
| KCab → MBON | 10,081 | 80,137 |
| KCapbp → MBON | 6,442 | 64,267 |
| KCg → KCg | 6,332 | 38,612 |
| KCapbp → KCapbp | 6,217 | 37,672 |
| APL → KCg | 2,489 | 62,498 |
| KCg → APL | 2,489 | 66,440 |
| APL → KCab | 1,772 | 50,628 |
| KCab → APL | 1,772 | 50,062 |
| KCapbp → PPL | 1,375 | 9,634 |
| KCab → PPL | 1,178 | 7,231 |
| KCapbp → PAM | 1,150 | 7,075 |
| KCg → PPL | 1,092 | 6,408 |
| PPL → KCg | 1,085 | 6,382 |
| KCapbp → APL | 918 | 33,147 |
| APL → KCapbp | 917 | 33,895 |
| PPL → KCapbp | 811 | 5,185 |
| KCab → PAM | 777 | 5,310 |
| PAM → MBON | 758 | 11,806 |

1. **KC→KC is the largest edge category** — ~24,000 recurrent KC edges across the three KC classes.
2. **APL closes a loop through every Kenyon cell.** Perfect symmetry: 2,489 each way for KCg, 1,772 each way for KCab, 918/917 for KCapbp. Two GABAergic neurons reciprocally wired to all 5,177 KCs, with strong edges (~25–29 synapses vs ~6.4 for KC→KC).
3. **DAN loops:** KC→PPL and PPL→KC both present.

**APL interpretation:** global inhibitory feedback hub — the fly's own gain control, enforcing sparse coding. For a reservoir this is close to ideal, a built-in stabiliser against runaway excitation. **Generates an experiment:** ablate APL (2 neurons), recompute SCC and forecasting performance. "How much of the fly's advantage rests on a single pair of cells?"

### Mushroom body extraction results
Pattern `^KC|^MBON|^PAM|^PPL1|^PPL2|^APL` against `primary_type`:

| Component | Count |
|---|---|
| KCg | 2,489 |
| KCab | 1,772 |
| KCapbp | 916 |
| PAM | 307 |
| MBON | 96 |
| PPL | 24 |
| APL | 2 |
| **Total** | **5,606** |

Sides: 2,811 right / 2,795 left (0.3% imbalance — good internal replication).
Subgraph: 73,099 edges, density 0.0023, mean in-degree 13.0.

**Biological sanity checks all pass:** ~2,590 KCs per hemisphere is the right magnitude; the γ > αβ > α'β' ordering is correct; exactly 2 APL (one per hemisphere) is a strong signal the filter is clean; 96 MBONs ≈ 35 types × 2 hemispheres.

### Glutamate sign decision
Settled on **glutamate inhibitory**. Biology: GluCl channels make glutamate frequently inhibitory in *Drosophila*, opposite to the vertebrate case. Modelling: glutamate-excitatory gives 85:13 excitation:inhibition, which saturates tanh; glutamate-inhibitory gives 68:30, a healthy regime. Run the flip as a robustness check.

### Code issues found
1. **DAN undercount.** 331 DANs by cell type (307 PAM + 24 PPL) vs 281 flagged by `nt_type` — ~50 carry a non-monoamine prediction. Select the Tier 3 reward channel by **cell type**, not neurotransmitter: cell-type identity is annotated and cross-validated, neurotransmitter is a prediction.
2. **Mean degree double-counted.** `2*len(sub)/len(ids)` is the undirected formula on a directed graph. Correct mean in-degree = 73,099 / 5,606 = **13.0**, not 26.1.
3. **Silent NaN signs.** Neurons absent from the presynaptic column get `nt_type` NaN → sign 0. Harmless (no outgoing edges means the sign is never used) but add an assertion.
4. **`MarksCSVRead` returns the first `rglob` match.** On OneDrive, sync conflicts produce files like `classification-DESKTOP-ABC.csv.gz`, and `rglob` has no stable ordering guarantee. Collect all matches and raise if more than one.
5. **No caching.** Re-reading and re-grouping 5.3M rows every run. Write `edges` and `meta` to parquet once.

### Gap identified: no input layer
The extraction has KCs, MBONs, DANs, and APL — but no **projection neurons**. In the fly, information enters the MB through PNs synapsing onto KC dendrites in the calyx, and PN→KC *is* the sparse random projection that justifies the whole mushroom-body framing. Without it, input would go through an invented $W_{in}$, discarding the strongest biological claim in the proposal.

**Next query** (find the input population empirically rather than guessing type names):
```python
kc = set(meta.loc[meta[type_col].astype(str).str.match(r"^KC", na=False), "root_id"])

upstream = (edges[edges["post_root_id"].isin(kc) & ~edges["pre_root_id"].isin(ids)]
            .groupby("pre_root_id", as_index=False)["syn_count"].sum())
upstream = upstream.merge(meta[["root_id", type_col, "class", "super_class", "side"]],
                          left_on="pre_root_id", right_on="root_id")

print(f"external neurons feeding KCs: {len(upstream):,}")
print(upstream.groupby("super_class")["syn_count"].agg(["count", "sum"])
              .sort_values("sum", ascending=False).head(10))
print(upstream.groupby("class")["syn_count"].agg(["count", "sum"])
              .sort_values("sum", ascending=False).head(15))
```
Expect a few hundred olfactory PNs plus thermo/hygrosensory and visual inputs.

### Finding: MBONs are a free experiment
No MBON→anything appears in the top-20 flows — they are near-pure sinks receiving ~27,000 KC edges. That is the biological readout layer, where learned valence is expressed. Gives a comparison at the cost of one extra ridge fit:
- Biological readout: ridge on 96 MBON states only
- Full-state readout: ridge on all ~5,600 states

If 96 dimensions approach 5,600, evolution picked a good 1.7% of the state space.

---

## 2026-10-01 — Data collection plan and validation design

### Decisions
- **Python throughout**, including the panel build.
- **UCDP GED 26.1 (frozen annual, 1989–2025)** as the modelling source, downloaded as a static file. The UCDP API now requires a token obtained by emailing the maintainer; the static downloads need none.
- **Candidate (monthly) data only for the live November 2026 check**, never for train/validate/test — it is preliminary and revised at each annual release.
- **Gleditsch–Ward state list** as the panel universe, zero-filling only within each state's lifespan.
- **Regression primary:** log(1 + state-based deaths) next month; onsets as an evaluated subset.
- **Pooled readout across per-country reservoir runs.**
- **Rolling-origin CV with a locked 2021–2025 test set.** Random year-mixed split considered and rejected (below).
- **Random/grid search on most hyperparameters, with an equal tuning budget across every model.**
- Covariate sources: V-Dem and COW first (previously used), then others as needed.

### Reasoning worth keeping: why not mix years across train and test
The appeal of a random split is real: the test set would look like the training set instead of being dominated by whichever wars happen to fall in the final years. It was rejected because it would make every result look better than it is.

**The leak.** Conflict deaths are extremely persistent month to month. With a random split, a test month (say, Syria March 2014) sits between training months February and April 2014. The model is no longer forecasting — it is interpolating between known neighbours, which is far easier.

**The reservoir makes it worse.** Each training row pairs the reservoir state at *t* with deaths at *t+1*, and the state at *t* includes deaths at *t* as input. So training row "state at Feb → deaths in March" literally contains the answer to a test row, and neighbouring rows carry test answers as inputs or targets.

**Consequences for this project specifically:**
1. **Inflated, compressed scores.** Persistence and every model look strong, and the gaps between them shrink or flip. The research question — does fly wiring beat random wiring — lives entirely in those gaps.
2. **Hyperparameter search selects for the leak.** Search rewards whatever exploits it best, typically long-memory, high-capacity configurations, which then fail on genuine future data (including the November check).
3. **It is the first thing a knowledgeable grader checks** in a forecasting project.

**Why rolling-origin recovers most of what the mixed split offered:**
- Every validation window comes strictly after its training data — no leak.
- The five validation windows plus the locked test together span 2006–2025 (Iraq, Syria, Libya, Tigray, Ukraine all land in some window), so no single period dominates.
- Mean-across-folds plus per-fold results shows stability over time, which is itself a finding.
- It is the time-series CV the guide already promised, and the hyperparameter search runs inside it.

**What still spans all years** (appendix only): country holdout (train on ~80% of countries over 1989–2025, test on the rest — tests generalisation to unseen countries, not to the future) and blocked time folds with ~12-month buffers (limits but does not remove the leak, since the model still uses the future to predict the past).

### Other issues found in planning
- **Onset definition mixes units.** UCDP's ≥25 battle-death threshold is per dyad per *year*; the guide applied it to months. Needs a monthly definition (k deaths after m zero months), stated and sensitivity-tested. Open.
- **Observation count:** ~80–85k country-months estimated, vs. course minimum of 10,000. Feature count of 10+ reachable from GED alone; V-Dem and COW add more.
- **Repo hygiene:** no `.gitignore` exists, and `connections_princeton.csv.gz` (66 MB) is committed. GitHub warns above 50 MB and rejects above 100 MB. The GED zip (~39 MB) and V-Dem files must not be committed — add a `.gitignore` covering `data/raw/` before any download script runs.

---

## 2026-10-01 — Panel built (notebooks 01–02)

### Data acquired (`data/raw/MANIFEST.json` holds versions + hashes)
GED 26.1 (417,968 events, 1989–2025) · GW state lists · V-Dem v16 Core (manual download; 1789–2025) · COW Direct Contiguity 3.2.

### Panel facts (verified by assertions in `02_clean_join_describe.ipynb`)
- **75,703 country-months, 178 states, 444 months.** Earlier estimate of 80–85k was high. 147 states have the full 444 months.
- **All 126 GED countries match the GW main list; 0 events fall outside a state's lifespan.** No microstate appears in GED → microstates excluded.
- **GW list censored at 31 Dec 2017** (174 spells) → extended to Dec 2025. Several states have multiple spells (pre-1989 occupations); lifespan checks test against *any* spell.
- **Death totals reconcile exactly** after spreading (state-based 2,574,474).
- **84.8% of country-months have < 1 state-based death.** 11,529 conflict months.
- **Persistence is strong:** P(conflict next | conflict now) = 85%; P(conflict next | none now) = 2.7%; corr of log1p(deaths) t vs t+1 = 0.89.

### Decisions
- **Multi-month events spread evenly by day.** They are 1.7% of events but **24% of deaths** (e.g. Ethiopia–TPLF 2021: 113k deaths coded Jan 1–Dec 31). Side effect: year-long events become flat plateaus, which slightly flatters persistence.
- **Conflict month = ≥ 1 death** for all yes/no descriptives. Spreading creates 200 months with 0 < deaths < 1 (83 deaths total); the continuous target keeps them as is.
- **V-Dem joined on V-Dem's stable `country_id`**, not COW code (handles Germany/Yemen/Czechia code changes). One override: GW 340 Serbia → COW 345.
- **V-Dem Core has no population or GDP** (`e_` variables are only in Full+Others). Seven indices used: polyarchy, libdem, civlib, clphy, rule, corr, regime.
- **V-Dem missing: 1,364 rows (1.8%)** — Bahamas, Belize, Brunei unscored (all months), plus 32 birth-year months with no y−1 value. Left NaN; fill-or-drop is a modelling decision.
- `{sb,ns,os}_events` = events **active** in the month (a 3-month event counts in each month).

### Data-quality findings
- **GED's own `low ≤ best ≤ high` fails for 5,075 events (1.2%)** — e.g. India 1990 low 17 / best 17 / high 6. Spread across all years, not only early ones. Source quirk; reported, not fixed. Treat low/high as a rough band only.
- ~12k events have month- or year-level date precision; they carry ~1.5M deaths. Kept; they are the main source of the plateaus.

### Onset groundwork (count of onset months, ≥ k deaths after ≥ m months below k)
| | m=1 | m=3 | m=6 | m=12 | m=24 |
|---|---|---|---|---|---|
| k=1 | 1727 | 737 | 425 | 258 | 142 |
| k=25 | 1212 | 587 | 367 | 216 | 138 |
Test period is ~13.5% of rows, so m=12 leaves only ~30 test onsets. Choose with that in mind.

### Code-level lessons
- **pandas 3 `stack()` keeps NaN** — range checks must `.dropna()` first.
- **Never run plain `git status` from the bridge** — it leaves `.git/index.lock`. Use `git --no-optional-locks status`.

---

## 2026-10-01 (late) — Review of notebooks 01–02; folds and naive baselines (notebook 03)

### Review of the first real run
Both notebooks ran cleanly on the real data; all invariant checks passed. Real COW contiguity loaded (35,902 directed dyad-years; 22,308 land).
- **Fixed — Macedonia had zero neighbours for 14 months (Nov 1991–Dec 1992):** COW codes it from 1993. Added `backfill_early_years` (copies the first coded year's borders back, both directions). Only Macedonia was affected; 16 dyad-years added.
- **Fixed — mislabeled output:** "states with no land neighbour in any month" actually listed states with *any* zero month. Now reports states that are *never* connected (20 island states) and the count of zero-neighbour months in connected states (0).
- **Fixed — pandas overflow warning** in the month-gap check (Period arithmetic); now uses an integer month index. Result unchanged.
- **Known, left as is:** COW codes 1991 as a transition year, so Russia's neighbour count peaks at 20 in Dec 1991 (USSR-era and successor-state borders both listed). One country-month.
- **Caution — thin final months:** Nov–Dec 2025 have fewer GED events than the 2025 average (state-based −23% / −14%; one-sided December −38%). They sit in the locked test window; report the test with and without them.
- Neighbour-conflict feature is now live: Spearman ρ with log1p(state-based deaths) = 0.265 (same row, unlagged).

### Onset counts per validation window (onset = ≥ k deaths after ≥ m quiet months)
| | f1 06–08 | f2 09–11 | f3 12–14 | f4 15–17 | f5 18–20 | TEST 21–25 | train ≤ 2005 |
|---|---|---|---|---|---|---|---|
| k=1, m=1 | 115 | 108 | 119 | 163 | 166 | 212 | 844 |
| k=1, m=3 | 55 | 46 | 52 | 55 | 61 | 100 | 368 |
| k=1, m=6 | 29 | 32 | 32 | 32 | 30 | 66 | 204 |
| k=1, m=12 | 25 | 20 | 19 | 23 | 18 | 39 | 114 |
| k=25, m=6 | 25 | 24 | 23 | 27 | 29 | 50 | 189 |
| k=25, m=12 | 16 | 14 | 15 | 16 | 20 | 25 | 110 |
**Consequence:** with m ≥ 6 each three-year validation window has ~20–30 onsets, too few for stable per-fold onset metrics; m ≤ 3 gives ~45–60. Pooling folds helps (150–300 onsets).

### Naive baselines (notebook 03; horizon 1; RMSE on log1p; folds 1–5 only — the locked test is never scored)
| Baseline | RMSE (mean of folds) | MAE |
|---|---|---|
| zero | 1.462 | 0.497 |
| persistence | 0.582 | 0.171 |
| **mean of last 3** | **0.526** | **0.163** |
| mean of last 12 | 0.552 | 0.179 |
- **Persistence is not the strongest naive baseline** — the 3-month mean beats it in all five folds. The bar for a model is 0.526 (h=1), and the 12-month mean from h=3 up.
- 83% of rows are quiet→quiet with near-zero error. Persistence's squared error: onset 32.6%, continuation 39.5%, cessation 27.9%.
- Onset rows: persistence RMSE 2.27 against mean target 1.94 — it predicts essentially nothing. Cessation: 2.11.
- Persistence skill vs zero decays 0.60 → 0.55 → 0.51 → 0.47 for h = 1, 3, 6, 12.

### New code
`src/eval/splits.py` (folds on target month; `test_rows` raises unless `final_evaluation_i_understand=True`), `src/eval/metrics.py`, `notebooks/03_folds_and_baselines.ipynb`.

---

## Features and model search (notebooks 04–05)
- Onset definition: k=25 deaths after m=6 quiet months (default). Stricter m=12 leaves only 14–20 onsets per fold; one-month blips stay 60–70% under every definition.
- Horizons: 1, 3, 6, 12 months. Onset events per fold (folds 1–5): h=1 25/24/23/27/29; h=3 34/34/32/39/38; h=6 45/46/43/51/49; h=12 63/51/67/70/69.
- V-Dem: rows missing any of the 7 indices are dropped (incl. Bahrain 1990–2005, 192 rows). Usable origins: 72,079.
- 04_features: 28 features, 12-month burn-in, truncation leakage audit passes (0 of 28 change), saves data/processed/features.parquet.
- 05_models_random_search: ridge, random forest, HistGradientBoosting, 40 random configs each, tuned naive window (1–24), 5 rolling-origin folds, resumable log. Test set never scored. Smoke test passed; full run completed 2026-10-02.
- Caveats: year-long GED events flatten into plateaus (flatters persistence); Nov–Dec 2025 GED undercount; configs are selected on the same folds reported (mildly optimistic).

## Full run of notebook 05 (2026-10-02)
Mean validation RMSE (folds 1–5), naive = best window:

| h | naive | ridge | RF | GB |
|---|---|---|---|---|
| 1 | 0.5282 (w4) | 0.5033 | 0.5048 | 0.5073 |
| 3 | 0.5892 (w6) | 0.5707 | 0.5783 | 0.5824 |
| 6 | 0.6320 (w9) | 0.6216 | 0.6322 | 0.6354 |
| 12 | 0.7093 (w6) | 0.6887 | 0.7012 | 0.7101 |

- Ridge is best at every horizon; skill vs naive 0.047 / 0.031 / 0.016 / 0.029. RF and GB are at best marginal and fall to or below naive at h=6 and 12.
- Gains are in stay-quiet and cessation rows; onset RMSE is about the same as naive (h=1: 1.82 ridge vs 1.79 naive).
- Onset AP 0.095–0.158 vs base rate 0.5–1.2%; models clearly beat naive only at h=12. Pooled events: 128 / 177 / 234 / 320 for h=1/3/6/12.
- Fold SDs (0.016–0.061) are comparable to model gaps: significance not established.
- Naive MAE is lower than all models' MAE.
- Bar for the reservoir and controls: beat ridge, not only naive.
- Next: paired fold-level comparison; reservoir and control notebooks; MLP baseline; seeds with confidence intervals; locked-test evaluation once at the end.

## Reservoir and control notebooks built (2026-10-02, quick-tested)
- Input layer found empirically: of 539 neurons outside the mushroom body that feed Kenyon cells, 285 are ALPN projection neurons (median 62 Kenyon cells each). Network = 5,606 + 285 = 5,891 neurons, 97,051 connections (92,642 carry signal; 7% inhibitory). Largest strongly connected component 5,548. Each Kenyon cell gets a median of 5 PN inputs (92% get at least one).
- Weights: fraction of postsynaptic input, sign by presynaptic neuron, glutamate inhibitory, modulators silent in W. All networks rescaled to a tuned spectral radius.
- Controls built and verified: shuffled keeps every connection (overlap 1.0); rewired keeps all in- and out-degrees (10% of connections shared with the fly); random keeps counts only.
- Readout validated: inputs-only through the new Gram ridge gives 0.5032 vs 0.5033 from notebook 05.
- Echo-state check: most sampled configurations forget their starting state; the one with rho 1.1 does not (chaotic regime).
- Quick tests (2 configs, horizon 1): fly, shuffled, rewired and random all land at 0.503–0.504, about the same as ridge on raw features alone. Not a result (2 configs, subsampled baselines), but it shows the bar is high and that a null outcome is plausible.
- Files: notebooks 06, 07, 08; `src/networks/`, `src/models/readout.py`, `reservoir_search.py`, `reservoir_data.py`, `common.py`; `src/eval/paired.py`.
- Caveats: tuning picks the best of 40 configs on the same folds that are reported; readout uses 1,024 random units, not all 5,891; one graph seed during tuning, five seeds afterwards.

## Full runs of notebooks 06–08 (2026-10-02)
**06, paired baselines (country-cluster bootstrap, folds 1–5).** Ridge beats the tuned naive baseline with an interval excluding 0 at horizons 1, 3 and 12 (differences -0.025, -0.019, -0.020); at horizon 6 the interval just spans 0 (-0.010, [-0.023, 0.001]). Random forest and gradient boosting are never reliably better than ridge or naive beyond horizon 1; their gaps to ridge all have intervals spanning 0.

**07/08, reservoirs.** Best mean validation RMSE (h = 1 / 3 / 6 / 12):

| network | h1 | h3 | h6 | h12 |
|---|---|---|---|---|
| ridge (raw features) | 0.5033 | 0.5707 | 0.6216 | 0.6887 |
| fly | 0.5026 | 0.5702 | 0.6206 | 0.6877 |
| weight-shuffled | 0.5027 | 0.5700 | 0.6208 | 0.6879 |
| rewired | 0.5019 | 0.5704 | 0.6209 | 0.6880 |
| random (ER) | 0.5025 | 0.5692 | 0.6197 | 0.6870 |

- Fly vs ridge: differences of -0.0006 to -0.0010 with intervals spanning 0 at every horizon. Fly vs each control: all within ±0.001, intervals span 0 (one h=1 fly-vs-shuffled interval excludes 0 but the difference is 0.0001).
- Seed spread (5 new W_in draws / graphs) is 0.0000–0.0014, the same size as the differences between networks.
- **Diagnosis: the readout switches the reservoir off.** Across the 40 configurations, RMSE rises steadily with `state_weight` (Spearman 0.94 at h=1 and h=12) and the best configurations all have state_weight 0.03–0.05, near the lower bound of the range (0.02). Spectral radius and input scale have no relationship with RMSE (|rho| < 0.15). The median configuration equals ridge on raw features alone (0.5033 vs 0.5032). The best-of-search gain of about 0.0007 is the size expected from selecting among 480 (config, penalty) pairs on the same folds.
- **Interpretation:** the 28 input features already contain lags and rolling means of the target, so the reservoir's fading memory duplicates information the raw features carry. This design cannot show a fly advantage or a lack of one; it shows only that states add nothing on top of the features. Do not report "wiring does not matter" from this.
- Onset average precision and row-type RMSE are the same for all reservoirs and ridge (e.g. h=12 AP 0.149–0.152).
- Next experiment under discussion: a memory test with history-free inputs (current-month values only), reservoir states as the only source of history, compared with ridge on the same inputs and ridge on the full feature set.
- Decision (2026-10-02): build notebook 09, a memory test with 12 history-free inputs; keep 07–08 as the full-feature experiment in the final report. Quick-tested; full run pending.

## Full run of notebook 09, review, and decisions (2026-10-02)

### Memory test results (history-free inputs; mean validation RMSE, folds 1–5)
| h | A: ridge, current values | B: ridge, 28 features | fly | weight-shuffled | rewired | random (ER) |
|---|---|---|---|---|---|---|
| 1 | 0.5672 | 0.5033 | 0.5051 | 0.5055 | 0.5064 | 0.5046 |
| 3 | 0.6491 | 0.5707 | 0.5754 | 0.5747 | 0.5749 | 0.5738 |
| 6 | 0.7066 | 0.6216 | 0.6267 | 0.6257 | 0.6261 | 0.6248 |
| 12 | 0.7569 | 0.6887 | 0.6960 | 0.6956 | 0.6957 | 0.6936 |

- **Memory recovered** = (A − network) / (A − B): fly 0.97 / 0.94 / 0.94 / 0.89; every network 0.89–0.98. The reservoirs supply almost all of the history the hand-built lags provide.
- **Fly vs A:** better at every horizon, intervals far from 0. **Fly vs B:** never significant (fly 0.002–0.007 worse).
- **Fly vs controls:** 11 of 12 intervals span 0. The one that excludes 0 is fly vs ER at h=3, and **ER is better** (+0.0017). ER is also best in the seed means at all four horizons (by 0.0008–0.0014). One significant result in 12 tests is what chance alone would produce.
- **Diagnosis from 07 confirmed:** RMSE no longer depends on `state_weight` (Spearman 0.01–0.23, was 0.94), so the readout uses the states.
- **New edge-of-range problem:** best `input_scale` 0.051–0.075 in a range of 0.05–5; RMSE rises with input scale (Spearman 0.87–0.93). The reservoirs want the near-linear regime. In that regime a reservoir is a linear filter bank whose memory depends on the eigenvalues of (1 − leak)I + leak·Wᵀ, and all networks are rescaled to the same spectral radius. This is the likely reason the controls tie.
- **MBON readout:** 96 MBONs recover about 84–86% of the gap vs 89–97% for 1,024 random units. Not a fair comparison (96 vs 1,024 units); needs a size-matched random readout.
- **Onset AP:** every reservoir ranks onsets better than B at h=6 (≈0.133 vs 0.115) and h=12 (≈0.180–0.186 vs 0.149); A (0.156) also beats B at h=12. Not yet tested for significance. A reservoir effect, not a fly effect.
- Seed spread: fly 0.0002–0.0007 (only W_in varies), controls up to 0.0012 (W_in and graph vary). The fly's smaller spread is partly by construction.

### Equivalence framing (for the write-up)
Worst-case fly advantage = −(lower CI bound) / (A − B gap, pooled):

| h | vs shuffled | vs rewired | vs ER |
|---|---|---|---|
| 1 | 4.0% | 6.4% | 1.6% |
| 3 | 1.0% | 0.5% | none |
| 6 | 1.6% | 1.6% | 0.1% |
| 12 | 4.4% | 4.3% | 3.1% |

Sentence for the report: **a fly advantage larger than about 6% of the skill explicit history provides is ruled out, and at most horizons the bound is under 2%.**

### Code review (no leakage or correctness bugs found)
Checked: scaler fit on ≤ 2005 only; reservoir causal (state at t uses inputs ≤ t); readout statistics from training rows only; folds split on target month; controls preserve what they claim; cluster bootstrap correct.
Smaller points, disclosed rather than fixed:
- Two RMSE definitions are in use: **mean of fold RMSEs** (leaderboards, memory recovered) and **pooled RMSE** (row-type tables, bootstraps). Label every table.
- Paired tests use the seed-0, best-of-40 configuration; seed means (seeds 1–5) are the unbiased headline.
- `lambda_id` 0 (bottom of the ridge grid) for shuffled and rewired at h=1. Left as is: changing `LAMBDAS` would invalidate every cached log.
- After degree-preserving rewiring the "fraction of postsynaptic input" normalisation no longer holds exactly (weights stay with the source). Harmless because every network is rescaled to the same spectral radius.
- The 71 neurons outside the main SCC are still in the network (not dropped). Low priority.

### Decisions
- **A null finding can be the headline.** The report frames it as a bound (above), with the mechanism explaining why.
- **Tier 2 stays in scope.** The goal is to test whether the connectome can learn when its connections and weights are allowed to change. Design to be settled after the mechanism results (see Next Steps).
- **Mechanism experiments go into notebook 09** (sections 7–10), not a new notebook. Each writes its own log, so re-running 01–09 reuses all cached tuning and computes only the new work.
- Housekeeping first, then push to GitHub before further testing.

### Code changes (2026-10-02)
- `src/networks/reservoir.py`: `activation` ("tanh" or "linear"); `leak` may be a per-unit array; divergence check.
- `src/models/reservoir_search.py`: `SPACE_LINEAR`, `SPACE_BANK`; no-recurrence banks `bank_single` / `bank_multi` (1,024 independent leaky units; one leak rate vs per-unit leak rates); `lab.add_random96(k)` for size-matched readouts; only the read-out nodes are stored (less memory, identical results — verified: fly and ER h=1 reproduce the logged 0.505084 and 0.504602 exactly); **log fingerprint guard**: `tune()` writes `<log>.fingerprint.json` (hash of inputs and config list) and refuses to resume a log written with different inputs or configs.
- `src/eval/paired.py`: `cluster_bootstrap_ap_diff` (country-cluster bootstrap for average precision; matches sklearn's AP exactly, ties included).
- Notebook 09: section 7 equivalence bounds; 8 onset AP bootstrap and per-fold AP; 9 linear reservoirs and no-recurrence banks with seeds and paired tests; 10 MBONs vs five random 96-unit readouts; 11 reading guide. Quick test passed.
- Smoke-test signals (3 random configs) were confirmed by the full run: see the next entry.

---

## Re-run of notebooks 01–09 and second review (2026-10-02, evening)

### Verification
- All nine notebooks re-run in order with `QUICK_TEST = False`: no errors, no stderr, every code cell executed (02 and 05, which had unsaved or stale cells, are now complete).
- **Reproducibility:** 05–08 and 09 §1–6 reproduced the earlier numbers exactly (leaderboard, paired CSVs, best configs). Pooled and fold-mean RMSE for all 9 models recomputed independently from the saved predictions match the tables.
- 02 and 04 outputs match this log (75,703 country-months, 178 states, 444 months; 28 features; 72,079 usable origins; leakage audit 0 of 28).
- Commit `1f455e8` ("First Computing Test (0-9 Jup)") is on origin. Later diffs in `MANIFEST.json` and `Planning/` are line endings only; every data hash is unchanged.

### Results, notebook 09 sections 7–10 (mean validation RMSE, folds 1–5)
| model | h1 | h3 | h6 | h12 |
|---|---|---|---|---|
| A: ridge, current values | 0.5672 | 0.6491 | 0.7066 | 0.7569 |
| B: ridge, 28 features | 0.5033 | 0.5707 | 0.6216 | 0.6887 |
| fly (tanh) | 0.5051 | 0.5754 | 0.6267 | 0.6960 |
| weight-shuffled (tanh) | 0.5055 | 0.5747 | 0.6257 | 0.6956 |
| rewired (tanh) | 0.5064 | 0.5749 | 0.6261 | 0.6957 |
| random ER (tanh) | 0.5046 | 0.5738 | 0.6248 | 0.6936 |
| fly (linear) | 0.5027 | 0.5729 | 0.6237 | 0.6926 |
| random ER (linear) | 0.5027 | 0.5722 | 0.6225 | 0.6916 |
| bank_single (no recurrence, one time scale) | 0.5084 | 0.5763 | 0.6285 | 0.6963 |
| **bank_multi (no recurrence, many time scales)** | **0.5023** | 0.5736 | 0.6241 | 0.6919 |

- **Memory recovered** (A − model)/(A − B): tanh 0.89–0.98; linear 0.94–1.01; bank_multi 0.95–1.02; bank_single 0.89–0.93. Values above 1 are within noise of B: do not write "better than the hand-built history".
- **Linear ≥ tanh** at every horizon (fly: 0.0024–0.0035 better; individually not significant, same sign everywhere). The linear search had 20 configurations against 40 for tanh, so this is conservative.
- **No-recurrence bank ties every network.** bank_multi vs B: −0.0010 to +0.0034, all intervals span 0. Fly (tanh) vs bank_multi: +0.0019 to +0.0040 (bank better), none significant. bank_multi beats bank_single at h=1 (−0.0061, significant) and nominally elsewhere.
- **Fly (linear) vs ER (linear):** +0.0000 to +0.0012, none significant; ER is nominally better at all four horizons. Worst-case fly advantage (lower interval bound / pooled A−B gap): 1.2%, 0.4%, 0.5%, 1.0% at h = 1/3/6/12. For tanh the maximum over all 12 fly-vs-control comparisons is 6.4%.
- **MBON readout:** the 96 MBONs rank last of six against five random 96-unit readouts at every horizon (e.g. h=1: 0.5125 vs random 0.5047–0.5064; 1,024 random units 0.5051). The biological readout is not special for this task. Caveat: the random sets are drawn from all nodes, about 88% Kenyon cells.
- **Onset AP (§8):** only h=12 vs B is significant: fly +0.032 [0.005, 0.054], ER +0.038 [0.003, 0.062]. Driven by folds 1, 3 and 4 (e.g. fold 3: fly 0.165 vs B 0.123); folds 2 and 5 flat. Fly vs A (no memory) not significant. A reservoir effect, not a fly effect; 2 of 16 correlated tests. Exploratory only.
- **Seeds:** mechanism seeds (3 new): sd 0.0001–0.0005.
- **Edge-of-range:** best ridge penalty at the bottom of the grid for all linear and bank models at h=1 (and shuffled/rewired tanh at h=1; effect probably negligible, not tested); bank_multi's slowest leak is 0.012 at h=12 against a lower bound of 0.01 and RMSE still improves with slower leaks (Spearman +0.65).
- **Exploratory baseline with no reservoir** (run outside the repo, now built into §9c): ridge on exponential moving averages of the 12 inputs, 3 leaks (0.5, 0.15, 0.04) → 0.5056 / 0.5733 / 0.6239 / 0.6916; 6 leaks → 0.5059 / 0.5737 / 0.6242 / 0.6917. Within about 0.0006 of the linear reservoirs at h = 3, 6, 12; 0.003 behind at h=1 (fewer time scales). Only the ridge penalty is chosen (12 values), so its budget is smaller than the reservoirs'. Confirm with the §9c output once notebook 09 is re-run.

### Interpretation
1. The reservoir mechanism works, but the memory it supplies is multi-time-scale leaky averaging of the inputs. A bank of independent leaky units supplies it just as well, and so does a plain set of moving averages. B's lags and rolling means already approximate it, so no reservoir beats B.
2. The connectome's wiring contributes nothing detectable: the fly ties shuffled, rewired and random networks, and a no-recurrence bank. The bound on any fly advantage is 1–6% of the history gap.
3. Random forest and gradient boosting did not beat ridge (05–06), tanh did not beat linear (09), and no reservoir beats B. The forecasting signal appears to be close to linear in a multi-scale history of log deaths. This matters for Tier 2: training weights on this target may also tie ridge. Tier 2b (a diagnostic task where nonlinear memory is required) is how to tell "no nonlinear signal in the data" apart from "the fly cannot learn".

### Disclosures for the methods section
- Tanh search: 40 configurations; linear and bank: 20 each. The tuning picks the best configuration on the same folds that are reported (mildly optimistic). Paired tests use the seed-0 best configuration; use seed means as the headline.
- Two RMSE definitions: mean of fold RMSEs (leaderboards, memory recovered) and pooled RMSE (row-type tables, bootstraps). Label each table.
- The fingerprint guard covers inputs and config lists, not reservoir or readout code, `LAMBDAS`, or the readout node set: delete the logs after changing those.

### Decisions
- Null result is the headline, framed as a bound with the mechanism as the explanation.
- Tier 2 stays in scope; design drafted in the Final Project Guide (2a trained weights, 2b diagnostic task, 2c structural plasticity as an extension).
- Added §9c to notebook 09 (no-reservoir baseline, mechanism bounds); `src/networks/reservoir.py::ema_bank` and `ReservoirLab.evaluate_ema` (reproduce the exploratory numbers exactly).
- Large tuning logs are no longer committed (`.gitignore`); `.gitattributes` normalises line endings.

---

## Current State
**Tier 1 is complete.** Panel, features (leakage-audited), rolling-origin folds with a locked test set, tuned baselines, paired comparisons, the fly reservoir with three controls (full features: states switched off; history-free: reservoirs recover 89–102% of the history gap), the mechanism checks (linear, no-recurrence banks, size-matched readout) and the no-reservoir baseline.
**Result:** a tightly bounded null on wiring (a fly advantage above 1–6% of the history gap is ruled out), explained by the mechanism: the reservoirs supply leaky averaging, which independent leaky units and plain moving averages supply equally.
**Pending:** run notebook 09 once more to execute §9c (a few minutes; every other cell reuses its log).
**Not started:** Tier 2 (trained weights, diagnostic task, structural plasticity), MLP baseline, robustness checks (APL ablation and glutamate flip deprioritised while the fly ties random networks), locked-test evaluation.

## Immediate Next Steps
1. Re-run notebook 09 (executes §9c); commit; push.
2. Optional one-off: extend the ridge penalty grid two decades lower for the six best linear and bank configurations at h=1 and confirm nothing moves.
3. Tier 2 feasibility: install PyTorch under Python 3.14 and time sparse backpropagation through 36-month windows for 178 countries with the 97k existing connections.
4. Build Tier 2a (see the guide): train existing weights from fly, shuffled, rewired and ER starts, plus a GRU of similar size, with an inner validation window for early stopping and 5 seeds.
5. Tier 2b: nonlinear memory benchmark (NARMA-10, delayed parity) for frozen and trained fly vs controls.
6. Tier 2c (extension): learnable connection gates and budgeted rewiring.
7. MLP baseline (same folds and budget).
8. Locked-test evaluation, once, at the end.
9. Email Dr. Adler re: social-phenomenon strictness (if not already done).
