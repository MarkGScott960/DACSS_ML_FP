---
aliases:
  - Tier 1 Report
  - Tier 1 Frozen Reservoir
---
# Tier 1 Report: A Frozen Fly-Connectome Reservoir for Conflict Forecasting

Consolidated record of Tier 1 of the DACSS 756 final project, written so that any section can be lifted into the final paper. The plan and settled decisions live in [[Final Project Guide]]; the day-by-day diagnostics live in [[ML FP Progress Log]]; background notes are in [[Connectome]] and [[Reservoir Computing]]. This report is the single place that says what Tier 1 was, what it found, and how to reproduce it.

**Status:** Tier 1 complete (2026-10-02). Nine notebooks run end to end with no errors; the locked 2021–2025 test set has not been touched. Notebook 09 §9c (the no-reservoir baseline) has been run and reproduces the exploratory numbers exactly (§9.4). **Tier 1 is final.**

---

## 1. Summary

**Question.** Does the wiring of the *Drosophila* mushroom body, used as a fixed recurrent network (a reservoir), forecast armed conflict better than random networks and conventional models?

**Tier 1 design.** Every connection is frozen; only a linear readout is trained. This is the cleanest test of "does the wiring itself help", because nothing can learn its way around a bad architecture.

**Findings.**
1. **Conventional baselines.** Ridge regression on 28 hand-built features beats a tuned moving-average baseline by 1.6–4.7% RMSE; random forest and gradient boosting never reliably beat ridge. The signal looks close to linear in a multi-scale history of log deaths.
2. **With all 28 features as inputs, every reservoir equals ridge** (differences ≤ 0.001). The features already contain the history, so the readout learns to switch the reservoir off. This design cannot say anything about wiring.
3. **With history-free inputs** (12 current-month values), the reservoirs must supply the memory themselves. They do: every network recovers **89–102%** of the forecasting skill that explicit lags and rolling means provide, and none is distinguishable from that ceiling.
4. **The fly's wiring contributes nothing detectable.** The fly ties weight-shuffled, degree-preserving rewired and random networks. Any fly advantage larger than about **6.4%** of the history gap is ruled out for tanh networks, and about **1.3%** for linear networks (against random networks; against the unconnected leaky-unit bank the widest bound is 8.9% at h=12, §9.2).
5. **The mechanism is leaky averaging, not recurrence.** Linear reservoirs are at least as good as tanh; a bank of 1,024 independent leaky units with no connections ties every network; ridge on plain moving averages of the inputs (no units at all) lands within about 0.001 of the linear reservoirs at every horizon, not significantly different (§9.4).
6. **The biological readout is not special.** The 96 mushroom-body output neurons (MBONs) read out worse than 96 random units at every horizon.
7. **Onset detection:** a modest reservoir (not fly) gain over the 28-feature ridge at the 12-month horizon only; exploratory.

**Headline sentence for the paper.** *A fixed fly-connectome reservoir recovers nearly all the forecasting skill that explicit history provides, but so does a random network, a bank of unconnected leaky units, and a set of exponential moving averages; a fly advantage larger than about 1–6% of the history gap is ruled out.*

---

## 2. Design logic

### 2.1 The three-tier structure
| Tier | What is fixed | What learns | Status |
|---|---|---|---|
| 1. Frozen reservoir | All connectome weights | Linear readout only (ridge) | **Done (this report)** |
| 2. Connectome-constrained | Which neurons connect (2a); nothing (2c) | Weights on existing connections (2a); the connections themselves (2c) | Next; design in [[Final Project Guide]] |
| 3. Mushroom-body plasticity (stretch) | Everything except MB synapses | MB synapses via dopamine-like reward | Not started |

Tier 1 is the control experiment for the whole project: if frozen wiring already matched random wiring, the open question for Tiers 2–3 is whether learning changes that.

### 2.2 Why the mushroom body
The mushroom body is independently reservoir-like: about 50 olfactory projection neurons feed about 2,000 Kenyon cells per hemisphere through sparse, quasi-random connections (a biological random expansion), and the output neurons (MBONs) are trained by dopamine. The project asks whether evolution's particular solution beats the random projections used by default.

### 2.3 Why conflict forecasting
Conflict was chosen over migration on data grounds: a country-month panel gives tens of thousands of examples, where a single migration series is about 300 points. UCDP also supports comparison against published forecasting benchmarks. The earlier ideas (Epstein sender matching, a fly brain trading stocks) were dropped on 2026-09-19: the first was not personally interesting and had been done by a collaborator, the second sits on near-efficient markets and is not social science.

---

## 3. Data

### 3.1 Sources
| Source | Version | Use |
|---|---|---|
| UCDP Georeferenced Event Dataset (GED) | 26.1, frozen annual release (417,968 events, 1989–2025) | Outcome and conflict features |
| Gleditsch–Ward state list | as published; extended from its 2017 end to Dec 2025 | Panel universe (matches GED `gwno` codes) |
| V-Dem Country-Year Core | v16 (manual download) | Seven democracy indices |
| COW Direct Contiguity | 3.2 (35,902 directed dyad-years; 22,308 over land) | Neighbour-conflict feature |
| FlyWire FAFB v783, Princeton synapses | Codex download | Network (see §5) |

Versions and SHA-256 hashes are recorded in `data/raw/MANIFEST.json`. The monthly UCDP Candidate data is deliberately not used for modelling (preliminary and revised at each release); it is reserved for a personal November 2026 check.

### 3.2 The panel
- **75,703 country-months, 178 states, 444 months (Jan 1989–Dec 2025);** 147 states have all 444 months. Peaceful months are zero-filled only inside each state's existence window.
- All 126 GED countries match the Gleditsch–Ward list; 0 events fall outside a state's lifespan; no microstate appears; death totals reconcile exactly (state-based total 2,574,474).
- 84.8% of country-months have fewer than 1 state-based death; 11,529 are conflict months (at least 1 death).
- Persistence is strong: P(conflict next month | conflict now) = 85%; P(conflict next | none now) = 2.7%; correlation of log(1+deaths) at *t* and *t+1* is 0.89.
- **Multi-month events are spread evenly by day** across the months they cover. They are 1.7% of events but 24% of deaths (e.g. Ethiopia–TPLF 2021: 113k deaths coded Jan 1–Dec 31). Side effect: year-long events become flat plateaus, which flatters persistence baselines.
- V-Dem is joined on V-Dem's own `country_id` (one override: GW 340 Serbia → COW 345) with a **one-year lag** (a year's score is not known until the year ends). V-Dem Core has no population or GDP. Macedonia's missing 1991–92 borders are back-filled from its first coded year.

### 3.3 Target, horizons and onsets
- **Target:** log(1 + state-based deaths) at the target month; horizons **1, 3, 6 and 12 months**, all reported. Non-state and one-sided deaths enter as input features.
- **Onset** (an evaluated subset, not the target): a row is *at risk* if the state had fewer than k = 25 deaths in each of the previous m = 6 months; the *event* is at least 25 deaths at the target month. With m = 12 only 14–20 onsets per fold remain, so m = 6 was chosen. Events per fold at h=1: 25 / 24 / 23 / 27 / 29; pooled over folds 1–5: **128 / 177 / 234 / 320** at h = 1 / 3 / 6 / 12. Scored by average precision on at-risk rows (base rate 0.5–1.2%).
- **Row types** for error decomposition: stay quiet (quiet → quiet; 83% of rows), onset, continuation, cessation.

### 3.4 Features (28)
| Group | Count | Features |
|---|---|---|
| Conflict history | 18 | `log_sb`, `log_ns`, `log_os` (current month, three violence types); `log_sb_lag1/2/3/6/12`; `log_sb_mean3/6/12`; `log_sb_max12`; `sb_trend` (mean3 − mean12); `log_ns_mean3`, `log_os_mean3`; `log_months_since_conflict`; `conflict_streak`; `never_conflict_seen` |
| Neighbours | 3 | `log_nbr` (neighbours' deaths this month), `log_nbr_mean3`, `n_land_nbrs` |
| V-Dem | 7 | polyarchy, libdem, civlib, clphy, rule, corr, regime |

- 72,079 of 75,703 rows (95.2%) are usable forecast origins: each needs a full 12-month history and all seven V-Dem indices (rows missing any index are dropped, not imputed: Bahamas, Belize, Brunei have none; Bahrain lacks two for 1990–2005).
- **No calendar-year or time-index feature** (a tree could split on it and then fail on years it has never seen).
- **Leakage audit:** every feature is computed from rows at or before the origin month; a truncation test (delete the future, recompute) changes 0 of 28 features.

---

## 4. Validation design and conventional baselines

### 4.1 Rolling-origin validation with a locked test set
| Fold | Train (targets in) | Validate (target months) |
|---|---|---|
| 1 | 1989–2005 | 2006–2008 |
| 2 | 1989–2008 | 2009–2011 |
| 3 | 1989–2011 | 2012–2014 |
| 4 | 1989–2014 | 2015–2017 |
| 5 | 1989–2017 | 2018–2020 |
| **Test (locked)** | 1989–2020 | **2021–2025, scored once, at the very end** |

- **Splits are on the target month.** A training row whose target falls in the validation window is excluded. `src/eval/splits.py` refuses to return the test mask unless the caller passes `final_evaluation_i_understand=True`.
- A random year-mixed split was considered and rejected: neighbouring months leak answers (a reservoir state at *t* contains deaths at *t*, so "state in Feb → deaths in March" literally contains the answer to a March test row), inflating every score and compressing the gaps the research question lives in.
- **Equal tuning budget for every model** (random search, same number of configurations), so a fly advantage cannot be extra tuning. All hyperparameter search happens inside folds 1–5.
- Rows per horizon: 71,904 / 71,554 / 71,030 / 69,988 (h = 1 / 3 / 6 / 12); validation rows pooled over the five folds: 30,632 / 30,622 / 30,607 / 30,577.

### 4.2 Metrics (two RMSE definitions, always label which)
- **Mean of fold RMSEs** (used for leaderboards and "memory recovered"). 
- **Pooled RMSE** over all validation rows (used for row-type tables and bootstraps). The two differ by about 0.0005–0.001.
- MAE is reported but ranks models differently: the naive baseline predicts exact zero in quiet cells and has the lowest MAE of all.
- **Paired comparisons** use a country-cluster bootstrap (resample countries with replacement, each keeps all its rows; 2,000 resamples; 95% interval), because a country's rows are strongly dependent. For onsets, the same bootstrap is applied to average precision (`cluster_bootstrap_ap_diff`; verified equal to scikit-learn's AP, ties included).

### 4.3 Naive baselines (notebook 03; h = 1)
| Baseline | RMSE (mean of folds) |
|---|---|
| zero | 1.462 |
| persistence | 0.582 |
| mean of last 3 months | 0.526 |
| mean of last 12 months | 0.552 |

Persistence is **not** the strongest naive baseline: the 3-month mean beats it in all five folds. The naive baseline is therefore tuned: a trailing-mean window per horizon from {1, 2, 3, 4, 6, 9, 12, 18, 24}, chosen **4 / 6 / 9 / 6** months for h = 1 / 3 / 6 / 12.

### 4.4 Standard models (notebooks 05–06)
Ridge (alpha 1e-3–1e4, log), random forest (100–400 trees, depth, leaf size, feature and sample fractions) and histogram gradient boosting (learning rate, iterations, depth, leaves, L2, feature fraction), **40 random configurations each**, identical draws at every horizon, scored on folds 1–5.

| h | naive (best window) | ridge | random forest | grad. boosting |
|---|---|---|---|---|
| 1 | 0.5282 | **0.5033** | 0.5048 | 0.5073 |
| 3 | 0.5892 | **0.5707** | 0.5783 | 0.5824 |
| 6 | 0.6320 | **0.6216** | 0.6322 | 0.6354 |
| 12 | 0.7093 | **0.6887** | 0.7012 | 0.7101 |

- Ridge skill vs naive: 0.047 / 0.031 / 0.016 / 0.029. Ridge wins everywhere with heavy regularisation; trees add no signal over linear.
- **Paired tests (06):** ridge beats naive with an interval excluding 0 at h = 1, 3, 12 (differences −0.025, −0.018, −0.020); at h = 6 the interval just spans 0 (−0.010, [−0.023, 0.001]). Random forest and boosting are never reliably better than ridge.
- **Where the gain is:** stay-quiet and cessation rows; onset RMSE is about the same as naive (h=1: 1.82 ridge vs 1.79 naive). Onset average precision 0.095–0.158 vs base rates 0.5–1.2% (about 8–20x chance); models clearly beat naive only at h = 12.
- **The bar for every reservoir is ridge, not naive.**
- Caveat: configurations are selected on the same folds that are reported (mildly optimistic).

---

## 5. The network

### 5.1 Dataset
**FlyWire FAFB v783 (female adult brain) with Princeton synapses**, over BANC, MCNS, MANC and MAOL. Reasons: signed edges (whole-brain neurotransmitter predictions ship in the connection table; a reservoir needs excitation and inhibition or tanh saturates), published simulation precedent (Shiu et al.; Lappalainen et al.), both hemispheres (free internal replication), direct CSV download, and a nerve cord that is irrelevant to forecasting. Princeton synapses are the Codex default since July 2025 (90.4% correct pairs on an expert-reviewed sample vs about 61.6% reported for the earlier detection), which matters because the weight matrix *is* the model. Network statistics will therefore not exactly match the 2024 papers.

Files: `connections_princeton.csv.gz`, `classification.csv.gz`, `consolidated_cell_types.csv.gz` (cell types live in the last one; join on `root_id`).

### 5.2 Pipeline conventions (settled and verified)
- **Aggregate before thresholding.** The connections file has one row per pair per neuropil: 5,342,446 rows collapse to **3,732,460 pairs** and **50,666,648 synapses**, matching the Codex headline figures exactly. Thresholding rows first would delete cells whose arbors span neuropils, i.e. Kenyon cells and MBONs.
- **The file is already thresholded at 5+** synapses per pair (pair counts are identical at thresholds 1–5; 10+ leaves 1,554,065). Threshold robustness can only be tested upward without the separate no-threshold file.
- **Sign by presynaptic neuron.** Transmitter predictions are per neuron and joined onto every row, so a "100% agreement" check is true by construction and validates nothing. Do not cite it.
- **Glutamate is inhibitory** (GluCl channels in *Drosophila*). Whole-brain neuron counts by transmitter: ACh 94,160 (68%), glutamate 22,657 (16%), GABA 18,374 (13%), serotonin 1,435, dopamine 786, octopamine 106. With glutamate excitatory the excitation:inhibition split is 85:13 (tanh saturates); inhibitory gives 68:30. The alternative is a planned robustness check, not yet run.
- **Modulators (serotonin, dopamine, octopamine) carry no fast signal:** their outgoing edges are removed from W but the neurons are kept (the dopaminergic neurons are the Tier 3 reward channel). Select dopaminergic neurons by **cell type**, not transmitter: 331 by type (307 PAM + 24 PPL) vs 281 flagged by transmitter.
- **Weights: fraction of postsynaptic input** (each neuron's inputs sum to 1); log1p, binary and raw counts are alternatives for robustness checks. Raw counts let APL's about 28-synapse edges dominate KC→KC's about 6.4 by 4x.
- **Spectral radius is a tuned hyperparameter, not a property of the fly.** All networks are rescaled to the same target radius; state this in methods.

### 5.3 Mushroom-body extraction (verified)
Pattern `^KC|^MBON|^PAM|^PPL1|^PPL2|^APL` on `primary_type` gives **5,606 neurons**: Kenyon cells 5,177 (KCg 2,489; KCab 1,772; KCapbp 916), PAM 307, MBON 96, PPL 24, APL 2. Sides: 2,811 right / 2,795 left (0.3% imbalance). Subgraph: 73,099 edges, density 0.0023, mean in-degree 13.0.

Biological sanity checks all pass: about 2,590 Kenyon cells per hemisphere; the γ > αβ > α'β' ordering; exactly 2 APL (one per hemisphere); about 35 MBON types x 2 hemispheres.

### 5.4 It is genuinely recurrent
A feedforward network is nilpotent (every eigenvalue 0) and cannot be scaled to a useful spectral radius, so recurrence had to be checked. **The largest strongly connected component is 5,535 neurons (98.7%).** Three independent sources: (1) KC→KC is the largest edge category (about 24,000 recurrent KC edges; the textbook "KCs do not interconnect" picture is hemibrain-era); (2) APL is reciprocally wired to every Kenyon cell (2,489 each way for KCg, 1,772 for KCab, 918/917 for KCapbp), a global inhibitory feedback hub that acts as built-in gain control; (3) dopaminergic loops (KC↔PPL, KC→PAM). The raw signed matrix has spectral radius 1,522.68, meaningless until rescaled.

### 5.5 The input layer
The extraction contains no projection neurons, yet PN→KC *is* the sparse random projection that justifies the framing. Found empirically: of 539 neurons outside the mushroom body that synapse onto Kenyon cells, **285 are ALPN projection neurons** (median 62 Kenyon cells each; 366,946 synapses). Each Kenyon cell receives a median of 5 PN inputs; 92% receive at least one.

**Final network: 5,891 neurons** (5,606 + 285 PNs), **97,051 connections, of which 92,642 carry signal** (7% inhibitory); largest strongly connected component 5,548 of 5,891. The 71 neurons outside the main component are retained.

### 5.6 Control networks
All controls take the fly's signed matrix and keep the node and edge counts; each isolates one thing about the wiring.

| Control | Preserves | Destroys | Question |
|---|---|---|---|
| Weight-shuffled | Exact wiring; per-neuron signs | Which synapses are strong or weak | Do strengths matter, or only topology? |
| Degree-rewired | Every neuron's in- and out-degree; per-neuron signs | Who connects to whom | Does specific wiring matter beyond hubs? |
| Random (Erdős–Rényi) | Node and edge counts; weight distribution; share of inhibitory edges | All structure including degrees and per-neuron signs | Does any structure matter? |

Verified (notebook 08): all have 92,642 connections and 7% inhibitory; shuffled shares 100% of connections with the fly, rewired 10.4%, random 0.3%; shuffled and rewired preserve degrees exactly; random does not.

---

## 6. The reservoir and its readout

**Update rule** (per country sequence, from a zero state at its first month):

x(t) = (1 − leak)·x(t−1) + leak·f(Wᵀ x(t−1) + drive(t)),  f = tanh (or identity for the linear reservoir)

with W = A·(ρ / ρ(A)); the drive reaches only the 285 PN nodes: drive(t) = input_scale · W_in u(t), with W_in a fixed random uniform(−1, 1) matrix. The state entering month *t* depends only on inputs up to *t* (information available at forecast time); nothing in the simulation looks at targets.

- **One reservoir run per country, one pooled readout** fit on all countries' states stacked. Known limitation: country *i*'s state never sees country *j*; cross-country effects enter only through the neighbour features.
- **Input scaling:** features standardised with mean and standard deviation from months ≤ 2005 (fold-1 training) only; clipped at ±5 for the reservoir drive; unclipped values go straight to the readout.
- **Readout:** ridge on `[states of 1,024 randomly chosen units, raw inputs]`, with a variant reading only the 96 MBONs. Fit from running Gram sums over nested folds (one eigendecomposition per fold gives the whole penalty path); column statistics from training rows only; penalty grid 1e-4 to 10^1.5 (12 values). A per-column `state_weight` scales how strongly states are penalised relative to raw inputs. Validated: inputs-only through this readout gives 0.5032 vs 0.5033 from scikit-learn ridge (notebook 05).
- **Hyperparameters tuned (random search, 40 draws, same seed for every network):** spectral radius ρ, input scale, leak, `state_weight`.
  - Experiment 1 (full features): ρ 0.3–1.4, input scale 0.02–2, leak 0.1–1, state weight 0.02–1.
  - Experiment 2 (history-free): ρ 0.3–1.4, input scale 0.05–5, leak 0.05–1, state weight 0.05–5.
- **Seeds:** the search uses one graph and one input-matrix seed; the best configuration is then re-run with 5 new seeds (new W_in, and new graphs for the controls).
- **Echo-state check (notebook 07):** most sampled configurations forget their starting state; one with ρ = 1.1 does not (chaotic regime). The best configuration's state gap decays to 0.03 after 60 months.

---

## 7. Experiment 1: all 28 features as inputs (notebooks 07–08)

Best mean validation RMSE (folds 1–5):

| network | h1 | h3 | h6 | h12 |
|---|---|---|---|---|
| ridge (raw features) | 0.5033 | 0.5707 | 0.6216 | 0.6887 |
| fly | 0.5026 | 0.5702 | 0.6206 | 0.6877 |
| weight-shuffled | 0.5027 | 0.5700 | 0.6208 | 0.6879 |
| rewired | 0.5019 | 0.5704 | 0.6209 | 0.6880 |
| random (ER) | 0.5025 | 0.5692 | 0.6197 | 0.6870 |

- Fly vs ridge: −0.0006 to −0.0010, intervals span 0 at every horizon. Fly vs each control: all within ±0.001 (one h=1 fly-vs-shuffled interval excludes 0, but the difference is 0.0001). Seed spread 0.0000–0.0014, the size of the between-network differences.
- **Diagnosis: the readout switches the reservoir off.** Across 40 configurations RMSE rises steadily with `state_weight` (Spearman 0.94 at h = 1 and 12), the best configurations sit at `state_weight` 0.03–0.05 near the lower bound, and spectral radius and input scale have no relationship with RMSE (|ρ| < 0.15). The median configuration equals ridge on raw features (0.5033 vs 0.5032); the best-of-search gain of about 0.0007 is what selecting among 480 (configuration, penalty) pairs on the same folds would produce.
- **Interpretation:** the 28 features contain lags and rolling means of the target, so the reservoir's fading memory duplicates information the features carry. This design **cannot show a fly advantage or a lack of one.** Do not report "wiring does not matter" from Experiment 1.

---

## 8. Experiment 2: history-free inputs (notebook 09)

### 8.1 Design
Inputs are reduced to **12 current-month values** (state-based, non-state and one-sided deaths this month; neighbour conflict this month; number of land neighbours; the seven V-Dem indices). All lags, rolling means, maxima, trends, months-since-conflict, streaks and the never-conflict flag are removed (asserted in code). Whatever the model knows about the past must come from the reservoir's own dynamics.

| Reference | What it is | Role |
|---|---|---|
| **A**: ridge, current values only | no memory | the floor |
| **B**: ridge, all 28 features | hand-built history | the ceiling |
| fly / shuffled / rewired / random reservoir | readout `[states, current values]` | how much of the A–B gap each recovers |

**Memory recovered** = (A − network) / (A − B): 0 = none, 1 = as good as hand-built history. The A–B gap is 0.064 / 0.078 / 0.085 / 0.068 RMSE at h = 1 / 3 / 6 / 12, so there is memory to recover at every horizon.

### 8.2 Results (mean validation RMSE, folds 1–5)
| model | h1 | h3 | h6 | h12 |
|---|---|---|---|---|
| A: ridge, current values | 0.5672 | 0.6491 | 0.7066 | 0.7569 |
| B: ridge, 28 features | 0.5033 | 0.5707 | 0.6216 | 0.6887 |
| fly (tanh) | 0.5051 | 0.5754 | 0.6267 | 0.6960 |
| weight-shuffled (tanh) | 0.5055 | 0.5747 | 0.6257 | 0.6956 |
| rewired (tanh) | 0.5064 | 0.5749 | 0.6261 | 0.6957 |
| random ER (tanh) | 0.5046 | 0.5738 | 0.6248 | 0.6936 |

Memory recovered (tanh): fly 0.97 / 0.94 / 0.94 / 0.89; shuffled 0.97 / 0.95 / 0.95 / 0.90; rewired 0.95 / 0.95 / 0.95 / 0.90; random 0.98 / 0.96 / 0.96 / 0.93.

### 8.3 Paired tests (pooled rows, country-cluster bootstrap; negative = first model better)
- **Fly vs A:** −0.0625 / −0.0742 / −0.0797 / −0.0606, intervals far from 0. The reservoir supplies real memory.
- **Fly vs B:** +0.0019 / +0.0048 / +0.0053 / +0.0074, all intervals span 0. The reservoir reaches the hand-built ceiling within noise but never beats it.
- **Fly vs weight-shuffled / rewired / random:** 11 of 12 intervals span 0. The exception, fly vs random at h = 3 (+0.0017, [0.0004, 0.0034]), **favours the random network**, which is also nominally best in the seed means at all four horizons. One significant result in 12 tests is what chance alone produces.
- **Seed means** (5 new seeds): fly 0.5052 / 0.5751 / 0.6261 / 0.6953; random 0.5044 / 0.5740 / 0.6251 / 0.6939; shuffled 0.5050 / 0.5745 / 0.6255 / 0.6943; rewired 0.5068 / 0.5746 / 0.6259 / 0.6949; standard deviations 0.0002–0.0012. The fly's spread is smaller partly by construction (only W_in varies; controls also redraw their graph).

### 8.4 How tight is the null? (equivalence bounds)
"Intervals span 0" invites the reply that the test lacked power. The lower end of each fly-vs-control interval is the largest fly advantage still consistent with the data; expressed as a share of the A–B gap:

| h | fly vs shuffled | fly vs rewired | fly vs random |
|---|---|---|---|
| 1 | 4.0% | 6.4% | 1.5% |
| 3 | 1.0% | 0.6% | 0% (random better) |
| 6 | 1.7% | 1.6% | 0.1% |
| 12 | 4.4% | 4.3% | 3.0% |

**A fly advantage larger than about 6% of the history gap is ruled out; at most horizons the bound is under 2%.** For the linear reservoirs (§9) the bound against random networks is about 1.3% or less.

### 8.5 Sensitivity
RMSE no longer depends on `state_weight` (|Spearman| ≤ 0.23, down from 0.94 in Experiment 1), so the readout now uses the states. RMSE still rises strongly with `input_scale` (Spearman 0.87–0.93; best values 0.05–0.08 at the bottom of the range), so the tanh networks prefer the near-linear regime. That motivated §9.

---

## 9. Where does the memory come from? (mechanism)

A linear reservoir computes weighted moving averages of its inputs, with memory set by the eigenvalues of (1 − leak)I + leak·Wᵀ; rescaling every network to the same spectral radius makes those spectra similar, so a tie would be expected. Four checks (notebook 09 §9, §9c):

| Model | Removes | Search |
|---|---|---|
| Linear reservoir (fly and controls) | the nonlinearity (identity activation; ρ ≤ 0.95) | 20 draws |
| `bank_single` | all recurrence: 1,024 independent leaky units, one shared leak rate | 20 draws |
| `bank_multi` | all recurrence; each unit has its own leak (log-uniform from the tuned slowest rate to 1) | 20 draws |
| EMA baseline | the reservoir entirely: ridge on exponential moving averages of the 12 inputs at 3 or 6 leak rates, plus raw inputs | ridge penalty only |

### 9.1 Results
| model | h1 | h3 | h6 | h12 |
|---|---|---|---|---|
| fly (linear) | 0.5027 | 0.5729 | 0.6237 | 0.6926 |
| random ER (linear) | 0.5027 | 0.5722 | 0.6225 | 0.6916 |
| bank_single | 0.5084 | 0.5763 | 0.6285 | 0.6963 |
| **bank_multi** | **0.5023** | 0.5736 | 0.6241 | 0.6919 |

Memory recovered: linear 0.94–1.01 (fly) and 0.96–1.01 (random); `bank_multi` 0.95–1.02; `bank_single` 0.89–0.93. Values above 1 are within noise of B; do not write "better than the hand-built history". Seeds (3 new): standard deviations 0.0001–0.0005.

### 9.2 What the paired tests say
- **Linear vs tanh (fly):** linear is better by 0.0024–0.0035, same sign at all horizons, individually not significant. The linear search had 20 draws against 40 for tanh, so this comparison is conservative.
- **Fly (linear) vs random (linear):** +0.0000 to +0.0012, none significant; random is nominally better at every horizon. Worst-case fly advantage 1.3% / 0.4% / 0.5% / 1.1% of the A–B gap.
- **Fly (tanh) vs `bank_multi`:** +0.0019 to +0.0040 (bank better), none individually significant. **`bank_multi` vs B:** −0.0010 to +0.0034, all intervals span 0: a network with no connections reaches the hand-built ceiling.
- **Equivalence bounds against the unconnected bank** (largest plausible fly advantage as a share of the A–B gap, h = 1/3/6/12): fly (tanh) vs `bank_multi` 0.1% / 0.6% / 0.1% / 6.1%; fly (linear) vs `bank_multi` 0.8% / 4.8% / 4.1% / 8.9%. The h=12 bound is the loosest because the 12-month errors are the noisiest.
- `bank_multi` beats `bank_single` (significant at h=1: −0.0061), so many time scales matter more than any recurrent structure.

### 9.3 Edge-of-range notes
The best ridge penalty is at the bottom of its grid for the linear and bank models at h=1 (and for shuffled and rewired tanh at h=1); the effect is probably negligible but untested. `bank_multi`'s slowest leak is 0.012 at h=12 against a lower bound of 0.01, and RMSE still improves with slower leaks (Spearman +0.65).

### 9.4 EMA baseline (notebook 09 §9c; confirmed)
Ridge on exponential moving averages of the 12 inputs plus the raw inputs. There are no reservoir units at all. Results from `results/memory_ema_rmse.csv` (they match the earlier exploratory run to four decimals):

| model | h1 | h3 | h6 | h12 |
|---|---|---|---|---|
| A: ridge, current values | 0.5672 | 0.6491 | 0.7066 | 0.7569 |
| fly (linear) | 0.5027 | 0.5729 | 0.6237 | 0.6926 |
| random ER (linear) | 0.5027 | 0.5722 | 0.6225 | 0.6916 |
| `bank_multi` | 0.5023 | 0.5736 | 0.6241 | 0.6919 |
| **EMA x3 (leaks 0.5, 0.15, 0.04)** | 0.5056 | 0.5733 | 0.6239 | 0.6916 |
| **EMA x6 (leaks 0.7 to 0.012, log-spaced)** | 0.5059 | 0.5737 | 0.6242 | 0.6917 |
| B: ridge, 28 features | 0.5033 | 0.5707 | 0.6216 | 0.6887 |

**Paired tests** (`results/paired_ema.csv`; RMSE(EMA x3) − RMSE(other), country-cluster bootstrap; positive = EMA worse):
- vs fly (linear): +0.0029 / +0.0004 / +0.0002 / −0.0009; vs random ER (linear): +0.0029 / +0.0011 / +0.0014 / +0.0001. None significant.
- vs `bank_multi`: +0.0033 at h=1 (95% CI 0.0002 to 0.0088, the only significant EMA comparison: the bank's many time scales help slightly at h=1); −0.0003 / −0.0001 / −0.0004 at h = 3/6/12, not significant.
- vs B (28 features): +0.0023 to +0.0030, none significant. EMA x6 vs EMA x3: within 0.0004, none significant.

**Reading.** Moving averages with no units, no connections and no nonlinearity recover about the same share of the history gap as every reservoir. The only measurable shortfall is at h=1 (about 0.003 behind the reservoirs, which have more time scales). Only the ridge penalty is chosen for the EMA baseline (12 values), a smaller selection budget than the reservoirs' (20–40 draws x 12 penalties), which if anything favours the reservoirs.

### 9.5 Conclusion of the mechanism checks
What the reservoirs recover is multi-time-scale leaky averaging of the inputs. A bank of unconnected leaky units supplies it equally, and so do plain moving averages. Reference B's lags and rolling means already approximate it, so no reservoir beats B. The connectome's wiring plays no detectable role.

---

## 10. The biological readout (MBONs)

The 96 MBONs are the fly's own readout layer (near-pure sinks receiving about 27,000 Kenyon-cell edges, where learned valence is expressed). Compared against five readouts of 96 randomly chosen units, tuned on the same 40 configurations (fly network, history-free inputs):

| h | 96 MBONs | random 96 (mean; best–worst of 5) | 1,024 random units |
|---|---|---|---|
| 1 | 0.5125 | 0.5058 (0.5047–0.5064) | 0.5051 |
| 3 | 0.5797 | 0.5752 (0.5746–0.5758) | 0.5754 |
| 6 | 0.6307 | 0.6266 (0.6259–0.6272) | 0.6267 |
| 12 | 0.7003 | 0.6954 (0.6946–0.6963) | 0.6960 |

The MBONs rank **last of six at every horizon**, 0.005–0.007 worse, several times the spread among the random sets. The biological readout is not special for this task. Caveat: the random sets are drawn from all 5,891 nodes, about 88% of which are Kenyon cells, so they sample the driven layer directly.

---

## 11. Onset detection (exploratory)

Average precision of regression predictions ranking at-risk rows (pooled folds 1–5; events 128 / 177 / 234 / 320).

- Only **h = 12 vs B** is significant: fly +0.032 [0.005, 0.054], random +0.038 [0.003, 0.062]. Not significant at h = 1, 3 or 6, and fly vs A (no memory) is not significant at any horizon.
- By fold at h=12 the gain comes from folds 1, 3 and 4 (e.g. fold 3: fly 0.165 vs B 0.123); folds 2 and 5 are flat.
- It is a reservoir effect, not a fly effect (random is as good), and two significant results among 16 correlated tests is thin. Report as exploratory.

---

## 12. Interpretation

**What Tier 1 shows.**
1. A fixed recurrent reservoir can supply the temporal memory of a conflict forecasting model; with history-free inputs it recovers 89–102% of what hand-built lags provide.
2. The fly connectome does this no better than shuffled, rewired or random networks. A fly advantage above 1–6% of the history gap is ruled out.
3. The memory is leaky integration at many time scales. It does not need recurrence (unconnected leaky units match), a nonlinearity (linear is as good), or a reservoir at all (moving averages match; §9.4).
4. The signal in this task looks close to linear in a multi-scale history of log deaths: random forest and gradient boosting did not beat ridge, tanh did not beat linear, and nothing beats B.

**What Tier 1 does not show.**
- That the fly cannot learn, or that learning would not make its wiring matter. Tier 1 froze everything; that is the point of Tier 2.
- That wiring never matters. It does not matter *for a frozen reservoir on this target*. A task with real nonlinear structure (Tier 2b) may differ.
- Anything about the locked test set, the ViEWS benchmark, or a neural-network baseline (not yet run).

**Implication for Tier 2.** Because the data show no sign of nonlinear structure to exploit, a trained network on this target may also tie ridge. Tier 2b (a diagnostic benchmark that requires nonlinear memory) is how to separate "no nonlinear signal in the data" from "the fly cannot learn".

---

## 13. Limitations and disclosures (for the methods section)

**Data and design**
- Per-country reservoirs: a country's state never sees other countries; cross-country effects enter only through the neighbour features.
- Year-long GED events flatten into plateaus, which flatters persistence-type baselines. Nov–Dec 2025 GED months are thin (state-based −23% / −14%; one-sided December −38%) and sit in the locked test window: report the test with and without them.
- Rows missing any V-Dem index are dropped (1.8% of rows had a missing index).
- GED's own low ≤ best ≤ high ordering fails for 1.2% of events (a source quirk, reported not fixed); low/high are not used.

**Tuning and statistics**
- Configurations are selected on the same folds that are reported (mildly optimistic). Paired tests use the seed-0 best configuration; use the seed means as the headline.
- Search budgets differ: tanh 40 draws; linear and bank 20 each; EMA baseline ridge penalty only.
- One graph and one input-matrix seed during tuning; the readout uses 1,024 random units (not all 5,891).
- Two RMSE definitions (§4.2): label every table.
- The fingerprint guard on tuning logs covers the inputs and configuration lists, not the reservoir or readout code, the penalty grid or the readout node set. Delete the logs after changing those.
- After degree-preserving rewiring the "fraction of postsynaptic input" normalisation no longer holds exactly (weights stay with the source neuron); harmless because every network is rescaled to the same spectral radius.

**Not yet done (planned robustness checks)**
APL ablation; glutamate sign flip; weight-scheme comparison (log1p, binary, raw); hemisphere replication; MCNS replication; upward threshold sweep; MLP baseline; ViEWS comparison; locked-test evaluation. The APL ablation and sign flip are deprioritised while the fly ties random networks.

---

## 14. Reproducibility

### 14.1 Environment and running
Python 3.14, Windows / VS Code; `pip install -r requirements.txt` (pandas, pyarrow, requests, matplotlib, scipy, scikit-learn, ipykernel). Run notebooks 01 to 09 in order. Notebook 01 needs the V-Dem Core CSV downloaded by hand (instructions inside). Every modelling notebook has a `QUICK_TEST` flag (default False) for a fast smoke test that writes `*_quicktest*` files (gitignored). Tuning logs are resumable; they are regenerable caches and are not committed, so a fresh clone recomputes them (several hours on a laptop; notebook 09's mechanism section alone about 2–3 hours).

### 14.2 Notebook map
| # | Notebook | Purpose | Key outputs |
|---|---|---|---|
| 01 | `01_download` | Fetch data, record versions and hashes | `data/raw/MANIFEST.json` |
| 02 | `02_clean_join_describe` | Build and verify the panel | `data/processed/panel_base.parquet` |
| 03 | `03_folds_and_baselines` | Rolling-origin folds, naive baselines | |
| 04 | `04_features` | 28 features, leakage audit | `features.parquet` |
| 05 | `05_models_random_search` | Naive, ridge, forest, boosting (40 draws each) | `tuning_log.csv`, `leaderboard.csv`, `best_configs.json`, `onset_ap.csv`, `baseline_predictions.parquet` |
| 06 | `06_paired_comparison` | Cluster-bootstrap tests of baselines | `paired_baselines.csv` |
| 07 | `07_fly_reservoir` | Network build, PN layer, fly tuning (full features) | `reservoir_log.csv`, `reservoir_best_fly.csv` |
| 08 | `08_controls_and_comparison` | Controls, tuning, seeds, paired tests | `paired_reservoir.csv`, `reservoir_by_kind.csv`, `reservoir_seeds.csv` |
| 09 | `09_memory_test` | History-free test; bounds; onset AP; mechanism; MBON vs random; EMA baseline | `memory_*` files, `paired_memory.csv`, `paired_mechanism.csv`, `memory_bounds*.csv`, `memory_onset_ap.csv`, `memory_mbon_vs_random96.csv`, `paired_ema.csv` |

### 14.3 Code map
`src/data/` (download, panel) · `src/features/build.py` (28 features) · `src/eval/` (`splits.py` folds and locked test; `metrics.py`; `paired.py` cluster bootstrap for RMSE and AP) · `src/models/` (`common.py`; `readout.py` ridge path; `reservoir_data.py`; `reservoir_search.py` lab, search, banks, EMA baseline, fingerprint guard) · `src/networks/` (`connectome.py` loader and weight matrix; `controls.py` the three controls; `reservoir.py` simulation and `ema_bank`).

### 14.4 Seeds and determinism
Search seed 42 (identical configuration draws for every network); input matrix `W_in` seeded `1000 + seed`; readout subsets seeded from the lab seed (42); control graphs seeded by `graph_seed`. A full re-run of notebooks 05–09 reproduced every earlier number exactly (2026-10-02). Commit `1f455e8` is the first full computing run.

### 14.5 Verification already performed
Pair and synapse totals match published Codex figures; the mushroom body is 98.7% one strongly connected component; truncation leakage audit (0 of 28); readout cross-check against scikit-learn (0.5032 vs 0.5033); refactored reservoir code reproduces logged results exactly; AP bootstrap equals scikit-learn's AP; all notebooks run clean; pooled and fold-mean RMSE for every model recomputed independently from saved predictions.

---

## Appendix A. Decision register

| Date | Decision | Reason |
|---|---|---|
| 2026-09-19 | Fly connectome + conflict is the class project; RL bargaining is the follow-on | Stocks too efficient; Epstein matching not interesting; three-tier design makes "how much the network learns" the experimental variable |
| 2026-09-19 | Conflict over migration | Tens of thousands of examples vs about 300; published benchmarks exist |
| 2026-10-01 | Load the connectome before the conflict data | The mushroom body could have been feedforward; find out before investing in data |
| 2026-10-01 | FAFB v783 with Princeton synapses | Signed edges, precedent, both hemispheres, access, accuracy (§5.1) |
| 2026-10-01 | Aggregate pairs before thresholding; sign by presynaptic neuron; glutamate inhibitory; modulators silent in W | §5.2 |
| 2026-10-01 | UCDP GED 26.1 frozen annual data; Candidate data only for the live check | Preliminary data is revised at each release |
| 2026-10-01 | Gleditsch–Ward universe; zero-fill only within lifespans | No fake zeros for states before they exist |
| 2026-10-01 | Multi-month events spread evenly by day | 24% of deaths sit in 1.7% of events; assigning to the start month would put years of deaths in one month |
| 2026-10-01 | Per-country reservoir runs with a pooled readout | One run with a 190-country vector gives only about 440 time steps |
| 2026-10-01 | Rolling-origin validation, locked 2021–2025 test; random split rejected | Neighbouring months leak answers (§4.1) |
| 2026-10-01 | Equal tuning budget for every model | A fly advantage must not be extra tuning |
| 2026-10-02 | Target: log(1 + state-based deaths); horizons 1, 3, 6, 12; onset k = 25, m = 6 | §3.3 |
| 2026-10-02 | Weights = fraction of postsynaptic input; spectral radius tuned | §5.2 |
| 2026-10-02 | Input layer = 285 ALPN projection neurons found empirically | PN→KC is the sparse random projection |
| 2026-10-02 | Bar for reservoirs is ridge, not naive | Ridge is the best conventional model |
| 2026-10-02 | Add the history-free memory test (notebook 09) | Experiment 1 could not separate the reservoir from the features |
| 2026-10-02 | Add mechanism checks, equivalence bounds, MBON size-matched readout, EMA baseline | Controls tied; need to explain why |
| 2026-10-02 | Null result is the headline, framed as a bound with the mechanism as the explanation | §1, §8.4, §9 |
| 2026-10-02 | Tier 2 stays in scope | The aim is to test whether the fly learns when its weights and connections can change |

## Appendix B. Glossary
**Reservoir / echo-state network:** a fixed recurrent network whose state summarises the input history; only a readout is trained. **Spectral radius (ρ):** the largest absolute eigenvalue of the recurrent matrix; controls how long memory lasts. **Leak:** the fraction of each unit's state replaced at every step; small leak means slow time scale. **Readout:** the trained linear map from states to the target. **Ridge:** least squares with an L2 penalty. **Strongly connected component:** a set of neurons each reachable from every other by directed paths. **MBON:** mushroom-body output neuron (the fly's own readout layer). **APL:** the single giant GABAergic neuron per hemisphere that inhibits every Kenyon cell. **ALPN:** antennal-lobe projection neuron (the input layer). **Memory recovered:** (A − network)/(A − B). **Country-cluster bootstrap:** resampling whole countries to get intervals that respect within-country dependence.
