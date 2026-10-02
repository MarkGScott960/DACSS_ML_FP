---
aliases:
  - Fly Conflict Project
  - Final Project Guide
---
# Final Project Guide: Fly Connectome Conflict Forecasting
Working guide for the DACSS 756 final project and the RL bargaining follow-on. See [[Final Project]] for the original sketches (Idea 1: fly trades stocks, Idea 2: fly forecasts conflict). See [[ML FP Progress Log]] for session-by-session diagnostics and verification results.

**Status as of 2026-10-01:** connectome loaded and verified; mushroom body confirmed as a viable reservoir; conflict data not yet acquired.

## The Project in One Paragraph
Use the fly [[Connectome]] (FlyWire FAFB v783) as a recurrent neural network to forecast armed conflict at the country-month level. Test whether biological wiring gives a useful inductive bias compared with random networks and conventional models, under three increasingly flexible learning regimes.

**Research question:** Does biological wiring improve learned forecasts of armed conflict, and does it still matter once the network is allowed to learn?

**Pitch line (for Dr. Adler / proposal):** I test whether a biologically evolved network architecture improves learned forecasts of armed conflict, compared with random and conventional architectures, under three increasingly flexible learning regimes: supervised readout, gradient-trained weights, and reward-modulated plasticity.

## Portfolio Arc
Theme: **learning systems applied to international conflict.**
1. **This project (class):** Can biological neural architectures forecast conflict?
2. **Follow-on (self-study):** Do RL agents rediscover rationalist theories of war? (see [[#RL Agent Thread]])

Split: fly + conflict is the class project; RL bargaining is the sequel, kept in mind throughout.

## Key Concepts
### [[Reservoir Computing]]
- A large recurrent network with **fixed** wiring (the reservoir) transforms an input time series into a high-dimensional state
- Only a **readout** (usually ridge regression) is trained, mapping reservoir states to the target
- Works because of:
	- **Fading memory:** recurrence means the state encodes recent input history. *This requires genuine recurrence — a feedforward (acyclic) network has all eigenvalues 0 and cannot be scaled to a useful spectral radius.*
	- **Nonlinear expansion:** few inputs become many nonlinear features, so patterns become linearly separable (same intuition as the kernel trick)
- Update rule: $x(t) = \tanh(W x(t-1) + W_{in} u(t))$, readout $y(t) = W_{out} x(t)$
- **Spectral radius** (largest eigenvalue of $W$) controls memory
	- Too low: echoes die fast, little memory
	- Above ~1: chaotic, state stops reflecting inputs
	- Just under 1: long memory still driven by input ("edge of chaos")
	- **Echo state property:** state depends on input history, not initial conditions
- Other hyperparameters: reservoir size, input scaling, sparsity, leak rate, ridge penalty
- Two families:
	- Echo state networks (Jaeger 2001): rate neurons, tanh, fast
	- Liquid state machines (Maass 2002): spiking neurons, realistic, slow
- In a standard ESN, $W$ is random. **Here, $W$ = the fly's adjacency matrix.**

### Why the Mushroom Body Specifically
The mushroom body is independently a reservoir-like structure, which is the theoretical hook for the whole project:
- ~50 olfactory projection neurons feed ~2,000 Kenyon cells per hemisphere through sparse, quasi-random connectivity — a biological **random projection / nonlinear expansion**
- Dasgupta, Stevens & Navlakha (*Science*, 2017) argued this implements a variant of locality-sensitive hashing
- The readout layer (MBONs) is trained by dopaminergic neurons signalling reward/punishment — a readout trained on a fixed expansion
- **Framing:** evolution converged on reservoir computing. The project tests whether its particular solution outperforms the random projections we use by default.

### Where the Machine "Learns"
- **Architecture** = structure chosen before training. The connectome is the architecture, designed by evolution (not ML).
- **Parameters** = numbers fit to data. This is the learning, and it differs by tier.

## Three Tiers of Learning
| Tier | What's fixed | What learns | Learning type | Course link |
|---|---|---|---|---|
| 1. Frozen reservoir | All connectome weights | Readout only | [[Supervised Learning]] (ridge regression) | Weeks 4–5 regression; [[Feature Extraction]] |
| 2. Connectome-constrained | Which neurons connect | Weights on existing connections | Gradient descent / backprop | Week 10 neural nets |
| 3. Mushroom body plasticity (stretch) | Everything except MB synapses | MB synapses via dopamine-like reward | [[Reinforcement Learning]] | Bridge to the RL follow-on |

- Tier 2 precedent: Lappalainen et al. (Nature 2024) trained a connectome-constrained fly visual system with backprop
- Tier 3: the mushroom body is where real flies learn; dopamine acts as reward/punishment
- **Deliverable = Tiers 1 + 2 + controls. Tier 3 = extension.**
- **Tier 2 kept in scope (decided 2026-10-02, after Tier 1 tied the controls).** The question: can the connectome *learn* when its weights (and possibly its connections) are allowed to change? Open design points, to settle after notebook 09's mechanism results:
  - which weights learn: all existing connections, only KC→MBON (where real flies learn), or also adding/removing connections (structural plasticity);
  - how: backpropagation through time on the forecasting loss, starting from the fly weights;
  - **the fair control:** the same training, same budget, starting from the shuffled / rewired / ER networks. "The fly learns" means nothing without "and a random network given the same training learns less".

## Dataset Decision (SETTLED)
**FAFB v783 — FlyWire Female Adult Fly Brain — with Princeton synapses.**

### Why FAFB over the alternatives
Codex currently offers FAFB (female brain), BANC (female brain + nerve cord), MCNS (male brain + nerve cord), MANC (male nerve cord), and MAOL (male optic lobe).

| Factor | Reasoning |
|---|---|
| **Signed edges** | A reservoir needs excitation *and* inhibition, or tanh saturates and the state carries no information. FAFB has whole-brain neurotransmitter predictions shipped in the connection tables. |
| **Published precedent** | Both simulation papers in the reading list (Shiu et al. whole-brain LIF; Lappalainen et al. connectome-constrained backprop) are built on FAFB. "I followed the published pipeline" beats "I made some choices." |
| **Both hemispheres** | Free internal replication: fit on one side, check on the other. Hemibrain is one hemisphere and only ~1/3 of the central brain. |
| **Access friction** | Codex offers direct downloadable CSVs for the FlyWire datasets. MCNS/MANC route through neuPrint (account + token). |
| **Nerve cord is dead weight** | The VNC is leg/wing motor circuitry. For forecasting a country-month panel it adds neurons that are hard to justify. Brain-only is the cleaner story. |

**BANC rejected:** newer, documented dissection damage to both antennal nerves, less proofread, far thinner precedent.
**MCNS reserved as a replication target** — see Robustness below.

### Why Princeton synapses
FlyWire has two generations of synapse detection on the same proofread segmentation:
- Original: Buhmann et al. method, refined with Heinrich et al. — what the 2024 Nature papers used
- **Princeton synapses (Yu et al.) — Codex default since July 2025**

Accuracy on an expert-reviewed sample of 1,000 predicted synapses: Princeton 90.4% correct pairs vs ~61.6% reported for the earlier connectome.

This matters more here than for most analyses because **the weight matrix *is* the model** — synapse-count errors propagate through the recurrent dynamics at every time step rather than averaging out into a summary statistic.

**Methods-section consequence:** network statistics will not exactly match the published 2024 figures (those are Buhmann-era). State the synapse detection, snapshot, and threshold explicitly.

### Files
From `https://storage.googleapis.com/flywire-data/codex/data/fafb/783/`:
- `connections_princeton.csv.gz` — the edge list. Columns: `pre_root_id, post_root_id, neuropil, syn_count, nt_type`
- `classification.csv.gz` — per-neuron metadata. Columns: `root_id, flow, super_class, class, sub_class, hemilineage, side, nerve`
- `consolidated_cell_types.csv.gz` — Columns: `root_id, primary_type, additional_type(s)`

**Cell types live in `consolidated_cell_types`, not `classification`.** Join on `root_id`.

Skip: Marked Neuron Coordinates (figures only), Synapse Table (per-synapse point cloud, ~11GB).

## Pipeline Conventions (SETTLED)
### Aggregate before thresholding
The connections file has **one row per pair per neuropil**, so 5,342,446 rows collapse to 3,732,460 unique pairs. Codex defines a connection by the pair's *total* synapse count, so:

> group by `(pre_root_id, post_root_id)` → sum `syn_count` → then threshold.

Thresholding rows first preferentially deletes neurons whose arbors span multiple neuropils — which in the mushroom body means Kenyon cells (calyx → peduncle → lobes) and MBONs/DANs tiling lobe compartments. Exactly the cells that matter.

### The file is already thresholded at 5+
Verified: pair totals at thresholds 1, 2, 3, and 5 all return 3,732,460. No pair below 5 exists. The `min syn_count = 1` visible in the raw rows is a per-neuropil fragment, not a pair total.

**Consequence:** the threshold-robustness check can only go *upward* (10 → 1,554,065 pairs, a 58% cut). Testing lower thresholds requires downloading the "Connections Princeton No Threshold" file separately. Optional extension, not already in hand.

### Sign by presynaptic neuron, not by edge
A neuron generally releases the same transmitter at all terminals, so mixed-sign outputs from one neuron reflect classifier noise, not biology. The `nt_type` column is already assigned per neuron and joined onto every connection row, so deduplicating by `pre_root_id` is sufficient.

### Glutamate is INHIBITORY (decided)
In *Drosophila*, glutamate frequently acts through glutamate-gated chloride channels (GluCl) — the opposite of the vertebrate intuition. The counts make the modeling case too:

| Convention | Excitatory | Inhibitory |
|---|---|---|
| Glutamate excitatory | 116,817 (85%) | 18,374 (13%) |
| **Glutamate inhibitory** | **94,160 (68%)** | **41,031 (30%)** |

85:13 is badly unbalanced — recurrent activity runs away and tanh saturates. 68:30 is a healthy regime. Run the alternative as a robustness check.

```python
SIGN = {"ACH": +1, "GLUT": -1, "GABA": -1,
        "SER": 0, "DA": 0, "OCT": 0}   # modulators excluded from W
```

### Modulators are a separate channel, not weights
Serotonin, dopamine, octopamine (2,327 neurons brain-wide) are neuromodulators, not fast transmitters. Keep them out of $W$ but **do not discard them** — the dopaminergic neurons are the Tier 3 reward pathway.

**Select DANs by cell type, not neurotransmitter.** Cell-type identity is annotated and cross-validated; neurotransmitter is a prediction. In the MB: 331 DANs by type (307 PAM + 24 PPL) vs only 281 flagged by `nt_type` — ~50 carry a non-monoamine prediction, either classifier error or documented co-transmission in PAM neurons.

```python
meta["is_dan"] = meta[type_col].astype(str).str.match(r"^(PAM|PPL)", na=False)
```

### Loader contract
Three outputs, so Tier 3 needs no refactor:
1. signed weight matrix $W$
2. mask of modulatory neurons (the reward channel)
3. neuron metadata (type, side, super_class)

Tiers 1–2 ignore the mask; Tier 3 uses it.

### Weight scaling (SETTLED for first runs 2026-10-02: fraction of postsynaptic input; others are robustness checks)
Raw synapse counts let APL's ~28-synapse edges dominate KC→KC's ~6.4 by 4×.

| Scheme | Effect |
|---|---|
| Raw counts | Strong edges dominate; APL overwhelms |
| `log1p(count)` | Compresses range, keeps ordering |
| Binary | Pure topology, discards strength |
| **Fraction of postsynaptic input** | Each neuron's inputs sum to 1; standard in connectome signal-propagation models |

**Lean:** fraction of postsynaptic input as default, `log1p` as robustness comparison, binary once as an ablation (pairs neatly with the weight-shuffled control — "does synapse count matter or only topology?").

### Spectral radius
Raw signed MB matrix has spectral radius **1,522.68**, which is meaningless on its own. Scale it:
```python
A_scaled = A * (rho_target / 1522.683)
```
Treat `rho_target` as a tuned hyperparameter over ~0.5–1.3. **State in methods that it is tuned, not a property of the fly** — this is the first thing a sharp reader will probe.

## The Circuit (VERIFIED)
### Mushroom body extraction
Pattern: `^KC|^MBON|^PAM|^PPL1|^PPL2|^APL` against `primary_type`.

| Component | Count | Note |
|---|---|---|
| Kenyon cells | 5,177 | 2,489 KCg + 1,772 KCab + 916 KCapbp (~2,590/hemisphere) |
| PAM dopaminergic | 307 | reward channel |
| MBON | 96 | ~35 types × 2 hemispheres |
| PPL dopaminergic | 24 | punishment channel |
| APL | 2 | exactly one per hemisphere |
| **Total** | **5,606** | 2,811 right / 2,795 left (0.3% imbalance) |

Subgraph: 73,099 edges, density 0.0023, **mean in-degree 13.0** (not 26.1 — that was the undirected formula applied to a directed graph).

### It is genuinely recurrent
**Largest strongly connected component: 5,535 neurons = 98.7% of the subgraph.** Three independent sources of recurrence:

1. **KC→KC is the largest edge category.** KCab→KCab alone is 11,590 edges; with KCg→KCg and KCapbp→KCapbp, ~24,000 recurrent KC edges. The textbook "KCs don't interconnect" picture is largely hemibrain-era; the whole-brain reconstruction at 5+ synapses shows otherwise.
2. **APL closes a loop through every Kenyon cell.** Perfectly symmetric: 2,489 KCg→APL / 2,489 APL→KCg; 1,772 each way for KCab; 918/917 for KCapbp. Two GABAergic neurons reciprocally wired to all 5,177 KCs, with strong edges (~25–29 synapses vs ~6.4 for KC→KC).
3. **DAN loops:** KC→PPL (1,092) and PPL→KC (1,085); KC→PAM (777+1,150).

**APL is the fly's own gain control** — global inhibitory feedback enforcing sparse coding. For a reservoir this is close to ideal: a built-in stabilizer against runaway excitation, tuned by evolution.

### Missing piece: no input layer
The extraction has no **projection neurons**. In the fly, information enters the MB through PNs synapsing onto KC dendrites in the calyx — and PN→KC *is* the famous sparse random projection. Without it, input would be injected through an invented $W_{in}$, discarding the most defensible biological claim in the proposal.

Find the input population empirically rather than guessing type names (see [[ML FP Progress Log]] for the query). Expect a few hundred olfactory PNs plus thermo/hygrosensory and visual inputs.

### MBONs are the biological readout
No MBON→anything appears in the top-20 flows — they are near-pure sinks receiving ~27,000 KC edges. That is not a defect: the MBON layer is where learned valence is expressed. Gives a free comparison:
- **Biological readout:** ridge on the 96 MBON states only
- **Full-state readout:** ridge on all ~5,600 states (standard RC)

If 96 dimensions approach 5,600, evolution picked a good 1.7% of the state space. One extra ridge fit, directly serving the research question.

## Data (BUILT — panel in `notebooks/02_clean_join_describe.ipynb`)
Settled 2026-10-01. Full reasoning in [[ML FP Progress Log#2026-10-01 — Data collection plan and validation design]].

- **Language:** Python 3.14 throughout. One environment (`requirements.txt`) for the panel, baselines, and reservoir.
- **Code format:** runnable pipeline steps are **numbered Jupyter notebooks** in `notebooks/` (`01_download.ipynb`, `02_panel.ipynb`, …), run in VS Code. Reusable functions live in `src/` and are imported by the notebooks, so logic is never copy-pasted between notebooks.
- **Primary source: UCDP GED 26.1, the frozen annual release** (events 1989–2025). Downloaded as a static file (not the API, which now requires a token). Version and file hash recorded at download.
- **Not the monthly Candidate data for modelling.** Candidate events are preliminary and are revised at each annual release; scoring test months against data that later changes is unsound. Candidate data is used **only** for the live November 2026 check (see Validation Design).
- **ViEWS:** benchmark forecasts for comparison, added after the baselines exist.
- **ACLED:** not used for now (licensing, and GED is sufficient).
- **Unit of analysis:** country-month panel. **Universe: Gleditsch–Ward state list** (matches GED's `gwno` codes), with each state present only between its entry and exit dates. Peaceful months are zero-filled **only while the state exists** — no fake zeros for South Sudan before 2011, post-Soviet states before 1991, etc.
- **Size (verified):** 178 states × up to 444 months = **75,703 country-months**. Clears the course minimum of 10,000 by 7.5×.
- **Primary target (SETTLED): regression on log(1 + state-based deaths) next month.** Onsets reported as an evaluated subset.
- **Violence type:** state-based (type 1) is the target. Non-state (2) and one-sided (3) enter as input features.
- **Fatalities:** `best` estimate; `low`/`high` as a one-off robustness check.
- **Multi-month events:** assigned to the month of `date_start`; report how many events span more than one month.
- **Forecast horizons (SETTLED 2026-10-02):** 1, 3, 6 and 12 months, all reported.
- **Onset definition (SETTLED 2026-10-02):** at risk if fewer than k=25 deaths in each of the previous m=6 months; event is >=25 deaths at the target month. Sensitivity: m=12 leaves only 14–20 onsets per fold, and one-month blips stay 60–70% under every definition. Onset results are an evaluated subset (average precision on at-risk rows), not the primary target.

### Features (target: 10+)
All features are lagged so the model only sees information available at forecast time.

| Source | Features |
|---|---|
| GED | State-based, non-state, one-sided deaths; event counts; 3/6/12-month rolling means; months since last state-based event |
| COW | Direct contiguity → neighbours' conflict (spatial lag) |
| V-Dem v16 Core | `v2x_polyarchy`, `v2x_libdem`, `v2x_civlib`, `v2x_clphy`, `v2x_rule`, `v2x_corr`, `v2x_regime` (Core has no population/GDP) |

- **Annual covariates are lagged one full year** when merged onto months — a year's V-Dem score is not known until the year ends. Most important for Tier 2, where a flexible model will find any leak.
- **Coverage end dates:** COW NMC likely ends 2016; V-Dem `e_` variables may end before 2025. Anything ending before the test period is dropped or given a stated fill rule. Verify actual end years at download.
- **COW ↔ Gleditsch–Ward crosswalk** needed (Germany, Yemen, Serbia/Yugoslavia are the usual trouble cases). Assert every merge matches the expected rows; silent mismatches drop or duplicate country-months.

### How the panel feeds the reservoir (SETTLED)
**One reservoir run per country, pooled readout.** Each country's monthly sequence drives the reservoir from a fresh state with a washout period; states are stacked across countries and one ridge readout is fit on all of them. The alternative — one run with a ~190-country input vector per month — gives only ~440 time steps, far too few for Tier 2.
- Requires the panel stored long, sorted by country then month, with no internal gaps.
- **Known limitation (name it in methods):** country *i*'s state never sees country *j*. Cross-country effects enter only through input features — one reason the neighbour-conflict feature matters.
- **New states** have short series partly consumed by washout. Decide: drop below a minimum length, or keep with shorter washout. (OPEN)

### Validation Design (SETTLED)
**Rolling-origin (expanding-window) cross-validation with a locked final test set.** Random splits across 1989–2025 were considered and rejected — see reasoning in the progress log.

| Fold | Train (targets in) | Validate |
|---|---|---|
| 1 | 1989–2005 | 2006–2008 |
| 2 | 1989–2008 | 2009–2011 |
| 3 | 1989–2011 | 2012–2014 |
| 4 | 1989–2014 | 2015–2017 |
| 5 | 1989–2017 | 2018–2020 |
| **Test (locked)** | 1989–2020 | **2021–2025** — touched once, at the very end |

Rules:
- **Split on the target month.** A training row whose target (*t+1*) falls inside the validation window is excluded from that fold's training set.
- **The reservoir runs continuously through history; only the readout fit is restricted.** The state entering a validation month is built from past inputs only, which is legitimate — it is information available at forecast time.
- **All hyperparameter search happens inside folds 1–5.** Report the mean across folds plus per-fold results, so no single period dominates.
- **Equal tuning budget for every model** — baselines, fly reservoir, and every control network get the same number of configurations. Otherwise a fly advantage could just be extra tuning.
- **Random search as the default** (Bergstra & Bengio 2012); **grid on spectral radius × input scaling** for the heat map. **Cache reservoir states** per reservoir configuration so the ridge-penalty search is nearly free.
- **Appendix robustness (optional):** (1) country holdout — train on ~80% of countries across all years, test on the rest (residual leak: the neighbour feature still uses held-out countries' deaths); (2) blocked time folds with ~12-month buffers.
- **Live November 2026 check (personal, not a graded goal):** feed Candidate data through October 2026 into the final model and forecast November. Caveat when reporting: trained on final data, predicting from preliminary data. The November Candidate release (for scoring) should arrive around mid-December.

## Evaluation
### Metrics
- Regression: RMSE, MAE
- Classification: onsets are rare → precision, recall, F1, PR-AUC, not accuracy
- **Time-series cross-validation:** rolling-origin, train on earlier months, test on later. No random splits (leakage). See [[#Validation Design (SETTLED)]].

### Baselines
- **Naive persistence** ("next month = this month"). Mandatory; very hard to beat in conflict data. Anyone who knows the literature will look for it.
- Linear regression / ARIMA
- Random forest / gradient boosting
- Small standard neural net (LSTM or MLP)
- Published ViEWS forecasts

### Control Networks
Each control isolates what about the fly matters. Run every control through every tier, trained identically.

| Control | Keeps | Tests |
|---|---|---|
| Random (Erdős–Rényi) | Size, density | Does any structure matter? |
| Degree-preserving rewired | Each neuron's degree | Does specific wiring matter beyond hubs? |
| Weight-shuffled | Exact wiring, scrambled strengths | Do synapse strengths matter, or only topology? |

### Robustness / "No Stone Unturned" Checklist
Principle: **clean spine, deep appendix.** Every extra analysis rules out an alternative explanation.
- [x] Multiple random seeds, confidence intervals on every result (seeds 1–5 + country-cluster bootstrap, notebooks 06, 08, 09)
- [ ] **Equivalence bounds** — largest fly advantage still consistent with the data, as a share of the history gap (09 §7)
- [ ] **Mechanism** — linear reservoir and no-recurrence leaky banks (09 §9): is the memory the network, or just leaky integration?
- [ ] Hyperparameter sweeps (spectral radius × input scaling heat maps)
- [ ] **APL ablation** — remove 2 neurons, recompute SCC and performance. "How much of the fly's advantage rests on a single pair of cells?"
- [ ] **Glutamate sign flip** — does the result survive the opposite convention?
- [ ] **Weight scheme comparison** — raw vs log1p vs binary vs input-fraction
- [~] **MBON-only vs full-state readout** — done vs 1,024 random units; size-matched 96 random units in 09 §10
- [ ] **Hemisphere replication** — fit left, test right
- [ ] **MCNS replication** — a second connectome, different consortium, different animal. Frame as *robustness across independent reconstructions*, NOT male vs female (a sex difference in forecasting skill would more plausibly reflect reconstruction pipelines than biology).
- [ ] Threshold sweep upward (10+); downward requires the no-threshold file
- [ ] Error analysis: which regions, conflict types, onset vs continuation
- [ ] PCA on reservoir states — what is the network encoding? (Week 9)
- [ ] Ethics section: early-warning systems shape aid and intervention; false positives stigmatise countries
- [ ] Public GitHub repo: configs, fixed seeds, environment file, README
- [ ] Polished Quarto write-up for the personal site

## Results So Far: Standard ML Baselines (notebook 05, folds 1–5 only; test set untouched)
Mean validation RMSE on log(1 + deaths), 40 random configs per family, tuned naive window.

| Horizon | Naive (best window) | Ridge | Random forest | Grad. boosting |
|---|---|---|---|---|
| 1 | 0.5282 (w=4) | **0.5033** | 0.5048 | 0.5073 |
| 3 | 0.5892 (w=6) | **0.5707** | 0.5783 | 0.5824 |
| 6 | 0.6320 (w=9) | **0.6216** | 0.6322 | 0.6354 |
| 12 | 0.7093 (w=6) | **0.6887** | 0.7012 | 0.7101 |

- **Target to beat for the reservoir:** ridge's numbers above. Skill vs the tuned naive baseline is only 1.6–4.7% for ridge and 0 or negative for boosting at horizons 6 and 12.
- Fold-to-fold SD of RMSE (0.016–0.061) is as large as the gaps between models, so differences are not established. Needs a paired fold comparison.
- Gains come from stay-quiet and cessation rows. Onset and continuation rows are barely improved.
- Onset average precision (at-risk rows, pooled folds): 0.095–0.158 against base rates of 0.5–1.2%, i.e. roughly 8–20x chance. Models beat naive clearly only at horizon 12.
- Ridge wins everywhere with heavy regularisation (alpha 455–1157). Trees do not add signal over linear.
- Naive MAE is lower than every model's (it predicts exact zero in quiet cells), so RMSE and MAE rank them differently.
- Caveat: configs are chosen on the same folds that are reported.

## Results So Far: Tier 1 Reservoirs (notebooks 07–09; folds 1–5 only)
- **07–08, all 28 features as inputs:** every reservoir equals ridge (differences ≤ 0.001). The features already hold the history, so the readout switches the states off. Uninformative about wiring.
- **09, 12 history-free inputs:** reservoirs recover 89–98% of the gap between ridge on current values (A) and ridge on all 28 features (B). The fly ties every control; random (ER) is nominally best. Any fly advantage is bounded at about 6% of the history gap at most, and under 2% at most horizons.
- Best `input_scale` sits at the bottom of its range → reservoirs prefer the near-linear regime, which likely explains the tie. Tested in 09 §9.
- Details and tables: [[ML FP Progress Log]].

## RL Agent Thread
Keep the follow-on in mind throughout so the class project builds toward it.

### The Follow-On Project
- States as RL agents playing a crisis-bargaining game based on Fearon (1995), *Rationalist Explanations for War*
- Mechanisms: power shifts, private information, commitment problems
- Actions: make offer, accept, reject, fight
- Question: do agents **learn** to fight under the conditions theory predicts? Do they find peaceful equilibria theory misses?
- Small agents, no big data, runs on a laptop

### Design Choices Now That Pay Off Later
- Structure the conflict data as a **gym-style environment** (observation → action → reward) even if Tiers 1–2 don't need it
- Keep **learning rules separate** from network definitions (readout / backprop / reward-modulated as swappable modules)
- The three-output loader contract above already does this
- Tier 3 = first real RL: reward signal, credit assignment, exploration vs exploitation
- Possible bridge: connectome-constrained network as the **policy network** of a bargaining agent
- Read Fearon (1995) and Sutton & Barto alongside the class work, not after

## Revised Timeline
**Deviation from the original plan:** connectome was loaded *before* the conflict data, to verify feasibility before investing in data acquisition. Justified — the mushroom body could have turned out to be feedforward and unusable, which would have invalidated the whole approach.

**Status (2026-10-02):** the conflict pipeline, baselines and Tier 1 (full-feature and history-free experiments, with controls) are done. Tier 1 is a tightly bounded null: the fly ties weight-shuffled, rewired and random networks. Next: the mechanism checks in notebook 09, then Tier 2.

| Week | Syllabus | Project work | RL thread |
|---|---|---|---|
| 4 | Check-in 1: Proposal | ✅ Proposal | Sutton & Barto ch. 1–2 |
| 5 | Regression Pt 2 | ✅ Connectome loaded, verified, MB extracted | |
| **6** | **Check-in 2: Data Collection; Lab 3** | **UCDP → country-month panel; persistence + linear baselines; PN input layer** | Sketch environment interface |
| 7 | Check-in 3: Hypothesis & Lit Review | Lit review; control networks built; Tier 1 prototype | Read Fearon (1995) |
| 8 | Lab 4 | Tier 1 full results + controls + time-series CV | |
| 9 | Check-in 4: Model Selection | Tier 2 training; PCA on states | |
| 10 | Lab 5 | Tier 2 + controls complete; robustness sweeps | Tier 3 design |
| 11 | Check-in 5: Results | Results write-up; error analysis; Tier 3 if on track | Tier 3 = first RL |
| 12 | | Ablations (APL, sign flip, weights); ethics section; figures | |
| 13 | Workshops | Final report + presentation | Bargaining game spec |
| 14 | Presentations | Present; publish repo + Quarto write-up | Begin follow-on |

## Repo Structure
```
fly-conflict/
├── data/
│   ├── raw/            # UCDP, ViEWS, FlyWire downloads (gitignored)
│   └── processed/      # country-month panel, cached edges/meta parquet
├── connectome/         # extraction, subnetwork definitions
├── src/
│   ├── data/           # loading, cleaning, panel construction
│   ├── networks/       # fly, random, rewired, weight-shuffled
│   ├── learning/       # readout, backprop, reward-modulated (swappable)
│   ├── env/            # gym-style environment (Tier 3 + RL follow-on)
│   └── eval/           # metrics, time-series CV, baselines
├── configs/            # hyperparameters, seeds
├── notebooks/          # runnable pipeline, numbered in run order; import functions from src/
└── README.md
```

## Fallback
If the connectome work stalls, **random reservoir vs baselines on conflict forecasting** is still a complete project. Tiers 1 + 2 + controls is the real deliverable.

## Open Decisions
- ~~Regression (fatalities) vs classification (onset) as primary target~~ → **regression primary** (settled 2026-10-01)
- ~~Monthly onset definition~~ → k=25 after m=6 quiet months (settled 2026-10-02)
- ~~Minimum series length / washout for new states~~ → each state's reservoir starts from zero at its first month; forecast origins need 13+ months of history (burn-in 12), so no extra washout (settled 2026-10-02)
- ~~Weight scaling scheme~~ → fraction of postsynaptic input; log1p, binary and raw are robustness variants (`weight_matrix(scheme=...)`)
- ~~Which neurons receive input~~ → the 285 olfactory projection neurons (class ALPN with at least one connection onto a Kenyon cell); found empirically in notebook 07. Network = 5,606 MB neurons + 285 PNs = 5,891
- Whether to add the central complex (see below)
- Covariates beyond conflict history
- **Email to Dr. Adler re: social-phenomenon strictness — still outstanding**

### Central complex as a possible addition
Not needed for recurrence (the MB has plenty), but worth considering for a second reason: the central complex (EB, FB, PB, NO) maintains a **ring attractor** encoding heading direction — a genuine recurrent working memory. "The fly's navigational memory circuit as a reservoir" is a strong argument for time-series forecasting. The combined version (CX for memory + MB for expansion and the dopaminergic learning channel) is a principled architecture rather than an arbitrary subgraph. Extract by **neuropil**, not cell type — CX type names are far less regular than `KC`/`MBON`.

## Reading List
**Reservoir computing**
- Jaeger (2001), echo state networks
- Maass, Natschläger & Markram (2002), liquid state machines
- Lukoševičius (2012), *A Practical Guide to Applying Echo State Networks*
- Suárez et al., conn2res (connectome-based reservoir computing)

**Mushroom body / random projection**
- Dasgupta, Stevens & Navlakha (2017, *Science*), *A neural algorithm for a fundamental computing problem*
- Li et al. (2020, *eLife*), the connectome of the adult Drosophila mushroom body

**Connectome**
- Dorkenwald et al. (2024, *Nature*), FlyWire whole-brain connectome
- Schlegel et al. (2024, *Nature*), whole-brain annotation and cell typing
- Yu et al. (2025), Princeton synapse detection
- Shiu et al. (2024, *Nature*), whole-brain fly LIF model
- Lappalainen et al. (2024, *Nature*), connectome-constrained visual system trained with backprop

**Conflict forecasting**
- Hegre et al. (2019), ViEWS early-warning system
- Sundberg & Melander (2013), UCDP GED

**RL / bargaining**
- Sutton & Barto, *Reinforcement Learning: An Introduction* (free online)
- Fearon (1995), *Rationalist Explanations for War*

## Reservoir Build (BUILT 2026-10-02, quick-tested; full runs pending)
- `src/networks/connectome.py` loader (MB + ALPN subgraph, signs, weight schemes), `controls.py` (weight-shuffled, degree-preserving rewired, Erdős–Rényi), `reservoir.py` (leaky tanh reservoir, all countries run in parallel).
- `src/models/readout.py`: ridge for 12 penalties and all folds from running Gram matrices. Verified against a float64 solve (1e-6) and against notebook 05's ridge (0.5032 vs 0.5033 at h=1).
- `src/models/reservoir_search.py`: `ReservoirLab`, random search over (rho 0.3–1.4, input scale 0.02–2, leak 0.1–1, state weight 0.02–1), 40 configs, same draws for every network. Readout = 1,024 random units (same for all networks) or the 96 MBONs, plus the 28 raw features.
- Notebooks: **06** paired comparison of baselines (country-cluster bootstrap), **07** fly reservoir, **08** controls, seeds, final comparison, **09** memory test (history-free inputs).
- **Result of 07–08 (full features): null.** Fly, shuffled, rewired, random and ridge all tie (RMSE within 0.001); the readout learned to ignore the states because the 28 features already hold lags and rolling means. This design cannot test the fly's wiring. Reported as one of two experiments.
- **09 memory test:** inputs are the 12 current-month values only (no lags, means, streaks); reference A = ridge on those 12 (no memory), reference B = ridge on all 28 (explicit history). Metric: share of the A-to-B gap each network recovers. This is the experiment that can show a fly advantage.
- Known behaviour: configurations with rho above about 1 do not forget their starting state (chaotic regime); notebook 07 flags the best configurations that do this.
- Design choices to defend: states plus raw features in the readout (reservoir can only add to ridge); 1,024-unit random readout for tuning (full 5,891-unit readout is a later robustness check); rho, leak and input scale are tuned, not properties of the fly.

## Next Steps
- [x] Run 06, 07, 08 (done 2026-10-02; null, see progress log)
- [ ] Run 09 (quick test first); decide from the memory-recovered table and paired intervals whether the fly beats the controls
- [ ] Robustness extensions on `ReservoirLab`: APL ablation, glutamate sign flip, weight schemes (log1p, binary), full 5,891-unit readout, hemisphere replication, multiple graph seeds
- [ ] MLP baseline; ViEWS benchmark
- [ ] Tier 2 (train weights on existing connections) once Tier 1 results are reviewed
- [ ] Locked-test evaluation, once, at the very end (with and without Nov–Dec 2025)
- [ ] Live November 2026 check using UCDP Candidate data (personal)
- [ ] Error analysis, PCA on reservoir states, ethics section
- [ ] Email Dr. Adler (outstanding)
