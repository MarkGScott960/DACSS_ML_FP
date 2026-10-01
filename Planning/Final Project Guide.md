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

### Weight scaling (OPEN — decide before Tier 1)
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

## Data (NOT YET STARTED)
- **UCDP Georeferenced Event Data (GED):** free, event-level fatalities, aggregate to country-month
- **ViEWS** ([viewsforecasting.org](https://viewsforecasting.org)): published monthly forecasts + benchmark data
- **ACLED:** richer events, check access/licensing terms first
- Covariates (add once the pipeline works): population, GDP, regime type (V-Dem), neighbouring conflict, past fatalities
- Unit of analysis: **country-month panel** (~190 countries × decades = tens of thousands of samples, enough for Tier 2)
- Target options: log fatalities next month (regression) or any conflict next month (classification; UCDP threshold is ≥25 battle deaths)
- **Onsets vs continuation:** conflict is highly persistent. Report onsets separately — that's where forecasting is hard and interesting.

## Evaluation
### Metrics
- Regression: RMSE, MAE
- Classification: onsets are rare → precision, recall, F1, PR-AUC, not accuracy
- **Time-series cross-validation:** train on earlier months, test on later. No random splits (leakage).

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
- [ ] Multiple random seeds, confidence intervals on every result
- [ ] Hyperparameter sweeps (spectral radius × input scaling heat maps)
- [ ] **APL ablation** — remove 2 neurons, recompute SCC and performance. "How much of the fly's advantage rests on a single pair of cells?"
- [ ] **Glutamate sign flip** — does the result survive the opposite convention?
- [ ] **Weight scheme comparison** — raw vs log1p vs binary vs input-fraction
- [ ] **MBON-only vs full-state readout**
- [ ] **Hemisphere replication** — fit left, test right
- [ ] **MCNS replication** — a second connectome, different consortium, different animal. Frame as *robustness across independent reconstructions*, NOT male vs female (a sex difference in forecasting skill would more plausibly reflect reconstruction pipelines than biology).
- [ ] Threshold sweep upward (10+); downward requires the no-threshold file
- [ ] Error analysis: which regions, conflict types, onset vs continuation
- [ ] PCA on reservoir states — what is the network encoding? (Week 9)
- [ ] Ethics section: early-warning systems shape aid and intervention; false positives stigmatise countries
- [ ] Public GitHub repo: configs, fixed seeds, environment file, README
- [ ] Polished Quarto write-up for the personal site

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

**The cost:** Check-in 2 is Data Collection, and the conflict data is not started. The persistence baseline — the number everything else is compared against — does not exist yet. This is now the critical path.

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
├── notebooks/          # exploration only; results come from src/
└── README.md
```

## Fallback
If the connectome work stalls, **random reservoir vs baselines on conflict forecasting** is still a complete project. Tiers 1 + 2 + controls is the real deliverable.

## Open Decisions
- Regression (fatalities) vs classification (onset) as primary target
- Weight scaling scheme (lean: fraction of postsynaptic input)
- Which neurons receive input, once the PN layer is identified
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

## Next Steps
- [ ] Run the upstream query to identify the PN input layer
- [ ] Add PNs to the subgraph; re-verify SCC
- [ ] Drop the 71 neurons outside the main SCC
- [ ] Fix DAN selection to cell-type based (331, not 281)
- [ ] Decide weight scaling scheme
- [ ] Cache `edges` and `meta` to parquet
- [ ] **Start UCDP data pipeline — this is the critical path**
- [ ] Email Dr. Adler
