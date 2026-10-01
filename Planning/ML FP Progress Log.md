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

## Current State
**Working:** connectome download, aggregation, NT signing, MB extraction, SCC and spectral diagnostics.
**Verified:** pair counts and synapse totals match published FAFB figures; MB is 98.7% recurrent; extraction passes biological sanity checks.
**Not started:** conflict data, baselines, control networks, any reservoir simulation.

## Immediate Next Steps
1. Run the upstream query; identify and add the PN input layer; re-verify SCC
2. Drop the 71 neurons outside the main SCC
3. Fix DAN selection to cell-type based (331, not 281)
4. Decide weight scaling (lean: fraction of postsynaptic input)
5. Cache `edges` and `meta` to parquet
6. **Start the UCDP pipeline and build the persistence baseline — critical path for Check-in 2**
7. Email Dr. Adler re: social-phenomenon strictness
