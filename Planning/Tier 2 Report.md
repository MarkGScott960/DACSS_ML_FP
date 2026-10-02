# Tier 2 Report: Can the Fly Connectome Learn?

**Status (2026-10-02): design agreed, feasibility confirmed, nothing trained yet.** This report is the single consolidated account of Tier 2 (same role as [[Tier 1 Report]]). Sections marked *PENDING* are filled in as work is done. The plan lives in [[Final Project Guide]] ("Tier 2 Design"); the chronology lives in [[ML FP Progress Log]].

## 1. Question and motivation
Tier 1 froze every connection and trained only a ridge readout. Result: a tightly bounded null. The fly's wiring tied the weight-shuffled, degree-preserving rewired and Erdős–Rényi controls, and even tied a bank of leaky units with *no* connections; the reservoirs supplied leaky averaging of the inputs, which hand-built lags already approximate (see Tier 1 Report §9, §12).

Tier 2 asks the question Tier 1 could not: **if the connections themselves are allowed to learn, does the fly's wiring learn better than random wiring given identical training?** A fly advantage only counts if the trained fly beats trained controls, not merely its own frozen version.

## 2. Design (agreed)
**2a. Trained weights on the conflict target (required).**
- *Trained:* the magnitude of each existing connection (92,642 signal-carrying of 97,051), the PN input matrix, and the readout. Each connection keeps its sign (from the neurotransmitter): W = sign ⊙ softplus(θ). Initialised at the Tier 1 fly weights rescaled to the tuned spectral radius.
- *Method:* backpropagation through time on MSE of log(1 + state-based deaths); 36-month windows; 178 country sequences per batch; AdamW; one multi-output head for horizons 1/3/6/12; one model per fold, network and seed.
- *Early stopping:* inner validation on the last 24 months of each fold's training window. Folds 1–5 remain the outer evaluation. **The locked 2021–2025 test set is not touched.**
- *Controls (same parameters, budget and stopping rule):* trained shuffled, trained rewired, trained Erdős–Rényi, and an unconstrained GRU of similar parameter count. Reference points: the frozen Tier 1 results.
- *Inputs:* the 12 history-free features first (main test: memory must come from the network), then the 28 features as a secondary check.
- *Reading the result:* trained fly vs trained controls (paired country-cluster bootstrap, at least 5 seeds), and trained vs frozen for each network.

**2b. Diagnostic tasks (can it learn at all?).** NARMA-10 and delayed parity: nonlinear-memory benchmarks with known answers. Frozen vs trained, fly vs controls. If the trained fly wins there but ties on conflict, the conflict data simply lacks nonlinear signal; if it does not win, that is a finding about the connectome.

**2c. Structural plasticity (extension; which neurons connect).** (i) learnable gate on each connection with a sparsity penalty (pruning); (ii) fixed connection budget with drop-and-grow (RigL-style). Compare learned wiring with the fly's; fly-start vs ER-start. Cut if time is short.

**Leakage rules.** Every training row's target month is at or before the fold's training end; the inner validation window is carved from training only; scaler and inputs as in Tier 1; the locked test set stays unscored until every model and hyperparameter is fixed.

## 3. Scope decisions (Mark, 2026-10-02)
- Compute: an overnight budget (10–20 h) is acceptable.
- Horizons: all four (1/3/6/12) from one multi-output head.
- Inputs: history-free first, 28 features second.
- Open: how the budget is split between hyperparameter tuning (one fold) and the final multi-fold, multi-seed runs.

## 4. Feasibility (script: `scripts/tier2_feasibility.py`)
Random inputs, no targets; checks speed and memory only.
- **Graph:** 5,891 nodes, 92,642 trainable connections, 285 input nodes; window 36 months; batch 178 countries; 1,024-unit readout; tanh; loss on the last month.
- **PyTorch:** installs; a `cp314` Windows wheel exists (torch 2.14.1, including `+cu126`).
- **CPU, 2 threads (container), seconds per window (forward + backward):** sparse-COO matmul 0.33 + 3.4 = **3.7**; masked dense matmul 7.1; gather/scatter 6.8 (5.9 GB). Gradients are finite in all three. One pass over 444 months is about 12 windows, so about 45 s per epoch on 2 threads.
- **Hardware (confirmed):** Intel i9-9900K (8 cores / 16 threads), 31.9 GB RAM, NVIDIA RTX 4070 with 12 GB, driver 560.94 (CUDA 12.6), so use the `cu126` build.
- **Cost model:** 5 folds × 5 architectures × 5 seeds × ~30 epochs is about 40 h on 2 threads; the full design needs the GPU (or a reduced design on the CPU).
- *PENDING:* GPU and 8-thread CPU timings of the three modes (Mark to run `python scripts/tier2_feasibility.py cuda` and `... cpu 8`).

## 5. Implementation *(PENDING)*
Planned module `src/models/trained_reservoir.py` (PyTorch): edge-vector parameterisation with fixed sign, fast matmul mode chosen from §4, multi-horizon head, windowed BPTT with state carried across windows, inner-validation early stopping, same fold masks and ridge-free readout as the conflict target. Controls built with the Tier 1 `controls.py` so graphs are identical to Tier 1.

## 6. Results *(PENDING)*
2a trained vs frozen; 2a trained fly vs trained controls and GRU; 2b diagnostics; 2c structural results.

## 7. Interpretation and limitations *(PENDING)*
Known risks to state: a flexible trained model finds any leak (hence the strict window and lag rules); training noise across seeds (hence at least 5 seeds and paired bootstraps); trained-fly vs trained-control differences must be judged against seed variance; the readout-only Tier 1 and weight-trained Tier 2 differ in what is tuned, so compare within tier first.

## 8. Reproducibility *(PENDING)*
Environment: Python 3.14, `torch 2.14.1+cu126` on Windows; seeds, run scripts and logs to be listed here once built.
