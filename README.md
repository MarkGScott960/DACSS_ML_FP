# Forecasting Armed Conflict with a Fly-Connectome Reservoir

DACSS 756 (Machine Learning) final project. Forecasts monthly state-based conflict deaths (UCDP GED, country-month panel, 1989–2025) with a reservoir computer built from the *Drosophila* mushroom-body connectome (FlyWire FAFB v783), compared against control networks and standard machine learning.

## Status
- Done: data download, cleaned country-month panel, feature set, rolling-origin validation, tuned baselines (naive, ridge, random forest, gradient boosting), paired comparisons, the fly reservoir and three control networks with full features (07–08), a history-free memory test with mechanism checks (09).
- **Tier 1 result:** a tightly bounded null. The fly ties its control networks and a bank of leaky units with no connections; the reservoirs supply leaky averaging of the inputs, which hand-built lags already approximate.
- Next: Tier 2 (training the connectome's weights, with trained controls and a diagnostic task), MLP baseline, one-time evaluation on the locked test set (2021–2025).

## Design in brief
- **Target:** log(1 + state-based deaths) at the target month, horizons 1, 3, 6 and 12 months. Onsets (>=25 deaths after 6 quiet months) are an evaluated subset.
- **Validation:** five expanding-window rolling-origin folds on the target month (validate 2006–08, 2009–11, 2012–14, 2015–17, 2018–20). 2021–2025 is locked; `src/eval/splits.py` refuses to return it unless explicitly asked. A random year-mixed split was rejected because it leaks future information.
- **Leakage controls:** annual covariates are lagged one year, no calendar-year features, 12-month burn-in, and a truncation test confirms no feature changes when future data is removed.
- **Tuning:** random search, 40 configs per model family, plus a tuned window for the naive baseline.

## Data sources
UCDP GED 26.1; Gleditsch–Ward state list; V-Dem v16 Core (manual download); COW Direct Contiguity 3.2. Raw data is not committed. `data/raw/MANIFEST.json` records versions and hashes.

## Results so far (validation folds only)
Mean RMSE on log(1 + deaths):

| Horizon | Naive (tuned) | Ridge | Random forest | Grad. boosting |
|---|---|---|---|---|
| 1 | 0.528 | 0.503 | 0.505 | 0.507 |
| 3 | 0.589 | 0.571 | 0.578 | 0.582 |
| 6 | 0.632 | 0.622 | 0.632 | 0.635 |
| 12 | 0.709 | 0.689 | 0.701 | 0.710 |

Gains over the tuned naive baseline are small (under 5%); ridge's are reliable at horizons 1, 3 and 12 (paired country-cluster bootstrap, notebook 06).

**Reservoirs, history-free inputs (notebook 09).** Ridge on current-month values only: 0.567 / 0.649 / 0.707 / 0.757 (h = 1/3/6/12); ridge on all 28 features: 0.503 / 0.571 / 0.622 / 0.689.

| Network | h1 | h3 | h6 | h12 |
|---|---|---|---|---|
| fly (tanh) | 0.505 | 0.575 | 0.627 | 0.696 |
| random ER (tanh) | 0.505 | 0.574 | 0.625 | 0.694 |
| fly (linear) | 0.503 | 0.573 | 0.624 | 0.693 |
| random ER (linear) | 0.503 | 0.572 | 0.623 | 0.692 |
| leaky-unit bank, no connections | 0.502 | 0.574 | 0.624 | 0.692 |

Every reservoir closes 89–102% of the gap to the 28-feature ridge. The fly ties every control; a fly advantage larger than about 6% of that gap (tanh) or 1.2% (linear) is ruled out. Linear reservoirs are at least as good as tanh, a bank of independent leaky units with no connections does as well, and the 96 mushroom-body output neurons read out worse than 96 random units. Details are in `Planning/ML FP Progress Log.md`.

## The reservoir model
A leaky echo-state network whose wiring is the fly's mushroom body (5,606 neurons) plus the 285 olfactory projection neurons that feed it, from FlyWire. Country-month features drive the projection neurons; each country is run separately and one ridge readout is fit on all of them. With all 28 features the reservoir adds nothing over ridge (the features already contain the history), so notebook 09 repeats the comparison with history-free inputs, where the reservoir's dynamics must supply the memory, and then asks where that memory comes from (linear reservoirs, banks of leaky units with no connections, moving averages with no units at all). Control networks (weight-shuffled, degree-preserving rewired, random) are tuned with identical budgets so any fly advantage can be attributed to specific wiring. Spectral radius, leak and input scale are tuned hyperparameters, not properties of the fly.

## Repository layout
- `notebooks/` numbered in run order: `01_download`, `02_clean_join_describe`, `03_folds_and_baselines`, `04_features`, `05_models_random_search`, `06_paired_comparison`, `07_fly_reservoir`, `08_controls_and_comparison`, `09_memory_test`
- `src/` reusable code: `data/` (download, panel), `features/`, `eval/` (splits, metrics, paired bootstrap), `models/` (readout, reservoir search), `networks/` (connectome, controls, reservoir)
- `results/` leaderboard, onset scores, paired comparisons, best configs. The large tuning logs (`memory_*log.csv`, `reservoir_log.csv`) are regenerable caches and are not committed
- `Planning/` project guide, progress log and connectome notes
- `Flywire_FAFBv783/` connectome files

## Running it
Python 3.14, Windows/VS Code:
```
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Run notebooks 01 to 09 in order (06 before 08 and 09). Tuning logs in `results/` are resumable: a re-run reuses finished configurations, and `tune()` refuses to resume a log written with different inputs (`*.fingerprint.json`). A fresh clone has no logs, so notebooks 07–09 recompute them (several hours on a laptop); delete a log after changing reservoir or readout code. Notebook 01 needs the V-Dem Core CSV downloaded manually (see its instructions). Notebook 05 has a `QUICK_TEST` flag for a fast check, is resumable from `results/tuning_log.csv`, and never scores the test set.
