# FlyPredictor · Forecasting Democratic Backsliding

DACSS 756 (Machine Learning) final project, Mark Scott, UMass Amherst.

Forecasting, for the world's democracies, whether a backsliding episode will begin within *k* years and whether democracy will break down. Data: V-Dem v16 and the Episodes of Regime Transformation (ERT) v16. Framing: Paxton's account of fascism as a process with warning signs.

* **Plan of record:** [`docs/plan_of_record.md`](docs/plan_of_record.md)
* **Decisions** (including two pending ones, D-07 and D-08): [`docs/decisions_log.md`](docs/decisions_log.md)
* **Progress:** [`docs/progress_log.md`](docs/progress_log.md)
* **Pre-registration (draft):** [`prereg/preregistration.md`](prereg/preregistration.md)

**Current phase:** data layer built and audited (zip 1). Next: evaluation harness and power simulation (zip 2), then freeze the pre-registration. **No model code exists until the freeze.**

---

## Setup (Windows, once)

Run everything from the repository root (`DACSS_ML_FP`).

### 0 · Archive the conflict study in git

The commit before "deleted everything" holds Study 1. Tag it so it is one command away:

```powershell
git log --oneline -5                     # find the commit just before the deletion
git tag conflict-final <that-hash>
git push origin conflict-final
```

### 1 · Python (3.14)

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip freeze > requirements-lock.txt       # commit this file
```

If `pyreadr` has no wheel for your Python version, remove it from `requirements.txt` and set `rdata_conversion: r` in `config/project.yaml`. R then does the conversion.

In VS Code, select the `.venv` interpreter as the notebook kernel.

### 2 · R (for ERT)

ERT's `get_eps` (used for the real-time states, threshold variants and v10 relabelling) is R-only and contains C++ code, so it needs **Rtools** matching your R version: <https://cran.r-project.org/bin/windows/Rtools/>.

```powershell
Rscript R/setup.R
```

This creates a project-local R library with `renv`, installs the pinned packages (ERT at commit `7fd51ce`, `nanoparquet`, `jsonlite`, …) and writes `renv.lock`. Commit `renv.lock`, `.Rprofile`, `renv/activate.R` and `renv/settings.json`.

If R cannot be set up, set `labels.state_source: rule` in `config/project.yaml`. The Python rule agrees with the ERT re-runs on 99.6% of democracy-years. The rest of the pipeline runs without R.

### 3 · OneDrive

`data/` grows to about 0.5 GB and is not committed. If the repository sits in a OneDrive folder, pause syncing while notebook 01 runs (OneDrive can lock files that are still being written), or mark `data/` "Always keep on this device".

---

## Run order

| Step | What | Time |
|---|---|---|
| `notebooks/0x_data/01_fetch.ipynb` | Pinned download, manifests, Parquet, real-time ERT re-runs (R) | ~1 min + ~10 min (R) |
| `notebooks/0x_data/02_panel_labels.ipynb` | Episodes, states, labels, panel → `data/processed/` | seconds |
| `notebooks/0x_data/03_audits.ipynb` | Pre-freeze audits → `results/dev/audits/*.csv` | seconds |
| `pytest` | 37 fixture tests + 3 real-data checks (`pytest -m realdata`) | seconds |

```text
GitHub (ERT @ 7fd51ce) ──► data/raw/ert/  ──► data/interim/*.parquet ──┐
                              │ manifest                                 │
                              ▼                                          ▼
                   R/ert_variants.R (get_eps on data truncated     episodes.py ─► states.py ─► labels.py
                   at every year) ──► real-time status ───────────────┘              │
                                                                                     ▼
                                                     panel.py (raw indicators + stage) ─► audits
```

---

## Layout

```
config/          every analytic choice: project.yaml (sources, labels, splits) · features.yaml (Paxton groups)
R/               setup.R · convert_vdem.R · ert_variants.R   (ERT and conversion only)
src/flypredictor/
  config.py paths.py manifest.py rbridge.py fetch.py
  data/          episodes.py · ert_jobs.py · states.py · labels.py · panel.py
  audit/         coverage.py · maturity.py · realtime.py
  eval/ models/ interpret/      empty until zip 2 / the pre-registration freeze
notebooks/       0x_data · 1x_development · 2x_locked · 3x_robustness · 4x_figures
tests/           synthetic fixtures with hand-computed answers + real-data checks
data/            raw/ interim/ processed/ (git-ignored) · manifests/ (committed)
results/         dev/ locked/ robustness/   (small CSVs committed, Parquet ignored)
prereg/          preregistration.md (frozen by tag prereg-v1)
docs/            plan of record · decisions log · progress log
reports/         paper, figures, talk
```

**Rules the layout enforces:**
- Logic lives in `src/`; notebooks only call it.
- `data/raw/` is never edited.
- Nothing in `1x_development` writes to `results/locked/`.
- The locked test runs once, through the lockbox (zip 2).

## Git tags

| Tag | Meaning |
|---|---|
| `conflict-final` | Study 1 (fly connectome, conflict forecasting), archived |
| `prereg-v1` | Pre-registration frozen; any later change to `config/` or `prereg/` is a logged deviation |
| `code-freeze` | Pipeline frozen before the locked test |
| `locked-run` | Commit that produced `results/locked/` |
