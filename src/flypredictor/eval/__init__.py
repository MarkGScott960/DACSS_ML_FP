"""Evaluation machinery (zip 2, built before any model is fit to real outcomes).

  splits.py         forecast-origin splits: development blocks with a k-year gap; locked window
                    with annual refits (decision D-07b); training cutoff t - 1 - k - lag
  metrics.py        PR-AUC / base rate, Brier skill, log loss, ROC-AUC, alert-budget P/R/F1,
                    event-level hit rate; all accept bootstrap weight matrices
  bootstrap.py      country-clustered bootstrap, paired contrasts for H1/H5 and drop-group (H2a)
  recalibration.py  intercept updating before each annual forecast (decision D-18)
  lockbox.py        the only route to locked-window rows: needs tag prereg-v1, unchanged config/
                    and prereg/ (or a logged deviation), one scoring per model x task (ledger)
  power.py          power simulation with planted effects on synthetic labels (notebook 04)
"""
