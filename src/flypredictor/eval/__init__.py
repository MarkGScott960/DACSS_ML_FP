"""Evaluation machinery (zip 2, before any model is fit).

Planned modules:
  splits.py    rolling-origin expanding-window folds with a k-year gap; the locked window
  metrics.py   PR-AUC / base rate, ROC-AUC, Brier skill, log loss, alert-budget P/R/F1, event-level hit rate
  bootstrap.py country-clustered bootstrap for intervals and paired model differences
  store.py     long-format prediction store (one row per model x fold x origin x horizon)
  lockbox.py   refuses to score locked-window rows unless tag prereg-v1 exists and results/locked/ is empty;
               records commit hash and time of the single locked run
"""
