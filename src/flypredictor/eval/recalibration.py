"""Intercept updating ("recalibration in the large") for annual refits (decision D-18).

The event rate roughly triples between development and the locked window. A model refit at
origin t is still trained mostly on low-rate history, so it under-predicts. Before scoring
origin t, every model's predictions (persistence included) are shifted on the logit scale
so that their mean over the most recent ``window`` years of KNOWN training labels equals the
observed rate there. Only labels known at t are used (the same training cutoff as the fit).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def intercept_shift(p_recent, y_recent, iters: int = 60) -> float:
    """Logit shift d with mean(sigmoid(logit(p_recent) + d)) == mean(y_recent)."""
    z = _logit(p_recent)
    target = float(np.mean(y_recent))
    if target <= 0 or target >= 1 or len(z) == 0:
        return 0.0
    lo, hi = -10.0, 10.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def apply_shift(p, d: float):
    return 1 / (1 + np.exp(-(_logit(p) + d)))
