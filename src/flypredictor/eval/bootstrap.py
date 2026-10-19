"""Country-clustered bootstrap (plan section 8).

Countries are resampled with replacement; every row of a drawn country enters with
weight = number of times its country was drawn. All metrics in ``metrics.py`` accept a
(B, n) weight matrix, so one matrix serves every model and every paired difference
(the same resamples for model and reference -> a proper paired interval).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import metrics as M


def country_weights(countries, reps: int, seed: int) -> np.ndarray:
    """(reps, n) matrix: row i = multiplicity of each row's country in resample i."""
    countries = np.asarray(countries)
    uniq, idx = np.unique(countries, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(reps, len(uniq)))
    counts = np.zeros((reps, len(uniq)))
    np.add.at(counts, (np.arange(reps)[:, None], draws), 1)
    return counts[:, idx]


@dataclass
class Interval:
    estimate: float
    lo: float
    hi: float
    se: float
    reps: int

    def excludes(self, value: float = 0.0, side: str = "above") -> bool:
        """Interval entirely above (or below) ``value``."""
        return self.lo > value if side == "above" else self.hi < value

    def as_dict(self) -> dict[str, float]:
        return {"estimate": self.estimate, "lo": self.lo, "hi": self.hi, "se": self.se, "reps": self.reps}


def interval(point: float, draws: np.ndarray, level: float = 0.95) -> Interval:
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    a = (1 - level) / 2
    lo, hi = (np.quantile(d, [a, 1 - a]) if len(d) else (np.nan, np.nan))
    return Interval(float(point), float(lo), float(hi), float(d.std(ddof=1)) if len(d) > 1 else np.nan, int(len(d)))


def boot(stat: Callable[[np.ndarray | None], np.ndarray | float], W: np.ndarray, level: float = 0.95) -> Interval:
    """``stat(w)`` must return a scalar for w=None and a (B,) array for a (B, n) matrix."""
    return interval(stat(None), stat(W), level)


def primary_contrasts(y, p, p_ref, W, level: float = 0.95) -> dict[str, Interval]:
    """The H1/H5 decision quantities for one model against a reference forecast.

    brier_skill      1 - Brier(p)/Brier(ref); passes if the interval is above 0
    ap_ratio_diff    AP/base(p) - AP/base(ref); passes if the interval is above 0
    """
    return {
        "brier_skill": boot(lambda w: M.brier_skill(y, p, p_ref, w), W, level),
        "ap_ratio_diff": boot(lambda w: M.ap_ratio(y, p, w) - M.ap_ratio(y, p_ref, w), W, level),
        "ap_ratio": boot(lambda w: M.ap_ratio(y, p, w), W, level),
    }


def log_loss_diff(y, p_reduced, p_full, W, level: float = 0.95) -> Interval:
    """Drop-group contrast (H2a): log loss of the reduced model minus the full model (> 0: the group helps)."""
    return boot(lambda w: M.log_loss(y, p_reduced, w) - M.log_loss(y, p_full, w), W, level)
