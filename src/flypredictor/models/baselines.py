"""Baselines (plan 6.1). Both are fit on training rows only.

PersistenceBaseline   base rate within regime subtype (electoral 2 / liberal 3) x 3-year EDI
                      change terciles; tercile cut points fixed once on development democracy
                      rows; empty or thin bins shrink toward the training base rate.
EDIMomentum           logistic regression on EDI level, 1-year and 3-year change: "the slide
                      predicts the slide" (the skeptic's baseline).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

TREND = "v2x_polyarchy__d3"


def tercile_cuts(features: pd.DataFrame, cfg: dict[str, Any]) -> np.ndarray:
    """3-year EDI change cut points on development democracy rows (outcome-free)."""
    d = features[(features["stage"] > 0) & (features["year"] <= cfg["splits"]["dev_last_origin"])]
    n = int(cfg["evaluation"]["persistence_bins"])
    return np.nanquantile(d[TREND].to_numpy(float), np.linspace(0, 1, n + 1)[1:-1])


class PersistenceBaseline:
    def __init__(self, cuts: np.ndarray, smoothing: float = 1.0):
        self.cuts = np.asarray(cuts, float)
        self.smoothing = float(smoothing)

    def _bin(self, X: pd.DataFrame) -> np.ndarray:
        trend = X[TREND].to_numpy(float)
        t = np.where(np.isnan(trend), -1, np.searchsorted(self.cuts, trend, side="right"))
        liberal = X["liberal"].fillna(0).to_numpy(int)
        return liberal * 10 + t

    def fit(self, X: pd.DataFrame, y) -> "PersistenceBaseline":
        y = np.asarray(y, float)
        self.base_ = float(y.mean())
        b = self._bin(X)
        df = pd.DataFrame({"b": b, "y": y}).groupby("b")["y"].agg(["sum", "count"])
        s = self.smoothing
        self.rates_ = ((df["sum"] + s * self.base_) / (df["count"] + s)).to_dict()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self.rates_.get(b, self.base_) for b in self._bin(X)], float)


class EDIMomentum:
    COLS = ["v2x_polyarchy", "v2x_polyarchy__d1", "v2x_polyarchy__d3"]

    def fit(self, X: pd.DataFrame, y) -> "EDIMomentum":
        Z = X[self.COLS].astype(float)
        self.med_ = Z.median()
        Z = Z.fillna(self.med_)
        self.mu_, self.sd_ = Z.mean(), Z.std().replace(0, 1)
        self.m_ = LogisticRegression(C=100.0, max_iter=2000).fit((Z - self.mu_) / self.sd_, np.asarray(y, int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Z = X[self.COLS].astype(float).fillna(self.med_)
        return self.m_.predict_proba((Z - self.mu_) / self.sd_)[:, 1]
