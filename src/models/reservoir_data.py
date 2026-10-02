"""Turn the feature table into reservoir inputs and align forecast rows to states."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.build import FEATURES

FIRST_MONTH = pd.Timestamp(1989, 1, 1)
SCALER_END = pd.Timestamp(2005, 12, 1)     # scaler is fit on fold-1 training months only
CLIP = 5.0


@dataclass
class Inputs:
    U: np.ndarray          # (T, S, F) standardised features clipped at +-5, NaN -> 0 (reservoir drive)
    R: np.ndarray          # (T, S, F) same but NOT clipped (passed straight to the readout)
    starts: np.ndarray     # (S,) first active month index per sequence
    ends: np.ndarray       # (S,) last active month index
    gwno: np.ndarray       # (S,)
    mean: np.ndarray
    std: np.ndarray

    def locate(self, d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(t, s) index arrays for the rows of a table with columns gwno and month."""
        t = ((d["month"].dt.year - 1989) * 12 + d["month"].dt.month - 1).to_numpy()
        s = pd.Index(self.gwno).get_indexer(d["gwno"])
        assert (s >= 0).all()
        return t, s


def build_inputs(feat: pd.DataFrame, features: list[str] | None = None) -> Inputs:
    features = list(FEATURES if features is None else features)
    f = feat.sort_values(["gwno", "month"], ignore_index=True)
    fit = f[f["usable"] & (f["month"] <= SCALER_END)]
    mean = fit[features].mean().to_numpy(); std = fit[features].std().to_numpy()
    std = np.where(std < 1e-8, 1.0, std)
    Xr = np.nan_to_num((f[features].to_numpy(float) - mean) / std, nan=0.0).astype(np.float32)
    X = Xr.clip(-CLIP, CLIP)                                    # burn-in / missing V-Dem -> 0
    t = ((f["month"].dt.year - 1989) * 12 + f["month"].dt.month - 1).to_numpy()
    gw = np.sort(f["gwno"].unique())
    s = pd.Index(gw).get_indexer(f["gwno"])
    T = int(t.max()) + 1
    U = np.zeros((T, len(gw), len(features)), np.float32)
    U[t, s] = X
    R = np.zeros_like(U); R[t, s] = Xr
    starts = np.full(len(gw), T); ends = np.full(len(gw), -1)
    np.minimum.at(starts, s, t); np.maximum.at(ends, s, t)
    # no internal gaps (asserted in notebook 02): months are consecutive per state
    assert ((ends - starts + 1) == np.bincount(s, minlength=len(gw))).all()
    return Inputs(U, R, starts, ends, gw, mean, std)
