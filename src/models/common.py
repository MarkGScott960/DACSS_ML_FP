"""Shared pieces for the model notebooks (05 keeps its own inline copies)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.paths import PROCESSED
from src.eval.splits import make_supervised
from src.eval.metrics import transition_type

K_ONSET, M_ONSET = 25, 6
CONFLICT_MIN = 1.0
HORIZONS = [1, 3, 6, 12]
NAIVE_WINDOWS = [1, 2, 3, 4, 6, 9, 12, 18, 24]
KINDS = ["stay quiet", "onset", "continuation", "cessation"]


def load_features() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "features.parquet")


def make_table(feat: pd.DataFrame, h: int) -> pd.DataFrame:
    """Usable forecast origins with their target `h` months ahead (same as notebook 05)."""
    d = make_supervised(feat, horizon=h)
    d = d[d["usable"]].reset_index(drop=True)
    d["event"] = d["at_risk"] & (d["raw_y"] >= K_ONSET)
    d["now_conflict"] = d["sb_best"] >= CONFLICT_MIN
    d["next_conflict"] = d["raw_y"] >= CONFLICT_MIN
    d["kind"] = transition_type(d["now_conflict"], d["next_conflict"])
    return d


def naive_pred(feat: pd.DataFrame, d: pd.DataFrame, w: int) -> np.ndarray:
    """Mean of the state's own log deaths over the last w months (origin month included)."""
    base = feat[["gwno", "month", "sb_best"]].copy()
    base["ls"] = np.log1p(base["sb_best"])
    base = base.sort_values(["gwno", "month"])
    base["pred"] = base.groupby("gwno")["ls"].transform(lambda s: s.rolling(w, min_periods=1).mean())
    return d[["gwno", "month"]].merge(base[["gwno", "month", "pred"]], on=["gwno", "month"], how="left")["pred"].to_numpy()
