"""Scoring helpers. All errors are on the log(1 + deaths) scale unless stated."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def transition_type(now_conflict: pd.Series, next_conflict: pd.Series) -> pd.Series:
    """Label each row by what happens between the origin month and the target month.

    quiet -> quiet      "stay quiet"
    quiet -> conflict   "onset"          (the hard case; simple, not the final onset definition)
    conflict -> conflict "continuation"
    conflict -> quiet   "cessation"
    Conflict = at least one state-based death in the month.
    """
    out = np.select(
        [~now_conflict & ~next_conflict, ~now_conflict & next_conflict,
         now_conflict & next_conflict, now_conflict & ~next_conflict],
        ["stay quiet", "onset", "continuation", "cessation"], default="")
    return pd.Series(out, index=now_conflict.index)


def score(y: pd.Series, pred: pd.Series, groups: pd.Series | None = None) -> pd.DataFrame:
    """RMSE / MAE overall and (optionally) within each group of `groups`."""
    rows = {"all rows": (y, pred)}
    if groups is not None:
        for g in ["stay quiet", "onset", "continuation", "cessation"]:
            m = groups == g
            if m.any():
                rows[g] = (y[m], pred[m])
    return pd.DataFrame(
        {k: {"n": len(v[0]), "RMSE": rmse(*v), "MAE": mae(*v)} for k, v in rows.items()}).T
