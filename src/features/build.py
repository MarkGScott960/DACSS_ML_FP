"""Model features for the country-month panel.

Imported by notebooks/04_features.ipynb. Functions only.

Timing rule: every feature in a row is computed from information available at
the ORIGIN month t (the month the forecast is made), using only rows at or
before t. Targets are built separately (src/eval/splits.make_supervised) and
sit `horizon` months after the origin.

- Conflict-history features use the state's own past months, including t.
- `log_nbr*` use neighbours' deaths in month t (known at t) and earlier.
- V-Dem columns were already joined with the y-1 rule in notebook 02, so they
  describe the previous calendar year.
- No calendar-year or time-index feature is used (a tree could split on it and
  then fail on future years it has never seen).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONFLICT_MIN = 1.0      # a "conflict month" has >= 1 state-based death
BURN_IN = 12            # first months of each state's series are history only

VDEM = ["v2x_polyarchy", "v2x_libdem", "v2x_civlib", "v2x_clphy",
        "v2x_rule", "v2x_corr", "v2x_regime"]

CONFLICT_FEATURES = [
    "log_sb", "log_ns", "log_os",
    "log_sb_lag1", "log_sb_lag2", "log_sb_lag3", "log_sb_lag6", "log_sb_lag12",
    "log_sb_mean3", "log_sb_mean6", "log_sb_mean12", "log_sb_max12", "sb_trend",
    "log_ns_mean3", "log_os_mean3",
    "log_months_since_conflict", "conflict_streak", "never_conflict_seen",
]
NEIGHBOUR_FEATURES = ["log_nbr", "log_nbr_mean3", "n_land_nbrs"]
FEATURES = CONFLICT_FEATURES + NEIGHBOUR_FEATURES + VDEM


def _roll(g, col: str, w: int, how: str) -> pd.Series:
    return g[col].transform(lambda s: getattr(s.rolling(w, min_periods=w), how)())


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add feature columns and a `usable` flag. Row order and count are unchanged."""
    df = panel.sort_values(["gwno", "month"], ignore_index=True).copy()

    for p in ("sb", "ns", "os"):
        df[f"log_{p}"] = np.log1p(df[f"{p}_best"])
    df["log_nbr"] = np.log1p(df["nbr_sb_best"])

    g = df.groupby("gwno")
    for lag in (1, 2, 3, 6, 12):
        df[f"log_sb_lag{lag}"] = g["log_sb"].shift(lag)
    for w in (3, 6, 12):
        df[f"log_sb_mean{w}"] = _roll(g, "log_sb", w, "mean")
    df["log_sb_max12"] = _roll(g, "log_sb", 12, "max")
    df["sb_trend"] = df["log_sb_mean3"] - df["log_sb_mean12"]
    df["log_ns_mean3"] = _roll(g, "log_ns", 3, "mean")
    df["log_os_mean3"] = _roll(g, "log_os", 3, "mean")
    df["log_nbr_mean3"] = _roll(g, "log_nbr", 3, "mean")

    # Months since the last conflict month, and the length of the current streak.
    pos = g.cumcount()                                   # 0 for a state's first month
    df["months_observed"] = pos + 1
    conflict = df["sb_best"] >= CONFLICT_MIN
    last_pos = pos.where(conflict).groupby(df["gwno"]).ffill()
    since = (pos - last_pos)
    df["never_conflict_seen"] = last_pos.isna().astype(int)
    # No conflict seen yet: all we know is "at least as long as the history we have".
    since = since.where(last_pos.notna(), df["months_observed"])
    df["log_months_since_conflict"] = np.log1p(since)
    run_id = (~conflict).groupby(df["gwno"]).cumsum()
    df["conflict_streak"] = conflict.groupby([df["gwno"], run_id]).cumsum()

    # Rows usable as forecast origins: full 12-month history (lag 12 exists) and every
    # V-Dem index present. Rows with any missing index are dropped, not imputed
    # (Bahamas, Belize, Brunei have none; Bahrain lacks two indices for 1990-2005).
    df["vdem_incomplete"] = df[VDEM].isna().any(axis=1)
    df["usable"] = (df["months_observed"] >= BURN_IN + 1) & ~df["vdem_incomplete"]
    return df


def at_risk_and_event(df: pd.DataFrame, k: float, m: int) -> tuple[pd.Series, pd.Series]:
    """Onset bookkeeping at the ORIGIN month.

    at_risk: the state had < k state-based deaths in each of the last m months up to
             and including the origin month t (known at forecast time).
    Returns (at_risk, quiet_run); the event is `target >= k` and is attached per
    horizon in the modelling notebook.
    """
    quiet = (df["sb_best"] < k)
    g = quiet.groupby(df["gwno"])
    # count of quiet months in the trailing window of m months (needs a full window)
    quiet_in_window = g.transform(lambda s: s.astype(int).rolling(m, min_periods=m).sum())
    at_risk = quiet_in_window == m
    return at_risk, quiet_in_window
