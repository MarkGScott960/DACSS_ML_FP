"""Assign every country-year to a forecasting stage, using only what was knowable at t.

    stage 1  democracy at t (RoW >= 2) and NOT in a known democratic episode  -> onset risk set
    stage 2  democracy at t and IN a known democratic episode, before breakdown -> breakdown risk set
    stage 0  not a democracy at t (no forecast is made)

"Democracy at t" = RoW (``v2x_regime``) >= ``democracy.min_regime`` and, when
``democracy.require_ert_democracy`` is true, ERT's own regime type ``reg_type == 1`` as well
(decision D-15): ERT only codes democratic episodes and breakdowns for spells it treats as
democratic, so rows outside them could never carry a positive label.

"Known" depends on ``labels.state_source`` (plan 3.3):

* ``ert_truncated``  ERT re-run on data truncated at t: exact real-time coding (primary).
* ``rule``           final-coded episodes, known from the rule-based confirmation year until the
                     final end year. Approximation: the end year is itself hindsight.
* ``hindsight``      final-coded episodes from their start year (plan v1; sensitivity only).

Every democracy row belongs to exactly one of stage 1 or stage 2, so the two risk
sets partition the democratic country-years.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _final_episode_flags(base: pd.DataFrame, eps: pd.DataFrame, start_col: str) -> np.ndarray:
    """True where year falls in [start_col, end_year] of a final episode and before its breakdown."""
    flag = np.zeros(len(base), dtype=bool)
    idx_by_country = {cid: g.index.to_numpy() for cid, g in base.groupby("country_id")}
    years = base["year"].to_numpy()
    for e in eps.itertuples(index=False):
        start = getattr(e, start_col)
        if pd.isna(start) or e.country_id not in idx_by_country:
            continue
        stop = e.end_year if pd.isna(e.breakdown_year) else min(e.end_year, int(e.breakdown_year) - 1)
        rows = idx_by_country[e.country_id]
        y = years[rows]
        flag[rows[(y >= int(start)) & (y <= stop)]] = True
    return flag


def assign_states(
    ert: pd.DataFrame,
    eps: pd.DataFrame,
    cfg: dict[str, Any],
    realtime: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Country-year table with ``stage`` (0/1/2) and diagnostic flags.

    Columns: country_id, year, v2x_regime, reg_type, row_democracy (RoW rule only),
    democracy (the risk-set definition actually used), in_known_episode, stage,
    in_unconfirmed_final_episode (inside a final-coded episode that has started but is not yet
    confirmed under the rule: used by the onset/'drop' option and for diagnostics).
    """
    source = cfg["labels"]["state_source"]
    min_regime = int(cfg["democracy"]["min_regime"])
    first, last = int(cfg["years"]["first"]), int(cfg["years"]["last"])

    require_ert = bool(cfg["democracy"].get("require_ert_democracy", False))
    cols = ["country_id", "year", "v2x_regime"]
    if require_ert:
        if "reg_type" not in ert.columns:
            raise ValueError("democracy.require_ert_democracy = true needs ERT's reg_type column")
        cols.append("reg_type")
    base = ert.loc[(ert["year"] >= first) & (ert["year"] <= last), cols].copy()
    base = base.sort_values(["country_id", "year"]).reset_index(drop=True)
    base["row_democracy"] = base["v2x_regime"].fillna(-1) >= min_regime
    if require_ert:
        base["democracy"] = base["row_democracy"] & (base["reg_type"].fillna(-1) == 1)
    else:
        base["democracy"] = base["row_democracy"]

    if source == "ert_truncated":
        if realtime is None:
            raise ValueError("state_source = ert_truncated needs the real-time ERT table (ert_jobs.load_realtime)")
        rt = realtime[["country_id", "year", "rt_in_dem_episode", "rt_post_breakdown"]]
        base = base.merge(rt, on=["country_id", "year"], how="left", validate="one_to_one")
        missing = base["rt_in_dem_episode"].isna() & base["democracy"]
        if missing.any():
            yrs = sorted(base.loc[missing, "year"].unique())[:10]
            raise ValueError(f"real-time ERT status missing for democratic country-years, e.g. years {yrs}")
        in_ep = base["rt_in_dem_episode"].fillna(False).astype(bool) & ~base["rt_post_breakdown"].fillna(False).astype(bool)
        base = base.drop(columns=["rt_in_dem_episode", "rt_post_breakdown"])
    elif source == "rule":
        in_ep = pd.Series(_final_episode_flags(base, eps, "confirm_year_rule"), index=base.index)
    else:  # hindsight
        in_ep = pd.Series(_final_episode_flags(base, eps, "start_year"), index=base.index)

    base["in_known_episode"] = in_ep.to_numpy()
    base["stage"] = np.where(base["democracy"], np.where(base["in_known_episode"], 2, 1), 0).astype("int8")

    # Diagnostic: started (final coding) but not yet confirmed under the rule at t.
    started = _final_episode_flags(base, eps, "start_year")
    confirmed = _final_episode_flags(base, eps, "confirm_year_rule")
    base["in_unconfirmed_final_episode"] = started & ~confirmed
    base["state_source"] = source
    return base
