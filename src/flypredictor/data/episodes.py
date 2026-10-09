"""Democratic autocratization episodes and the country-level events they generate.

Definitions (plan section 3.2; ERT v16 codebook):

* A **democratic episode** is an ERT autocratization episode with ``aut_ep_prch == 1``:
  it begins in a democracy (ERT regime type in the pre-episode year) and could end
  in breakdown. Episodes with ``aut_ep_outcome == 5`` ("regressed autocracy") started
  from autocracy and are excluded by construction.
* **Onset year** = ERT ``aut_ep_start_year`` (first year of a decline of at least start_incl).
* **Confirmation year (rule)** = first year y >= onset in which the EDI range over
  [pre-episode year, y] reaches cum_incl. This mirrors ERT's manifest test
  (max - min of v2x_polyarchy over the episode >= cum_incl) using only data up to y.
  The exact real-time version comes from re-running ERT on truncated data
  (``ert_jobs``); this rule is the fallback and a cross-check.
* **Breakdown year** = first year inside the episode with ``row_regch_event == -1``
  (RoW moves from democracy to autocracy) for episodes with ``aut_ep_outcome == 1``.
  ERT's ``aut_ep_pbr`` switches on the year after this.
* **Breakdown confirmation year** = when ERT could first code the breakdown as genuine:
  the breakdown year itself for a move to closed autocracy, otherwise the earlier of the
  founding autocratic election and breakdown year + tolerance. Used for the stage-2
  maturity audit.
* **Severity** = EDI(pre-episode year) - EDI(end year); also the maximum drop.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import Paths

EPS = 1e-9  # tolerance for floating-point threshold comparisons

OUTCOME_LABELS = {
    1: "democratic breakdown",
    2: "preempted breakdown",
    3: "diminished democracy",
    4: "averted regression",
    5: "regressed autocracy",
    6: "outcome censored",
}


def load_ert(paths: Paths) -> pd.DataFrame:
    """ERT v16 country-year table with integer keys, sorted by country and year."""
    df = pd.read_parquet(paths.ert_parquet)
    df["country_id"] = df["country_id"].astype("int64")
    df["year"] = df["year"].astype("int64")
    return df.sort_values(["country_id", "year"]).reset_index(drop=True)


def _confirmation_year(edi: pd.Series, start: int, end: int, cum_incl: float, tolerance: int) -> float:
    """First year y in [start, end + tolerance] with range(EDI[start-1 .. y]) >= cum_incl."""
    last = min(end + tolerance, int(edi.index.max()))
    lo = hi = edi.get(start - 1, np.nan)
    if np.isnan(lo):
        return np.nan
    for y in range(start, last + 1):
        v = edi.get(y, np.nan)
        if np.isnan(v):
            continue
        lo, hi = min(lo, v), max(hi, v)
        if hi - lo >= cum_incl - EPS:
            return float(y)
    return np.nan


def democratic_episodes(ert: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """One row per democratic episode (``aut_ep_prch == 1``) with timing, outcome and severity."""
    p = cfg["ert"]["defaults"]
    cum_incl, tolerance = float(p["cum_incl"]), int(p["tolerance"])
    min_regime = int(cfg["democracy"]["min_regime"])

    ep_rows = ert[(ert["aut_ep"] == 1) & (ert["aut_ep_prch"] == 1)]
    by_country = {cid: g.set_index("year") for cid, g in ert.groupby("country_id")}
    records = []
    for ep_id, g in ep_rows.groupby("aut_ep_id", sort=True):
        cid = int(g["country_id"].iloc[0])
        c = by_country[cid]
        start = int(g["aut_ep_start_year"].iloc[0])
        end = int(g["aut_ep_end_year"].iloc[0])
        outcome = int(g["aut_ep_outcome"].max())
        edi = c["v2x_polyarchy"]
        window = c.loc[start:end]

        breakdown = np.nan
        breakdown_confirm = np.nan
        if outcome == 1:
            hits = window.index[window["row_regch_event"] == -1]
            if len(hits) == 0:  # fallback: first RoW-autocratic year in the episode
                hits = window.index[window["v2x_regime"] < min_regime]
            if len(hits) == 0:
                raise ValueError(f"{ep_id}: outcome is breakdown but no breakdown year found")
            breakdown = float(hits.min())
            b = int(breakdown)
            if c.loc[b, "v2x_regime"] == 0:  # closed autocracy: genuine at once
                breakdown_confirm = breakdown
            else:
                fe = c.loc[b:, "aut_founding_elec"].dropna()
                fe_year = float(fe.iloc[0]) if len(fe) else np.inf
                breakdown_confirm = float(min(fe_year, b + tolerance))

        edi_pre = float(edi.get(start - 1, np.nan))
        records.append(
            {
                "episode_id": ep_id,
                "country_id": cid,
                "country_text_id": g["country_text_id"].iloc[0],
                "country_name": g["country_name"].iloc[0],
                "pre_year": start - 1,
                "start_year": start,
                "end_year": end,
                "duration": end - start + 1,
                "confirm_year_rule": _confirmation_year(edi, start, end, cum_incl, tolerance),
                "breakdown_year": breakdown,
                "breakdown_confirm_year": breakdown_confirm,
                "outcome": outcome,
                "outcome_label": OUTCOME_LABELS[outcome],
                "outcome_agg": int(g["aut_ep_outcome_agg"].max()),
                "censored": outcome == 6,
                "duration_censored": bool(g["aut_ep_censored"].max() == 1),
                "regime_pre": float(c["v2x_regime"].get(start - 1, np.nan)),
                "edi_pre": edi_pre,
                "edi_end": float(edi.get(end, np.nan)),
                "edi_min": float(window["v2x_polyarchy"].min()),
            }
        )
    eps = pd.DataFrame.from_records(records)
    eps["severity_total"] = eps["edi_pre"] - eps["edi_end"]
    eps["severity_max"] = eps["edi_pre"] - eps["edi_min"]
    eps["confirm_lag"] = eps["confirm_year_rule"] - eps["start_year"]
    eps["breakdown_confirm_lag"] = eps["breakdown_confirm_year"] - eps["breakdown_year"]
    for col in ("confirm_year_rule", "breakdown_year", "breakdown_confirm_year", "confirm_lag", "breakdown_confirm_lag"):
        eps[col] = eps[col].astype("Int64")
    return eps.sort_values(["country_id", "start_year"]).reset_index(drop=True)


def check_episodes(eps: pd.DataFrame) -> list[str]:
    """Consistency checks on the episode table. Returns a list of problems (empty = OK)."""
    problems = []
    if (eps["outcome"] == 5).any():
        problems.append("outcome 5 (regressed autocracy) present among democratic episodes")
    if eps["episode_id"].duplicated().any():
        problems.append("duplicate episode ids")
    bd = eps[eps["outcome"] == 1]
    if bd["breakdown_year"].isna().any():
        problems.append("breakdown episode without a breakdown year")
    if ((bd["breakdown_year"] < bd["start_year"]) | (bd["breakdown_year"] > bd["end_year"])).any():
        problems.append("breakdown year outside the episode")
    if eps["confirm_year_rule"].isna().any():
        problems.append(f"{int(eps['confirm_year_rule'].isna().sum())} episodes never reach cum_incl under the rule")
    if (eps["regime_pre"] < 2).any():
        problems.append("episode whose pre-episode year is not a democracy on RoW")
    return problems


def country_events(
    eps: pd.DataFrame, realtime: pd.DataFrame | None = None, ert: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Country-level event table: one row per (country, year, event type).

    Event types
    -----------
    onset             final-coded democratic episode starts
    confirmation_rule episode confirmed under the Python rule
    declaration       ERT, re-run on data truncated at year T, first codes the country as being in a
                      democratic episode (status flips 0 -> 1 between T-1 and T); needs ``realtime``
    breakdown         democratic breakdown (RoW change to autocracy inside a breakdown episode)
    row_breakdown     any RoW change from democracy to autocracy (``row_regch_event == -1``),
                      whether or not ERT codes a democratic episode; needs ``ert``.
                      Sensitivity outcome (decision D-15).
    """
    frames = [
        eps[["country_id", "start_year"]].rename(columns={"start_year": "year"}).assign(event="onset"),
        eps[["country_id", "confirm_year_rule"]]
        .dropna()
        .rename(columns={"confirm_year_rule": "year"})
        .assign(event="confirmation_rule"),
        eps[["country_id", "breakdown_year"]].dropna().rename(columns={"breakdown_year": "year"}).assign(event="breakdown"),
    ]
    if realtime is not None:
        rt = realtime.sort_values(["country_id", "year"])
        prev = rt.groupby("country_id")["rt_in_dem_episode"].shift(1)
        prev_year = rt.groupby("country_id")["year"].shift(1)
        flips = rt[(rt["rt_in_dem_episode"]) & (prev == False) & (prev_year == rt["year"] - 1)]  # noqa: E712
        frames.append(flips[["country_id", "year"]].assign(event="declaration"))
    if ert is not None:
        rb = ert.loc[ert["row_regch_event"] == -1, ["country_id", "year"]]
        frames.append(rb.assign(event="row_breakdown"))
    ev = pd.concat(frames, ignore_index=True)
    ev["year"] = ev["year"].astype("int64")
    ev["country_id"] = ev["country_id"].astype("int64")
    return ev.drop_duplicates().sort_values(["country_id", "year", "event"]).reset_index(drop=True)
