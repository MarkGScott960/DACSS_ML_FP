"""Forecast labels for stage 1 (onset or confirmation) and stage 2 (breakdown).

For a row with origin year t and horizon k:

    y(t, k) = 1  if the country has an event of the stage's type in years t+1 .. t+k
              0  otherwise

Event types (from ``episodes.country_events``):
    stage 1, stage1_event = confirmation:  'declaration' (state_source ert_truncated)
                                           'confirmation_rule' (state_source rule / hindsight)
    stage 1, stage1_event = onset:         'onset'
    stage 2:                               'breakdown'
    optional stage-1 targets:              'breakdown' (stage1_breakdown_target),
                                           'row_breakdown' (stage1_row_breakdown_target; sensitivity)

A label is **observable** only if (a) every year t+1 .. t+k exists for the country in
the data (no gap, the country still exists) and (b) t + k <= last_year - maturity buffer.
The rule is symmetric: positives and negatives are dropped alike, so the base rate is
not distorted near the end of the data. Unobservable rows are kept with
``observable = False`` so that the audits can count them; models must filter on it.

Leakage guarantee (tested in tests/test_labels.py): y(t, k) depends only on events in
years strictly after t. Features (later zip) use only years <= t.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def stage1_event_type(cfg: dict[str, Any]) -> str:
    lab = cfg["labels"]
    if lab["stage1_event"] == "onset":
        return "onset"
    return "declaration" if lab["state_source"] == "ert_truncated" else "confirmation_rule"


def maturity_buffer(cfg: dict[str, Any], stage: int, override: int | None = None) -> int:
    if override is not None:
        return int(override)
    m = cfg["labels"]["maturity"]
    if stage == 2:
        return int(m["stage2"]["default"])
    key = "stage1_onset" if cfg["labels"]["stage1_event"] == "onset" else "stage1"
    return int(m[key]["default"])


def _window_labels(
    years: np.ndarray, present: np.ndarray, events: np.ndarray, origins: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """For one country on a contiguous year grid: (any event in (t, t+k], all years in (t, t+k] present).

    years   : contiguous grid first..last (int)
    present : bool, country-year exists in the data
    events  : bool, event in that year
    origins : origin years (subset of grid)
    """
    first = years[0]
    ev_cum = np.concatenate([[0], np.cumsum(events)])  # ev_cum[i] = events in years[:i]
    pr_cum = np.concatenate([[0], np.cumsum(present)])
    i = origins - first  # index of t on the grid
    j = np.minimum(i + k, len(years) - 1)  # index of t+k (clipped to grid)
    n_ev = ev_cum[j + 1] - ev_cum[i + 1]
    n_pr = pr_cum[j + 1] - pr_cum[i + 1]
    complete = (i + k <= len(years) - 1) & (n_pr == k)
    return n_ev > 0, complete


def build_labels(
    states: pd.DataFrame,
    events: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    include_exploratory: bool = True,
    buffer_override: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Long label table: one row per (country, origin year, stage, horizon).

    Columns: country_id, year (origin t), stage, k, horizon_set, event, y, observable.
    Only stage-1 and stage-2 rows are emitted; with stage1_event = onset and
    onset_unconfirmed_rows = drop, stage-1 rows inside a started-but-unconfirmed final episode are removed.
    """
    lab = cfg["labels"]
    last_year = int(cfg["years"]["last"])
    buffer_override = buffer_override or {}

    h1 = [(k, "confirmatory") for k in lab["horizons_stage1"]]
    if include_exploratory:
        h1 += [(k, "exploratory") for k in lab["horizons_stage1_exploratory"]]
    plan = [(1, stage1_event_type(cfg), h1), (2, "breakdown", [(k, "confirmatory") for k in lab["horizons_stage2"]])]
    if lab.get("stage1_breakdown_target", False):
        # Optional: breakdown within k years forecast from every stage-1 row (real-time
        # breakdown forecasting across all democracies; decision D-08 in docs/decisions_log.md).
        plan.append((1, "breakdown", h1))
    if lab.get("stage1_row_breakdown_target", False):
        # Sensitivity (decision D-15): any RoW democracy -> autocracy change within k years.
        if not (events["event"] == "row_breakdown").any():
            raise ValueError(
                "labels.stage1_row_breakdown_target is on but the event table has no 'row_breakdown' "
                "events: build it with episodes.country_events(eps, rt, ert)"
            )
        plan.append((1, "row_breakdown", h1))
    if lab.get("stage1_declaration_split", False) and stage1_event_type(cfg) == "declaration":
        # H1b (decision D-07b): gradual vs abrupt declarations, scored with the H1 models.
        n_decl = int((events["event"] == "declaration").sum())
        n_split = int(events["event"].isin(["declaration_dem", "declaration_abrupt"]).sum())
        if n_decl != n_split:
            raise ValueError(
                f"labels.stage1_declaration_split is on but {n_split} of {n_decl} declarations are split: "
                "run episodes.split_declarations(ev, states) before build_labels"
            )
        for ev_type in ("declaration_dem", "declaration_abrupt"):
            plan.append((1, ev_type, h1))

    st = states.sort_values(["country_id", "year"])
    if lab["stage1_event"] == "onset" and lab["onset_unconfirmed_rows"] == "drop":
        st = st[~((st["stage"] == 1) & st["in_unconfirmed_final_episode"])]

    out = []
    for stage, ev_type, horizons in plan:
        ev = events.loc[events["event"] == ev_type, ["country_id", "year"]]
        ev_sets = ev.groupby("country_id")["year"].apply(set).to_dict()
        # Breakdown labels need the breakdown-confirmation buffer whichever stage they are scored on.
        buffer = maturity_buffer(cfg, 2 if ev_type == "breakdown" else stage, buffer_override.get(stage))
        for cid, g in states.groupby("country_id", sort=True):
            origins = st.loc[(st["country_id"] == cid) & (st["stage"] == stage), "year"].to_numpy()
            if len(origins) == 0:
                continue
            grid = np.arange(int(g["year"].min()), int(g["year"].max()) + 1)
            present = np.isin(grid, g["year"].to_numpy())
            evs = ev_sets.get(cid, set())
            is_ev = np.isin(grid, list(evs)) if evs else np.zeros(len(grid), dtype=bool)
            for k, hset in horizons:
                y, complete = _window_labels(grid, present, is_ev, origins, k)
                mature = origins + k <= last_year - buffer
                out.append(
                    pd.DataFrame(
                        {
                            "country_id": cid,
                            "year": origins,
                            "stage": stage,
                            "k": k,
                            "horizon_set": hset,
                            "event": ev_type,
                            "y": y.astype("int8"),
                            "observable": complete & mature,
                        }
                    )
                )
    labels = pd.concat(out, ignore_index=True)
    labels["stage"] = labels["stage"].astype("int8")
    labels["k"] = labels["k"].astype("int8")
    return labels.sort_values(["stage", "k", "country_id", "year"]).reset_index(drop=True)


def label_summary(labels: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Rows, positives and base rate per stage x horizon x period (development / locked / later).

    Outcome-only counts: allowed before the pre-registration freeze.
    """
    sp = cfg["splits"]
    obs = labels[labels["observable"]].copy()
    obs["period"] = np.select(
        [obs["year"] <= sp["dev_last_origin"], obs["year"] <= sp["locked"]["last_origin"]],
        ["development", "locked"],
        default="after_locked",
    )
    g = obs.groupby(["stage", "event", "k", "horizon_set", "period"])
    summ = g.agg(rows=("y", "size"), positives=("y", "sum"), countries=("country_id", "nunique")).reset_index()
    summ["base_rate"] = summ["positives"] / summ["rows"]
    return summ
