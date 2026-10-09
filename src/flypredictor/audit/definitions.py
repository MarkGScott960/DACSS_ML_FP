"""Risk-set and event definitions: RoW vs ERT (outcome-only; review finding, decision D-15).

The risk sets use Regimes of the World (``v2x_regime``) at t; ERT decides whether an
episode starts in a democracy with its own ``reg_type``, and only codes a democratic
breakdown inside such an episode. These audits count where the two disagree.
"""

from __future__ import annotations

import pandas as pd


def risk_set_vs_ert(states: pd.DataFrame, ert: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab of RoW democracy at t (``row_democracy``) against ERT ``reg_type``."""
    reg = ert[["country_id", "year", "reg_type"]]
    m = states[["country_id", "year", "row_democracy", "democracy", "stage"]].merge(
        reg, on=["country_id", "year"], how="left", validate="one_to_one"
    )
    return pd.crosstab(m["row_democracy"].rename("RoW democracy"), m["reg_type"].fillna(-1).rename("ERT reg_type"))


def row_transitions(ert: pd.DataFrame, eps: pd.DataFrame, first_year: int = 1900, revert_window: int = 2) -> pd.DataFrame:
    """Every RoW democracy -> autocracy change, and whether ERT counts it as a democratic breakdown.

    Columns: country_text_id, year, v2x_regime, v2x_polyarchy, reg_type_prev (ERT regime type the
    year before), ert_breakdown (matches a breakdown year in ``eps``), aut_ep_outcome (ERT episode
    outcome in that year, 5 = regressed autocracy), reverts (RoW back to democracy within
    ``revert_window`` years: a near-threshold flicker rather than a lasting change).
    """
    e = ert.sort_values(["country_id", "year"])
    prev_reg_type = e.groupby("country_id")["reg_type"].shift(1)
    t = e[(e["row_regch_event"] == -1) & (e["year"] >= first_year)].copy()
    t["reg_type_prev"] = prev_reg_type.loc[t.index]
    bk = {(int(c), int(y)) for c, y in zip(eps["country_id"], eps["breakdown_year"]) if pd.notna(y)}
    t["ert_breakdown"] = [(int(c), int(y)) in bk for c, y in zip(t["country_id"], t["year"])]
    regime = e.set_index(["country_id", "year"])["v2x_regime"]
    t["reverts"] = [
        any(regime.get((c, y + j), -1) >= 2 for j in range(1, revert_window + 1))
        for c, y in zip(t["country_id"], t["year"])
    ]
    cols = ["country_id", "country_text_id", "year", "v2x_regime", "v2x_polyarchy", "reg_type_prev",
            "aut_ep_outcome", "ert_breakdown", "reverts"]
    return t[cols].reset_index(drop=True)


def declaration_timing(events: pd.DataFrame, states: pd.DataFrame, dev_last_origin: int) -> pd.DataFrame:
    """Is the country still a democracy in the year ERT first declares its episode? (decision D-07)

    One row per real-time declaration, with the stage in the declaration year
    (0 = already not a democracy, i.e. the slide is first recognised at or after breakdown;
    2 = recognised while the country is still a democracy) and the period
    (development: declaration year <= dev_last_origin + 1).
    """
    d = events.loc[events["event"] == "declaration", ["country_id", "year"]]
    d = d.merge(states[["country_id", "year", "stage"]], on=["country_id", "year"], how="left")
    d["still_democracy"] = d["stage"] == 2
    d["period"] = (d["year"] <= dev_last_origin + 1).map({True: "development", False: "later"})
    return d.reset_index(drop=True)
