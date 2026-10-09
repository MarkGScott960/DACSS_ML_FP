"""Risk-set definition (RoW + ERT reg_type), the RoW-breakdown sensitivity event and the
coverage audits added after the zip-1 review (decisions D-15, D-16)."""

import numpy as np
import pandas as pd
import pytest

from flypredictor.audit import coverage, definitions
from flypredictor.data import episodes, labels, states


def _stage(st, cid, year):
    return int(st.loc[(st.country_id == cid) & (st.year == year), "stage"].iloc[0])


def test_require_ert_democracy_removes_rows(ert, cfg):
    # Country B is a RoW democracy throughout; pretend ERT treats 1995-1999 as autocratic.
    ert.loc[(ert.country_id == 2) & ert.year.between(1995, 1999), "reg_type"] = 0
    eps = episodes.democratic_episodes(ert, cfg)

    cfg["democracy"]["require_ert_democracy"] = True
    st = states.assign_states(ert, eps, cfg)
    assert [_stage(st, 2, y) for y in (1994, 1995, 1999, 2000)] == [1, 0, 0, 1]
    row = st[(st.country_id == 2) & (st.year == 1995)].iloc[0]
    assert bool(row["row_democracy"]) and not bool(row["democracy"])

    cfg["democracy"]["require_ert_democracy"] = False
    st = states.assign_states(ert, eps, cfg)
    assert _stage(st, 2, 1995) == 1


def test_require_ert_democracy_needs_reg_type(ert, cfg):
    cfg["democracy"]["require_ert_democracy"] = True
    with pytest.raises(ValueError):
        states.assign_states(ert.drop(columns="reg_type"), episodes.democratic_episodes(ert, cfg), cfg)


def test_row_breakdown_event_and_label(ert, cfg):
    # D: a RoW drop with no ERT episode (near-threshold), 2005, ERT reg_type stays democratic before it.
    ert.loc[(ert.country_id == 4) & (ert.year == 2005), ["v2x_regime", "row_regch_event"]] = [1, -1]
    ert.loc[(ert.country_id == 4) & (ert.year > 2005), "v2x_regime"] = 1
    ert.loc[(ert.country_id == 4) & (ert.year >= 2005), "reg_type"] = 0
    eps = episodes.democratic_episodes(ert, cfg)
    ev = episodes.country_events(eps, None, ert)
    rb = ev[ev.event == "row_breakdown"]
    assert set(zip(rb.country_id, rb.year)) == {(1, 2004), (4, 2005)}
    assert set(zip(*ev[ev.event == "breakdown"][["country_id", "year"]].values.T)) == {(1, 2004)}

    st = states.assign_states(ert, eps, cfg)
    cfg["labels"]["stage1_row_breakdown_target"] = True
    lab = labels.build_labels(st, ev, cfg, include_exploratory=False)
    y = lab[(lab.event == "row_breakdown") & (lab.k == 1)].set_index(["country_id", "year"])["y"]
    yb = lab[(lab.event == "breakdown") & (lab.stage == 1) & (lab.k == 1)].set_index(["country_id", "year"])["y"]
    assert y[(4, 2004)] == 1 and yb[(4, 2004)] == 0      # counted only by the sensitivity outcome
    assert y[(4, 2003)] == 0


def test_row_breakdown_target_requires_events(ert, cfg):
    eps = episodes.democratic_episodes(ert, cfg)
    st = states.assign_states(ert, eps, cfg)
    cfg["labels"]["stage1_row_breakdown_target"] = True
    with pytest.raises(ValueError):
        labels.build_labels(st, episodes.country_events(eps), cfg)


def test_row_transitions_audit(ert, cfg):
    eps = episodes.democratic_episodes(ert, cfg)
    t = definitions.row_transitions(ert, eps, first_year=1990)
    assert len(t) == 1
    assert bool(t.loc[0, "ert_breakdown"]) and not bool(t.loc[0, "reverts"])


def _toy_panel():
    years = np.arange(1990, 2026)
    p = pd.DataFrame({"country_id": 1, "year": years, "stage": 1})
    p["full"] = 1.0
    p["ends_2019"] = np.where(years <= 2019, 1.0, np.nan)           # GDP-like
    p["sparse_ends_2019"] = np.where((years >= 2010) & (years <= 2019), 1.0, np.nan)  # v2xpas-like
    return p


def test_era_proxy_ends_early():
    f = coverage.era_proxy_flags(_toy_panel(), ["full", "ends_2019", "sparse_ends_2019"]).set_index("variable")
    assert not f.loc["full", "era_proxy"]
    assert f.loc["ends_2019", "era_proxy"] and "ends_early" in f.loc["ends_2019", "reason"]
    assert f.loc["sparse_ends_2019", "era_proxy"] and "ends_early" in f.loc["sparse_ends_2019", "reason"]


def test_window_coverage(cfg):
    cfg["splits"] = {"dev_last_origin": 2004, "locked": {"first_origin": 2005, "last_origin": 2024}}
    w = coverage.window_coverage(_toy_panel(), ["full", "ends_2019"], cfg).set_index("variable")
    assert not w.loc["full", "locked_gap"]
    assert w.loc["ends_2019", "locked_gap"] and w.loc["ends_2019", "last_year_democracy_rows"] == 2019
