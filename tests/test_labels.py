"""Labels: hand-computed values, observability, maturity, and the no-leak property."""

import numpy as np
import pytest

from flypredictor.data import episodes, labels, states


def _build(ert, cfg, realtime=None):
    eps = episodes.democratic_episodes(ert, cfg)
    ev = episodes.country_events(eps, realtime)
    st = states.assign_states(ert, eps, cfg, realtime)
    return labels.build_labels(st, ev, cfg)


def _y(lab, cid, year, stage, k, event=None):
    r = lab[(lab.country_id == cid) & (lab.year == year) & (lab.stage == stage) & (lab.k == k)]
    if event is not None:
        r = r[r.event == event]
    assert len(r) == 1, f"expected one row, got {len(r)}"
    return int(r["y"].iloc[0]), bool(r["observable"].iloc[0])


def test_stage1_confirmation_labels(ert, cfg):
    lab = _build(ert, cfg)
    ev = "confirmation_rule"
    assert _y(lab, 1, 2001, 1, 1, ev) == (1, True)   # confirmed in 2002
    assert _y(lab, 1, 2000, 1, 1, ev) == (0, True)   # 2001: nothing yet
    assert _y(lab, 1, 2000, 1, 2, ev) == (1, True)
    assert _y(lab, 1, 1999, 1, 3, ev) == (1, True)
    assert _y(lab, 1, 1996, 1, 5, ev) == (0, True)   # 1997-2001: no confirmation


def test_stage2_breakdown_labels(ert, cfg):
    lab = _build(ert, cfg)
    assert _y(lab, 1, 2002, 2, 1) == (0, True)
    assert _y(lab, 1, 2002, 2, 2) == (1, True)
    assert _y(lab, 1, 2003, 2, 1) == (1, True)
    assert _y(lab, 3, 1997, 2, 3) == (0, True)       # averted episode: no breakdown


def test_stage1_breakdown_target(ert, cfg):
    lab = _build(ert, cfg)
    assert _y(lab, 1, 2001, 1, 3, "breakdown") == (1, True)
    assert _y(lab, 1, 2001, 1, 2, "breakdown") == (0, True)


def test_event_in_origin_year_is_not_counted(ert, cfg):
    """No leak: an event in year t itself never makes y(t, k) = 1."""
    lab = _build(ert, cfg)
    # Origin 2004 is not a democracy for A, so use C: confirmed 1997; origin 1998 is stage 1 again.
    assert _y(lab, 3, 1998, 1, 1, "confirmation_rule") == (0, True)


def test_observability_gap_and_end(ert, cfg):
    lab = _build(ert, cfg)
    ev = "confirmation_rule"
    assert _y(lab, 4, 1995, 1, 1, ev)[1] is False    # 1996 missing for D
    assert _y(lab, 4, 1994, 1, 3, ev)[1] is False    # window crosses the gap
    assert _y(lab, 4, 1998, 1, 1, ev)[1] is True
    assert _y(lab, 2, 2005, 1, 5, ev)[1] is False    # 2010 is past the last year
    assert _y(lab, 2, 2004, 1, 5, ev)[1] is True


def test_maturity_buffer_is_symmetric(ert, cfg):
    cfg["labels"]["maturity"]["stage2"]["default"] = 3
    lab = _build(ert, cfg)
    # Breakdown labels: last observable origin is 2009 - 3 - k.
    s1b = lab[(lab.event == "breakdown") & (lab.stage == 1) & (lab.k == 1) & lab.observable]
    assert s1b["year"].max() == 2005
    # The buffer drops positives and negatives alike (no row past the cut-off is observable).
    late = lab[(lab.event == "breakdown") & (lab.year + lab.k > 2006)]
    assert not late["observable"].any()


def test_onset_mode_drop(ert, cfg):
    cfg["labels"]["stage1_event"] = "onset"
    cfg["labels"]["onset_unconfirmed_rows"] = "drop"
    lab = _build(ert, cfg)
    a = lab[(lab.country_id == 1) & (lab.stage == 1) & (lab.event == "onset")]
    assert not a["year"].isin([2000, 2001]).any()   # started-but-unconfirmed rows removed
    assert _y(lab, 1, 1999, 1, 1, "onset") == (1, True)


def test_onset_mode_zero_keeps_rows(ert, cfg):
    cfg["labels"]["stage1_event"] = "onset"
    cfg["labels"]["onset_unconfirmed_rows"] = "zero"
    lab = _build(ert, cfg)
    assert _y(lab, 1, 2000, 1, 1, "onset") == (0, True)


def test_declaration_labels_with_realtime(ert, cfg, realtime):
    cfg["labels"]["state_source"] = "ert_truncated"
    lab = _build(ert, cfg, realtime)
    assert set(lab.loc[lab.stage == 1, "event"]) == {"declaration", "breakdown"}
    assert _y(lab, 1, 2001, 1, 1, "declaration") == (1, True)


def test_window_labels_match_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(5, 40))
        grid = np.arange(1950, 1950 + n)
        present = rng.random(n) > 0.1
        events = (rng.random(n) > 0.85) & present
        origins = grid[present]
        for k in (1, 2, 3, 5):
            y, complete = labels._window_labels(grid, present, events, origins, k)
            for t, yy, cc in zip(origins, y, complete):
                win = (grid > t) & (grid <= t + k)
                assert yy == bool(events[win].any())
                assert cc == (t + k <= grid[-1] and bool(present[win].all()) and win.sum() == k)


def test_label_summary_periods(ert, cfg):
    lab = _build(ert, cfg)
    s = labels.label_summary(lab, cfg)
    assert set(s["period"]) <= {"development", "locked", "after_locked"}
    assert (s["positives"] <= s["rows"]).all()


# ---------------------------------------------------------------------------
# Training populations P1-P3 (decision D-24)
# ---------------------------------------------------------------------------

def _build_pop(ert, cfg, realtime):
    cfg["labels"]["state_source"] = "ert_truncated"
    cfg["training_population"]["enabled"] = True
    eps = episodes.democratic_episodes(ert, cfg)
    ev = episodes.country_events(eps, realtime)
    st = states.assign_states(ert, eps, cfg, realtime)
    return st, ev, labels.build_labels(st, ev, cfg)


def test_declarations_any(realtime):
    d = episodes.declarations_any(realtime).set_index("country_id")
    assert d.loc[1, "year"] == 2002 and d.loc[1, "start_year"] == 2000   # A: same as its declaration
    assert d.loc[3, "year"] == 1997
    assert d.loc[5, "year"] == 2002 and d.loc[5, "start_year"] == 2001   # E: autocratic episode
    assert len(d) == 3


def test_population_groups(ert, cfg, realtime):
    st, _, _ = _build_pop(ert, cfg, realtime)
    g = st.set_index(["country_id", "year"])["pop_group"]
    assert g[(2, 1995)] == "democracy"
    assert g[(5, 2000)] == "electoral_autocracy"
    assert g[(5, 2004)] == "electoral_autocracy"
    assert g.isna()[(5, 2002)] and g.isna()[(5, 2003)]    # inside E's known episode: not at risk
    assert g.isna()[(1, 2002)]                              # stage 2 is never a training row here
    assert g.isna()[(1, 2004)] and g.isna()[(1, 2005)]    # autocracy, still in A's known episode
    assert g[(1, 2006)] == "electoral_autocracy"
    # every stage-1 row is 'democracy' and nothing else is
    assert (st.loc[st["stage"] == 1, "pop_group"] == "democracy").all()
    assert not (st.loc[st["stage"] != 1, "pop_group"] == "democracy").any()


def test_declaration_any_labels(ert, cfg, realtime):
    _, _, lab = _build_pop(ert, cfg, realtime)
    assert _y(lab, 5, 2001, 0, 1, "declaration_any") == (1, True)   # E declared in 2002
    assert _y(lab, 5, 2000, 0, 1, "declaration_any") == (0, True)
    assert _y(lab, 5, 2000, 0, 2, "declaration_any") == (1, True)
    assert lab[(lab.country_id == 5) & lab.year.isin([2002, 2003])].empty
    assert set(lab.loc[lab.stage == 0, "event"]) == {"declaration_any"}
    assert "declaration_any" not in set(lab.loc[lab.stage == 1, "event"])   # democracies keep H1


def test_training_rows_nest_and_keep_democracy_labels(ert, cfg, realtime):
    st, _, lab = _build_pop(ert, cfg, realtime)
    p = {name: labels.training_rows(lab, st, cfg, name, 1) for name in ("P1", "P2", "P3")}
    h1 = lab[(lab.stage == 1) & (lab.event == "declaration") & (lab.k == 1) & lab.observable]
    assert len(p["P3"]) == len(h1) and (p["P3"]["pop_group"] == "democracy").all()
    key = ["country_id", "year"]
    s1, s2, s3 = (set(map(tuple, p[n][key].to_numpy())) for n in ("P1", "P2", "P3"))
    assert s3 <= s2 <= s1
    dem = [d[d.pop_group == "democracy"].set_index(key)["y"] for d in p.values()]
    assert dem[0].equals(dem[1]) and dem[1].equals(dem[2])        # same democracy labels everywhere
    assert set(p["P2"]["pop_group"]) == {"democracy", "electoral_autocracy"}
    assert p["P1"].loc[(p["P1"].country_id == 5) & (p["P1"].year == 2001), "y"].item() == 1


def test_training_population_requires_any_events(ert, cfg, realtime):
    cfg["labels"]["state_source"] = "ert_truncated"
    cfg["training_population"]["enabled"] = True
    eps = episodes.democratic_episodes(ert, cfg)
    rt = realtime.drop(columns="rt_in_any_episode")
    ev = episodes.country_events(eps, rt)
    st = states.assign_states(ert, eps, cfg, rt)
    with pytest.raises(ValueError):
        labels.build_labels(st, ev, cfg)
