"""Democratic episodes and events on the synthetic fixture (answers in conftest.py)."""

from flypredictor.data import episodes


def test_only_democratic_episodes(ert, cfg):
    eps = episodes.democratic_episodes(ert, cfg)
    assert set(eps["episode_id"]) == {"AAA_2000_2005", "CCC_1995_1997"}  # EEE (outcome 5) excluded
    assert episodes.check_episodes(eps) == []


def test_confirmation_breakdown_and_severity(ert, cfg):
    eps = episodes.democratic_episodes(ert, cfg).set_index("episode_id")
    a, c = eps.loc["AAA_2000_2005"], eps.loc["CCC_1995_1997"]
    assert a["confirm_year_rule"] == 2002 and a["confirm_lag"] == 2
    assert a["breakdown_year"] == 2004
    assert a["breakdown_confirm_year"] == 2006 and a["breakdown_confirm_lag"] == 2
    assert abs(a["severity_total"] - (0.80 - 0.40)) < 1e-9
    assert c["confirm_year_rule"] == 1997 and c["outcome"] == 4
    assert c["breakdown_year"] is not None and str(c["breakdown_year"]) == "<NA>"


def test_events_without_realtime(ert, cfg):
    ev = episodes.country_events(episodes.democratic_episodes(ert, cfg))
    got = {(r.country_id, r.year, r.event) for r in ev.itertuples()}
    assert got == {
        (1, 2000, "onset"), (1, 2002, "confirmation_rule"), (1, 2004, "breakdown"),
        (3, 1995, "onset"), (3, 1997, "confirmation_rule"),
    }


def test_declaration_events_from_realtime(ert, cfg, realtime):
    ev = episodes.country_events(episodes.democratic_episodes(ert, cfg), realtime)
    decl = {(r.country_id, r.year) for r in ev[ev.event == "declaration"].itertuples()}
    assert decl == {(1, 2002), (3, 1997)}
