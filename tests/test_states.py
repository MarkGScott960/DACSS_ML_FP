"""Stage assignment under each state source."""

import pytest

from flypredictor.data import episodes, states


def _stages(st, cid):
    s = st[st.country_id == cid].set_index("year")["stage"]
    return {y: int(v) for y, v in s.items()}


@pytest.mark.parametrize("source", ["rule", "hindsight", "ert_truncated"])
def test_partition(ert, cfg, realtime, source):
    cfg["labels"]["state_source"] = source
    eps = episodes.democratic_episodes(ert, cfg)
    st = states.assign_states(ert, eps, cfg, realtime if source == "ert_truncated" else None)
    dem = st["democracy"]
    assert set(st.loc[dem, "stage"]) <= {1, 2}       # every democracy row is in exactly one risk set
    assert (st.loc[~dem, "stage"] == 0).all()
    assert _stages(st, 5) == {y: 0 for y in range(1990, 2010)}  # autocracy never at risk


def test_rule_states(ert, cfg):
    st = states.assign_states(ert, episodes.democratic_episodes(ert, cfg), cfg)
    a = _stages(st, 1)
    assert all(a[y] == 1 for y in range(1990, 2002))   # before confirmation: stage 1 (real time)
    assert a[2002] == 2 and a[2003] == 2                 # confirmed, not yet broken down
    assert all(a[y] == 0 for y in range(2004, 2010))   # RoW autocracy from 2004
    c = _stages(st, 3)
    assert c[1996] == 1 and c[1997] == 2 and c[1998] == 1
    # started but not yet confirmed (diagnostic flag)
    flag = st[(st.country_id == 1)].set_index("year")["in_unconfirmed_final_episode"]
    assert flag[2000] and flag[2001] and not flag[2002] and not flag[1999]


def test_hindsight_states(ert, cfg):
    cfg["labels"]["state_source"] = "hindsight"
    st = states.assign_states(ert, episodes.democratic_episodes(ert, cfg), cfg)
    a = _stages(st, 1)
    assert a[1999] == 1 and a[2000] == 2 and a[2001] == 2 and a[2003] == 2


def test_truncated_requires_realtime(ert, cfg):
    cfg["labels"]["state_source"] = "ert_truncated"
    with pytest.raises(ValueError):
        states.assign_states(ert, episodes.democratic_episodes(ert, cfg), cfg, None)
