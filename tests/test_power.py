"""Power-simulation helpers and intercept updating."""

import numpy as np
import pandas as pd
import pytest

from flypredictor.eval import power, recalibration as R


def test_window_matches_label_rule():
    c = np.array([1, 1, 1, 1, 2, 2])
    t = np.array([2000, 2001, 2002, 2003, 2000, 2001])
    ev = np.array([0, 1, 0, 0, 0, 0], bool)        # event in 2002 for country 1
    assert power._window(ev, c, t, 1).tolist() == [False, True, False, False, False, False]
    assert power._window(ev, c, t, 2).tolist() == [True, True, False, False, False, False]


def test_solve_alpha_hits_target():
    lin = np.random.default_rng(0).normal(size=5000)
    a = power._solve_alpha(lin, 0.03)
    assert power._sigmoid(a + lin).mean() == pytest.approx(0.03, rel=1e-3)


def test_target_rates():
    states = pd.DataFrame({"country_id": [1] * 4 + [2] * 4, "year": [2003, 2004, 2005, 2006] * 2, "stage": 1})
    events = pd.DataFrame({"country_id": [1, 2], "year": [2005, 2006], "event": "declaration"})
    cfg = {"splits": {"dev_last_origin": 2004}}
    r = power.target_rates(events, states, cfg, "declaration")
    assert r == {"development": pytest.approx(1 / 4), "locked": pytest.approx(1 / 4)}


def test_intercept_shift():
    p = np.full(100, 0.01)
    y = np.r_[np.ones(3), np.zeros(97)]
    d = R.intercept_shift(p, y)
    assert R.apply_shift(p, d).mean() == pytest.approx(0.03, rel=1e-3)
    assert R.intercept_shift(p, np.zeros(100)) == 0.0          # no events: no shift


def test_summarise_intersection_and_mde():
    rows = []
    for (b, m), pw in {(0.0, 0.0): 0.0, (0.0, 0.5): 0.1, (1.0, 0.5): 0.5, (2.0, 0.5): 0.9}.items():
        for rep in range(10):
            for k in (1, 2):
                ok = rep < round(pw * 10) if k == 1 else True
                rows.append({"scenario": f"{b},{m}", "beta": b, "beta_mech": m, "rep": rep, "task": "H1", "k": k,
                             "pass": ok, "brier_skill": b / 10, "oracle_auc": 0.5 + b / 10 if k == 1 else np.nan})
    agg, mde = power.summarise(pd.DataFrame(rows))
    allk = mde[(mde.task == "H1") & (mde.k == "all")].iloc[0]
    assert allk["mde_beta"] == 2.0 and allk["false_positive_rate"] == 0.0 and allk["leading_null_rate"] == pytest.approx(0.1)
    assert allk["oracle_auc_at_mde"] == pytest.approx(0.7)
