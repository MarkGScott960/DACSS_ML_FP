"""Feature construction: hand-checked transformations, coverage rules, no look-ahead,
and the training-only preprocessor. Also the gradual/abrupt declaration split (H1b)."""

import numpy as np
import pandas as pd
import pytest

from flypredictor.config import all_feature_vars
from flypredictor.data import episodes, features, labels, states


def _panel(cfg, years=range(1940, 1961), countries=(1, 2), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in countries:
        for y in years:
            r = {"country_id": c, "year": y, "stage": 1}
            for v in all_feature_vars(cfg):
                r[v] = float(rng.normal())
            r["v2x_regime"] = 3.0 if y % 2 else 2.0
            r["e_pt_coup"] = float(y in (1951, 1955))
            r["e_pt_coup_attempts"] = 0.0
            rows.append(r)
    p = pd.DataFrame(rows)
    p.loc[p.year < 1950, ["e_pt_coup", "e_pt_coup_attempts"]] = np.nan
    return p


def test_changes_and_gaps(cfg):
    p = _panel(cfg)
    p = p[~((p.country_id == 2) & (p.year == 1950))]          # coding gap
    f = features.build_features(p, cfg).set_index(["country_id", "year"])
    x = p.set_index(["country_id", "year"])["v2cacamps"]
    assert f.loc[(1, 1955), "v2cacamps__d1"] == pytest.approx(x[(1, 1955)] - x[(1, 1954)])
    assert f.loc[(1, 1955), "v2cacamps__d3"] == pytest.approx(x[(1, 1955)] - x[(1, 1952)])
    assert np.isnan(f.loc[(2, 1951), "v2cacamps__d1"])      # 1950 missing
    assert np.isnan(f.loc[(2, 1953), "v2cacamps__d3"])      # 1950 missing
    assert np.isnan(f.loc[(1, 1940), "v2cacamps__d1"])      # first year


def test_drop_from_peak(cfg):
    p = _panel(cfg)
    p.loc[p.country_id == 1, "v2x_polyarchy"] = np.linspace(0.9, 0.5, 21)   # steady decline
    f = features.build_features(p, cfg).set_index(["country_id", "year"])
    # 1955: max over 1946..1955 is the 1946 value
    e = p.set_index(["country_id", "year"])["v2x_polyarchy"]
    assert f.loc[(1, 1955), "v2x_polyarchy__drop10"] == pytest.approx(e[(1, 1946)] - e[(1, 1955)])
    assert f.loc[(1, 1940), "v2x_polyarchy__drop10"] == pytest.approx(0.0)


def test_coup_coverage_rule_and_liberal_flag(cfg):
    f = features.build_features(_panel(cfg), cfg).set_index(["country_id", "year"])
    assert f.loc[(1, 1945), "coup_precoverage"] == 1 and f.loc[(1, 1945), "e_pt_coup"] == 0
    assert f.loc[(1, 1950), "coup_precoverage"] == 0
    assert f.loc[(1, 1953), "e_pt_coup__sum3"] == 1 and f.loc[(1, 1954), "e_pt_coup__sum3"] == 0
    assert f.loc[(1, 1941), "liberal"] == 1 and f.loc[(1, 1942), "liberal"] == 0


def test_no_lookahead_by_truncation(cfg):
    p = _panel(cfg)
    full = features.build_features(p, cfg).set_index(["country_id", "year"])
    for T in (1945, 1952, 1958):
        cut = features.build_features(p[p.year <= T], cfg).set_index(["country_id", "year"])
        pd.testing.assert_frame_equal(cut.loc[cut.index.get_level_values("year") == T],
                                      full.loc[full.index.get_level_values("year") == T])


def test_feature_sets(cfg):
    clean = features.feature_columns(cfg, "clean")
    full = features.feature_columns(cfg, "full")
    assert set(clean) < set(full)
    assert not any(c.startswith(("v2x_polyarchy", "v2me", "v2el")) or c == "liberal" for c in clean)
    no_lead = features.feature_columns(cfg, "full", drop_roles=["leading"])
    assert "v2cacamps" not in no_lead and "v2mecrit" in no_lead
    with pytest.raises(KeyError):
        features.feature_columns(cfg, "clean", drop_groups=["nope"])


def test_preprocessor_uses_training_rows_only(cfg):
    f = features.build_features(_panel(cfg), cfg)
    cols = features.feature_columns(cfg, "clean")
    train, test = f[f.year <= 1950].copy(), f[f.year > 1950].copy()
    test.loc[:, "v2cacamps"] = 1000.0                        # extreme test values must not move the fit
    pre = features.FeaturePreprocessor(cols, features.column_groups(cfg), features.coverage_rule_groups(cfg))
    Xtr = pre.fit_transform(train)
    assert np.allclose(Xtr.mean(axis=0), 0, atol=1e-9)
    med_before = pre.medians_["v2cacamps"]
    pre.transform(test)
    assert pre.medians_["v2cacamps"] == med_before
    # missingness indicator for a group that is missing in training (d1 in the first year)
    assert "missing__polarization" in pre.output_names
    assert not any(n == "missing__structural" for n in pre.output_names)   # coverage-rule group


def test_split_declarations(ert, cfg, realtime):
    cfg["labels"]["state_source"] = "ert_truncated"
    cfg["labels"]["stage1_declaration_split"] = True
    eps = episodes.democratic_episodes(ert, cfg)
    ev = episodes.country_events(eps, realtime)
    st = states.assign_states(ert, eps, cfg, realtime)
    with pytest.raises(ValueError):
        labels.build_labels(st, ev, cfg)
    ev2 = episodes.split_declarations(ev, st)
    d = ev2[ev2.event.str.startswith("declaration")]
    # A declared 2002 (still a democracy), C declared 1997 (democracy): both gradual
    assert set(d[d.event == "declaration_dem"][["country_id", "year"]].itertuples(index=False, name=None)) == {(1, 2002), (3, 1997)}
    assert (d.event == "declaration_abrupt").sum() == 0
    lab = labels.build_labels(st, ev2, cfg)
    assert {"declaration_dem", "declaration_abrupt"} <= set(lab.event)
    a = lab[(lab.event == "declaration") & (lab.k == 1)].set_index(["country_id", "year"]).y
    g = lab[(lab.event == "declaration_dem") & (lab.k == 1)].set_index(["country_id", "year"]).y
    assert (a == g).all()        # all declarations here are gradual
