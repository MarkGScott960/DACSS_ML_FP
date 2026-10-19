"""Checks on the real ERT v16 / V-Dem v16 outputs. Skipped until notebooks 01-02 have run."""

import pandas as pd
import pytest

from flypredictor import Paths, load_config

paths = Paths.from_root()
pytestmark = pytest.mark.realdata

needs_processed = pytest.mark.skipif(
    not (paths.episodes_parquet.is_file() and paths.labels_parquet.is_file() and paths.states_parquet.is_file()),
    reason="run notebooks/0x_data/01 and 02 first",
)


@needs_processed
def test_episode_counts():
    eps = pd.read_parquet(paths.episodes_parquet)
    assert len(eps) == 101
    assert eps["outcome"].value_counts().to_dict() == {1: 71, 6: 15, 4: 13, 2: 1, 3: 1}
    assert eps["breakdown_year"].notna().sum() == 71


@needs_processed
def test_states_partition_democracies():
    st = pd.read_parquet(paths.states_parquet)
    assert set(st.loc[st["democracy"], "stage"]) <= {1, 2}
    assert (st.loc[~st["democracy"], "stage"] == 0).all()
    assert not st.duplicated(["country_id", "year"]).any()


@needs_processed
def test_labels_shape():
    cfg = load_config()
    lab = pd.read_parquet(paths.labels_parquet)
    expected = {0, 1, 2} if cfg.get("training_population", {}).get("enabled") else {1, 2}
    assert set(lab["stage"]) == expected, "labels are stale: re-run notebook 02"
    assert set(lab.loc[lab["stage"] == 2, "k"]) == set(cfg["labels"]["horizons_stage2"])
    obs = lab[lab["observable"]]
    assert (obs["year"] + obs["k"] <= cfg["years"]["last"]).all()
    assert not lab.duplicated(["country_id", "year", "stage", "k", "event"]).any()


needs_panel = pytest.mark.skipif(not paths.panel_parquet.is_file(), reason="run notebooks/0x_data/02 first")


@needs_panel
def test_features_no_lookahead_real_panel():
    from flypredictor.data import features

    cfg = load_config()
    panel = pd.read_parquet(paths.panel_parquet)
    full = features.build_features(panel, cfg)
    for T in (1975, 2004, 2015):
        cut = features.build_features(panel[panel["year"] <= T], cfg)
        pd.testing.assert_frame_equal(cut[cut["year"] == T].reset_index(drop=True),
                                      full[full["year"] == T].reset_index(drop=True))


@needs_processed
def test_risk_set_and_declaration_split():
    cfg = load_config()
    st = pd.read_parquet(paths.states_parquet)
    lab = pd.read_parquet(paths.labels_parquet)
    if cfg["democracy"].get("require_ert_democracy"):
        assert int((st["stage"] == 1).sum()) == 5031
    if cfg["labels"].get("stage1_declaration_split"):
        assert {"declaration_dem", "declaration_abrupt"} <= set(lab["event"]), "labels are stale: re-run notebook 02"
        a = lab[(lab.event == "declaration") & lab.observable].set_index(["country_id", "year", "k"]).y
        g = lab[(lab.event == "declaration_dem") & lab.observable].set_index(["country_id", "year", "k"]).y
        b = lab[(lab.event == "declaration_abrupt") & lab.observable].set_index(["country_id", "year", "k"]).y
        assert ((g | b) == a).all()                     # every declaration is gradual or abrupt


@needs_processed
def test_training_populations():
    """Decision D-24: P1/P2 rows exist, democracy rows are untouched, groups nest."""
    cfg = load_config()
    if not cfg.get("training_population", {}).get("enabled"):
        pytest.skip("training populations disabled")
    from flypredictor.data import labels as L

    st = pd.read_parquet(paths.states_parquet)
    lab = pd.read_parquet(paths.labels_parquet)
    assert "pop_group" in st, "states are stale: re-run notebook 02"
    assert int((st["pop_group"] == "democracy").sum()) == int((st["stage"] == 1).sum()) == 5031
    assert set(lab.loc[lab["stage"] == 0, "event"]) == {"declaration_any"}
    n = {p: len(L.training_rows(lab, st, cfg, p, 1)) for p in ("P1", "P2", "P3")}
    assert n["P3"] < n["P2"] < n["P1"]
    assert n["P1"] > 10_000
