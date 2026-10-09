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
    assert set(lab["stage"]) == {1, 2}
    assert set(lab.loc[lab["stage"] == 2, "k"]) == set(cfg["labels"]["horizons_stage2"])
    obs = lab[lab["observable"]]
    assert (obs["year"] + obs["k"] <= cfg["years"]["last"]).all()
    assert not lab.duplicated(["country_id", "year", "stage", "k", "event"]).any()
