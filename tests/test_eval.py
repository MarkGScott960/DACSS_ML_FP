"""Evaluation harness: splits never leak, metrics match scikit-learn, bootstrap is paired
and country-clustered, the lockbox refuses to open without the pre-registration tag."""

import json
import subprocess

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss as sk_log_loss

from flypredictor.eval import bootstrap as B
from flypredictor.eval import lockbox, metrics as M, splits


@pytest.fixture
def real_cfg(base_cfg):
    return base_cfg


def test_training_cutoff_and_lags(real_cfg):
    assert splits.training_cutoff(2005, 1, 0) == 2003
    assert splits.event_lag(real_cfg, "declaration") == 0
    assert splits.event_lag(real_cfg, "breakdown") == real_cfg["labels"]["maturity"]["stage2"]["default"]
    assert splits.event_lag(real_cfg, "row_breakdown") == 0


def test_dev_folds_never_overlap_eval(real_cfg):
    for k in (1, 3, 5):
        for ev in ("declaration", "breakdown"):
            for f in splits.dev_folds(real_cfg, k, ev):
                # the last training label window (origin + k) ends before the first eval origin
                assert f.train_last_origin + k < f.eval_first_origin
                assert f.eval_last_origin <= real_cfg["splits"]["dev_last_origin"]
    blocks = splits.validation_blocks(real_cfg)
    assert blocks[0] == (1970, 1974) and blocks[-1] == (2000, 2004)


def test_locked_schedules(real_cfg):
    ann = splits.locked_folds(real_cfg, 2, "breakdown", "annual")
    assert len(ann) == 20 and ann[0].eval_first_origin == 2005
    assert all(f.train_last_origin == f.eval_first_origin - 1 - 2 - 5 for f in ann)
    fr = splits.locked_folds(real_cfg, 1, "declaration", "frozen")
    assert len(fr) == 1 and fr[0].train_last_origin == 2003
    rows = pd.DataFrame({"year": [2000, 2003, 2006]})
    with pytest.raises(PermissionError):
        splits.fold_masks(rows, ann[0], real_cfg)


@pytest.mark.parametrize("ties", [False, True])
def test_average_precision_matches_sklearn(ties):
    rng = np.random.default_rng(1)
    y = rng.random(400) < 0.08
    s = rng.random(400)
    if ties:
        s = np.round(s * 5) / 5          # 6 distinct scores, like the persistence baseline
    w = rng.integers(0, 3, 400).astype(float)
    assert M.average_precision(y, s) == pytest.approx(average_precision_score(y, s))
    assert M.average_precision(y, s, w) == pytest.approx(average_precision_score(y, s, sample_weight=w))
    W = np.vstack([w, np.ones(400)])
    assert M.average_precision(y, s, W) == pytest.approx([average_precision_score(y, s, sample_weight=w),
                                                          average_precision_score(y, s)])


def test_brier_logloss_skill():
    rng = np.random.default_rng(2)
    y = (rng.random(300) < 0.1).astype(int)
    p = np.clip(rng.random(300) * 0.3, 0.01, 0.99)
    assert M.brier(y, p) == pytest.approx(brier_score_loss(y, p))
    assert M.log_loss(y, p) == pytest.approx(sk_log_loss(y, p))
    assert M.brier_skill(y, p, p) == pytest.approx(0.0)


def test_alerts_and_event_hits():
    s = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0])
    a = M.alerts(s, 0.2)
    assert a.tolist() == [True, True] + [False] * 8
    rows = pd.DataFrame({"country_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3], "year": [2000, 2001, 2002] * 2 + [2000, 2001, 2002, 2003]})
    events = pd.DataFrame({"country_id": [1, 2, 4], "year": [2002, 2003, 2001]})
    h = M.event_hit_rate(rows, s, events, k=2, budget=0.2)
    # country 1 event 2002: leads 2000, 2001 alerted -> hit; country 2 event 2003: leads 2001, 2002 not alerted;
    # country 4 has no scored rows -> not eligible
    assert h == {"events": 2, "caught": 1, "hit_rate": 0.5}


def test_country_weights_cluster():
    c = np.array([1, 1, 2, 3, 3, 3])
    W = B.country_weights(c, 50, seed=0)
    assert W.shape == (50, 6)
    assert np.all(W[:, 0] == W[:, 1]) and np.all(W[:, 3] == W[:, 5])     # rows of a country move together
    assert np.allclose(W[:, [0, 2, 3]].sum(1), 3)                         # 3 countries drawn each time


def test_paired_bootstrap_detects_a_real_gain():
    rng = np.random.default_rng(3)
    n, countries = 3000, np.repeat(np.arange(100), 30)
    z = rng.normal(size=n)
    p_true = 1 / (1 + np.exp(-(-3 + 1.2 * z)))
    y = (rng.random(n) < p_true).astype(int)
    p_ref = np.full(n, y.mean())
    W = B.country_weights(countries, 400, seed=1)
    res = B.primary_contrasts(y, p_true, p_ref, W)
    assert res["brier_skill"].excludes(0) and res["ap_ratio_diff"].excludes(0)
    null = B.primary_contrasts(y, p_ref + 1e-9 * rng.random(n), p_ref, W)
    assert not null["brier_skill"].excludes(0)


def _git(tmp, *a):
    subprocess.run(["git", "-C", str(tmp), *a], check=True, capture_output=True)


def test_lockbox(tmp_path, real_cfg):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "config").mkdir(); (tmp_path / "prereg").mkdir()
    (tmp_path / "config" / "project.yaml").write_text("a: 1\n")
    (tmp_path / "prereg" / "preregistration.md").write_text("# prereg\n")
    _git(tmp_path, "add", "."); _git(tmp_path, "commit", "-qm", "x")
    with pytest.raises(lockbox.LockboxError):
        lockbox.Lockbox(tmp_path, real_cfg)                       # no tag yet
    _git(tmp_path, "tag", "prereg-v1")
    lb = lockbox.Lockbox(tmp_path, real_cfg)
    assert len(lb.folds(1, "declaration")) == 20
    lb.record("enet", "clean", "declaration", 1, "annual")
    with pytest.raises(lockbox.LockboxError):
        lb.record("enet", "clean", "declaration", 1, "annual")    # second scoring refused
    lb.record("enet", "clean", "declaration", 1, "annual", rerun_reason="bug fix in metrics, see log")
    assert len(json.loads((tmp_path / "results" / "locked" / "ledger.json").read_text())) == 2
    (tmp_path / "config" / "project.yaml").write_text("a: 2\n")     # change after the freeze
    with pytest.raises(lockbox.LockboxError):
        lockbox.Lockbox(tmp_path, real_cfg)
    (tmp_path / "prereg" / "preregistration.md").write_text("# prereg\n| 2026 | 4 | a: 2 | reason |\n")
    lockbox.Lockbox(tmp_path, real_cfg, deviation="| 2026 | 4 | a: 2 | reason |")
