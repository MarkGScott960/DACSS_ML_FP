"""Paired comparison of two models' predictions on the same rows."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cluster_bootstrap_rmse_diff(y, pa, pb, groups, B: int = 2000, seed: int = 0) -> dict:
    """RMSE(a) - RMSE(b) with a country-cluster bootstrap CI (resample countries with
    replacement; each country keeps all its rows). Negative = a is better.
    Returns the point estimate, 95% CI and the share of resamples where a beats b."""
    y, pa, pb = (np.asarray(v, float) for v in (y, pa, pb))
    g = pd.factorize(np.asarray(groups))[0]
    C = g.max() + 1
    n = np.bincount(g, minlength=C).astype(float)
    sa = np.bincount(g, weights=(y - pa) ** 2, minlength=C)
    sb = np.bincount(g, weights=(y - pb) ** 2, minlength=C)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(C, np.full(C, 1 / C), size=B).astype(float)      # (B, C)
    d = np.sqrt(counts @ sa / (counts @ n)) - np.sqrt(counts @ sb / (counts @ n))
    point = np.sqrt(sa.sum() / n.sum()) - np.sqrt(sb.sum() / n.sum())
    return {"diff": point, "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5),
            "p_a_better": float((d < 0).mean())}


def _weighted_ap(event_sorted: np.ndarray, thr: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Average precision for each row of W (bootstrap weights, shape (b, n)), rows already
    sorted by descending score. `thr` = last index of each block of tied scores, so ties
    are handled as sklearn's average_precision_score does (one threshold per distinct score)."""
    tp = np.cumsum(W * event_sorted, axis=1)[:, thr]
    allp = np.cumsum(W, axis=1)[:, thr]
    prec = np.divide(tp, allp, out=np.zeros_like(tp), where=allp > 0)
    rec = tp / tp[:, -1:]
    dr = np.diff(rec, axis=1, prepend=0.0)
    return (dr * prec).sum(axis=1)


def cluster_bootstrap_ap_diff(event, sa, sb, groups, B: int = 2000, seed: int = 0,
                              chunk: int = 100) -> dict:
    """Average precision AP(a) - AP(b) for ranking events (e.g. onsets among at-risk rows),
    with the same country-cluster bootstrap as `cluster_bootstrap_rmse_diff`.
    POSITIVE = a is better (higher AP). Pass only the at-risk rows."""
    e = np.asarray(event, float); sa = np.asarray(sa, float); sb = np.asarray(sb, float)
    g = pd.factorize(np.asarray(groups))[0]
    C = g.max() + 1
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(C, np.full(C, 1 / C), size=B).astype(float)      # (B, C)

    prepared = []
    for s in (sa, sb):
        o = np.argsort(-s, kind="mergesort")
        ss = s[o]
        thr = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]
        prepared.append((o, e[o], g[o], thr))

    def ap(weights_by_country):
        out = []
        for o, es, gs, thr in prepared:
            out.append(_weighted_ap(es, thr, weights_by_country[:, gs]))
        return out[0] - out[1]

    point = float(ap(np.ones((1, C)))[0])
    d = np.concatenate([ap(counts[i:i + chunk]) for i in range(0, B, chunk)])
    return {"diff": point, "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5),
            "p_a_better": float((d > 0).mean())}


def per_fold_rmse(df: pd.DataFrame, pred_col: str) -> pd.Series:
    return df.groupby("fold").apply(lambda s: float(np.sqrt(np.mean((s["y"] - s[pred_col]) ** 2))),
                                    include_groups=False)
