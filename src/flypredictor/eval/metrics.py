"""Forecast metrics (plan section 8). All accept optional row weights so the
country-clustered bootstrap can reuse them (``bootstrap.py``).

Primary:    PR-AUC / base rate  (``ap_ratio``),  Brier skill vs a reference forecast.
Secondary:  ROC-AUC, log loss, precision / recall / F1 at a fixed alert budget,
            event-level hit rate (an event counts as caught if any of its k lead-year rows
            raised an alert; corrects for one event making up to k positive rows).

``average_precision`` follows scikit-learn's definition (step-wise, tied scores form one
threshold) and is vectorised over a matrix of bootstrap weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EPS = 1e-12


def _as2d(w, n):
    if w is None:
        return np.ones((1, n)), True
    w = np.asarray(w, float)
    return (w[None, :], True) if w.ndim == 1 else (w, False)


def average_precision(y, score, w=None):
    """Weighted average precision; ``w`` may be (n,) or (B, n) -> scalar or (B,)."""
    y = np.asarray(y, float)
    s = np.asarray(score, float)
    W, squeeze = _as2d(w, len(y))
    order = np.argsort(-s, kind="mergesort")
    s, y, W = s[order], y[order], W[:, order]
    last_of_tie = np.r_[np.diff(s) != 0, True]
    tp = np.cumsum(W * y, axis=1)[:, last_of_tie]
    fp = np.cumsum(W * (1 - y), axis=1)[:, last_of_tie]
    total_pos = tp[:, -1:]
    precision = tp / np.maximum(tp + fp, EPS)
    recall = tp / np.maximum(total_pos, EPS)
    d_recall = np.diff(np.hstack([np.zeros((W.shape[0], 1)), recall]), axis=1)
    ap = (d_recall * precision).sum(axis=1)
    ap = np.where(total_pos[:, 0] > 0, ap, np.nan)
    return float(ap[0]) if squeeze else ap


def base_rate(y, w=None):
    y = np.asarray(y, float)
    W, squeeze = _as2d(w, len(y))
    r = (W * y).sum(1) / np.maximum(W.sum(1), EPS)
    return float(r[0]) if squeeze else r


def ap_ratio(y, score, w=None):
    return average_precision(y, score, w) / base_rate(y, w)


def brier(y, p, w=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    W, squeeze = _as2d(w, len(y))
    b = (W * (p - y) ** 2).sum(1) / np.maximum(W.sum(1), EPS)
    return float(b[0]) if squeeze else b


def brier_skill(y, p, p_ref, w=None):
    """1 - Brier(p) / Brier(p_ref): > 0 means p beats the reference."""
    return 1.0 - brier(y, p, w) / brier(y, p_ref, w)


def log_loss(y, p, w=None, clip=1e-6):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), clip, 1 - clip)
    W, squeeze = _as2d(w, len(y))
    ll = -(W * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum(1) / np.maximum(W.sum(1), EPS)
    return float(ll[0]) if squeeze else ll


def roc_auc(y, score, w=None):
    y = np.asarray(y)
    if y.min() == y.max():
        return np.nan
    return float(roc_auc_score(y, score, sample_weight=w))


def alerts(score, budget: float = 0.10) -> np.ndarray:
    """True for the top ``budget`` share of scores (ties at the cut-off are all alerted)."""
    s = np.asarray(score, float)
    n_alert = int(np.ceil(budget * len(s)))
    if n_alert == 0:
        return np.zeros(len(s), bool)
    cut = np.sort(s)[::-1][n_alert - 1]
    return s >= cut


def alert_metrics(y, score, budget: float = 0.10) -> dict[str, float]:
    y = np.asarray(y, int)
    a = alerts(score, budget)
    tp = int((a & (y == 1)).sum())
    prec = tp / a.sum() if a.sum() else np.nan
    rec = tp / y.sum() if y.sum() else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and (prec + rec) > 0 else 0.0
    return {"alerts": int(a.sum()), "precision": prec, "recall": rec, "f1": f1}


def event_hit_rate(rows: pd.DataFrame, score, events: pd.DataFrame, k: int, budget: float = 0.10) -> dict[str, float]:
    """Share of events with an alert in at least one of their k lead years.

    rows    scored rows with columns country_id, year (origin)
    events  one row per event: country_id, year (event year)
    An event is eligible if at least one of its lead-year origins (year-k .. year-1) is scored.
    """
    r = rows[["country_id", "year"]].copy()
    r["alert"] = alerts(score, budget)
    alerted = set(map(tuple, r.loc[r["alert"], ["country_id", "year"]].to_numpy()))
    scored = set(map(tuple, r[["country_id", "year"]].to_numpy()))
    eligible = hits = 0
    for c, e in events[["country_id", "year"]].itertuples(index=False):
        leads = [(c, t) for t in range(e - k, e)]
        if not any(x in scored for x in leads):
            continue
        eligible += 1
        hits += any(x in alerted for x in leads)
    return {"events": eligible, "caught": hits, "hit_rate": hits / eligible if eligible else np.nan}


def summary(y, p, p_ref=None, budget: float = 0.10) -> dict[str, float]:
    """Point estimates of every row-level metric for one model on one scoring set."""
    out = {
        "rows": int(len(y)),
        "positives": int(np.sum(y)),
        "base_rate": base_rate(y),
        "pr_auc": average_precision(y, p),
        "ap_ratio": ap_ratio(y, p),
        "roc_auc": roc_auc(y, p),
        "brier": brier(y, p),
        "log_loss": log_loss(y, p),
    }
    if p_ref is not None:
        out["brier_skill"] = brier_skill(y, p, p_ref)
    out.update({f"alert_{k}": v for k, v in alert_metrics(y, p, budget).items()})
    return out
