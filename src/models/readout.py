"""Ridge readout for many alphas and all rolling-origin folds from running Gram sums.

Folds are nested (each training set contains the previous one), so the Gram matrix is
built incrementally: add the newly eligible block, solve, predict the validation window,
move on. One eigendecomposition per fold gives the whole alpha path.

Columns are centred and scaled with TRAINING-row statistics only; lambda is in
standardised-correlation units: sklearn Ridge(alpha) on standardised X equals
lambda = alpha / n_train.
"""
from __future__ import annotations

import numpy as np

LAMBDAS = 10 ** np.linspace(-4, 1.5, 12)


def ridge_path(Z: np.ndarray, y: np.ndarray, train_masks, val_masks, lambdas=LAMBDAS,
               col_scale: np.ndarray | None = None):
    """Returns a list (one entry per fold) of prediction arrays (n_val, n_lambda).

    col_scale multiplies the standardised columns, which divides their effective ridge
    penalty by col_scale**2 (used to penalise reservoir states more than raw inputs)."""
    d = Z.shape[1]
    G = np.zeros((d, d)); s1 = np.zeros(d); sxy = np.zeros(d); sy = 0.0; n = 0
    used = np.zeros(len(y), bool)
    c0 = None; y0 = None
    preds = []
    for tr, va in zip(train_masks, val_masks):
        assert not (used & ~tr).any(), "folds must be nested"
        new = tr & ~used
        if c0 is None:
            c0 = Z[new].mean(axis=0, dtype=np.float64).astype(np.float32); y0 = float(y[new].mean())
        for lo in range(0, int(new.sum()), 20000):          # chunked to bound memory
            idx = np.flatnonzero(new)[lo:lo + 20000]
            Zc = Z[idx] - c0; yc = (y[idx] - y0).astype(np.float64)
            if d <= 2500:                       # float64 Gram: collinear lags need the precision
                Zc = Zc.astype(np.float64)
            G += (Zc.T @ Zc).astype(np.float64)
            s1 += Zc.sum(axis=0, dtype=np.float64); sxy += (Zc.T @ yc).astype(np.float64)
            sy += yc.sum(); n += len(idx)
        used |= new
        mu = s1 / n; ybar = sy / n
        cov = G / n - np.outer(mu, mu)
        sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        if col_scale is not None:
            sd = sd / col_scale
        R = cov / np.outer(sd, sd)
        zxy = (sxy / n - mu * ybar) / sd
        w, V = np.linalg.eigh(R)
        w = np.clip(w, 0, None)
        B = V @ ((V.T @ zxy)[:, None] / (w[:, None] + np.asarray(lambdas)[None, :]))
        Zv = (Z[va] - c0 - mu.astype(np.float32)) / sd.astype(np.float32)
        preds.append(Zv @ B.astype(np.float32) + np.float32(y0 + ybar))
    return preds
