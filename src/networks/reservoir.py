"""Leaky echo-state reservoir run over many country sequences at once.

Update (per country sequence, from a zero state at its first month):
    x(t) = (1 - leak) * x(t-1) + leak * f( W^T x(t-1) + drive(t) )
with W = A * (rho / rho(A)), so rho is the spectral radius and is tuned, not a
property of the fly. f is tanh (default) or the identity ("linear" reservoir, the
input_scale -> 0 limit of tanh). `drive` is non-zero only on the input (PN) nodes:
    drive(t)[pn] = input_scale * W_in @ u(t),  W_in fixed random uniform(-1, 1).
`leak` may be a scalar or an (N, 1) array of per-unit leak rates (used by the
no-recurrence integrator bank). The state entering month t depends only on inputs
up to month t (information available at forecast time). Nothing here looks at targets.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

ACTIVATIONS = {"tanh": np.tanh, "linear": lambda z: z}


def scale_to_rho(A: sp.spmatrix, rho_target: float, rho0: float) -> sp.csr_matrix:
    if rho0 == 0:                      # empty graph (no-recurrence bank): nothing to scale
        return sp.csr_matrix(A.shape, dtype=np.float32)
    return (A * (rho_target / rho0)).tocsr().astype(np.float32)


def simulate(W: sp.csr_matrix, pn_idx: np.ndarray, W_in: np.ndarray, U: np.ndarray,
             starts: np.ndarray, ends: np.ndarray, readout_idx: np.ndarray,
             leak, input_scale: float, x0: np.ndarray | None = None,
             activation: str = "tanh") -> np.ndarray:
    """U: (T, S, F) inputs; sequence s is active for months starts[s] <= t <= ends[s].
    Returns states (T, S, len(readout_idx)) float32 for the chosen readout nodes
    (inactive months are zero)."""
    T, S, F = U.shape
    N = W.shape[0]
    f = ACTIVATIONS[activation]
    WT = W.T.tocsr()
    X = np.zeros((N, S), np.float32) if x0 is None else x0.astype(np.float32).copy()
    out = np.zeros((T, S, len(readout_idx)), np.float32)
    Win = (input_scale * W_in).astype(np.float32)
    leak = np.asarray(leak, np.float32)
    for t in range(T):
        begin = starts == t
        if begin.any():
            X[:, begin] = 0.0
        active = (starts <= t) & (t <= ends)
        drive = np.zeros((N, S), np.float32)
        drive[pn_idx] = Win @ U[t].T
        X = (1.0 - leak) * X + leak * f(WT @ X + drive)
        X[:, ~active] = 0.0
        out[t] = X[readout_idx].T
    if not np.isfinite(out).all():
        raise FloatingPointError("reservoir states diverged (linear reservoir with rho too large?)")
    return out


def echo_state_gap(W, pn_idx, W_in, U, starts, ends, leak, input_scale, seed=0, months=60):
    """Fading-memory check: run the same inputs from two different random initial
    states and report the largest state difference at the end of `months` months."""
    rng = np.random.default_rng(seed)
    N, S = W.shape[0], U.shape[1]
    s = np.zeros(S, int); e = np.full(S, months)
    U = U[:months]
    a = simulate(W, pn_idx, W_in, U, s - 1, e, np.arange(0, N, 7), leak, input_scale,
                 x0=rng.uniform(-1, 1, (N, S)))
    b = simulate(W, pn_idx, W_in, U, s - 1, e, np.arange(0, N, 7), leak, input_scale,
                 x0=rng.uniform(-1, 1, (N, S)))
    return np.abs(a - b).max(axis=(1, 2))      # (months,) gap over time
