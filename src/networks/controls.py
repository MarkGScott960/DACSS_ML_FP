"""Control networks. Each isolates one thing about the fly wiring (Planning guide).

All controls take the fly's signed sparse matrix A (A[i, j] = pre i -> post j) and keep
node count and edge count. What each one preserves / destroys:

| kind      | keeps                                   | destroys                              |
|-----------|-----------------------------------------|---------------------------------------|
| fly       | everything                              | nothing                               |
| shuffled  | exact wiring, per-neuron signs          | which synapse is strong or weak       |
| rewired   | every neuron's in- and out-degree,      | who connects to whom                  |
|           | per-neuron signs                        |                                       |
| er        | node/edge counts, weight distribution,  | all structure incl. degrees and       |
|           | the share of inhibitory edges           | per-neuron signs (Dale's law)         |
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

KINDS = ["fly", "shuffled", "rewired", "er"]


def _parts(A: sp.spmatrix):
    c = A.tocoo()
    return c.row.astype(np.int64), c.col.astype(np.int64), c.data.astype(float)


def weight_shuffled(A, seed: int):
    r, c, w = _parts(A)
    rng = np.random.default_rng(seed)
    mag = rng.permutation(np.abs(w))
    return sp.csr_matrix((mag * np.sign(w), (r, c)), shape=A.shape)


def degree_preserving_rewire(A, seed: int, swaps_per_edge: int = 10):
    """Double-edge swaps: (a->b, c->d) becomes (a->d, c->b). Weights stay with their
    source neuron (so signs follow the presynaptic neuron) and in/out-degrees are
    unchanged. Swaps that would create a self-loop or a duplicate edge are skipped."""
    r, c, w = _parts(A)
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    E = len(r)
    keys = set((r * n + c).tolist())
    pairs = rng.integers(0, E, size=(swaps_per_edge * E, 2))
    done = 0
    for i, j in pairs.tolist():
        a, b, cc, d = r[i], c[i], r[j], c[j]
        if i == j or a == cc or b == d or a == d or cc == b:
            continue
        k1, k2 = a * n + d, cc * n + b
        if k1 in keys or k2 in keys:
            continue
        keys.discard(a * n + b); keys.discard(cc * n + d)
        keys.add(k1); keys.add(k2)
        c[i], c[j] = d, b
        done += 1
    out = sp.csr_matrix((w, (r, c)), shape=A.shape)
    assert out.nnz == E, "duplicate edges created"
    out.swap_fraction = done / len(pairs)
    return out


def erdos_renyi(A, seed: int):
    """Random directed graph with the same node and edge counts. The signed weights are
    permuted onto random node pairs, so the weight distribution and the share of
    inhibitory edges match the fly. Only neurons that send signals in the fly (nonzero
    out-degree) are allowed as presynaptic nodes."""
    r, c, w = _parts(A)
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    senders = np.unique(r)
    E = len(r)
    keys = set()
    pre = np.empty(E, np.int64); post = np.empty(E, np.int64)
    filled = 0
    while filled < E:
        m = int((E - filled) * 1.3) + 10
        a = rng.choice(senders, m); b = rng.integers(0, n, m)
        for x, y in zip(a.tolist(), b.tolist()):
            k = x * n + y
            if x == y or k in keys:
                continue
            keys.add(k); pre[filled] = x; post[filled] = y; filled += 1
            if filled == E:
                break
    return sp.csr_matrix((rng.permutation(w), (pre, post)), shape=A.shape)


def make_control(A, kind: str, seed: int = 0):
    if kind == "fly":
        return A.tocsr()
    return {"shuffled": weight_shuffled, "rewired": degree_preserving_rewire,
            "er": erdos_renyi}[kind](A, seed)
