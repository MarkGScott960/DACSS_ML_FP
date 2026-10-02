"""Tier 2 feasibility: can we backpropagate through the fly reservoir on a laptop?

Times forward+backward through a 36-month window for B country sequences with the
97k connection weights trainable (sign fixed). Uses random inputs: this checks speed
and memory only; it reads no targets and never touches the locked test set.
Run:  python scripts/tier2_feasibility.py
"""
import time, resource, sys
import numpy as np, torch, scipy.sparse as sp
from src.networks.connectome import build_connectome, weight_matrix, spectral_radius

torch.manual_seed(0)
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else torch.get_num_threads())
c = build_connectome(); A = weight_matrix(c).tocoo()
N, E = c.n, A.nnz
pn = torch.as_tensor(c.idx("PN")); F, B, T = 28, 178, 36
rho0 = spectral_radius(A.tocsr()); scale = 0.9 / rho0
pre = torch.as_tensor(A.row, dtype=torch.long); post = torch.as_tensor(A.col, dtype=torch.long)
sign = torch.as_tensor(np.sign(A.data), dtype=torch.float32)
mag0 = torch.as_tensor(np.abs(A.data) * scale, dtype=torch.float32)
theta = torch.nn.Parameter(torch.log(torch.expm1(mag0)))           # softplus^-1, so |w| starts at the fly value
W_in = torch.nn.Parameter(torch.empty(len(pn), F).uniform_(-1, 1) * 0.5)
read = torch.nn.Linear(1024 + F, 1)
rand_idx = torch.as_tensor(np.sort(np.random.default_rng(0).choice(N, 1024, replace=False)))
U = torch.randn(T, B, F); y = torch.randn(B)
leak = 0.5
print(f"N={N}  edges={E}  window={T}  batch={B}  threads={torch.get_num_threads()}  torch={torch.__version__}")

def run(mode):
    w = sign * torch.nn.functional.softplus(theta)
    idx = torch.stack([post, pre])                                  # (N x N) with row=post so W^T @ x
    Wt = torch.sparse_coo_tensor(idx, w, (N, N)).coalesce() if mode == "coo" else None
    X = torch.zeros(N, B)
    for t in range(T):
        drive = torch.zeros(N, B).index_add(0, pn, W_in @ U[t].T)
        if mode == "coo":
            rec = torch.sparse.mm(Wt, X)
        else:                                                       # gather / scatter
            rec = torch.zeros(N, B).index_add(0, post, w[:, None] * X[pre])
        X = (1 - leak) * X + leak * torch.tanh(rec + drive)
    z = torch.cat([X[rand_idx].T, U[-1]], 1)
    return ((read(z).squeeze(1) - y) ** 2).mean()

for mode in ("coo", "gather"):
    try:
        for rep in range(3):
            t0 = time.time(); loss = run(mode); t1 = time.time()
            loss.backward(); t2 = time.time()
            if rep: print(f"{mode:7s} forward {t1-t0:5.2f}s  backward {t2-t1:5.2f}s  "
                          f"grad finite={bool(torch.isfinite(theta.grad).all())}  "
                          f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB")
            theta.grad = None; W_in.grad = None
    except RuntimeError as e:
        print(mode, "failed:", str(e)[:200])
