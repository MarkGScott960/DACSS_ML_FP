"""Random search for reservoir networks (fly and controls) inside rolling-origin folds 1-5.

One *config* = (rho, input_scale, leak, state_weight[, activation]). The reservoir states
for a config are computed once and reused for every horizon, readout variant and ridge
penalty. The same configs (same seed) are used for every network, so tuning budgets are
equal.

Kinds of reservoir:
- "fly", "shuffled", "rewired", "er": the connectome and its controls (src.networks.controls)
- "bank_single": NO recurrence. 1,024 independent leaky units, each a leaky average of
  tanh(random projection of the inputs), all with the same leak rate.
- "bank_multi": NO recurrence, but each unit draws its own leak rate log-uniformly
  between `leak` and 1, so the bank holds many time scales at once. This is the
  strongest "memory without a network" baseline.

Readout variants: "random" (1,024 fixed random units), "mbon" (the 96 MBONs) and, after
`lab.add_random96(k)`, "random96_0" ... "random96_{k-1}" (96 random units, the
size-matched comparison for the MBON readout).

Logs are resumable by config index. Each log has a sidecar `<log>.fingerprint.json`
holding a hash of the reservoir inputs and of the config list; `tune` refuses to resume
a log written with different inputs or configs (stale cached results).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import loguniform, uniform
from sklearn.model_selection import ParameterSampler

from src.eval.splits import FOLDS, fold_masks
from src.features.build import FEATURES
from src.models.common import HORIZONS, load_features, make_table
from src.models.readout import LAMBDAS, ridge_path
from src.models.reservoir_data import build_inputs
from src.networks.connectome import build_connectome, spectral_radius, weight_matrix
from src.networks.controls import make_control
from src.networks.reservoir import ema_bank, scale_to_rho, simulate

SPACE = {"rho": uniform(0.3, 1.1),            # spectral radius 0.3 - 1.4
         "input_scale": loguniform(0.02, 2.0),
         "leak": uniform(0.1, 0.9),           # leak rate 0.1 - 1.0
         "state_weight": loguniform(0.02, 1.0)}  # readout penalty on states vs raw inputs


# Wider space for the history-free "memory test" (notebook 09): states must carry the
# history, so they may be weighted up to 5x the raw inputs, and inputs can drive tanh harder.
SPACE_MEMORY = {"rho": uniform(0.3, 1.1),
                "input_scale": loguniform(0.05, 5.0),
                "leak": uniform(0.05, 0.95),
                "state_weight": loguniform(0.05, 5.0)}

# Linear reservoir (notebook 09, mechanism section). input_scale is irrelevant: states are
# linear in it and the readout standardises every column, so it is fixed at 1. rho stays
# below 1 so the linear system cannot diverge over 444 months.
SPACE_LINEAR = {"rho": uniform(0.3, 0.65),    # 0.3 - 0.95
                "leak": uniform(0.05, 0.95),
                "state_weight": loguniform(0.05, 5.0)}

# No-recurrence banks. input_scale reaches lower than SPACE_MEMORY to probe the
# near-linear regime the tanh networks preferred.
SPACE_BANK = {"input_scale": loguniform(0.005, 5.0),
              "leak": loguniform(0.01, 1.0),          # bank_multi: the slowest leak
              "state_weight": loguniform(0.05, 5.0)}

BANKS = ("bank_single", "bank_multi")
N_BANK = 1024


def sample_configs(n_iter: int, seed: int = 42, space: dict | None = None,
                   fixed: dict | None = None) -> list[dict]:
    """`fixed` adds constant entries to every config (e.g. activation="linear")."""
    out = []
    for c in ParameterSampler(space or SPACE, n_iter=n_iter, random_state=seed):
        c = {k: float(v) for k, v in c.items()}
        c.update(fixed or {})
        out.append(c)
    return out


class ReservoirLab:
    def __init__(self, horizons=HORIZONS, n_random: int = 1024, seed: int = 0, features=None):
        self.feat = load_features()
        self.features = list(FEATURES if features is None else features)
        self.inp = build_inputs(self.feat, self.features)
        self.conn = build_connectome()
        self.A = weight_matrix(self.conn)
        self.pn, self.mbon = self.conn.idx("PN"), self.conn.idx("MBON")
        rng = np.random.default_rng(seed)
        self.seed = seed
        self.rand_idx = np.sort(rng.choice(self.conn.n, n_random, replace=False))
        self.vnodes = {"random": self.rand_idx, "mbon": self.mbon}   # readout variant -> node indices
        self.horizons = list(horizons)
        self.tables, self.loc, self.masks, self.y = {}, {}, {}, {}
        for h in self.horizons:
            d = make_table(self.feat, h)
            self.tables[h] = d
            self.loc[h] = self.inp.locate(d)
            m = [fold_masks(d, f) for f in FOLDS]
            self.masks[h] = ([a.to_numpy() for a, _ in m], [b.to_numpy() for _, b in m])
            self.y[h] = d["y"].to_numpy()
        self._nets = {}

    # ------------------------------------------------------------------ readouts
    @property
    def cols(self) -> np.ndarray:
        """Nodes stored for the default readouts (1,024 random + MBONs; overlaps counted once)."""
        return np.unique(np.r_[self.rand_idx, self.mbon])

    def add_random96(self, k: int = 5, size: int | None = None) -> list[str]:
        """Add k readout variants of `size` (default: number of MBONs) random units,
        the size-matched comparison for the MBON readout. Drawn from a separate RNG
        so the 1,024-unit readout is unchanged."""
        size = size or len(self.mbon)
        rng = np.random.default_rng(10_000 + self.seed)
        names = []
        for i in range(k):
            name = f"random96_{i}"
            self.vnodes[name] = np.sort(rng.choice(self.conn.n, size, replace=False))
            names.append(name)
        return names

    def fingerprint(self) -> str:
        """Hash of the reservoir inputs, rounded so float noise from a regenerated
        features.parquet does not trigger a false alarm."""
        h = hashlib.sha256()
        h.update(np.round(self.inp.U, 4).tobytes())
        h.update(np.round(self.inp.R, 4).tobytes())
        h.update(json.dumps(self.features).encode())
        for k in self.horizons:
            h.update(np.round(self.y[k], 6).tobytes())
        return h.hexdigest()[:16]

    # ------------------------------------------------------------------ networks
    def network(self, kind: str, graph_seed: int = 0):
        key = (kind, graph_seed if kind != "fly" else 0)
        if key not in self._nets:
            A = make_control_matrix(self.A, kind, key[1])
            self._nets[key] = (A, spectral_radius(A))
        return self._nets[key]

    def _setup(self, kind: str, cfg: dict, w_seed: int, graph_seed: int, variants):
        """Returns W, input nodes, W_in, leak, {variant: node indices}."""
        F = self.inp.U.shape[2]
        rng = np.random.default_rng(1000 + w_seed)
        if kind in BANKS:
            if any(v != "random" for v in variants):
                raise ValueError("banks have a single readout variant, 'random' (all 1,024 units)")
            W = sp.csr_matrix((N_BANK, N_BANK), dtype=np.float32)
            inp = np.arange(N_BANK)
            W_in = rng.uniform(-1, 1, (N_BANK, F))
            if kind == "bank_single":
                leak = cfg["leak"]
            else:   # per-unit time scales, log-uniform between the slowest leak and 1
                lo = np.log(cfg["leak"])
                leak = np.exp(rng.uniform(lo, 0.0, (N_BANK, 1)))
            return W, inp, W_in, leak, {"random": np.arange(N_BANK)}
        A, rho0 = self.network(kind, graph_seed)
        W = scale_to_rho(A, cfg["rho"], rho0)
        W_in = rng.uniform(-1, 1, (len(self.pn), F))
        return W, self.pn, W_in, cfg["leak"], {v: self.vnodes[v] for v in variants}

    def evaluate(self, kind: str, cfg: dict, w_seed: int = 0, graph_seed: int = 0,
                 variants=("random", "mbon"), horizons=None, predictions: bool = False):
        """Fold-wise validation errors for every ridge penalty.
        Returns {(variant, h): (rmse[fold, lam], mae[fold, lam], preds or None)}."""
        W, inp, W_in, leak, vn = self._setup(kind, cfg, w_seed, graph_seed, variants)
        cols = np.unique(np.concatenate(list(vn.values())))      # only simulate what is read out
        pos = pd.Series(np.arange(len(cols)), index=cols)
        S = simulate(W, inp, W_in, self.inp.U, self.inp.starts, self.inp.ends, cols,
                     leak, cfg.get("input_scale", 1.0), activation=cfg.get("activation", "tanh"))
        out = {}
        for h in (horizons or self.horizons):
            ti, si = self.loc[h]; Zs = S[ti, si]; Zin = self.inp.R[ti, si]
            trm, vam = self.masks[h]; y = self.y[h]
            for v in variants:
                st = Zs[:, pos[vn[v]].to_numpy()]
                Z = np.concatenate([st, Zin], axis=1)
                cs = np.r_[np.full(st.shape[1], cfg["state_weight"]), np.ones(Zin.shape[1])].astype(np.float32)
                P = ridge_path(Z, y, trm, vam, col_scale=cs)
                rm = np.array([np.sqrt(np.mean((y[va][:, None] - p) ** 2, axis=0)) for va, p in zip(vam, P)])
                ma = np.array([np.mean(np.abs(y[va][:, None] - p), axis=0) for va, p in zip(vam, P)])
                out[(v, h)] = (rm, ma, P if predictions else None)
        return out

    def evaluate_ema(self, leaks, horizons=None, predictions: bool = False):
        """No-reservoir baseline: ridge on [moving averages of the inputs at each leak, raw inputs].
        Same folds, rows and ridge path as `evaluate`. Returns {h: (rmse[fold, lam], mae, preds or None)}."""
        E = ema_bank(self.inp.U, self.inp.starts, self.inp.ends, leaks)
        out = {}
        for h in (horizons or self.horizons):
            ti, si = self.loc[h]
            Z = np.concatenate([E[ti, si], self.inp.R[ti, si]], axis=1)
            trm, vam = self.masks[h]; y = self.y[h]
            P = ridge_path(Z, y, trm, vam)
            rm = np.array([np.sqrt(np.mean((y[va][:, None] - p) ** 2, axis=0)) for va, p in zip(vam, P)])
            ma = np.array([np.mean(np.abs(y[va][:, None] - p), axis=0) for va, p in zip(vam, P)])
            out[h] = (rm, ma, P if predictions else None)
        return out


def make_control_matrix(A, kind, seed):
    return make_control(A, kind, seed)


def _check_fingerprint(lab: ReservoirLab, configs: list[dict], log_path: Path) -> None:
    """Refuse to resume a log written with different inputs or a different config list."""
    side = Path(str(log_path) + ".fingerprint.json")
    now = {"inputs": lab.fingerprint(),
           "configs": hashlib.sha256(json.dumps(configs, sort_keys=True).encode()).hexdigest()[:16]}
    if side.exists():
        old = json.loads(side.read_text(encoding="utf-8"))
        for k in ("inputs", "configs"):
            if old.get(k) != now[k]:
                raise RuntimeError(
                    f"{log_path.name} was written with different {k} (stored {old.get(k)}, now {now[k]}). "
                    f"Its cached results are stale. Delete {log_path.name} and {side.name} to re-run, "
                    f"or point this run at a new log file.")
    elif log_path.exists():
        print(f"note: {log_path.name} has no fingerprint yet; recording the current inputs and configs.")
    side.write_text(json.dumps(now, indent=2) + "\n", encoding="utf-8")


def tune(lab: ReservoirLab, kind: str, configs: list[dict], log_path, variants=("random", "mbon"),
         graph_seed: int = 0, w_seed: int = 0, tag: str | None = None) -> pd.DataFrame:
    """Evaluate every config for one network; append to a resumable CSV log."""
    tag = tag or kind
    log_path = Path(log_path)
    _check_fingerprint(lab, configs, log_path)
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    done = set(log.loc[log["network"] == tag, "config_id"]) if len(log) else set()
    t0 = time.time()
    for i, cfg in enumerate(configs):
        if i in done:
            continue
        res = lab.evaluate(kind, cfg, w_seed=w_seed, graph_seed=graph_seed, variants=variants)
        rows = []
        for (v, h), (rm, ma, _) in res.items():
            for k, f in enumerate(FOLDS):
                for j, lam in enumerate(LAMBDAS):
                    rows.append((tag, v, h, i, j, f.name, rm[k, j], ma[k, j], json.dumps(cfg)))
        log = pd.concat([log, pd.DataFrame(rows, columns=["network", "variant", "horizon", "config_id",
                                                          "lambda_id", "fold", "rmse", "mae", "params"])],
                        ignore_index=True)
        log.to_csv(log_path, index=False)
        print(f"[{tag}] config {i + 1}/{len(configs)} done  ({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)
    return log


def best_configs(log: pd.DataFrame) -> pd.DataFrame:
    """Per (network, variant, horizon): the (config, lambda) with the lowest mean fold RMSE."""
    g = (log.groupby(["network", "variant", "horizon", "config_id", "lambda_id"])
         .agg(rmse=("rmse", "mean"), rmse_sd=("rmse", "std"), mae=("mae", "mean"), params=("params", "first"))
         .reset_index())
    return g.loc[g.groupby(["network", "variant", "horizon"])["rmse"].idxmin()].reset_index(drop=True)
