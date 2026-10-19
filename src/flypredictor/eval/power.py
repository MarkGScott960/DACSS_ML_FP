"""Power simulation (plan 7.2 item 6; decisions D-07b, D-08b).

Plants known effects in SYNTHETIC labels on the real panel structure (real stage-1 risk
sets, real features, real observability, real country clustering, real event rates per
period) and runs the confirmatory pipeline to see how often each decision rule passes.
No real label value is read: only the event RATES per period (already in the audits) are
used, to calibrate the synthetic event process.

Data-generating process (one simulated dataset)
-----------------------------------------------
For each stage-1 country-year t, an event occurs in year t+1 with probability

    h(t) = sigmoid(alpha_period + beta * s(t) + beta_mech * m(t))

  s(t)  "leading" signal: standardised mean of a fixed set of clean leading features
        (``SIGNAL_FEATURES``); the effect under test, beta in ``power.effect_sizes``
  m(t)  "mechanical" signal: minus the standardised 3-year EDI change (a slide predicts the
        slide); fixed at ``power.mechanical_beta`` so persistence has realistic skill
  alpha_period is solved so that the expected event rate equals the real rate, separately
        for development and locked origins (keeps the real base-rate shift).

Window labels y(t, k) = any event in t+1 .. t+k (same rule as data/labels.py), on the real
observable rows. Each event is tagged gradual with the real per-period gradual share (H1b).

Scenarios: a true null (beta = beta_mech = 0: no signal at all; checks the false-positive
rate) and beta in ``power.effect_sizes`` with beta_mech fixed.

Tests per dataset (annual refits over the locked window, as in the real design; persistence
baseline refit alongside):
  H1   clean EN vs persistence, every k in {1,2,3,5}: Brier skill CI > 0 and AP-ratio diff CI > 0
  H1b  same models, gradual events only (abrupt-only positives removed)
  H5   breakdown process (own rate, lag = breakdown buffer), clean EN vs persistence, per k
  H2a  full EN minus (full EN without leading groups): delta log loss CI > 0 at k = 3 and 5

Model: ``power.model`` = ``ridge`` (default) uses an L2 logit as a fast stand-in for the
elastic net (about 300x faster; annual refits need ~240 fits per dataset). ``stand_in_check``
compares the two on a few datasets. Hyperparameters are fixed; the real analysis tunes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..data.features import feature_columns
from ..models.baselines import PersistenceBaseline, tercile_cuts
from ..models.linear import ElasticNetLogit, RidgeLogit
from . import bootstrap as B
from . import metrics as M
from .recalibration import apply_shift, intercept_shift
from .splits import event_lag, training_cutoff

SIGNAL_FEATURES = [
    "v2cacamps",              # polarization (level)
    "v2caviol__d3",           # political violence (change)
    "v2x_jucon__d3",          # judicial constraints (change, negative = erosion)
    "v2xlg_legcon__d3",       # legislative constraints (change)
    "v2exrescon__d3",         # executive respects constitution (change)
    "v2regsupgroupssize__d3", # elite alignment (change)
]
SIGNAL_SIGNS = np.array([+1, +1, -1, -1, -1, -1], float)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _solve_alpha(lin: np.ndarray, target: float) -> float:
    lo, hi = -20.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _sigmoid(mid + lin).mean() > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def target_rates(events: pd.DataFrame, states: pd.DataFrame, cfg: dict[str, Any], event: str) -> dict[str, float]:
    """Events per stage-1 country-year (event in t+1 after a stage-1 year t), per period."""
    s1 = states.loc[states["stage"] == 1, ["country_id", "year"]].copy()
    ev = events.loc[events["event"] == event, ["country_id", "year"]].copy()
    ev["year"] = ev["year"] - 1
    s1["hit"] = s1.merge(ev.assign(h=1), on=["country_id", "year"], how="left")["h"].fillna(0).to_numpy()
    dev = s1["year"] <= cfg["splits"]["dev_last_origin"]
    return {"development": float(s1.loc[dev, "hit"].mean()), "locked": float(s1.loc[~dev, "hit"].mean())}


@dataclass
class PowerData:
    """Everything the simulation needs, without any real label value."""

    feats: pd.DataFrame          # stage-1 rows: keys + all engineered features (sorted)
    observable: dict[tuple[str, int], np.ndarray]   # (event, k) -> bool mask over feats rows (real observability)
    signal: np.ndarray           # s(t), standardised on development rows
    mech: np.ndarray             # m(t)
    countries: np.ndarray
    years: np.ndarray


def prepare(features: pd.DataFrame, labels: pd.DataFrame, cfg: dict[str, Any],
            events: tuple[str, ...] = ("declaration", "breakdown")) -> PowerData:
    """Stage-1 feature rows plus the real observability mask per (event, k). Reads no label value."""
    lab = labels.loc[(labels["stage"] == 1) & labels["event"].isin(events), ["country_id", "year", "event", "k", "observable"]]
    f = features[features["stage"] == 1].sort_values(["country_id", "year"]).reset_index(drop=True)
    obs = {}
    for (ev, k), g in lab.groupby(["event", "k"]):
        m = f[["country_id", "year"]].merge(g[["country_id", "year", "observable"]], on=["country_id", "year"], how="left")["observable"]
        obs[(str(ev), int(k))] = m.fillna(False).astype(bool).to_numpy()
    dev = f["year"] <= cfg["splits"]["dev_last_origin"]

    def z(x):
        x = pd.Series(x, dtype=float)
        x = x.fillna(x[dev].median())
        return ((x - x[dev].mean()) / x[dev].std()).to_numpy()

    s = np.mean([sign * z(f[c]) for c, sign in zip(SIGNAL_FEATURES, SIGNAL_SIGNS)], axis=0)
    s = (s - s[dev.to_numpy()].mean()) / s[dev.to_numpy()].std()
    m = -z(f["v2x_polyarchy__d3"])
    return PowerData(f, obs, s, m, f["country_id"].to_numpy(), f["year"].to_numpy())


def _window(events_next: np.ndarray, countries: np.ndarray, years: np.ndarray, k: int) -> np.ndarray:
    """y(t,k) = any simulated event in t+1..t+k; events_next[i] = event in years[i]+1 (stage-1 rows only)."""
    df = pd.DataFrame({"c": countries, "t": years, "e": events_next.astype(int)})
    ev_years = set(zip(df.loc[df["e"] == 1, "c"], df.loc[df["e"] == 1, "t"] + 1))
    out = np.zeros(len(df), bool)
    for j in range(1, k + 1):
        out |= np.fromiter(((c, t + j) in ev_years for c, t in zip(df["c"], df["t"])), bool, len(df))
    return out


def _simulate_events(pdat: PowerData, cfg, rates: dict[str, float], beta: float, beta_mech: float, rng):
    dev = pdat.years <= cfg["splits"]["dev_last_origin"]
    lin = beta * pdat.signal + beta_mech * pdat.mech
    h = np.empty(len(lin))
    for mask, key in ((dev, "development"), (~dev, "locked")):
        h[mask] = _sigmoid(_solve_alpha(lin[mask], rates[key]) + lin[mask])
    return rng.random(len(h)) < h, h


def _model(cfg, cols, kind: str, kw: dict[str, Any]):
    return RidgeLogit(cfg, cols, **kw) if kind == "ridge" else ElasticNetLogit(cfg, cols, **kw)


def _annual_predict(pdat: PowerData, cfg, y, obs, lag: int, k: int, feature_sets: dict[str, list[str]],
                    kind: str, kw: dict[str, Any], cuts, refit: str = "annual", recal_window: int = 0):
    """Annual (or frozen) refits over the locked window -> (test mask, {name: predictions}).

    recal_window > 0: intercept updating on the last ``recal_window`` years of known training labels.
    """
    lk = cfg["splits"]["locked"]
    first, last = int(lk["first_origin"]), int(lk["last_origin"])
    years, X = pdat.years, pdat.feats
    te = obs & (years >= first) & (years <= last)
    preds = {name: np.full(len(years), np.nan) for name in ["persistence", *feature_sets]}
    schedule = [(t, t) for t in range(first, last + 1)] if refit == "annual" else [(first, last)]
    for a, b in schedule:
        rows = te & (years >= a) & (years <= b)
        if not rows.any():
            continue
        tr = obs & (years <= training_cutoff(a, k, lag))
        ytr = y[tr].astype(int)
        if ytr.sum() < 2:
            raise ValueError("too few training events")
        Xtr, Xte = X[tr], X[rows]
        cut = training_cutoff(a, k, lag)
        recent = (years[tr] > cut - recal_window) if recal_window else None
        fitted = {"persistence": PersistenceBaseline(cuts, cfg["evaluation"]["persistence_smoothing"]).fit(Xtr, ytr)}
        for name, cols in feature_sets.items():
            fitted[name] = _model(cfg, cols, kind, kw).fit(Xtr, ytr)
        for name, m in fitted.items():
            p = m.predict_proba(Xte)
            if recal_window:
                p = apply_shift(p, intercept_shift(m.predict_proba(Xtr[recent]), ytr[recent]))
            preds[name][rows] = p
    return te, {n: p[te] for n, p in preds.items()}


def one_rep(pdat: PowerData, cfg: dict[str, Any], scenario: tuple[float, float], rep: int,
            rates: dict[str, dict[str, float]], gradual_share: dict[str, float], cuts: np.ndarray,
            kind: str = "ridge", kw: dict[str, Any] | None = None, refit: str = "annual",
            tasks: tuple[str, ...] = ("H1", "H5")) -> list[dict[str, Any]]:
    beta, beta_mech = scenario
    kw = kw or {}
    pw = cfg["power"]
    rng = np.random.default_rng([int(pw["seed"]), int(round(beta * 1000)), int(round(beta_mech * 1000)), rep])
    W = B.country_weights(pdat.countries, int(pw["bootstrap_reps"]), seed=int(rng.integers(1 << 31)))
    level = float(cfg["evaluation"]["ci_level"])
    sets = {"clean": feature_columns(cfg, "clean"), "full": feature_columns(cfg, "full"),
            "full_no_leading": feature_columns(cfg, "full", drop_roles=["leading"])}
    base = {"scenario": f"beta={beta:g}, mech={beta_mech:g}", "beta": beta, "beta_mech": beta_mech, "rep": rep}
    out: list[dict[str, Any]] = []
    dev = pdat.years <= cfg["splits"]["dev_last_origin"]

    for task, event in (("H1", "declaration"), ("H5", "breakdown")):
        if task not in tasks:
            continue
        ev_next, h = _simulate_events(pdat, cfg, rates[event], beta, beta_mech, rng)
        if task == "H1":
            share = np.where(dev, gradual_share["development"], gradual_share["locked"])
            gradual = ev_next & (rng.random(len(ev_next)) < share)
        lag = event_lag(cfg, event)
        for k in cfg["labels"]["horizons_stage1"]:
            y = _window(ev_next, pdat.countries, pdat.years, k)
            obs = pdat.observable[(event, k)]
            want = {"clean": sets["clean"]}
            if task == "H1" and k in (3, 5):
                want.update({"full": sets["full"], "full_no_leading": sets["full_no_leading"]})
            try:
                te, P = _annual_predict(pdat, cfg, y, obs, lag, k, want, kind, kw, cuts, refit,
                                        int(cfg["evaluation"].get("recalibration_window", 0)))
            except ValueError as e:
                out.append({**base, "task": task, "k": k, "pass": False, "note": str(e)})
                continue
            yte, Wte = y[te].astype(int), W[:, te]
            if yte.sum() == 0:
                out.append({**base, "task": task, "k": k, "pass": False, "note": "no test events"})
                continue
            res = B.primary_contrasts(yte, P["clean"], P["persistence"], Wte, level)
            ok = res["brier_skill"].excludes(0) and res["ap_ratio_diff"].excludes(0)
            row = {**base, "task": task, "k": k, "pass": ok, "positives": int(yte.sum()),
                   "brier_skill": res["brier_skill"].estimate, "ap_ratio_diff": res["ap_ratio_diff"].estimate,
                   "ap_ratio": res["ap_ratio"].estimate}
            if k == 1:   # h is the exact event probability for k = 1: how strong is this scenario?
                row["oracle_auc"] = M.roc_auc(yte, h[te])
                row["oracle_brier_skill"] = M.brier_skill(yte, h[te], P["persistence"])
            out.append(row)
            if task == "H1":
                yg = _window(gradual, pdat.countries, pdat.years, k)[te]
                keep = (yg == 1) | (yte == 0)                     # drop abrupt-only positives
                if yg[keep].sum() >= 1:
                    rb = B.primary_contrasts(yg[keep].astype(int), P["clean"][keep], P["persistence"][keep], Wte[:, keep], level)
                    okb = rb["brier_skill"].excludes(0) and rb["ap_ratio_diff"].excludes(0)
                    out.append({**base, "task": "H1b", "k": k, "pass": okb, "positives": int(yg[keep].sum()),
                                "brier_skill": rb["brier_skill"].estimate, "ap_ratio_diff": rb["ap_ratio_diff"].estimate,
                                "ap_ratio": rb["ap_ratio"].estimate})
                if k in (3, 5):
                    d = B.log_loss_diff(yte, P["full_no_leading"], P["full"], Wte, level)
                    out.append({**base, "task": "H2a", "k": k, "pass": d.excludes(0), "positives": int(yte.sum()),
                                "delta_log_loss": d.estimate})
    return out


def scenarios(cfg: dict[str, Any]) -> list[tuple[float, float]]:
    pw = cfg["power"]
    mech = float(pw.get("mechanical_beta", 0.5))
    return [(0.0, 0.0)] + [(float(b), mech) for b in pw["effect_sizes"]]


def run(pdat: PowerData, cfg: dict[str, Any], rates, gradual_share, cuts, reps: int | None = None,
        scenario_list=None, n_jobs: int = -1, kind: str | None = None, kw: dict[str, Any] | None = None,
        refit: str = "annual", checkpoint_dir=None, progress: bool = True) -> pd.DataFrame:
    """Run every scenario; one scenario at a time so progress and an ETA can be printed.

    checkpoint_dir: if given, each finished scenario is saved there as a parquet file and
    skipped on the next call (an interrupted run resumes; delete the folder to start over).
    Seeds depend only on (scenario, rep), so resumed and uninterrupted runs give the same result.
    """
    import time
    from pathlib import Path

    pw = cfg["power"]
    reps = int(reps or pw["reps"])
    scenario_list = scenario_list if scenario_list is not None else scenarios(cfg)
    kind = kind or pw.get("model", "ridge")
    if kw is None:
        kw = {"C": float(pw.get("ridge_C", 0.05))} if kind == "ridge" else {"C": float(pw["enet_C"]), "l1_ratio": float(pw["enet_l1_ratio"])}
    ck = Path(checkpoint_dir) if checkpoint_dir else None
    if ck:
        ck.mkdir(parents=True, exist_ok=True)
    parts, t_start, done_here = [], time.time(), 0
    for i, sc in enumerate(scenario_list, 1):
        f = ck / f"beta{sc[0]:g}_mech{sc[1]:g}_reps{reps}_{kind}_{refit}.parquet" if ck else None
        if f is not None and f.is_file():
            parts.append(pd.read_parquet(f))
            if progress:
                print(f"[{i}/{len(scenario_list)}] beta={sc[0]:g}, mech={sc[1]:g}: loaded from checkpoint", flush=True)
            continue
        t0 = time.time()
        jobs = [delayed(one_rep)(pdat, cfg, sc, r, rates, gradual_share, cuts, kind, kw, refit) for r in range(reps)]
        part = pd.DataFrame([x for rr in Parallel(n_jobs=n_jobs, verbose=0)(jobs) for x in rr])
        if f is not None:
            part.to_parquet(f, index=False)
        parts.append(part)
        done_here += 1
        if progress:
            took = time.time() - t0
            left = len(scenario_list) - i
            print(f"[{i}/{len(scenario_list)}] beta={sc[0]:g}, mech={sc[1]:g}: {reps} reps in {took / 60:.1f} min"
                  f" | about {left * took / 60:.0f} min left", flush=True)
    return pd.concat(parts, ignore_index=True)


def stand_in_check(pdat, cfg, rates, gradual_share, cuts, reps: int = 5, beta: float = 1.0, n_jobs: int = -1) -> pd.DataFrame:
    """Same simulated datasets scored with the ridge stand-in and the real elastic net
    (frozen fit, to keep the elastic-net cost manageable). Returns paired estimates."""
    mech = float(cfg["power"].get("mechanical_beta", 0.5))
    out = []
    for kind, kw in (("ridge", {"C": float(cfg["power"].get("ridge_C", 0.05))}),
                     ("enet", {"C": float(cfg["power"]["enet_C"]), "l1_ratio": float(cfg["power"]["enet_l1_ratio"])})):
        jobs = [delayed(one_rep)(pdat, cfg, (beta, mech), r, rates, gradual_share, cuts, kind, kw, "frozen", ("H1",))
                for r in range(reps)]
        res = pd.DataFrame([x for rr in Parallel(n_jobs=n_jobs)(jobs) for x in rr]).assign(model=kind)
        out.append(res)
    return pd.concat(out, ignore_index=True)


def summarise(sim: pd.DataFrame, target: float = 0.80) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(power table by task x k x scenario, one row per task x k with the minimum detectable effect).

    'H1 / all' is the intersection test (every k passes in the same dataset); 'H2a / 3+5' likewise.
    false_positive_rate = pass rate in the true null (no signal at all).
    leading_null_rate   = pass rate with the mechanical signal only (beta = 0): for H2a this is
                          its null (leading groups add nothing); for H1 it shows how often the
                          clean model beats persistence without any leading signal.
    mde_beta            = smallest beta > 0 with power >= target; metric_at_mde = median estimate there.
    """
    s = sim.copy()
    keys = ["scenario", "beta", "beta_mech", "rep"]
    h1 = s[s["task"] == "H1"].groupby(keys)["pass"].all().reset_index().assign(task="H1", k="all")
    h2 = s[s["task"] == "H2a"].groupby(keys)["pass"].all().reset_index().assign(task="H2a", k="3+5")
    s["k"] = s["k"].astype(str)
    allrows = pd.concat([s, h1, h2], ignore_index=True)
    metric = {"H1": "brier_skill", "H1b": "brier_skill", "H5": "brier_skill", "H2a": "delta_log_loss"}
    g = ["task", "k", "scenario", "beta", "beta_mech"]
    agg = allrows.groupby(g).agg(power=("pass", "mean"), reps=("pass", "size")).reset_index()
    for col in ("brier_skill", "ap_ratio", "ap_ratio_diff", "delta_log_loss", "positives", "oracle_auc", "oracle_brier_skill"):
        if col in allrows:
            med = allrows.groupby(g)[col].median()
            agg[f"median_{col}"] = [med.get(tuple(x), np.nan) for x in agg[g].itertuples(index=False)]
    rows = []
    for (t, k), d in agg.groupby(["task", "k"], sort=False):
        null = d[(d["beta"] == 0) & (d["beta_mech"] == 0)]
        lead0 = d[(d["beta"] == 0) & (d["beta_mech"] > 0)]
        eff = d[(d["beta"] > 0)].sort_values("beta")
        ok = eff[eff["power"] >= target]
        r = {"task": t, "k": k,
             "false_positive_rate": float(null["power"].iloc[0]) if len(null) else np.nan,
             "leading_null_rate": float(lead0["power"].iloc[0]) if len(lead0) else np.nan,
             "metric": metric[t]}
        if len(ok):
            b = ok.iloc[0]
            r.update({"mde_beta": b["beta"], "power_at_mde": b["power"], "metric_at_mde": b.get(f"median_{metric[t]}", np.nan)})
            ref = agg[(agg["task"] == ("H5" if t == "H5" else "H1")) & (agg["k"] == "1") & (agg["beta"] == b["beta"])
                      & (agg["beta_mech"] == b["beta_mech"])]
            if len(ref) and "median_oracle_auc" in ref:
                r["oracle_auc_at_mde"] = float(ref["median_oracle_auc"].iloc[0])
        else:
            r.update({"mde_beta": np.nan, "power_at_mde": float(eff["power"].max()) if len(eff) else np.nan,
                      "metric_at_mde": np.nan})
        rows.append(r)
    return agg, pd.DataFrame(rows)
