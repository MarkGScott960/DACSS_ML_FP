"""Feature matrix: transformations of the raw panel indicators (plan section 4.2).

Every feature at origin *t* uses only years <= *t* (tested by truncation in
tests/test_features.py and tests/test_realdata.py).

Transformations
---------------
default (every indicator)     level, 1-year change (``__d1``), 3-year change (``__d3``).
                              A change is missing unless both years exist for the country.
outcome_trajectory (EDI)      the default three, plus ``__drop10``: the highest EDI in
                              t-9 .. t minus EDI at t ("decline since last peak", 10-year window).
regime_subtype (v2x_regime)   ``liberal`` = 1 if v2x_regime == 3 (liberal democracy), else 0.
coups (coverage rule)         ``__sum3`` = coups in t-2 .. t; before 1950 the series does not
                              exist, so the values are 0 and ``coup_precoverage`` = 1
                              (structurally missing, not imputed; decision D-14).

Fit-time preprocessing (``FeaturePreprocessor``, fit on training rows only): median
imputation, one missingness indicator per feature group (groups with a coverage rule get
none), and standardisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

KEYS = ["country_id", "year"]
COUP_COVERAGE_START = 1950


@dataclass(frozen=True)
class FeatureSpec:
    """One engineered column and where it belongs."""

    name: str
    variable: str
    group: str
    set: str  # clean | full
    role: str  # leading | lagging | context | trajectory


def _group_transform(name: str, grp: dict[str, Any], var: str) -> list[str]:
    rules = grp.get("coverage_rules", {}) or {}
    if rules.get(var) == "structurally_missing_before_1950":
        return [var, f"{var}__sum3"]
    if name == "regime_subtype":
        return ["liberal"]
    if name == "outcome_trajectory":
        return [var, f"{var}__d1", f"{var}__d3", f"{var}__drop10"]
    return [var, f"{var}__d1", f"{var}__d3"]


def feature_specs(cfg: dict[str, Any]) -> list[FeatureSpec]:
    """Every engineered column defined by config/features.yaml (clean and full groups)."""
    specs: list[FeatureSpec] = []
    for gname, grp in cfg["features"]["groups"].items():
        if grp["set"] not in {"clean", "full"}:
            continue
        for var in grp["vars"]:
            for col in _group_transform(gname, grp, var):
                specs.append(FeatureSpec(col, var, gname, grp["set"], grp["role"]))
        if any(r == "structurally_missing_before_1950" for r in (grp.get("coverage_rules") or {}).values()):
            specs.append(FeatureSpec("coup_precoverage", "coverage", gname, grp["set"], grp["role"]))
    seen: dict[str, FeatureSpec] = {}
    for s in specs:
        seen.setdefault(s.name, s)
    return list(seen.values())


def feature_columns(
    cfg: dict[str, Any], feature_set: str, drop_groups: Iterable[str] = (), drop_roles: Iterable[str] = ()
) -> list[str]:
    """Columns of a feature set (clean = clean groups; full = clean + full), minus dropped groups/roles."""
    if feature_set not in {"clean", "full"}:
        raise ValueError("feature_set must be 'clean' or 'full'")
    allowed = {"clean"} if feature_set == "clean" else {"clean", "full"}
    drop_groups, drop_roles = set(drop_groups), set(drop_roles)
    unknown = drop_groups - set(cfg["features"]["groups"])
    if unknown:
        raise KeyError(f"unknown feature groups: {sorted(unknown)}")
    return [
        s.name
        for s in feature_specs(cfg)
        if s.set in allowed and s.group not in drop_groups and s.role not in drop_roles
    ]


def column_groups(cfg: dict[str, Any]) -> dict[str, str]:
    """Engineered column -> feature group."""
    return {s.name: s.group for s in feature_specs(cfg)}


def _lag(df: pd.DataFrame, cols: list[str], lag: int) -> pd.DataFrame:
    """Values of ``cols`` at year - lag for the same country (NaN if that country-year is absent)."""
    lagged = df[KEYS + cols].copy()
    lagged["year"] = lagged["year"] + lag
    out = df[KEYS].merge(lagged, on=KEYS, how="left", validate="one_to_one")
    return out[cols]


def build_features(panel: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """One row per panel country-year: keys, stage and every engineered column.

    Uses only the panel itself, and for row t only rows t-9 .. t of the same country.
    """
    df = panel.sort_values(KEYS).reset_index(drop=True)
    specs = feature_specs(cfg)
    out = {k: df[k].to_numpy() for k in KEYS}
    if "stage" in df:
        out["stage"] = df["stage"].to_numpy()

    by_var: dict[str, list[str]] = {}
    for s in specs:
        by_var.setdefault(s.variable, []).append(s.name)

    raw_vars = [v for v in by_var if v not in {"coverage", "v2x_regime"}]
    lag1 = _lag(df, raw_vars, 1)
    lag3 = _lag(df, raw_vars, 3)
    for v in raw_vars:
        x = df[v].astype(float)
        for col in by_var[v]:
            if col == v:
                out[col] = x.to_numpy()
            elif col.endswith("__d1"):
                out[col] = (x - lag1[v].astype(float)).to_numpy()
            elif col.endswith("__d3"):
                out[col] = (x - lag3[v].astype(float)).to_numpy()
            elif col.endswith("__drop10"):
                lags = pd.concat([x] + [_lag(df, [v], j)[v].astype(float) for j in range(1, 10)], axis=1)
                out[col] = (lags.max(axis=1, skipna=True) - x).to_numpy()
            elif col.endswith("__sum3"):
                pass  # filled below (coverage rule)

    # Coups: structurally missing before 1950 -> 0 plus a pre-coverage indicator.
    coup_vars = [v for v in raw_vars if f"{v}__sum3" in by_var[v]]
    if coup_vars:
        pre = df["year"] < COUP_COVERAGE_START
        out["coup_precoverage"] = pre.astype(float).to_numpy()
        for v in coup_vars:
            x = df[v].astype(float).where(~pre, 0.0)
            l1 = _lag(df.assign(**{v: x}), [v], 1)[v]
            l2 = _lag(df.assign(**{v: x}), [v], 2)[v]
            out[v] = x.to_numpy()
            out[f"{v}__sum3"] = (x + l1.fillna(0.0) + l2.fillna(0.0)).where(x.notna()).to_numpy()

    if "liberal" in by_var.get("v2x_regime", []):
        r = df["v2x_regime"].astype(float)
        out["liberal"] = np.where(r.isna(), np.nan, (r == 3).astype(float))

    feats = pd.DataFrame(out)
    cols = [s.name for s in specs]
    missing = [c for c in cols if c not in feats]
    if missing:
        raise KeyError(f"features not built: {missing}")
    return feats[[c for c in feats.columns if c not in cols] + cols]


@dataclass
class FeaturePreprocessor:
    """Median imputation + group missingness indicators + standardisation, fit on training rows.

    ``groups`` maps each column to its feature group. Groups listed in ``no_indicator``
    (coverage-rule groups) get no missingness indicator. Implemented in numpy: it runs
    hundreds of times per simulated dataset in the power simulation.
    """

    columns: list[str]
    groups: dict[str, str]
    no_indicator: set[str] = field(default_factory=set)
    medians_: pd.Series | None = None
    indicator_groups_: list[str] | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def _array(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X[self.columns].to_numpy(dtype=float)
        return np.asarray(X, dtype=float)

    def _raw(self, A: np.ndarray) -> np.ndarray:
        miss = np.isnan(A)
        filled = np.where(miss, self._med, A)
        if self._ind_idx:
            ind = np.column_stack([miss[:, idx].any(axis=1) for idx in self._ind_idx]).astype(float)
            return np.hstack([filled, ind])
        return filled

    def fit(self, X) -> "FeaturePreprocessor":
        A = self._array(X)
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN column -> median 0
                med = np.nanmedian(A, axis=0)
        self._med = np.where(np.isnan(med), 0.0, med)
        self.medians_ = pd.Series(self._med, index=self.columns)
        groups = sorted({self.groups[c] for c in self.columns} - set(self.no_indicator))
        miss = np.isnan(A)
        self.indicator_groups_, self._ind_idx = [], []
        for g in groups:
            idx = [i for i, c in enumerate(self.columns) if self.groups[c] == g]
            if miss[:, idx].any():          # keep an indicator only if the group is ever missing in training
                self.indicator_groups_.append(g)
                self._ind_idx.append(idx)
        Z = self._raw(A)
        self.mean_ = Z.mean(axis=0)
        sd = Z.std(axis=0)
        self.scale_ = np.where(sd > 0, sd, 1.0)
        return self

    def transform(self, X) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("fit the preprocessor first")
        return (self._raw(self._array(X)) - self.mean_) / self.scale_

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def output_names(self) -> list[str]:
        return list(self.columns) + [f"missing__{g}" for g in (self.indicator_groups_ or [])]


def coverage_rule_groups(cfg: dict[str, Any]) -> set[str]:
    """Groups whose missingness is handled by a coverage rule (no missingness indicator)."""
    return {n for n, g in cfg["features"]["groups"].items() if g.get("coverage_rules")}
