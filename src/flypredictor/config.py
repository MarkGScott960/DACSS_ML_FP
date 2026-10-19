"""Load config/project.yaml and config/features.yaml into one dictionary.

Usage::

    from flypredictor import load_config
    cfg = load_config()
    cfg["labels"]["horizons_stage1"]          # [1, 2, 3, 5]
    feature_vars(cfg, "clean")                 # every raw indicator in the clean set
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import find_root

VALID_SETS = {"clean", "full", "party"}


def load_config(root: Path | str | None = None) -> dict[str, Any]:
    """Return project.yaml with features.yaml under the key ``features``."""
    base = find_root(Path(root)) if root is not None else find_root()
    with open(base / "config" / "project.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    with open(base / "config" / "features.yaml", encoding="utf-8") as fh:
        cfg["features"] = yaml.safe_load(fh)
    validate(cfg)
    return cfg


def validate(cfg: dict[str, Any]) -> None:
    """Fail early on settings that would silently change the analysis."""
    lab = cfg["labels"]
    if lab["stage1_event"] not in {"confirmation", "onset"}:
        raise ValueError(f"labels.stage1_event must be 'confirmation' or 'onset', got {lab['stage1_event']!r}")
    if lab["onset_unconfirmed_rows"] not in {"zero", "drop"}:
        raise ValueError("labels.onset_unconfirmed_rows must be 'zero' or 'drop'")
    if lab["state_source"] not in {"ert_truncated", "rule", "hindsight"}:
        raise ValueError("labels.state_source must be 'ert_truncated', 'rule' or 'hindsight'")
    for key in ("horizons_stage1", "horizons_stage2", "horizons_stage1_exploratory"):
        if any((not isinstance(k, int)) or k < 1 for k in lab[key]):
            raise ValueError(f"labels.{key} must be positive integers")
    sp = cfg["splits"]
    if not sp["dev_last_origin"] < sp["locked"]["first_origin"] <= sp["locked"]["last_origin"]:
        raise ValueError("splits: dev_last_origin < locked.first_origin <= locked.last_origin is required")
    tp = cfg.get("training_population")
    if tp and tp.get("enabled"):
        groups = {"democracy", "electoral_autocracy", "closed_autocracy"}
        for name, members in tp["candidates"].items():
            if "democracy" not in members or not set(members) <= groups:
                raise ValueError(f"training_population.candidates.{name} must contain 'democracy' and only {sorted(groups)}")
        if tp["breakdown_population"] not in tp["candidates"]:
            raise ValueError("training_population.breakdown_population must name one of the candidates")
        if lab["state_source"] != "ert_truncated":
            raise ValueError("training_population needs labels.state_source = ert_truncated (real-time status for all regimes)")
    for name, grp in cfg["features"]["groups"].items():
        if grp["set"] not in VALID_SETS:
            raise ValueError(f"features.groups.{name}.set must be one of {sorted(VALID_SETS)}")
        if grp["set"] == "clean" and grp.get("in_edi"):
            raise ValueError(f"features.groups.{name}: an EDI component cannot be in the clean set")


def feature_groups(cfg: dict[str, Any], feature_set: str) -> dict[str, list[str]]:
    """Groups (name -> raw variables) that belong to a feature set.

    clean -> groups with set == clean
    full  -> clean + full groups
    party -> clean + party groups (the party-augmented arm uses the clean set plus party indices)
    """
    if feature_set not in VALID_SETS:
        raise ValueError(f"feature_set must be one of {sorted(VALID_SETS)}")
    include = {"clean": {"clean"}, "full": {"clean", "full"}, "party": {"clean", "party"}}[feature_set]
    return {name: list(g["vars"]) for name, g in cfg["features"]["groups"].items() if g["set"] in include}


def feature_vars(cfg: dict[str, Any], feature_set: str) -> list[str]:
    """Flat, de-duplicated list of raw variables in a feature set (order preserved)."""
    seen: dict[str, None] = {}
    for vars_ in feature_groups(cfg, feature_set).values():
        for v in vars_:
            seen.setdefault(v, None)
    return list(seen)


def all_feature_vars(cfg: dict[str, Any]) -> list[str]:
    """Every raw variable named anywhere in features.yaml (all sets)."""
    seen: dict[str, None] = {}
    for g in cfg["features"]["groups"].values():
        for v in g["vars"]:
            seen.setdefault(v, None)
    return list(seen)
