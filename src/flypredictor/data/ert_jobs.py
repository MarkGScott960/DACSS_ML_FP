"""Run ERT's get_eps from Python (via R) and load the results.

Two jobs, both executed by R/ert_variants.R:

* ``run_realtime``  re-runs get_eps on V-Dem truncated at every year T in
  ``ert.realtime_years`` (codingend capped at T). The row for year T then holds
  each country's episode status exactly as ERT would have coded it in T: the
  real-time state used for stage assignment (plan 3.3). About 5 seconds per year.
* ``run_thresholds`` re-runs get_eps on the full data with the stricter/looser
  parameter sets in ``ert.variants`` (plan 10.5; after the locked test).
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ..paths import Paths
from ..rbridge import run_r

REALTIME_FILE = "ert_realtime_truncated.parquet"
THRESHOLD_FILE = "ert_threshold_variants.parquet"


def _run_job(job: dict[str, Any], name: str, cfg: dict[str, Any], paths: Paths) -> None:
    jobs_dir = paths.interim / "ert_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_file = jobs_dir / f"{name}.json"
    job_file.write_text(json.dumps(job, indent=2), encoding="utf-8")
    run_r(
        paths.r_scripts / "ert_variants.R",
        [job_file],
        cwd=paths.root,
        expected_outputs=[job["output"]],
        rscript=cfg.get("rscript"),
    )


def run_realtime(cfg: dict[str, Any], paths: Paths, force: bool = False) -> None:
    out = paths.interim / REALTIME_FILE
    if out.is_file() and not force:
        print(f"[skip] {out.name} exists (force=True to rebuild)")
        return
    first, last = cfg["ert"]["realtime_years"]
    job = {
        "mode": "truncated",
        "params": cfg["ert"]["defaults"],
        "years": list(range(int(first), int(last) + 1)),
        "output": str(out),
    }
    _run_job(job, "realtime", cfg, paths)


def run_thresholds(cfg: dict[str, Any], paths: Paths, force: bool = False) -> None:
    out = paths.interim / THRESHOLD_FILE
    if out.is_file() and not force:
        print(f"[skip] {out.name} exists (force=True to rebuild)")
        return
    job = {"mode": "thresholds", "variants": cfg["ert"]["variants"], "output": str(out)}
    _run_job(job, "thresholds", cfg, paths)


def load_realtime(paths: Paths) -> pd.DataFrame:
    """Real-time status per country-year (year = truncation year T).

    Columns: country_id, year, rt_in_dem_episode (in a democratic episode as coded at T),
    rt_post_breakdown (ERT already codes a breakdown inside that episode at T),
    rt_in_any_episode (in an autocratization episode of any kind as coded at T; training
    populations P1/P2, decision D-24), rt_episode_id, rt_start_year.
    """
    raw = pd.read_parquet(paths.interim / REALTIME_FILE)
    out = pd.DataFrame(
        {
            "country_id": raw["country_id"].astype("int64"),
            "year": raw["trunc_year"].astype("int64"),
            "rt_in_dem_episode": (raw["aut_ep"].fillna(0) == 1) & (raw["aut_ep_prch"].fillna(0) == 1),
            "rt_post_breakdown": raw["aut_ep_pbr"].fillna(0) == 1,
            "rt_in_any_episode": raw["aut_ep"].fillna(0) == 1,
            "rt_episode_id": raw["aut_ep_id"],
            "rt_start_year": raw["aut_ep_start_year"],
        }
    )
    if out.duplicated(["country_id", "year"]).any():
        raise ValueError("duplicate country-years in the real-time ERT file")
    return out.sort_values(["country_id", "year"]).reset_index(drop=True)
