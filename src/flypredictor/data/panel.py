"""The country-year panel: identifiers, raw feature indicators and stage, one row per country-year.

No transformations happen here (levels, changes and coverage rules are built in
features.py, later zip). The panel is the single table that features and models read.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from ..config import all_feature_vars
from ..paths import Paths


def panel_columns(cfg: dict[str, Any]) -> list[str]:
    f = cfg["features"]
    cols = list(f["id_vars"]) + list(f["grouping_vars"]) + all_feature_vars(cfg) + list(f["external_outcomes"])
    return list(dict.fromkeys(cols))  # de-duplicate, keep order


def build_panel(paths: Paths, cfg: dict[str, Any], states: pd.DataFrame) -> pd.DataFrame:
    """Read only the needed V-Dem columns and attach each country-year's stage.

    Raises if any configured variable is missing from V-Dem, or if the merge would
    duplicate or drop country-years.
    """
    wanted = panel_columns(cfg)
    available = set(pq.read_schema(paths.vdem_parquet).names)
    missing = [c for c in wanted if c not in available]
    if missing:
        raise KeyError(f"variables in config/features.yaml not found in V-Dem v16: {missing}")

    vdem = pd.read_parquet(paths.vdem_parquet, columns=wanted)
    vdem["country_id"] = vdem["country_id"].astype("int64")
    vdem["year"] = vdem["year"].astype("int64")
    first, last = int(cfg["years"]["first"]), int(cfg["years"]["last"])
    vdem = vdem[(vdem["year"] >= first) & (vdem["year"] <= last)]

    st = states[["country_id", "year", "democracy", "in_known_episode", "stage"]]
    panel = vdem.merge(st, on=["country_id", "year"], how="left", validate="one_to_one", indicator=True)
    # Country-years in V-Dem but not in ERT (ERT drops rows without EDI) get stage 0.
    panel["in_ert"] = panel.pop("_merge") == "both"
    panel["stage"] = panel["stage"].fillna(0).astype("int8")
    panel["democracy"] = panel["democracy"].fillna(False).astype(bool)
    panel["in_known_episode"] = panel["in_known_episode"].fillna(False).astype(bool)
    if panel.duplicated(["country_id", "year"]).any():
        raise ValueError("duplicate country-years in the panel")
    n_states_missing = len(st) - int(panel["in_ert"].sum())
    if n_states_missing:
        raise ValueError(f"{n_states_missing} ERT country-years have no V-Dem row")
    return panel.sort_values(["country_id", "year"]).reset_index(drop=True)
