"""Forecast-origin splits (plan 7.1; decision D-07b).

All splits are on the forecast ORIGIN year t of a label row (country, t, k).

Training-label cutoff
---------------------
A forecaster standing at origin t knows the label of an earlier origin s at horizon k only
once every year s+1 .. s+k has happened, and (for breakdowns) once ERT can code the event as
genuine. Training for origin t therefore uses origins

    s <= t - 1 - k - lag        (``training_cutoff``)

where lag is the event's maturity buffer (0 for declarations and RoW breakdowns, 5 for ERT
breakdowns). The extra "- 1" is the plan's conservative year (V-Dem year-t data appear in t+1).

Development folds: 5-year validation blocks over 1970-2004, each with its own cutoff at
the block start, expanding window.

Locked test (origins 2005-2024): ``annual`` refits one model per origin year with its own
cutoff; ``frozen`` fits once at the first locked origin's cutoff. Rows of the locked window
are only reachable through ``flypredictor.eval.lockbox``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pandas as pd

from ..data.labels import maturity_buffer

BREAKDOWN_EVENTS = {"breakdown"}


def event_lag(cfg: dict[str, Any], event: str) -> int:
    """Maturity buffer (years) that must pass before a label of this event type is known."""
    if cfg["splits"].get("training_label_lag", "maturity") != "maturity":
        return 0
    return maturity_buffer(cfg, 2 if event in BREAKDOWN_EVENTS else 1)


def training_cutoff(origin: int, k: int, lag: int) -> int:
    """Last training origin usable when forecasting from ``origin``."""
    return int(origin) - 1 - int(k) - int(lag)


@dataclass(frozen=True)
class Fold:
    name: str
    train_last_origin: int
    eval_first_origin: int
    eval_last_origin: int


def validation_blocks(cfg: dict[str, Any]) -> list[tuple[int, int]]:
    v = cfg["splits"]["validation"]
    starts = range(int(v["first_block_start"]), int(v["last_block_end"]) + 1, int(v["block_length"]))
    return [(s, min(s + int(v["block_length"]) - 1, int(v["last_block_end"]))) for s in starts]


def dev_folds(cfg: dict[str, Any], k: int, event: str) -> list[Fold]:
    lag = event_lag(cfg, event)
    last = int(cfg["splits"]["dev_last_origin"])
    folds = []
    for a, b in validation_blocks(cfg):
        if b > last:
            raise ValueError("validation block runs past dev_last_origin")
        folds.append(Fold(f"dev_{a}_{b}", training_cutoff(a, k, lag), a, b))
    return folds


def locked_folds(cfg: dict[str, Any], k: int, event: str, refit: str | None = None) -> list[Fold]:
    """Fit/score schedule for the locked window. Used by the lockbox, not directly."""
    lk = cfg["splits"]["locked"]
    refit = refit or lk["refit"]
    lag = event_lag(cfg, event)
    first, last = int(lk["first_origin"]), int(lk["last_origin"])
    if refit == "annual":
        return [Fold(f"locked_{t}", training_cutoff(t, k, lag), t, t) for t in range(first, last + 1)]
    if refit == "frozen":
        return [Fold(f"locked_frozen_{first}_{last}", training_cutoff(first, k, lag), first, last)]
    raise ValueError("refit must be 'annual' or 'frozen'")


def task_rows(labels: pd.DataFrame, stage: int, event: str, k: int) -> pd.DataFrame:
    """Observable label rows of one task (stage x event x horizon), sorted by country and origin."""
    r = labels[(labels["stage"] == stage) & (labels["event"] == event) & (labels["k"] == k) & labels["observable"]]
    return r.sort_values(["country_id", "year"]).reset_index(drop=True)


def fold_masks(rows: pd.DataFrame, fold: Fold, cfg: dict[str, Any], allow_locked: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (train, eval) masks over ``rows`` for one fold.

    Refuses folds that touch locked-window origins unless ``allow_locked`` (set only by the lockbox).
    """
    dev_last = int(cfg["splits"]["dev_last_origin"])
    if not allow_locked and (fold.eval_last_origin > dev_last or fold.train_last_origin > dev_last):
        raise PermissionError("this fold touches locked-window origins: use flypredictor.eval.lockbox")
    y = rows["year"].to_numpy()
    train = y <= fold.train_last_origin
    ev = (y >= fold.eval_first_origin) & (y <= fold.eval_last_origin)
    return train, ev


def iter_dev(rows: pd.DataFrame, cfg: dict[str, Any], k: int, event: str) -> Iterator[tuple[Fold, np.ndarray, np.ndarray]]:
    for f in dev_folds(cfg, k, event):
        tr, ev = fold_masks(rows, f, cfg)
        yield f, tr, ev
