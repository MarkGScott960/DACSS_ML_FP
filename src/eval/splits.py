"""Rolling-origin folds with a locked final test set.

Design (see Planning/Final Project Guide.md, "Validation Design"):
- Every split is on the TARGET month, never on the origin month.
- A training row's target month must be <= the fold's train_end, so no training
  target ever falls inside (or after) the validation window.
- Hyperparameter search happens inside folds 1-5 only.
- The 2021-2025 test window is locked: `test_rows` refuses to run unless the
  caller states, in code, that this is the single final evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: pd.Timestamp   # last target month allowed in training
    val_start: pd.Timestamp   # first target month in validation
    val_end: pd.Timestamp     # last target month in validation


def _fold(i: int, train_end_year: int, val_years: tuple[int, int]) -> Fold:
    return Fold(
        name=f"fold {i}",
        train_end=pd.Timestamp(train_end_year, 12, 1),
        val_start=pd.Timestamp(val_years[0], 1, 1),
        val_end=pd.Timestamp(val_years[1], 12, 1),
    )


FOLDS: list[Fold] = [
    _fold(1, 2005, (2006, 2008)),
    _fold(2, 2008, (2009, 2011)),
    _fold(3, 2011, (2012, 2014)),
    _fold(4, 2014, (2015, 2017)),
    _fold(5, 2017, (2018, 2020)),
]

TEST = Fold(name="TEST (locked)", train_end=pd.Timestamp(2020, 12, 1),
            val_start=pd.Timestamp(2021, 1, 1), val_end=pd.Timestamp(2025, 12, 1))


def make_supervised(panel: pd.DataFrame, horizon: int = 1,
                    value_col: str = "sb_best") -> pd.DataFrame:
    """One row per (state, origin month t) with the target `horizon` months ahead.

    `y` = log(1 + value_col) at month t + horizon, taken within the same state.
    Rows whose target month does not exist (end of a state's series, end of the
    panel) are dropped. Requires consecutive months within each state, which
    02_clean_join_describe.ipynb asserts.
    """
    df = panel.sort_values(["gwno", "month"], ignore_index=True).copy()
    g = df.groupby("gwno")
    df["target_month"] = g["month"].shift(-horizon)
    df["y"] = np.log1p(g[value_col].shift(-horizon))
    df["raw_y"] = g[value_col].shift(-horizon)
    df = df.dropna(subset=["target_month", "y"]).reset_index(drop=True)

    # Guard: the shifted month really is `horizon` calendar months later.
    gap = ((df["target_month"].dt.year - df["month"].dt.year) * 12
           + (df["target_month"].dt.month - df["month"].dt.month))
    assert (gap == horizon).all(), "months are not consecutive within a state"
    return df


def fold_masks(df: pd.DataFrame, fold: Fold) -> tuple[pd.Series, pd.Series]:
    """Boolean (train, validation) masks for a fold, on target month."""
    train = df["target_month"] <= fold.train_end
    val = (df["target_month"] >= fold.val_start) & (df["target_month"] <= fold.val_end)
    assert not (train & val).any()
    assert df.loc[val, "target_month"].min() > df.loc[train, "target_month"].max()
    return train, val


def test_rows(df: pd.DataFrame, *, final_evaluation_i_understand: bool = False) -> pd.Series:
    """Mask for the locked 2021-2025 test targets. Use once, at the very end."""
    if not final_evaluation_i_understand:
        raise PermissionError(
            "The 2021-2025 test set is locked. It is evaluated once, after every "
            "model and hyperparameter has been fixed. Pass "
            "final_evaluation_i_understand=True only for that run.")
    return (df["target_month"] >= TEST.val_start) & (df["target_month"] <= TEST.val_end)
