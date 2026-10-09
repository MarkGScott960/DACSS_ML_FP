"""Episode recount and label-maturity lags (outcome-only; plan 3.5 and 7.2)."""

from __future__ import annotations

import math

import pandas as pd


def episode_recount(eps: pd.DataFrame) -> pd.DataFrame:
    """Democratic episodes by decade of onset x outcome."""
    d = eps.assign(decade=(eps["start_year"] // 10) * 10)
    tab = pd.crosstab(d["decade"], d["outcome_label"], margins=True, margins_name="total")
    return tab


def lag_distribution(eps: pd.DataFrame, col: str) -> pd.DataFrame:
    s = eps[col].dropna().astype(int)
    tab = s.value_counts().sort_index().rename("episodes").to_frame()
    tab["cumulative_share"] = tab["episodes"].cumsum() / tab["episodes"].sum()
    return tab.rename_axis(col)


def recommended_buffer(eps: pd.DataFrame, col: str, quantile: float = 0.9) -> int:
    """Smallest whole number of years covering ``quantile`` of the historical lags."""
    s = eps[col].dropna().astype(float)
    return int(math.ceil(s.quantile(quantile))) if len(s) else 0


def maturity_report(eps: pd.DataFrame, quantile: float = 0.9) -> dict[str, int]:
    """Buffers implied by the data.

    onset_label_buffer   : years from episode start to confirmation (needed only if stage 1 predicts onset)
    breakdown_buffer     : years from breakdown to ERT being able to code it as genuine
    confirmation_buffer  : 0, a confirmation/declaration is observable the year it happens
    """
    return {
        "onset_label_buffer": recommended_buffer(eps, "confirm_lag", quantile),
        "breakdown_buffer": recommended_buffer(eps[eps["outcome"] == 1], "breakdown_confirm_lag", quantile),
        "confirmation_buffer": 0,
    }
