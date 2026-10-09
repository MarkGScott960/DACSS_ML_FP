"""Compare stage assignments from different state sources (plan 3.3 gold-standard check)."""

from __future__ import annotations

import pandas as pd


def compare_states(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str) -> dict[str, object]:
    """Cross-tab of stage under two sources, on democracy rows, plus the disagreeing country-years."""
    cols = ["country_id", "year", "stage"]
    m = a[cols].merge(b[cols], on=["country_id", "year"], suffixes=(f"_{name_a}", f"_{name_b}"))
    sa, sb = f"stage_{name_a}", f"stage_{name_b}"
    dem = m[(m[sa] > 0) | (m[sb] > 0)]
    diff = dem[dem[sa] != dem[sb]]
    return {
        "crosstab": pd.crosstab(dem[sa], dem[sb]),
        "agreement": float((dem[sa] == dem[sb]).mean()),
        "disagreements": diff.sort_values(["country_id", "year"]).reset_index(drop=True),
    }
