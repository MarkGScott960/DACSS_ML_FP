"""Coverage and missingness of the raw indicators (feature-only; no outcome involved)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

ERAS = [(1900, 1944, "1900-1944"), (1945, 1989, "1945-1989"), (1990, 2100, "1990-")]


def _democracy_rows(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[panel["stage"].isin([1, 2])]


def coverage_table(panel: pd.DataFrame, variables: Iterable[str]) -> pd.DataFrame:
    """Per variable: first and last year with data, non-missing share overall and in democracy rows, by era."""
    dem = _democracy_rows(panel)
    rows = []
    for v in variables:
        s = panel.loc[panel[v].notna(), "year"]
        rec = {
            "variable": v,
            "first_year": int(s.min()) if len(s) else None,
            "last_year": int(s.max()) if len(s) else None,
            "share_all": float(panel[v].notna().mean()),
            "share_democracy": float(dem[v].notna().mean()),
        }
        for lo, hi, name in ERAS:
            d = dem[(dem["year"] >= lo) & (dem["year"] <= hi)]
            rec[f"share_dem_{name}"] = float(d[v].notna().mean()) if len(d) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def missingness_by_decade(panel: pd.DataFrame, variables: Iterable[str], democracy_only: bool = True) -> pd.DataFrame:
    """Share MISSING per variable x decade (rows: variables, columns: decades)."""
    d = _democracy_rows(panel) if democracy_only else panel
    decade = (d["year"] // 10) * 10
    miss = d[list(variables)].isna()
    return miss.groupby(decade).mean().T.rename_axis(index="variable", columns="decade")


def era_proxy_flags(
    panel: pd.DataFrame,
    variables: Iterable[str],
    corr_threshold: float = 0.3,
    min_share: float = 0.02,
    tail_years: int = 6,
    tail_missing: float = 0.95,
) -> pd.DataFrame:
    """Does a variable's missingness indicator mostly encode *when* a row is?

    A missingness flag that tracks time (e.g. GDP missing only after 2019) would let a
    model learn "recent years" through the flag, and recent years have the highest base
    rate. Computed on democracy rows only. Flagged (``era_proxy``) when either

    * ``corr``: |corr(missing, year)| >= corr_threshold and the variable is missing in at least
      ``min_share`` and at most 1 - ``min_share`` of rows; or
    * ``ends_early``: missing in >= ``tail_missing`` of the last ``tail_years`` years while observed
      somewhere earlier. A series that stops before the data end is a time proxy even when it is
      sparse overall, so the correlation test misses it (v2xpas_*: corr -0.16, 100% missing
      2020-2025; review finding, decision D-16).

    ``reason`` names the rule(s) that fired.
    """
    d = _democracy_rows(panel)
    year = d["year"].astype(float)
    tail = d["year"] >= d["year"].max() - (tail_years - 1)
    out = []
    for v in variables:
        m = d[v].isna().astype(float)
        share = float(m.mean())
        corr = float(np.corrcoef(m, year)[0, 1]) if 0 < share < 1 else np.nan
        recent = float(d.loc[tail, v].isna().mean())
        by_corr = bool(min_share <= share <= 1 - min_share and abs(corr) >= corr_threshold)
        ends_early = bool(recent >= tail_missing and d.loc[~tail, v].notna().any())
        reasons = [r for r, hit in (("corr", by_corr), ("ends_early", ends_early)) if hit]
        out.append(
            {
                "variable": v,
                "share_missing": share,
                "corr_missing_year": corr,
                f"share_missing_last{tail_years}y": recent,
                "era_proxy": by_corr or ends_early,
                "reason": "+".join(reasons),
            }
        )
    res = pd.DataFrame(out)
    order = res["era_proxy"].astype(int) * 10 + res["corr_missing_year"].abs().fillna(0)
    return res.loc[order.sort_values(ascending=False).index]


def window_coverage(panel: pd.DataFrame, variables: Iterable[str], cfg: dict) -> pd.DataFrame:
    """Missing share in development vs locked-test democracy rows, and the last year observed.

    A variable that is observed in training but missing (or only carried forward) in the
    locked window gives the model a feature it never learned from: e.g. ``e_gdppc`` ends in
    2019, so a "years since observed" feature is ~0 in training and 1-5 in 2020-2024.
    ``locked_gap`` is True when the series ends before the last locked origin, or when the
    locked missing share exceeds the development one by ``0.10`` or more. Feature-only.
    """
    sp = cfg["splits"]
    dev_last, lk_first, lk_last = sp["dev_last_origin"], sp["locked"]["first_origin"], sp["locked"]["last_origin"]
    d = _democracy_rows(panel)
    dev = d[d["year"] <= dev_last]
    lk = d[(d["year"] >= lk_first) & (d["year"] <= lk_last)]
    rows = []
    for v in variables:
        obs_years = d.loc[d[v].notna(), "year"]
        last = int(obs_years.max()) if len(obs_years) else None
        m_dev, m_lk = float(dev[v].isna().mean()), float(lk[v].isna().mean())
        rows.append(
            {
                "variable": v,
                "last_year_democracy_rows": last,
                "missing_dev": m_dev,
                "missing_locked": m_lk,
                "locked_minus_dev": m_lk - m_dev,
                "locked_gap": bool(last is None or last < lk_last or m_lk - m_dev >= 0.10),
            }
        )
    res = pd.DataFrame(rows)
    return res.sort_values(["locked_gap", "locked_minus_dev"], ascending=False).reset_index(drop=True)
