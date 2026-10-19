"""Synthetic fixtures with hand-computed answers.

Years 1990-2009. Five countries:

  A (1)  democracy; episode 2000-2005 (start 2000), EDI 0.80 -> 0.78 -> 0.75 -> 0.70, so the
         range since the pre-episode year (1999) reaches 0.10 in 2002 = confirmation year.
         RoW drops to electoral autocracy in 2004 (row_regch_event = -1) = breakdown year;
         founding autocratic election 2006 -> breakdown confirmation year 2006 (lag 2). Outcome 1.
  B (2)  stable democracy, no episode.
  C (3)  democracy; episode 1995-1997, EDI 0.70 -> 0.68 -> 0.64 -> 0.60: confirmed 1997.
         Recovers, averted regression (outcome 4).
  D (4)  stable democracy with a coding gap: no rows for 1996 and 1997.
  E (5)  electoral autocracy with an autocratic-regression episode 2001-2003
         (aut_ep_prch = 0, outcome 5): must NOT be a democratic episode.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from flypredictor import load_config

YEARS = list(range(1990, 2010))


def _country(cid, code, regime, edi, years=YEARS):
    return pd.DataFrame(
        {
            "country_id": cid,
            "country_text_id": code,
            "country_name": code,
            "year": years,
            "v2x_regime": regime,
            "reg_type": [1 if r >= 2 else 0 for r in (regime if isinstance(regime, list) else [regime] * len(years))],
            "v2x_polyarchy": edi,
            "aut_ep": 0,
            "aut_ep_id": None,
            "aut_ep_start_year": np.nan,
            "aut_ep_end_year": np.nan,
            "aut_ep_prch": 0,
            "aut_ep_pbr": 0,
            "aut_ep_outcome": 0,
            "aut_ep_outcome_agg": 0,
            "aut_ep_censored": 0,
            "row_regch_event": 0,
            "aut_founding_elec": np.nan,
        }
    )


def _mark_episode(df, ep_id, start, end, prch, outcome, agg):
    in_ep = df["year"].between(start, end)
    pre = df["year"] == start - 1
    df.loc[in_ep, "aut_ep"] = 1
    df.loc[in_ep | pre, "aut_ep_id"] = ep_id
    df.loc[in_ep | pre, "aut_ep_start_year"] = start
    df.loc[in_ep | pre, "aut_ep_end_year"] = end
    df.loc[in_ep, "aut_ep_prch"] = prch
    df.loc[in_ep, "aut_ep_outcome"] = outcome
    df.loc[in_ep, "aut_ep_outcome_agg"] = agg
    return df


def make_ert() -> pd.DataFrame:
    # A
    edi_a = [0.80] * 10 + [0.78, 0.75, 0.70, 0.62, 0.45, 0.40] + [0.40] * 4
    reg_a = [2] * 14 + [1] * 6
    a = _country(1, "AAA", reg_a, edi_a)
    a = _mark_episode(a, "AAA_2000_2005", 2000, 2005, prch=1, outcome=1, agg=1)
    a.loc[a["year"] == 2004, "row_regch_event"] = -1
    a.loc[a["year"] >= 2005, "aut_ep_pbr"] = np.where(a.loc[a["year"] >= 2005, "aut_ep"] == 1, 1, 0)
    a.loc[a["year"] >= 2004, "aut_founding_elec"] = 2006
    # B
    b = _country(2, "BBB", [3] * 20, [0.85] * 20)
    # C
    edi_c = [0.70] * 5 + [0.68, 0.64, 0.60] + [0.66, 0.70] + [0.70] * 10
    c = _country(3, "CCC", [2] * 20, edi_c)
    c = _mark_episode(c, "CCC_1995_1997", 1995, 1997, prch=1, outcome=4, agg=2)
    # D (gap 1996-1997)
    yrs_d = [y for y in YEARS if y not in (1996, 1997)]
    d = _country(4, "DDD", [2] * len(yrs_d), [0.75] * len(yrs_d), years=yrs_d)
    # E (autocracy)
    edi_e = [0.40] * 11 + [0.36, 0.33, 0.29] + [0.29] * 6
    e = _country(5, "EEE", [1] * 20, edi_e)
    e = _mark_episode(e, "EEE_2001_2003", 2001, 2003, prch=0, outcome=5, agg=3)
    return pd.concat([a, b, c, d, e], ignore_index=True)


def make_realtime() -> pd.DataFrame:
    """Real-time status as ERT re-runs would give it: A known from 2002-2003, C in 1997."""
    ert = make_ert()
    rt = ert[["country_id", "year"]].copy()
    rt["rt_in_dem_episode"] = False
    rt.loc[(rt.country_id == 1) & rt.year.between(2002, 2005), "rt_in_dem_episode"] = True
    rt.loc[(rt.country_id == 3) & (rt.year == 1997), "rt_in_dem_episode"] = True
    rt["rt_post_breakdown"] = (rt.country_id == 1) & (rt.year >= 2005)
    rt["rt_episode_id"] = None
    rt["rt_start_year"] = np.nan
    return rt


@pytest.fixture(scope="session")
def base_cfg():
    return load_config()


@pytest.fixture
def cfg(base_cfg):
    c = copy.deepcopy(base_cfg)
    c["years"] = {"first": 1990, "last": 2009}
    c["labels"]["state_source"] = "rule"
    c["labels"]["stage1_event"] = "confirmation"
    c["labels"]["maturity"]["stage1"]["default"] = 0
    c["labels"]["maturity"]["stage2"]["default"] = 0
    c["labels"]["stage1_breakdown_target"] = True
    c["labels"]["stage1_row_breakdown_target"] = False   # switched on in tests/test_definitions.py
    c["labels"]["stage1_declaration_split"] = False      # switched on in tests/test_features.py
    c["democracy"]["require_ert_democracy"] = True
    c["splits"] = {
        "dev_last_origin": 1999,
        "validation": {"first_block_start": 1995, "block_length": 5, "last_block_end": 1999},
        "locked": {"first_origin": 2000, "last_origin": 2008},
        "gap_equals_horizon": True,
    }
    return c


@pytest.fixture
def ert():
    return make_ert()


@pytest.fixture
def realtime():
    return make_realtime()
