"""Build the country-month conflict panel: grid, GED aggregation, V-Dem and
COW contiguity joins.

Imported by notebooks/02_clean_join_describe.ipynb. Functions only.

Key conventions (see Planning/Final Project Guide.md, "Data"):
- Universe: Gleditsch-Ward independent states (main list, no microstates).
  The list was last updated at 31 Dec 2017; that date is a censoring date,
  not an exit, and is extended to the panel end.
- GED deaths of events spanning several months are spread evenly across
  the days the event covers, then summed by month.
- Annual covariates use the y-1 rule: every month of year y gets year y-1.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PANEL_START = pd.Period("1989-01", freq="M")
PANEL_END = pd.Period("2025-12", freq="M")

GW_CENSOR_DATE = pd.Timestamp("2017-12-31")

VIOLENCE_TYPES = {1: "sb", 2: "ns", 3: "os"}  # state-based, non-state, one-sided
FATALITY_COLS = ["best", "low", "high"]


# ---------------------------------------------------------------- GW states
def load_gw(path: Path, panel_end: pd.Period = PANEL_END) -> pd.DataFrame:
    """Gleditsch-Ward state spells (one row per spell; a state can have several)."""
    gw = pd.read_csv(
        path, sep="\t", header=None, encoding="latin-1",
        names=["gwno", "abb", "name", "start", "end"],
    )
    gw["name"] = gw["name"].str.strip()
    gw["end"] = gw["end"].str.strip()
    gw["start"] = pd.to_datetime(gw["start"], format="%d:%m:%Y")
    gw["end"] = pd.to_datetime(gw["end"], format="%d:%m:%Y")
    censored = gw["end"] == GW_CENSOR_DATE
    gw["censored_2017"] = censored
    gw.loc[censored, "end"] = panel_end.end_time.normalize()
    return gw


def build_grid(gw: pd.DataFrame, start: pd.Period = PANEL_START,
               end: pd.Period = PANEL_END) -> pd.DataFrame:
    """One row per (gwno, month) for every month a state exists on at least one day."""
    rows = []
    for spell in gw.itertuples(index=False):
        s = max(spell.start.to_period("M"), start)
        e = min(spell.end.to_period("M"), end)
        if s > e:
            continue
        months = pd.period_range(s, e, freq="M")
        rows.append(pd.DataFrame({"gwno": spell.gwno, "month": months}))
    grid = pd.concat(rows, ignore_index=True)
    # Multiple spells of one state must not overlap inside the panel window.
    assert not grid.duplicated(["gwno", "month"]).any(), "overlapping GW spells"
    names = gw.sort_values("start").groupby("gwno")["name"].last()
    grid["country"] = grid["gwno"].map(names)
    return grid.sort_values(["gwno", "month"], ignore_index=True)


# ---------------------------------------------------------------------- GED
GED_COLS = ["id", "year", "type_of_violence", "country", "country_id",
            "date_start", "date_end", "date_prec", *FATALITY_COLS]


def load_ged(zip_path: Path, member: str = "GEDEvent_v26_1.csv") -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        ev = pd.read_csv(z.open(member), usecols=GED_COLS)
    ev["date_start"] = pd.to_datetime(ev["date_start"])
    ev["date_end"] = pd.to_datetime(ev["date_end"])
    ev = ev.rename(columns={"country_id": "gwno"})
    assert (ev["date_end"] >= ev["date_start"]).all(), "event ends before it starts"
    return ev


def events_outside_lifespan(ev: pd.DataFrame, gw: pd.DataFrame) -> pd.DataFrame:
    """Events whose dates are not fully inside any GW spell of their country."""
    m = ev[["id", "gwno", "date_start", "date_end"]].merge(
        gw[["gwno", "start", "end"]], on="gwno", how="left")
    m["inside"] = (m["date_start"] >= m["start"]) & (m["date_end"] <= m["end"])
    ok = m.groupby("id")["inside"].any()
    return ev[ev["id"].isin(ok[~ok].index)]


def spread_events(ev: pd.DataFrame) -> pd.DataFrame:
    """Split each event into (event, month) pieces weighted by days in that month.

    Returns one row per event-month with `share` (fraction of the event's days
    falling in that month) and fatalities multiplied by `share`. Single-month
    events get one row with share = 1.
    """
    ev = ev.copy()
    ev["m_start"] = ev["date_start"].dt.to_period("M")
    ev["m_end"] = ev["date_end"].dt.to_period("M")
    single = ev["m_start"] == ev["m_end"]

    one = ev.loc[single].assign(month=lambda d: d["m_start"], share=1.0)

    multi = ev.loc[~single]
    pieces = []
    for r in multi.itertuples(index=False):
        days = pd.date_range(r.date_start, r.date_end, freq="D")
        per_month = days.to_period("M").value_counts(sort=False)
        pieces.append(pd.DataFrame({
            "id": r.id, "month": per_month.index,
            "share": per_month.to_numpy() / len(days),
        }))
    multi = (pd.concat(pieces, ignore_index=True)
             .merge(multi.drop(columns=["m_start", "m_end"]), on="id")) if pieces else multi

    out = pd.concat([one.drop(columns=["m_start", "m_end"]), multi], ignore_index=True)
    for c in FATALITY_COLS:
        out[c] = out[c] * out["share"]
    return out


def aggregate_ged(pieces: pd.DataFrame) -> pd.DataFrame:
    """Country-month totals by violence type.

    Columns: {sb,ns,os}_{best,low,high} (deaths, may be fractional after
    spreading) and {sb,ns,os}_events (events active in the month).
    """
    g = pieces.groupby(["gwno", "month", "type_of_violence"])
    agg = g[FATALITY_COLS].sum()
    agg["events"] = g["id"].nunique()
    wide = agg.unstack("type_of_violence", fill_value=0)
    wide.columns = [f"{VIOLENCE_TYPES[t]}_{v}" for v, t in wide.columns]
    order = [f"{p}_{v}" for p in VIOLENCE_TYPES.values() for v in [*FATALITY_COLS, "events"]]
    return wide.reindex(columns=order, fill_value=0).reset_index()


# -------------------------------------------------------------------- V-Dem
VDEM_VARS = ["v2x_polyarchy", "v2x_libdem", "v2x_civlib", "v2x_clphy",
             "v2x_rule", "v2x_corr", "v2x_regime"]

# GW states whose V-Dem series is filed under a different COW code.
# Serbia is GW 340 from June 2006, but V-Dem keeps it as COW 345 throughout.
GW_TO_VDEM_COW_OVERRIDE = {340: 345}


def load_vdem(path: Path, vars_: list[str] = VDEM_VARS) -> pd.DataFrame:
    v = pd.read_csv(path, usecols=["country_name", "country_id", "year", "COWcode", *vars_])
    return v.rename(columns={"country_id": "vdem_id"})


def gw_to_vdem_map(gw: pd.DataFrame, vdem: pd.DataFrame) -> pd.DataFrame:
    """Map each GW state to one stable V-Dem country_id via COW codes.

    V-Dem keeps one country_id across code changes (Germany 260->255,
    Yemen 678->679, Czechia 315->316), so joining on vdem_id handles those.
    """
    cow_to_vdem = (vdem.dropna(subset=["COWcode"])
                   .assign(COWcode=lambda d: d["COWcode"].astype(int))
                   .groupby("COWcode")["vdem_id"].unique())
    assert cow_to_vdem.apply(len).eq(1).all(), "a COW code maps to several V-Dem ids"
    cow_to_vdem = cow_to_vdem.str[0]

    states = gw[["gwno", "name"]].drop_duplicates("gwno")
    states["cow_lookup"] = states["gwno"].replace(GW_TO_VDEM_COW_OVERRIDE)
    states["vdem_id"] = states["cow_lookup"].map(cow_to_vdem)
    return states[["gwno", "vdem_id"]]


def join_vdem_lagged(panel: pd.DataFrame, vdem: pd.DataFrame,
                     xwalk: pd.DataFrame, vars_: list[str] = VDEM_VARS) -> pd.DataFrame:
    """Attach V-Dem values from year y-1 to every month of year y."""
    out = panel.merge(xwalk, on="gwno", how="left", validate="many_to_one")
    out["vdem_year"] = out["month"].dt.year - 1
    vals = vdem[["vdem_id", "year", *vars_]].rename(columns={"year": "vdem_year"})
    n = len(out)
    out = out.merge(vals, on=["vdem_id", "vdem_year"], how="left", validate="many_to_one")
    assert len(out) == n
    out["vdem_missing"] = out[vars_[0]].isna()
    return out


# --------------------------------------------------------- COW contiguity
CONTIGUITY_LAST_YEAR = 2016

# COW -> GW code differences relevant after 1989.
COW_TO_GW = {255: 260, 679: 678}


def load_contiguity_gw(zip_path: Path, gw: pd.DataFrame,
                       panel_end_year: int = PANEL_END.year) -> pd.DataFrame:
    """Directed GW-coded contiguity dyads per year, 1989..panel end.

    Years after the last COW year reuse the last year's borders (no new
    states or border changes between 2017 and 2025). Keeps `conttype`
    (1 = land/river; 2-5 = water at increasing distance).
    """
    with zipfile.ZipFile(zip_path) as z:
        member = next(n for n in z.namelist() if n.lower().endswith("contdird.csv"))
        c = pd.read_csv(z.open(member))
    c = c.rename(columns=str.lower)
    c = c.loc[c["year"] >= PANEL_START.year, ["state1no", "state2no", "year", "conttype"]]

    def to_gw(code: pd.Series, year: pd.Series) -> pd.Series:
        g = code.replace(COW_TO_GW)
        # COW 345 is Yugoslavia, then Serbia; GW renumbers Serbia to 340 in June 2006.
        return g.mask((code == 345) & (year >= 2006), 340)

    c["gw1"] = to_gw(c["state1no"], c["year"])
    c["gw2"] = to_gw(c["state2no"], c["year"])
    # In 2006 both Yugoslavia (GW 345, to June) and Serbia (GW 340, from June)
    # exist: the 2006 rows above map to 340; add a copy coded 345.
    y06 = c[(c["year"] == 2006) & ((c["state1no"] == 345) | (c["state2no"] == 345))].copy()
    y06["gw1"] = y06["state1no"].replace(COW_TO_GW)
    y06["gw2"] = y06["state2no"].replace(COW_TO_GW)
    c = pd.concat([c, y06], ignore_index=True)

    last = c[c["year"] == CONTIGUITY_LAST_YEAR]
    fill = [last.assign(year=y) for y in range(CONTIGUITY_LAST_YEAR + 1, panel_end_year + 1)]
    c = pd.concat([c, *fill], ignore_index=True)

    gw_codes = set(gw["gwno"])
    c = c[c["gw1"].isin(gw_codes) & c["gw2"].isin(gw_codes)]
    return (c[["gw1", "gw2", "year", "conttype"]]
            .drop_duplicates()
            .sort_values(["gw1", "year", "gw2"], ignore_index=True))


def backfill_early_years(dyads: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Give states born before COW codes them the borders of their first COW year.

    Example: Macedonia is a GW state from Nov 1991 but COW codes it from 1993,
    so 1991-92 would show no neighbours (and its neighbours would not see it).
    Only the borders of the first coded year are copied back, in both directions.
    """
    first_panel = grid.groupby("gwno")["month"].min().dt.year
    first_dyad = dyads.groupby("gw1")["year"].min()
    extra = []
    for gw_code, y0 in first_dyad.items():
        start = first_panel.get(gw_code)
        if start is None or start >= y0:
            continue
        base = dyads[((dyads["gw1"] == gw_code) | (dyads["gw2"] == gw_code)) & (dyads["year"] == y0)]
        extra += [base.assign(year=y) for y in range(int(start), int(y0))]
    if not extra:
        return dyads
    out = pd.concat([dyads, *extra], ignore_index=True).drop_duplicates()
    return out.sort_values(["gw1", "year", "gw2"], ignore_index=True)


def neighbour_sum(panel: pd.DataFrame, dyads: pd.DataFrame, col: str,
                  max_conttype: int = 1) -> pd.Series:
    """Same-month sum of `col` over contiguous neighbours that exist that month.

    NOT lagged — this describes the month. Lag it in the feature step before
    any model sees it.
    """
    d = dyads[dyads["conttype"] <= max_conttype][["gw1", "gw2", "year"]]
    src = panel[["gwno", "month", col]].rename(columns={"gwno": "gw2"})
    src["year"] = src["month"].dt.year
    m = d.merge(src, on=["gw2", "year"])
    s = m.groupby(["gw1", "month"])[col].sum()
    key = pd.MultiIndex.from_frame(panel[["gwno", "month"]])
    return pd.Series(s.reindex(key, fill_value=0.0).to_numpy(), index=panel.index)
