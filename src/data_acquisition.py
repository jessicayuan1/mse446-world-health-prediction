"""
Data acquisition and integration.

Builds a single tidy country-year panel by joining three public sources:

    1. World Bank World Development Indicators (WDI)  -- via the `wbgapi` client
    2. World Happiness Report "Data for Table 2.1"    -- Gallup World Poll panel
    3. World Bank income groups + regions             -- economy metadata

The output is one row per (country, year) with hard economic/health indicators
plus subjective/social variables, indexed by ISO-3 code so everything joins on
stable keys.

Run directly to (re)build the cached panel:

    python -m src.data_acquisition
"""
from __future__ import annotations

import io
import os
import urllib.request

import pandas as pd

try:  # allow both "python -m src.data_acquisition" and notebook "import"
    from . import config as C
except ImportError:  # pragma: no cover
    import config as C


# --------------------------------------------------------------------------- #
# 1. World Happiness Report panel
# --------------------------------------------------------------------------- #
def load_whr() -> pd.DataFrame:
    """Download + concatenate the WHR train/test halves into the full panel."""
    frames = []
    for url in C.WHR_URLS:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=60).read()
        frames.append(pd.read_csv(io.BytesIO(raw)))
    whr = pd.concat(frames, ignore_index=True)

    # Keep + rename only the columns we model with.
    keep = [c for c in C.WHR_COLUMNS if c in whr.columns]
    whr = whr[keep].rename(columns=C.WHR_COLUMNS)
    whr = whr.drop_duplicates(subset=["country_name", "year"])

    # Cache the reconstructed raw panel for provenance / offline reuse.
    whr.to_csv(os.path.join(C.RAW_DIR, "whr_panel.csv"), index=False)
    print(f"[WHR]  {whr.shape[0]} rows, "
          f"{whr['country_name'].nunique()} countries, "
          f"years {int(whr.year.min())}-{int(whr.year.max())}")
    return whr


# --------------------------------------------------------------------------- #
# 2. World Bank economy metadata (ISO-3, income group, region)
# --------------------------------------------------------------------------- #
def load_economy_metadata() -> pd.DataFrame:
    """Return one row per real country (aggregates removed) with income tier."""
    import wbgapi as wb

    econ = wb.economy.DataFrame().reset_index()  # 'id' becomes a column
    econ = econ.rename(columns={"id": "iso3"})
    econ = econ[~econ["aggregate"]]              # drop regional/income aggregates
    econ = econ[["iso3", "name", "region", "incomeLevel"]].copy()

    income_map = {
        "LIC": "Low income",
        "LMC": "Lower-middle income",
        "UMC": "Upper-middle income",
        "HIC": "High income",
    }
    econ["income_group"] = econ["incomeLevel"].map(income_map)
    econ.to_csv(os.path.join(C.RAW_DIR, "economy_metadata.csv"), index=False)
    print(f"[META] {econ.shape[0]} countries with income groups")
    return econ


# --------------------------------------------------------------------------- #
# 3. WDI indicators
# --------------------------------------------------------------------------- #
def load_wdi() -> pd.DataFrame:
    """Pull all configured WDI series into a tidy (iso3, year) panel."""
    import wbgapi as wb

    series = list(C.WDI_INDICATORS)
    df = wb.data.DataFrame(
        series,
        time=range(C.YEAR_MIN, C.YEAR_MAX + 1),
        columns="series",
        skipBlanks=True,
    )
    df = df.reset_index().rename(columns={"economy": "iso3", "time": "year"})
    # 'time' comes back as 'YR2015' -> integer 2015.
    df["year"] = df["year"].str.replace("YR", "", regex=False).astype(int)
    df = df.rename(columns=C.WDI_INDICATORS)

    df.to_csv(os.path.join(C.RAW_DIR, "wdi_panel.csv"), index=False)
    print(f"[WDI]  {df.shape[0]} rows x {len(series)} indicators")
    return df


# --------------------------------------------------------------------------- #
# 4. Join everything
# --------------------------------------------------------------------------- #
def _whr_to_iso3(whr: pd.DataFrame, econ: pd.DataFrame) -> pd.DataFrame:
    """Attach an ISO-3 code to each WHR row via name matching (+ manual fixes)."""
    name_to_iso = dict(zip(econ["name"], econ["iso3"]))

    def resolve(name: str):
        fixed = C.WHR_NAME_FIXES.get(name, name)
        return name_to_iso.get(fixed)

    whr = whr.copy()
    whr["iso3"] = whr["country_name"].map(resolve)

    unmatched = sorted(whr.loc[whr["iso3"].isna(), "country_name"].unique())
    if unmatched:
        print(f"[JOIN] {len(unmatched)} WHR countries unmatched (dropped): "
              f"{unmatched}")
    return whr.dropna(subset=["iso3"])


def build_panel(save: bool = True) -> pd.DataFrame:
    """Full acquisition pipeline -> merged, provenance-tagged country-year panel."""
    whr = load_whr()
    econ = load_economy_metadata()
    wdi = load_wdi()

    whr = _whr_to_iso3(whr, econ)

    # Outer-merge WDI (hard indicators) with WHR (subjective/social) on iso3+year.
    panel = wdi.merge(
        whr.drop(columns=["country_name"]),
        on=["iso3", "year"],
        how="outer",
    )
    # Attach static country attributes (name, region, income group).
    panel = panel.merge(econ.drop(columns=["incomeLevel"]), on="iso3", how="left")

    # Keep only rows that correspond to a real, named country.
    panel = panel.dropna(subset=["name"])
    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)

    # Reorder: identifiers first, then features.
    id_cols = ["iso3", "name", "region", "income_group", "year"]
    feat_cols = [c for c in panel.columns if c not in id_cols]
    panel = panel[id_cols + feat_cols]

    if save:
        out = os.path.join(C.PROCESSED_DIR, "panel.csv")
        panel.to_csv(out, index=False)
        print(f"[DONE] merged panel: {panel.shape[0]} rows, "
              f"{panel['iso3'].nunique()} countries -> {out}")
    return panel


if __name__ == "__main__":
    build_panel()
