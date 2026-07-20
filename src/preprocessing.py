"""
Preprocessing and supervised-dataset construction.

Two responsibilities:

    1. `clean_panel`      -- light, leakage-safe cleaning: within-country
                             interpolation of feature series (each country lives
                             entirely inside one CV fold, so this is safe for the
                             grouped-by-country evaluation that we headline).
    2. `build_supervised` -- turn the country-year panel into an (X, y) forecasting
                             problem: features observed in year t predict a target
                             observed in year t + horizon.  Returns the group
                             labels (ISO-3) and target years needed for
                             leakage-free cross-validation and temporal hold-outs.

Feature groups are defined so we can cleanly test the central hypothesis:
economic inputs alone vs. economic + health + social inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from . import config as C
except ImportError:  # pragma: no cover
    import config as C


# --------------------------------------------------------------------------- #
# Feature groups (target-agnostic definitions)
# --------------------------------------------------------------------------- #
FEATURES_ECON = ["gdp_pc_ppp", "urban_pop_pct", "gini_index", "whr_log_gdp_pc"]

FEATURES_HEALTH_INPUTS = [
    "health_exp_pc",
    "dtp3_immunization",
    "measles_immunization",
    "physicians_per_1000",
    "sanitation_access",
    "water_access",
    "primary_enrollment",
]

FEATURES_DEMOG = ["under5_mortality", "infant_mortality", "fertility_rate"]

FEATURES_SOCIAL = [
    "social_support",
    "freedom",
    "generosity",
    "corruption_perception",
    "positive_affect",
    "negative_affect",
]

FEATURES_FULL = (
    FEATURES_ECON + FEATURES_HEALTH_INPUTS + FEATURES_DEMOG + FEATURES_SOCIAL
)


def feature_sets_for(target: str) -> dict[str, list[str]]:
    """
    Return the named feature sets used in the hypothesis experiment, excluding
    any variable that is the target itself or a near-mechanical duplicate of it
    (e.g. WHR "healthy life expectancy" is essentially the life-expectancy target).
    """
    drop = {target}
    if target == "life_expectancy":
        drop |= {"healthy_life_expectancy"}
    if target == "life_ladder":
        # positive/negative affect are alternative *subjective* survey items
        # collected at the same time as the ladder; keep them out of "social"
        # so the beyond-GDP story rests on structural social factors.
        drop |= {"positive_affect", "negative_affect"}

    def clean(cols):
        return [c for c in cols if c not in drop]

    econ = clean(FEATURES_ECON)
    full = clean(FEATURES_FULL)
    return {
        "Economic only": econ,
        "Full (econ + health + social)": full,
    }


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def _despike(series: pd.Series, thresh: float = 5.0, min_obs: int = 5) -> pd.Series:
    """
    Replace within-country outliers with NaN using a robust median/MAD rule.

    A point is flagged when it lies more than `thresh` scaled MADs from the
    country's median for that indicator.  This removes obvious source errors
    (e.g. a life-expectancy value of 14.7 sandwiched between values near 50)
    without touching mild, plausible year-to-year variation.
    """
    s = series.astype(float)
    if s.notna().sum() < min_obs:
        return s
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return s
    z = (s - med).abs() / (1.4826 * mad)
    return s.mask(z > thresh)


def clean_panel(panel: pd.DataFrame, despike: bool = True) -> pd.DataFrame:
    """
    1. (optional) Robustly despike every numeric indicator within each country,
       turning obvious source errors into NaN.
    2. Interpolate interior gaps of each *feature* within a country's own
       time series, then forward/backward fill the ragged ends.

    Targets are despiked but never interpolated, so a corrupted label is dropped
    (via the missing-target filter downstream) rather than fabricated.  Because
    every country lives entirely inside a single CV fold, within-country
    interpolation cannot leak information across the grouped evaluation.
    """
    panel = panel.sort_values(["iso3", "year"]).copy()
    feature_cols = [c for c in FEATURES_FULL if c in panel.columns]
    target_cols = [c for c in C.TARGETS if c in panel.columns]
    numeric_cols = feature_cols + [c for c in target_cols if c not in feature_cols]

    if despike:
        panel[numeric_cols] = panel.groupby("iso3", group_keys=False)[
            numeric_cols
        ].transform(_despike)

    def _interp(group: pd.DataFrame) -> pd.DataFrame:
        group[feature_cols] = (
            group[feature_cols]
            .interpolate(method="linear", limit_area="inside")
            .ffill()
            .bfill()
        )
        return group

    panel = (
        panel.groupby("iso3", group_keys=False)[panel.columns.tolist()]
        .apply(_interp)
    )
    return panel


# --------------------------------------------------------------------------- #
# Supervised construction (t -> t + horizon)
# --------------------------------------------------------------------------- #
def build_supervised(
    panel: pd.DataFrame,
    target: str,
    horizon: int = C.FORECAST_HORIZON,
    feature_cols: list[str] | None = None,
    add_persistence: bool = True,
) -> dict:
    """
    Build a forecasting dataset.

    For every (country, t) we take the feature vector at year t and attach the
    target observed at year t + horizon.  Rows with a missing target are dropped.

    Returns a dict with:
        X            : feature DataFrame (may still contain NaNs -> imputed in-pipeline)
        y            : target Series (level at t+horizon)
        y_now        : target level at t (persistence baseline / for gains)
        delta        : y - y_now  (the multi-year "gain")
        groups       : ISO-3 codes (for GroupKFold)
        target_year  : the year the target is measured (for temporal hold-out)
        feature_cols : the feature columns actually used
    """
    if feature_cols is None:
        feature_cols = feature_sets_for(target)["Full (econ + health + social)"]
    feature_cols = [c for c in feature_cols if c in panel.columns]

    panel = panel.sort_values(["iso3", "year"]).copy()

    # Future target: shift the target back by `horizon` years within each country.
    fut = panel[["iso3", "year", target]].copy()
    fut["year"] = fut["year"] - horizon
    fut = fut.rename(columns={target: "_target_future"})

    df = panel.merge(fut, on=["iso3", "year"], how="left")

    # Optionally expose the current target level for persistence / gains analysis.
    df["_target_now"] = df[target]

    # Keep rows with an observed future target and an observed current value.
    df = df.dropna(subset=["_target_future"])

    X = df[feature_cols].reset_index(drop=True)
    y = df["_target_future"].reset_index(drop=True)
    y_now = df["_target_now"].reset_index(drop=True)
    groups = df["iso3"].reset_index(drop=True)
    target_year = (df["year"] + horizon).reset_index(drop=True)

    out = {
        "X": X,
        "y": y,
        "y_now": y_now,
        "delta": (y - y_now),
        "groups": groups,
        "target_year": target_year,
        "feature_cols": feature_cols,
        "meta": df[["iso3", "name", "region", "income_group", "year"]]
        .reset_index(drop=True),
    }
    if add_persistence:
        out["persistence_pred"] = y_now
    return out


def coverage_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Small helper: per-column non-missing counts + share, for the report."""
    n = len(panel)
    rep = pd.DataFrame(
        {
            "non_missing": panel.notna().sum(),
            "pct_present": (panel.notna().mean() * 100).round(1),
        }
    ).sort_values("pct_present")
    rep.attrs["n_rows"] = n
    return rep
