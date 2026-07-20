"""
Central configuration for the "Development Trajectories" project.

Everything that another module might need to know about *where data lives*,
*which indicators we pull*, and *how the pipeline is parameterised* is defined
here so the rest of the code stays declarative and easy to audit.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Resolve everything relative to the project root (the parent of /src) so the
# code runs identically from a notebook, a terminal, or an IDE.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

for _d in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------- #
# World Development Indicators (WDI) pulled through the `wbgapi` client.
# The keys are the official World Bank series codes; the values are the short,
# human-readable names we use everywhere downstream.
# --------------------------------------------------------------------------- #
WDI_INDICATORS: dict[str, str] = {
    # --- Health outcomes -------------------------------------------------- #
    "SP.DYN.LE00.IN": "life_expectancy",          # Life expectancy at birth (years)
    "SH.DYN.MORT":    "under5_mortality",          # Under-5 mortality (per 1,000)
    "SP.DYN.IMRT.IN": "infant_mortality",          # Infant mortality (per 1,000)
    "SP.DYN.TFRT.IN": "fertility_rate",            # Total fertility (births/woman)
    # --- Health inputs / systems ----------------------------------------- #
    "SH.XPD.CHEX.PC.CD": "health_exp_pc",          # Current health exp. per capita (US$)
    "SH.IMM.IDPT":       "dtp3_immunization",      # DTP3 immunization (% of infants)
    "SH.IMM.MEAS":       "measles_immunization",   # Measles immunization (% of infants)
    "SH.MED.PHYS.ZS":    "physicians_per_1000",    # Physicians (per 1,000)
    "SH.STA.BASS.ZS":    "sanitation_access",      # Basic sanitation (% of pop.)
    "SH.H2O.BASW.ZS":    "water_access",           # Basic drinking water (% of pop.)
    # --- Economic / structural ------------------------------------------- #
    "NY.GDP.PCAP.PP.KD": "gdp_pc_ppp",             # GDP per capita, PPP (const 2021 $)
    "SP.URB.TOTL.IN.ZS": "urban_pop_pct",          # Urban population (% of total)
    "SE.PRM.ENRR":       "primary_enrollment",     # Primary school enrolment (% gross)
    "SI.POV.GINI":       "gini_index",             # Gini index (income inequality)
}

# Economic-only feature set used to test the "economics alone is weak" hypothesis.
ECONOMIC_ONLY_FEATURES = [
    "gdp_pc_ppp",
    "urban_pop_pct",
    "gini_index",
]

# --------------------------------------------------------------------------- #
# World Happiness Report (WHR) panel.
# The full "Data for Table 2.1" panel (2005-2020, ~166 countries) is mirrored
# on GitHub split into train/test halves; concatenated they reconstruct the
# complete official panel sourced from the Gallup World Poll.
# --------------------------------------------------------------------------- #
WHR_URLS = [
    "https://raw.githubusercontent.com/Manish3591/"
    "world-happiness-report-regression/main/train_data.csv",
    "https://raw.githubusercontent.com/Manish3591/"
    "world-happiness-report-regression/main/test_data.csv",
]

WHR_COLUMNS: dict[str, str] = {
    "Country name": "country_name",
    "year": "year",
    "Life Ladder": "life_ladder",
    "Log GDP per capita": "whr_log_gdp_pc",
    "Social support": "social_support",
    "Healthy life expectancy at birth": "healthy_life_expectancy",
    "Freedom to make life choices": "freedom",
    "Generosity": "generosity",
    "Perceptions of corruption": "corruption_perception",
    "Positive affect": "positive_affect",
    "Negative affect": "negative_affect",
}

# WHR-only social/subjective features (the "beyond GDP" additions).
WHR_SOCIAL_FEATURES = [
    "social_support",
    "freedom",
    "generosity",
    "corruption_perception",
    "positive_affect",
    "negative_affect",
]

# --------------------------------------------------------------------------- #
# Pipeline parameters
# --------------------------------------------------------------------------- #
YEAR_MIN = 2005
YEAR_MAX = 2020
FORECAST_HORIZON = 4          # predict outcome at year t+k (t -> t+k)
TEST_YEARS = (2017, 2018, 2019, 2020)  # temporal hold-out (targets land here)
RANDOM_STATE = 42

# Prediction targets.
TARGETS = {
    "life_expectancy": "Life expectancy at birth (years)",
    "life_ladder": "Cantril life-ladder score (0-10)",
}

# Country-name fixes so WHR names join cleanly onto World Bank ISO-3 codes.
WHR_NAME_FIXES = {
    "Bolivia": "Bolivia",
    "Congo (Brazzaville)": "Congo, Rep.",
    "Congo (Kinshasa)": "Congo, Dem. Rep.",
    "Czech Republic": "Czechia",
    "Czechia": "Czechia",
    "Egypt": "Egypt, Arab Rep.",
    "Somalia": "Somalia, Fed. Rep.",
    "Gambia": "Gambia, The",
    "Hong Kong S.A.R. of China": "Hong Kong SAR, China",
    "Iran": "Iran, Islamic Rep.",
    "Ivory Coast": "Cote d'Ivoire",
    "Kyrgyzstan": "Kyrgyz Republic",
    "Laos": "Lao PDR",
    "North Macedonia": "North Macedonia",
    "Palestinian Territories": "West Bank and Gaza",
    "Russia": "Russian Federation",
    "Slovakia": "Slovak Republic",
    "South Korea": "Korea, Rep.",
    "Swaziland": "Eswatini",
    "Syria": "Syrian Arab Republic",
    "Taiwan Province of China": "Taiwan, China",
    "Turkey": "Turkiye",
    "Turkiye": "Turkiye",
    "Venezuela": "Venezuela, RB",
    "Vietnam": "Viet Nam",
    "Yemen": "Yemen, Rep.",
}
