# Predicting and Clustering National Development Trajectories

**MSE 446 project — Joey Suh, Jessica Yuan, Maham Ali, Farzad Rahman**

We build an original country–year panel (217 countries, 2005–2020) by joining the
World Bank *World Development Indicators* (WDI) with the *World Happiness Report*
(WHR / Gallup World Poll), then (1) forecast life expectancy and the subjective
life-ladder score four years ahead, and (2) cluster countries by the *shape* of
their development trajectories and compare the clusters to World Bank income
tiers.

The full write-up is **`MSE446_Project_Report.ipynb`** (and
`MSE446_Project_Report.html`).

## Quick start

```bash
# 1. install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. run the whole pipeline: download data, clean, model, cluster
python run_all.py --rebuild        # omit --rebuild to reuse data/processed/panel.csv

# 3. (optional) regenerate + execute the report notebook
python build_report.py
python -m nbconvert --to notebook --execute --inplace MSE446_Project_Report.ipynb
```

`run_all.py` writes every figure to `figures/` and every results table to
`results/` (plus `results/summary.json`). The notebook loads those cached
artifacts, so it renders quickly and deterministically.

## Repository layout

```
mse446/
├── src/
│   ├── config.py            # paths, indicator codes, feature groups, parameters
│   ├── data_acquisition.py  # pull WDI (wbgapi) + WHR, reconcile names, merge panel
│   ├── preprocessing.py     # despiking, interpolation, t->t+k targets, feature sets
│   ├── prediction.py        # models, grouped CV, hypothesis test, permutation/SHAP
│   ├── clustering.py        # trajectory features, k-selection, KMeans/Ward, stability
│   └── utils.py             # metrics, leakage-safe pipelines, plotting
├── run_all.py               # end-to-end pipeline -> figures/ + results/
├── build_report.py          # assembles MSE446_Project_Report.ipynb
├── data/  (raw/, processed/) # cached downloads and the merged panel
├── figures/                 # all generated figures
├── results/                 # all generated tables + summary.json
├── MSE446_Project_Report.ipynb / .html
├── requirements.txt
└── README.md
```

## Methodology highlights

* **Leakage control** — headline metric is 5-fold `GroupKFold` **grouped by
  country**; imputation (MICE) and scaling are fit *inside* each fold. A
  persistence baseline and a forward-in-time temporal hold-out are also reported.
* **Data cleaning** — robust within-country median/MAD despiking removes obvious
  source errors; targets are never imputed (missing-outcome rows are dropped).
* **Hypothesis test** — economics-only vs. full (economic + health + social)
  feature sets on the same rows, for both the future *level* and the multi-year
  *gain*.
* **Clustering** — countries represented by per-indicator *level / slope /
  volatility*; k chosen by silhouette + elbow; stability via bootstrap Adjusted
  Rand Index; clusters cross-tabulated against income tiers.

## Data sources

* World Bank WDI via the [`wbgapi`](https://pypi.org/project/wbgapi/) client
  (indicator codes listed in `src/config.py`).
* World Happiness Report "Data for Table 2.1" panel (Gallup World Poll),
  reconstructed from a public mirror; see `WHR_URLS` in `src/config.py`.
* World Bank income groups and regions from `wbgapi` economy metadata.

All randomness is seeded (`RANDOM_STATE = 42`).
