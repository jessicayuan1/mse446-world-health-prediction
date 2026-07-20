"""
End-to-end pipeline: data -> preprocessing -> EDA -> prediction -> clustering.

Running this script regenerates every figure (into figures/) and every results
table (into results/) that the report notebook displays.  It is the single
reproducible entry point:

    python run_all.py            # uses cached data/processed/panel.csv if present
    python run_all.py --rebuild  # re-download WDI + WHR from source first
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning

# Keep the console readable: the MICE imputer's early-stopping notice and a few
# all-NaN slice warnings are expected and handled, not errors.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from src import config as C
from src import data_acquisition as acq
from src import preprocessing as pp
from src import prediction as pred
from src import clustering as clu
from src import utils as U


# --------------------------------------------------------------------------- #
def get_panel(rebuild: bool) -> pd.DataFrame:
    cache = os.path.join(C.PROCESSED_DIR, "panel.csv")
    if rebuild or not os.path.exists(cache):
        panel = acq.build_panel(save=True)
    else:
        panel = pd.read_csv(cache)
        print(f"[panel] loaded cache: {panel.shape[0]} rows, "
              f"{panel['iso3'].nunique()} countries")
    return panel


# --------------------------------------------------------------------------- #
# Exploratory data analysis figures
# --------------------------------------------------------------------------- #
def run_eda(panel_clean: pd.DataFrame):
    # (a) Coverage / missingness bar
    cov = pp.coverage_report(panel_clean)
    fig, ax = plt.subplots(figsize=(7, 6))
    cov["pct_present"].plot.barh(ax=ax, color="#4C72B0")
    ax.set_xlabel("% of rows present (after cleaning)")
    ax.set_title("Data coverage by variable", fontweight="bold")
    U.savefig(fig, "eda_coverage.png")

    # (b) Correlation heatmap of features + targets
    num = panel_clean[[c for c in pp.FEATURES_FULL if c in panel_clean.columns]
                      + list(C.TARGETS)]
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax, square=False,
                cbar_kws={"shrink": 0.7})
    ax.set_title("Correlation among indicators and targets", fontweight="bold")
    U.savefig(fig, "eda_correlation.png")

    # (c) Target distributions
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (tgt, lab) in zip(axes, C.TARGETS.items()):
        sns.histplot(panel_clean[tgt].dropna(), kde=True, ax=ax, color="#55A868")
        ax.set_title(lab, fontsize=10)
    fig.suptitle("Distribution of prediction targets", fontweight="bold")
    U.savefig(fig, "eda_targets.png")

    # (d) Example trajectories: life expectancy for a few countries
    fig, ax = plt.subplots(figsize=(8, 4.5))
    picks = ["USA", "CHN", "IND", "RWA", "KOR", "NGA"]
    for iso in picks:
        g = panel_clean[panel_clean.iso3 == iso]
        if len(g):
            ax.plot(g["year"], g["life_expectancy"], marker="o", ms=3,
                    label=g["name"].iloc[0])
    ax.set_ylabel("Life expectancy (years)"); ax.set_xlabel("Year")
    ax.set_title("Example national trajectories", fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    U.savefig(fig, "eda_trajectories.png")

    cov.to_csv(os.path.join(C.RESULTS_DIR, "coverage.csv"))
    print("[eda] figures written")


# --------------------------------------------------------------------------- #
def run_prediction(panel_clean: pd.DataFrame, results: dict):
    results["prediction"] = {}
    for tgt, lab in C.TARGETS.items():
        print(f"\n[prediction] target = {tgt}")
        # 1. Model comparison
        cmp_df, data, oof = pred.experiment_model_comparison(panel_clean, tgt)
        pred.plot_model_comparison(cmp_df, lab, f"pred_models_{tgt}.png")
        cmp_df.to_csv(os.path.join(C.RESULTS_DIR, f"pred_models_{tgt}.csv"))

        # best model by R2 (excluding baseline)
        best = cmp_df.drop(index="Persistence (y_t)")["R2"].idxmax()

        # pred-vs-actual for best model
        pred.plot_pred_vs_actual(data["y"], oof[best], lab,
                                 f"pred_scatter_{tgt}.png")

        # 2. Feature-set hypothesis test
        fs_df = pred.experiment_feature_sets(panel_clean, tgt, model_key=best)
        pred.plot_feature_sets(fs_df, lab, f"pred_featuresets_{tgt}.png")
        fs_df.to_csv(os.path.join(C.RESULTS_DIR, f"pred_featuresets_{tgt}.csv"))

        # 3. Importance + SHAP
        imp = pred.experiment_importance(panel_clean, tgt, model_key=best)
        pred.plot_permutation(imp["importance"], lab, f"pred_perm_{tgt}.png")
        imp["importance"].to_csv(
            os.path.join(C.RESULTS_DIR, f"pred_perm_{tgt}.csv"), index=False)
        if imp["shap_values"] is not None:
            pred.plot_shap(imp["shap_values"], imp["X_proc"], lab,
                           f"pred_shap_{tgt}.png")

        # 4. Temporal generalisation
        temp_df = pred.experiment_temporal(panel_clean, tgt)
        temp_df.to_csv(os.path.join(C.RESULTS_DIR, f"pred_temporal_{tgt}.csv"))

        # 5. Autoregressive check (adding current outcome should beat persistence)
        ar = pred.experiment_autoregressive(panel_clean, tgt, model_key=best)

        results["prediction"][tgt] = {
            "best_model": best,
            "model_comparison": cmp_df.round(4).to_dict(),
            "feature_sets": fs_df.round(4).to_dict(),
            "feature_sets_n_rows": fs_df.attrs.get("n_rows"),
            "feature_sets_n_countries": fs_df.attrs.get("n_countries"),
            "temporal": temp_df.round(4).to_dict(),
            "autoregressive": {k: round(v, 4) for k, v in ar.items()},
            "top_features": imp["importance"].head(10)["feature"].tolist(),
        }
        print(f"   autoregressive (best + current outcome): "
              f"R2={ar['R2']:.3f}, RMSE={ar['RMSE']:.3f}")
        print(f"   best model: {best}")
        print(cmp_df.round(3).to_string())
        print(fs_df.round(3).to_string())


# --------------------------------------------------------------------------- #
def run_clustering(panel_clean: pd.DataFrame, results: dict):
    print("\n[clustering]")
    # k is chosen by the silhouette score (=2 here: a robust developed/developing
    # split, bootstrap ARI ~0.96).  We also fit k=3 purely to characterise the
    # finer structure that emerges (a small fragile-state group) and report it.
    res = clu.run_clustering(panel_clean, k=None)
    res_k3 = clu.run_clustering(panel_clean, k=3)
    res_k3["traj"][["iso3", "name", "income_group", "region", "cluster"]].to_csv(
        os.path.join(C.RESULTS_DIR, "clu_assignments_k3.csv"), index=False)
    res_k3["crosstab"].to_csv(os.path.join(C.RESULTS_DIR, "clu_crosstab_k3.csv"))
    clu.plot_k_selection(res["k_selection"], "clu_kselection.png")
    clu.plot_pca_scatter(res["traj"], res["k"], "clu_pca.png")
    clu.plot_dendrogram(res["Xs"], "clu_dendrogram.png")
    clu.plot_profile_heatmap(res["profile"], "clu_profile.png")
    clu.plot_crosstab(res["crosstab"], "clu_crosstab.png")

    res["k_selection"].to_csv(os.path.join(C.RESULTS_DIR, "clu_kselection.csv"),
                              index=False)
    res["profile"].to_csv(os.path.join(C.RESULTS_DIR, "clu_profile.csv"))
    res["crosstab"].to_csv(os.path.join(C.RESULTS_DIR, "clu_crosstab.csv"))
    res["divergent"]["over_performers"].to_csv(
        os.path.join(C.RESULTS_DIR, "clu_overperformers.csv"), index=False)
    res["divergent"]["under_performers"].to_csv(
        os.path.join(C.RESULTS_DIR, "clu_underperformers.csv"), index=False)
    res["traj"][["iso3", "name", "income_group", "region", "cluster"]].to_csv(
        os.path.join(C.RESULTS_DIR, "clu_assignments.csv"), index=False)

    results["clustering"] = {
        "k": res["k"],
        "silhouette": round(res["silhouette_final"], 4),
        "stability_ari_mean": round(res["stability"][0], 4),
        "stability_ari_std": round(res["stability"][1], 4),
        "ari_kmeans_vs_ward": round(res["ari_km_ward"], 4),
        "pca_explained": [round(float(x), 4) for x in res["pca_explained"]],
        "crosstab": res["crosstab"].to_dict(),
        "n_countries": int(len(res["traj"])),
        "k3_crosstab": res_k3["crosstab"].to_dict(),
        "k3_stability_ari_mean": round(res_k3["stability"][0], 4),
    }
    print(f"   k={res['k']}, silhouette={res['silhouette_final']:.3f}, "
          f"stability ARI={res['stability'][0]:.3f}+/-{res['stability'][1]:.3f}, "
          f"kmeans-vs-ward ARI={res['ari_km_ward']:.3f}")
    print(res["crosstab"].to_string())


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="re-download WDI + WHR before running")
    args = ap.parse_args()

    panel = get_panel(args.rebuild)
    panel_clean = pp.clean_panel(panel)
    panel_clean.to_csv(os.path.join(C.PROCESSED_DIR, "panel_clean.csv"),
                       index=False)

    results = {"n_rows": int(len(panel_clean)),
               "n_countries": int(panel_clean["iso3"].nunique()),
               "years": [int(panel_clean.year.min()), int(panel_clean.year.max())]}

    run_eda(panel_clean)
    run_prediction(panel_clean, results)
    run_clustering(panel_clean, results)

    with open(os.path.join(C.RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n[ALL DONE] results/summary.json written")


if __name__ == "__main__":
    main()
