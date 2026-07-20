"""
Unsupervised clustering of national development *trajectories*.

Each country is represented not by a single snapshot but by the shape of its
path over time.  For a core set of development indicators we engineer three
trajectory descriptors:

    level       -- the average standing over the study period,
    slope       -- the per-decade linear trend (are things improving?),
    volatility  -- the std of year-to-year changes (how bumpy was the path?).

We then cluster countries in this trajectory space with K-means and Ward
hierarchical clustering, pick k with the silhouette score + elbow, check
stability by bootstrapping, profile each cluster, and finally cross-tabulate the
data-driven archetypes against World Bank income tiers to surface countries that
over- or under-perform their income group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

try:
    from . import config as C
    from . import utils as U
except ImportError:  # pragma: no cover
    import config as C
    import utils as U

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# Broadly-covered WDI indicators -> keeps ~all countries in the clustering.
TRAJ_INDICATORS = [
    "life_expectancy",
    "gdp_pc_ppp",
    "under5_mortality",
    "fertility_rate",
    "sanitation_access",
    "dtp3_immunization",
    "urban_pop_pct",
    "primary_enrollment",
]


# --------------------------------------------------------------------------- #
# Trajectory feature engineering
# --------------------------------------------------------------------------- #
def _slope(years, values):
    """Per-decade OLS slope, robust to missing values."""
    m = np.isfinite(values)
    if m.sum() < 3:
        return np.nan
    b = np.polyfit(years[m], values[m], 1)[0]
    return b * 10.0  # per decade


def build_trajectory_features(panel, indicators=TRAJ_INDICATORS, min_years=6):
    """One row per country: level / slope / volatility for each indicator."""
    rows = []
    for iso, g in panel.sort_values("year").groupby("iso3"):
        if g["year"].nunique() < min_years:
            continue
        rec = {"iso3": iso, "name": g["name"].iloc[0],
               "income_group": g["income_group"].iloc[0],
               "region": g["region"].iloc[0]}
        yr = g["year"].to_numpy(dtype=float)
        for ind in indicators:
            if ind not in g:
                continue
            v = g[ind].to_numpy(dtype=float)
            rec[f"{ind}__level"] = np.nan if not np.isfinite(v).any() else np.nanmean(v)
            rec[f"{ind}__slope"] = _slope(yr, v)
            rec[f"{ind}__vol"] = np.nanstd(np.diff(v[np.isfinite(v)])) \
                if np.isfinite(v).sum() > 2 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def _prep_matrix(traj):
    feat_cols = [c for c in traj.columns
                 if c.endswith(("__level", "__slope", "__vol"))]
    X = traj[feat_cols].copy()
    # Median-impute residual gaps (a handful of countries miss an indicator).
    X = X.fillna(X.median(numeric_only=True))
    Xs = StandardScaler().fit_transform(X)
    return Xs, feat_cols


# --------------------------------------------------------------------------- #
# Choosing k
# --------------------------------------------------------------------------- #
def choose_k(Xs, k_range=range(2, 9)):
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=C.RANDOM_STATE)
        labels = km.fit_predict(Xs)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(Xs, labels)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Stability via bootstrap (ARI between bootstrap fit and full fit)
# --------------------------------------------------------------------------- #
def bootstrap_stability(Xs, k, n_boot=100):
    rng = np.random.default_rng(C.RANDOM_STATE)
    base = KMeans(n_clusters=k, n_init=20,
                  random_state=C.RANDOM_STATE).fit_predict(Xs)
    aris = []
    n = Xs.shape[0]
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        uniq = np.unique(idx)
        lab = KMeans(n_clusters=k, n_init=10,
                     random_state=int(rng.integers(1e6))).fit_predict(Xs[uniq])
        aris.append(adjusted_rand_score(base[uniq], lab))
    return float(np.mean(aris)), float(np.std(aris))


# --------------------------------------------------------------------------- #
# Full clustering pipeline
# --------------------------------------------------------------------------- #
def run_clustering(panel, k=None):
    traj = build_trajectory_features(panel)
    Xs, feat_cols = _prep_matrix(traj)

    ksel = choose_k(Xs)
    if k is None:
        k = int(ksel.loc[ksel["silhouette"].idxmax(), "k"])

    km = KMeans(n_clusters=k, n_init=50, random_state=C.RANDOM_STATE)
    traj["cluster"] = km.fit_predict(Xs)

    # Ward hierarchical labels for comparison.
    ward = AgglomerativeClustering(n_clusters=k, linkage="ward")
    traj["cluster_ward"] = ward.fit_predict(Xs)
    ari_km_ward = adjusted_rand_score(traj["cluster"], traj["cluster_ward"])

    # Order clusters by mean life-expectancy level so labels are interpretable
    # (0 = lowest development ... k-1 = highest).
    order = (traj.groupby("cluster")["life_expectancy__level"].mean()
             .sort_values().index.tolist())
    remap = {old: new for new, old in enumerate(order)}
    traj["cluster"] = traj["cluster"].map(remap)

    pca = PCA(n_components=2, random_state=C.RANDOM_STATE)
    coords = pca.fit_transform(Xs)
    traj["pc1"], traj["pc2"] = coords[:, 0], coords[:, 1]

    stab_mean, stab_std = bootstrap_stability(Xs, k)

    profile = _cluster_profile(traj, feat_cols)
    crosstab = pd.crosstab(traj["cluster"], traj["income_group"])
    divergent = _find_divergent(traj)

    return {
        "traj": traj, "Xs": Xs, "feat_cols": feat_cols, "k": k,
        "k_selection": ksel, "silhouette_final": silhouette_score(Xs, traj["cluster"]),
        "stability": (stab_mean, stab_std), "ari_km_ward": ari_km_ward,
        "profile": profile, "crosstab": crosstab, "divergent": divergent,
        "pca_explained": pca.explained_variance_ratio_,
    }


def _cluster_profile(traj, feat_cols):
    """Mean of the (raw) trajectory features per cluster, plus size."""
    prof = traj.groupby("cluster")[feat_cols].mean()
    prof.insert(0, "n_countries", traj.groupby("cluster").size())
    return prof


def _find_divergent(traj):
    """Countries whose data-driven cluster disagrees with their income tier."""
    income_rank = {"Low income": 0, "Lower-middle income": 1,
                   "Upper-middle income": 2, "High income": 3}
    t = traj.copy()
    t["income_rank"] = t["income_group"].map(income_rank)
    k = t["cluster"].nunique()
    # Normalise both ranks to 0..1 and flag large gaps.
    t["cluster_norm"] = t["cluster"] / (k - 1)
    t["income_norm"] = t["income_rank"] / 3.0
    t["gap"] = t["cluster_norm"] - t["income_norm"]
    over = t.nlargest(10, "gap")[["name", "income_group", "cluster", "gap"]]
    under = t.nsmallest(10, "gap")[["name", "income_group", "cluster", "gap"]]
    return {"over_performers": over, "under_performers": under}


# ======================= plotting ========================================== #
def plot_k_selection(ksel, fname):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ksel["k"], ksel["inertia"], "o-", color="#4C72B0", label="Inertia")
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("Inertia (elbow)", color="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(ksel["k"], ksel["silhouette"], "s--", color="#C44E52",
             label="Silhouette")
    ax2.set_ylabel("Silhouette", color="#C44E52")
    ax1.set_title("Choosing k: elbow + silhouette", fontweight="bold")
    return U.savefig(fig, fname)


def plot_pca_scatter(traj, k, fname):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.scatterplot(data=traj, x="pc1", y="pc2", hue="cluster",
                    palette="viridis", s=55, ax=ax, legend="full")
    ax.set_title(f"Development archetypes in trajectory space (k={k})",
                 fontweight="bold")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    return U.savefig(fig, fname)


def plot_dendrogram(Xs, fname):
    Z = linkage(Xs, method="ward")
    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(Z, no_labels=True, color_threshold=0.7 * max(Z[:, 2]), ax=ax)
    ax.set_title("Ward hierarchical clustering dendrogram", fontweight="bold")
    ax.set_ylabel("Merge distance")
    return U.savefig(fig, fname)


def plot_profile_heatmap(profile, fname):
    # z-score each feature (column) across clusters for readability.
    cols = [c for c in profile.columns if c != "n_countries"]
    z = (profile[cols] - profile[cols].mean()) / profile[cols].std()
    z.columns = [c.replace("__", "\n") for c in z.columns]
    fig, ax = plt.subplots(figsize=(min(1 + 0.42 * len(cols), 16), 4))
    sns.heatmap(z, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "z-score"})
    ax.set_title("Cluster profiles (standardised trajectory features)",
                 fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("cluster")
    return U.savefig(fig, fname)


def plot_crosstab(crosstab, fname):
    order = ["Low income", "Lower-middle income",
             "Upper-middle income", "High income"]
    ct = crosstab.reindex(columns=[c for c in order if c in crosstab.columns])
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Data-driven clusters vs. World Bank income tiers",
                 fontweight="bold")
    ax.set_xlabel("World Bank income group"); ax.set_ylabel("cluster")
    return U.savefig(fig, fname)
