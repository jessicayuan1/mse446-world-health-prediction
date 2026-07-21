"""
Supervised prediction experiments.

Design choices that matter for validity:

  * Evaluation is **grouped by country** (GroupKFold): a country is never split
    across train and test, so temporal autocorrelation within a country cannot
    inflate scores.  We report pooled out-of-fold RMSE / MAE / R^2.
  * A naive **persistence baseline** (predict y_{t+k} = y_t) is always shown so
    that "good" R^2 is judged against the hard-to-beat do-nothing forecast.
  * The **hypothesis test** (economics-alone vs. economics+health+social) is run
    on the *same* rows -- those with observed social variables -- so the extra
    predictors get a fair chance rather than being mostly imputed.
  * A separate **temporal hold-out** (train on target years <= 2016, test on
    2017-2020) checks generalisation forward in time, not just across countries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (GroupKFold, cross_val_predict,
                                     cross_val_score, GridSearchCV)
from sklearn.inspection import permutation_importance

try:
    from . import config as C
    from . import preprocessing as pp
    from . import utils as U
except ImportError:  # pragma: no cover
    import config as C
    import preprocessing as pp
    import utils as U

import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------------------------------- #
# Cross-validated out-of-fold predictions (grouped by country)
# --------------------------------------------------------------------------- #
def _oof_predict(model, X, y, groups, n_splits=5):
    cv = GroupKFold(n_splits=n_splits)
    preds = cross_val_predict(model, X, y, groups=groups, cv=cv, n_jobs=-1)
    return preds


# --------------------------------------------------------------------------- #
# Hyperparameter tuning (GridSearchCV with GroupKFold)
# --------------------------------------------------------------------------- #
def tune_models(X, y, groups, n_splits=5):
    """
    Tune each model's hyperparameters with GridSearchCV under GroupKFold, so the
    search itself never splits a country across train/validation.  Returns:
        tuned  : {name -> best_estimator_ pipeline (tuned hyperparameters)}
        params : DataFrame of chosen params + best grouped-CV RMSE per model
    Scoring is negative RMSE; the reported CV_RMSE is the grouped cross-validated
    error at the selected settings.
    """
    cv = GroupKFold(n_splits=n_splits)
    grids = U.param_grids()
    tuned, rows = {}, []
    for name, model in U.make_models().items():
        grid = grids.get(name)
        if grid:
            gs = GridSearchCV(model, grid, cv=cv,
                              scoring="neg_root_mean_squared_error", n_jobs=-1)
            gs.fit(X, y, groups=groups)
            tuned[name] = gs.best_estimator_
            best = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
            rows.append({"Model": name, "CV_RMSE": -gs.best_score_,
                         "best_params": best})
        else:  # Linear regression: no hyperparameters to tune
            sc = -cross_val_score(model, X, y, groups=groups, cv=cv,
                                  scoring="neg_root_mean_squared_error").mean()
            tuned[name] = clone(model)
            rows.append({"Model": name, "CV_RMSE": sc, "best_params": {}})
    params = pd.DataFrame(rows).set_index("Model")
    return tuned, params


# --------------------------------------------------------------------------- #
# Experiment 1 -- model comparison on the full feature set
# --------------------------------------------------------------------------- #
def experiment_model_comparison(panel, target, horizon=C.FORECAST_HORIZON):
    """
    Tune every model (grouped CV), then report pooled out-of-fold metrics using
    the tuned hyperparameters, alongside the persistence baseline.  Returns the
    metrics table, the built dataset, the OOF predictions, the tuned estimators
    and the chosen-hyperparameter table.
    """
    data = pp.build_supervised(panel, target, horizon)
    X, y, groups = data["X"], data["y"], data["groups"]

    tuned, params = tune_models(X, y, groups)

    rows = []
    # Persistence baseline: y_hat(t+k) = y(t).  Some current-year values were
    # removed by despiking, so score the baseline only where y_t is observed.
    pmask = data["persistence_pred"].notna()
    base = U.regression_metrics(y[pmask], data["persistence_pred"][pmask])
    base["Model"] = "Persistence (y_t)"
    rows.append(base)

    oof_store = {}
    for name, est in tuned.items():
        preds = _oof_predict(clone(est), X, y, groups)
        m = U.regression_metrics(y, preds)
        m["Model"] = name
        rows.append(m)
        oof_store[name] = preds

    df = pd.DataFrame(rows).set_index("Model")[["RMSE", "MAE", "R2"]]
    return df, data, oof_store, tuned, params


def experiment_autoregressive(panel, target, estimator,
                              horizon=C.FORECAST_HORIZON):
    """
    Sanity check on the persistence baseline: add the current outcome value as a
    feature and confirm the learned model then *exceeds* naive persistence,
    i.e. the indicators carry signal beyond simple carry-forward.
    """
    data = pp.build_supervised(panel, target, horizon)
    X = data["X"].copy()
    X["outcome_now"] = data["y_now"].to_numpy()
    m = X["outcome_now"].notna().to_numpy()
    preds = _oof_predict(clone(estimator), X[m], data["y"][m], data["groups"][m])
    return U.regression_metrics(data["y"][m], preds)


# --------------------------------------------------------------------------- #
# Experiment 2 -- the central hypothesis: economics alone vs. + health + social
# --------------------------------------------------------------------------- #
def experiment_feature_sets(panel, target, estimator,
                            horizon=C.FORECAST_HORIZON):
    """
    Compare feature sets on a common set of rows (those with observed social
    variables) for both the LEVEL target y_{t+k} and the multi-year GAIN
    Delta y = y_{t+k} - y_t.  `estimator` is the tuned pipeline to reuse.
    """
    sets = pp.feature_sets_for(target)
    # Build once on the full feature list so rows align, then subset columns.
    full = pp.build_supervised(panel, target, horizon,
                               feature_cols=sets["Full (econ + health + social)"])
    X_all, y, groups = full["X"], full["y"], full["groups"]
    delta = full["delta"]

    # Common-rows mask: require the core social variables to be observed so the
    # comparison is not dominated by imputation.
    core_social = [c for c in ["social_support", "freedom"] if c in X_all.columns]
    mask = X_all[core_social].notna().all(axis=1) if core_social else pd.Series(True, index=X_all.index)
    Xc, yc, gc, dc = X_all[mask], y[mask], groups[mask], delta[mask]

    # The GAIN target needs an observed current value; keep those rows only.
    gmask = dc.notna().to_numpy()

    rows = []
    for set_name, cols in sets.items():
        cols = [c for c in cols if c in Xc.columns]
        # Target = future LEVEL
        preds_lvl = _oof_predict(clone(estimator), Xc[cols], yc, gc)
        m_lvl = U.regression_metrics(yc, preds_lvl)
        # Target = multi-year GAIN (rows with observed y_t only)
        preds_dlt = _oof_predict(clone(estimator), Xc[cols][gmask], dc[gmask], gc[gmask])
        m_dlt = U.regression_metrics(dc[gmask], preds_dlt)
        rows.append({
            "Feature set": set_name, "n_features": len(cols),
            "Level RMSE": m_lvl["RMSE"], "Level R2": m_lvl["R2"],
            "Gain RMSE": m_dlt["RMSE"], "Gain R2": m_dlt["R2"],
        })
    df = pd.DataFrame(rows).set_index("Feature set")
    df.attrs["n_rows"] = int(mask.sum())
    df.attrs["n_countries"] = int(gc.nunique())
    return df


# --------------------------------------------------------------------------- #
# Experiment 3 -- feature importance (permutation + SHAP) on a temporal hold-out
# --------------------------------------------------------------------------- #
def experiment_importance(panel, target, estimator,
                          horizon=C.FORECAST_HORIZON):
    data = pp.build_supervised(panel, target, horizon)
    X, y, ty = data["X"], data["y"], data["target_year"]
    train = ty <= (min(C.TEST_YEARS) - 1)
    test = ~train

    model = clone(estimator)
    model.fit(X[train], y[train])

    # Permutation importance on the held-out (future) years.
    perm = permutation_importance(
        model, X[test], y[test], n_repeats=20,
        random_state=C.RANDOM_STATE, n_jobs=-1, scoring="r2",
    )
    imp = (pd.DataFrame({"feature": X.columns,
                         "importance": perm.importances_mean,
                         "std": perm.importances_std})
           .sort_values("importance", ascending=False)
           .reset_index(drop=True))

    # SHAP on the tree model (fast TreeExplainer); transform X through the
    # impute+scale stages first so SHAP sees exactly what the model saw.
    shap_vals, X_proc = None, None
    try:
        import shap
        pre = model[:-1]
        X_proc = pd.DataFrame(pre.transform(X[test]), columns=X.columns)
        explainer = shap.TreeExplainer(model[-1])
        shap_vals = explainer.shap_values(X_proc)
    except Exception as e:  # pragma: no cover
        print(f"[SHAP] skipped for {target}: {e!r}")

    return {"importance": imp, "shap_values": shap_vals,
            "X_proc": X_proc, "model": model, "data": data,
            "train_mask": train, "test_mask": test}


# --------------------------------------------------------------------------- #
# Experiment 4 -- temporal generalisation
# --------------------------------------------------------------------------- #
def experiment_temporal(panel, target, models=None, horizon=C.FORECAST_HORIZON):
    """Forward-in-time hold-out. `models` may be the tuned estimators dict."""
    data = pp.build_supervised(panel, target, horizon)
    X, y, ty = data["X"], data["y"], data["target_year"]
    train = ty <= (min(C.TEST_YEARS) - 1)
    test = ~train
    models = models if models is not None else U.make_models()

    rows = []
    pm = test & data["persistence_pred"].notna()
    base = U.regression_metrics(y[pm], data["persistence_pred"][pm])
    base["Model"] = "Persistence (y_t)"
    rows.append(base)
    for name, model in models.items():
        est = clone(model)
        est.fit(X[train], y[train])
        m = U.regression_metrics(y[test], est.predict(X[test]))
        m["Model"] = name
        rows.append(m)
    df = pd.DataFrame(rows).set_index("Model")[["RMSE", "MAE", "R2"]]
    df.attrs["n_train"] = int(train.sum())
    df.attrs["n_test"] = int(test.sum())
    return df


# ======================= plotting helpers ================================== #
def plot_model_comparison(df, target_label, fname):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    d = df.reset_index()
    sns.barplot(data=d, y="Model", x="RMSE", ax=axes[0], color="#4C72B0")
    axes[0].set_title(f"RMSE (lower is better)")
    r2 = d[d["R2"] > -1]
    sns.barplot(data=r2, y="Model", x="R2", ax=axes[1], color="#55A868")
    axes[1].set_title("R^2 (higher is better)")
    fig.suptitle(f"Model comparison - predicting {target_label}", fontweight="bold")
    return U.savefig(fig, fname)


def plot_feature_sets(df, target_label, fname):
    d = df.reset_index().melt(
        id_vars="Feature set",
        value_vars=["Level R2", "Gain R2"],
        var_name="Task", value_name="R2",
    )
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.barplot(data=d, x="Task", y="R2", hue="Feature set", ax=ax)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(f"Adding health + social factors - predicting {target_label}",
                 fontweight="bold")
    ax.set_ylabel("Out-of-fold R^2 (grouped by country)")
    return U.savefig(fig, fname)


def plot_permutation(imp, target_label, fname, top=15):
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(d["feature"], d["importance"], xerr=d["std"], color="#C44E52")
    ax.set_xlabel("Drop in R^2 when feature is shuffled")
    ax.set_title(f"Permutation importance - {target_label}", fontweight="bold")
    return U.savefig(fig, fname)


def plot_shap(shap_values, X_proc, target_label, fname, top=12):
    import shap
    fig = plt.figure()
    shap.summary_plot(shap_values, X_proc, max_display=top, show=False)
    plt.title(f"SHAP summary - {target_label}", fontweight="bold")
    fig = plt.gcf()
    return U.savefig(fig, fname)


def plot_pred_vs_actual(y_true, y_pred, target_label, fname, meta=None):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=12, alpha=0.4, edgecolor="none")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(f"Actual {target_label}")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Out-of-fold predictions - {target_label}", fontweight="bold")
    return U.savefig(fig, fname)
