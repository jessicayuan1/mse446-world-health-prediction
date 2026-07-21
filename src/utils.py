"""Shared helpers: plotting style, metrics, and model factory."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: write figures to disk, never open a window
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

try:
    from . import config as C
except ImportError:  # pragma: no cover
    import config as C

sns.set_theme(context="notebook", style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "figure.autolayout": True})


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def regression_metrics(y_true, y_pred) -> dict:
    """RMSE, MAE and R^2 in one call."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------- #
# Model factory  (imputer -> scaler -> estimator)
# --------------------------------------------------------------------------- #
def make_models(random_state: int = C.RANDOM_STATE) -> dict:
    """
    Return the model zoo, each as a leakage-safe Pipeline:
        IterativeImputer (MICE) -> StandardScaler -> estimator.

    The imputer and scaler are fit *inside* cross-validation folds, so no
    information from held-out countries/years leaks into preprocessing.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    def pipe(estimator):
        return Pipeline(
            [
                ("impute", IterativeImputer(random_state=random_state, max_iter=10,
                                            sample_posterior=False)),
                ("scale", StandardScaler()),
                ("model", estimator),
            ]
        )

    return {
        "Linear": pipe(LinearRegression()),
        "Ridge": pipe(Ridge(alpha=10.0, random_state=random_state)),
        "Lasso": pipe(Lasso(alpha=0.05, random_state=random_state, max_iter=10000)),
        "Random Forest": pipe(
            RandomForestRegressor(
                n_estimators=400, max_depth=None, min_samples_leaf=3,
                n_jobs=-1, random_state=random_state,
            )
        ),
        "XGBoost": pipe(
            XGBRegressor(
                n_estimators=500, learning_rate=0.03, max_depth=4,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=random_state, n_jobs=-1,
            )
        ),
    }


# --------------------------------------------------------------------------- #
# Hyperparameter search spaces  (keys target the "model" step of each pipeline)
# --------------------------------------------------------------------------- #
def param_grids() -> dict:
    """
    Modest, sensible grids searched with GroupKFold.  Linear regression has no
    hyperparameters, so it is left out (evaluated at its defaults).
    """
    return {
        "Ridge": {"model__alpha": [0.1, 1.0, 10.0, 50.0, 100.0]},
        "Lasso": {"model__alpha": [0.001, 0.01, 0.05, 0.1, 0.5]},
        "Random Forest": {
            "model__n_estimators": [400],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_leaf": [1, 3, 5],
        },
        "XGBoost": {
            "model__n_estimators": [300, 600],
            "model__max_depth": [3, 4, 6],
            "model__learning_rate": [0.03, 0.1],
            "model__subsample": [0.8],
        },
    }


# --------------------------------------------------------------------------- #
# Figure saving
# --------------------------------------------------------------------------- #
def savefig(fig, name: str):
    """Save a figure into the project's figures/ directory and close it."""
    import os

    path = os.path.join(C.FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
