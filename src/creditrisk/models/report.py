"""Phase 5 on training data: fit the final models and measure them by cross-validation (ADR-012).

Two models are built with the same chain (``WoeLogitModel``):

- the champion, on the candidates of ADR-011 (without the September status);
- the sensitivity model of ADR-009, on the same candidates plus ``STATUS_SEP``.

Cross-validation refits the whole chain in each of the 15 folds of ``make_cv``. The final models
are then fitted on all of train. Nothing here touches the test set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditrisk.data.schema import TARGET
from creditrisk.data.split import make_cv
from creditrisk.features.engineering import FEATURE_SPECS, build_features
from creditrisk.features.selection import champion_specs
from creditrisk.models.evaluation import evaluate_predictions
from creditrisk.models.logistic import coefficient_table
from creditrisk.models.pipeline import WoeLogitModel, cross_validate
from creditrisk.validation.metrics import auc


def model_specs() -> dict[str, dict[str, dict]]:
    """Candidate features of the champion and of the sensitivity model."""
    champion = champion_specs()
    return {
        "champion": champion,
        "sensitivity_with_status_sep": {**champion, "STATUS_SEP": FEATURE_SPECS["STATUS_SEP"]},
    }


def _records(table: pd.DataFrame) -> list[dict]:
    return table.astype(object).where(table.notna(), None).to_dict(orient="records")


def _cv_summary(cv: dict, y: pd.Series, thresholds, n_boot: int, seed: int) -> dict:
    """Metrics of the out-of-fold predictions: repeat 1 with bootstrap intervals, plus the spread
    across the three repeats and across the 15 folds."""
    oof = cv["oof"]
    folds = pd.DataFrame(cv["folds"])
    return {
        "repeat_1": evaluate_predictions(y, oof[1], n_boot, seed, thresholds=thresholds),
        "auc_by_repeat": {int(r): auc(y, oof[r]) for r in oof.columns},
        "fold_auc_min": float(folds["auc"].min()),
        "fold_auc_max": float(folds["auc"].max()),
        "fold_ks_min": float(folds["ks"].min()),
        "fold_ks_max": float(folds["ks"].max()),
        "n_folds": int(len(folds)),
        "selection_counts": {k: int(v) for k, v in cv["selection_counts"].items()},
        "n_features_by_fold": [len(f) for f in folds["features"]],
    }


def train_report(train: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> tuple[dict, dict, dict]:
    """Cross-validate and fit both models on ``train``.

    Returns the JSON-friendly report, the fitted final models (for the test evaluation) and the
    out-of-fold PD of the first repeat of every model (for plots).
    """
    X, y = build_features(train), train[TARGET]
    folds = list(make_cv().split(np.zeros(len(y)), y))
    n_folds = make_cv().cvargs["n_splits"]
    train_rate = float(y.mean())
    thresholds = (train_rate, 0.5)

    report = {
        "n_train_rows": int(len(train)),
        "train_default_rate": train_rate,
        "operating_thresholds": list(thresholds),
        "n_boot": n_boot,
        "models": {},
    }
    models = {}
    oof_repeat_1 = {}
    for name, specs in model_specs().items():
        cv = cross_validate(specs, X, y, folds, n_folds=n_folds)
        model = WoeLogitModel(specs).fit(X, y)
        models[name] = model
        oof_repeat_1[name] = cv["oof"][1]
        summary = model.selection_summary().reset_index().rename(columns={"index": "feature"})
        report["models"][name] = {
            "candidates": list(specs),
            "final_features": list(model.features_),
            "coefficients": _records(coefficient_table(model.result_).reset_index()),
            "selection": _records(summary),
            "train_fit_auc": auc(y, model.predict_pd(X)),
            "cv": _cv_summary(cv, y, thresholds, n_boot, seed),
        }
    return report, models, oof_repeat_1
