"""Final evaluation of frozen models on the hold-out test set (ADR-010, ADR-012).

This is the only library module that loads the test set (``tests/test_test_set_guard.py``). It
evaluates each model once, after every modelling decision has been taken on training data:

- the models must match, variable by variable and coefficient by coefficient, the frozen models
  recorded in ``reports/metrics/pd_logistic_cv.json``; otherwise it refuses to run;
- the result is saved to ``reports/metrics/pd_logistic_test.json`` together with the model it
  scored. If that file already exists for a different model, it refuses to run, so the test set
  can never be used to compare a second version of a model. Running it again on the same frozen
  model (for example when the notebook is re-executed) reproduces the same evaluation.

Every model is reported on all test rows and, as fixed in ADR-010, on the test rows without a
twin in train. Operating points: the PD cut-offs fixed on training data (train default rate, 0.5
and the cut-offs that flagged 10%, 20% and 30% of the out-of-fold predictions), plus the capture
rate when the 10%, 20% or 30% of test accounts with the highest PD are flagged (a capacity rule,
which uses the ranking of the test scores but not their outcomes).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from creditrisk.config import load_config
from creditrisk.data.schema import TARGET
from creditrisk.data.split import has_twin_in, load_test, load_train
from creditrisk.features.engineering import build_features
from creditrisk.models.evaluation import evaluate_predictions
from creditrisk.models.pipeline import WoeLogitModel
from creditrisk.reporting import METRICS_DIR, save_metrics
from creditrisk.validation.metrics import auc, paired_bootstrap_diff

CV_METRICS = "pd_logistic_cv"
TEST_METRICS = "pd_logistic_test"


def model_spec(model: WoeLogitModel) -> dict:
    """Variables and coefficients of a fitted model: what identifies it."""
    return {
        "final_features": list(model.features_),
        "coefficients": {k: float(v) for k, v in model.result_.params.items()},
    }


def _same_spec(spec: dict, frozen: dict) -> bool:
    if spec["final_features"] != frozen["final_features"]:
        return False
    frozen_coefs = frozen["coefficients"]
    if isinstance(frozen_coefs, list):  # the CV report stores the coefficient table as rows
        frozen_coefs = {row["term"]: row["coef"] for row in frozen_coefs}
    if set(spec["coefficients"]) != set(frozen_coefs):
        return False
    return all(
        np.isclose(spec["coefficients"][k], frozen_coefs[k], rtol=1e-6, atol=1e-9)
        for k in frozen_coefs
    )


def _read_metrics(folder: Path, name: str) -> dict | None:
    path = folder / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def evaluate_on_test(
    models: dict[str, WoeLogitModel],
    compare: tuple[str, str],
    n_boot: int = 1000,
    seed: int | None = None,
    metrics_dir: Path | None = None,
) -> dict:
    """Check the models are the frozen ones, evaluate them on test once, save and return the result.

    ``compare`` names two models whose AUC difference (first minus second) gets a paired
    bootstrap interval, on all test rows and on the rows without a twin in train.
    """
    folder = METRICS_DIR if metrics_dir is None else Path(metrics_dir)
    config = load_config("project")
    seed = config["seed"] if seed is None else seed

    frozen = _read_metrics(folder, CV_METRICS)
    if frozen is None:
        raise FileNotFoundError(f"{CV_METRICS}.json not found: freeze the models first")
    specs = {name: model_spec(model) for name, model in models.items()}
    for name, spec in specs.items():
        if not _same_spec(spec, frozen["models"][name]):
            raise ValueError(f"Model '{name}' is not the frozen model of {CV_METRICS}.json")
    previous = _read_metrics(folder, TEST_METRICS)
    if previous is not None:
        for name, spec in specs.items():
            if not _same_spec(spec, previous["frozen_models"][name]):
                raise ValueError(
                    f"The test set was already evaluated with a different '{name}' model; "
                    "it cannot be used to evaluate a second version."
                )

    train, test = load_train(), load_test()
    twin = has_twin_in(test, train)
    X_test, y_test = build_features(test), test[TARGET]
    predictions = {name: model.predict_pd(X_test) for name, model in models.items()}

    results = {}
    for name, p in predictions.items():
        oof_points = frozen["models"][name]["cv"]["repeat_1"]["operating_points"]
        oof_cutoffs = [r["threshold"] for r in oof_points if r["point"].startswith("flag")]
        thresholds = (*frozen["operating_thresholds"], *oof_cutoffs)
        results[name] = {
            "all_rows": evaluate_predictions(y_test, p, n_boot, seed, thresholds=thresholds),
            "without_twins": evaluate_predictions(
                y_test[~twin], p[~twin], n_boot, seed, thresholds=thresholds
            ),
        }

    first, second = compare
    differences = {}
    for subset, rows in {"all_rows": np.ones(len(test), bool), "without_twins": ~twin}.items():
        estimate, low, high = paired_bootstrap_diff(
            auc,
            y_test[rows],
            predictions[first][rows],
            predictions[second][rows],
            n_boot=n_boot,
            seed=seed,
        )
        differences[subset] = {"estimate": estimate, "low": low, "high": high}

    out = {
        "n_test_rows": int(len(test)),
        "n_test_rows_with_twin_in_train": int(twin.sum()),
        "bootstrap_seed": seed,
        "n_boot": n_boot,
        "frozen_models": specs,
        "models": results,
        "auc_difference": {"first": first, "second": second, **differences},
    }
    # load_train() already checked the raw file against this hash.
    save_metrics(TEST_METRICS, out, config["data"]["raw_sha256"], metrics_dir=folder)
    return out
