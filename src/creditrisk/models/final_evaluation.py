"""Final evaluation of finished models on the hold-out test set (ADR-010, ADR-012).

This is the only module allowed to load the test set (``tests/test_test_set_guard.py``). It is
called once per final model, after every modelling decision has been taken on training data, and
nothing is changed afterwards because of what it shows.

Every model is reported on all test rows and, as the sensitivity fixed in ADR-010, on the test
rows that have no twin in train.
"""

from __future__ import annotations

from creditrisk.data.schema import TARGET
from creditrisk.data.split import has_twin_in, load_test, load_train
from creditrisk.features.engineering import build_features
from creditrisk.models.evaluation import evaluate_predictions
from creditrisk.models.pipeline import WoeLogitModel
from creditrisk.validation.metrics import auc, paired_bootstrap_diff


def evaluate_on_test(
    models: dict[str, WoeLogitModel],
    thresholds: tuple[float, ...],
    compare: tuple[str, str] | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Evaluate fitted models on the test set; return a JSON-friendly dict.

    ``thresholds`` are the fixed PD cut-offs of the operating-point table (chosen on train).
    ``compare`` names two models whose AUC difference (first minus second) gets a paired
    bootstrap interval on the same test rows.
    """
    train, test = load_train(), load_test()
    twin = has_twin_in(test, train)
    X_test, y_test = build_features(test), test[TARGET]
    predictions = {name: model.predict_pd(X_test) for name, model in models.items()}

    results = {}
    for name, p in predictions.items():
        results[name] = {
            "all_rows": evaluate_predictions(y_test, p, n_boot, seed, thresholds=thresholds),
            "without_twins": evaluate_predictions(
                y_test[~twin], p[~twin], n_boot, seed, thresholds=thresholds
            ),
        }
    out = {
        "n_test_rows": int(len(test)),
        "n_test_rows_with_twin_in_train": int(twin.sum()),
        "models": results,
    }
    if compare:
        first, second = compare
        estimate, low, high = paired_bootstrap_diff(
            auc, y_test, predictions[first], predictions[second], n_boot=n_boot, seed=seed
        )
        out["auc_difference"] = {
            "first": first,
            "second": second,
            "estimate": estimate,
            "low": low,
            "high": high,
        }
    return out
