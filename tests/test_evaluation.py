import numpy as np
import pandas as pd
import pytest

from creditrisk.models import evaluation as ev


def test_operating_points_known_answer():
    y = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    table = ev.operating_points(y, p, flag_rates=[0.2, 0.4], thresholds=[0.55])
    top_20 = table.loc["flag 20%"]
    assert top_20["flag_rate"] == 0.2
    assert top_20["recall"] == pytest.approx(2 / 3)
    assert top_20["precision"] == 1.0
    top_40 = table.loc["flag 40%"]
    assert top_40["recall"] == 1.0
    assert top_40["precision"] == 0.75
    fixed = table.loc["PD >= 0.55"]
    assert fixed["flag_rate"] == 0.4
    assert fixed["threshold"] == 0.55


def test_evaluate_predictions_reports_every_metric_with_an_interval():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.22, size=3000)
    p = 1 / (1 + np.exp(-(rng.normal(size=3000) + 2.5 * y - 2)))
    out = ev.evaluate_predictions(y, p, n_boot=100, seed=1)
    for name in ["auc", "gini", "ks", "pr_auc"]:
        assert out[name]["low"] <= out[name]["estimate"] <= out[name]["high"], name
    assert set(out["calibration"]) == {
        "brier",
        "calibration_in_the_large",
        "calibration_intercept",
        "calibration_slope",
    }
    assert len(out["calibration_by_decile"]) == 10
    assert out["n_rows"] == 3000
    assert out["n_defaults"] == int(y.sum())
    assert {"flag 10%", "flag 20%", "flag 30%"} <= {r["point"] for r in out["operating_points"]}


def test_evaluate_predictions_is_reproducible():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.binomial(1, 0.3, size=500))
    p = pd.Series(rng.random(500))
    assert ev.evaluate_predictions(y, p, n_boot=50, seed=3) == ev.evaluate_predictions(
        y, p, n_boot=50, seed=3
    )
