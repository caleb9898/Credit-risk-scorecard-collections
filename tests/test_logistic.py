import numpy as np
import pandas as pd
import pytest

from creditrisk.models import logistic as lg


def _simulated(n=20_000, seed=0):
    """Logistic data with known coefficients: intercept -1, a = -1.0, b = -0.5."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    logit = -1.0 - 1.0 * X["a"] - 0.5 * X["b"]
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X, y


def test_fit_logit_recovers_known_coefficients():
    X, y = _simulated()
    result = lg.fit_logit(X, y)
    assert result.params["const"] == pytest.approx(-1.0, abs=0.06)
    assert result.params["a"] == pytest.approx(-1.0, abs=0.06)
    assert result.params["b"] == pytest.approx(-0.5, abs=0.06)


def test_coefficient_table_has_intervals_around_the_estimate():
    X, y = _simulated()
    table = lg.coefficient_table(lg.fit_logit(X, y))
    assert list(table.index) == ["const", "a", "b"]
    assert list(table.columns) == ["coef", "std_err", "z", "p_value", "ci_low", "ci_high"]
    assert (table["ci_low"] < table["coef"]).all() and (table["coef"] < table["ci_high"]).all()
    assert table.loc["a", "p_value"] < 1e-10


def test_select_features_drops_a_wrong_sign_first():
    X, _ = _simulated()
    rng = np.random.default_rng(1)
    # "wrong" has a true positive coefficient: with WoE inputs that is a wrong sign.
    wrong = rng.normal(size=len(X))
    logit = -1.0 - 1.0 * X["a"] - 0.5 * X["b"] + 0.4 * wrong
    y = pd.Series((rng.random(len(X)) < 1 / (1 + np.exp(-logit))).astype(int))
    result, kept, dropped = lg.select_features(X.assign(wrong=wrong), y)
    assert kept == ["a", "b"]
    assert dropped == [("wrong", "wrong sign")]
    assert set(result.params.index) == {"const", "a", "b"}


def test_select_features_drops_a_variable_without_evidence():
    # Pure noise whose fitted coefficient happens to be negative (-0.018, p = 0.30 with this
    # fixed draw), so it is removed by the p-value rule and not by the sign rule.
    X, y = _simulated()
    noise = np.random.default_rng(100).normal(size=len(X))
    _, kept, dropped = lg.select_features(X.assign(noise=noise), y, max_p=0.05)
    assert kept == ["a", "b"]
    assert dropped == [("noise", "p-value above 0.05")]


def test_select_features_keeps_everything_when_nothing_is_wrong():
    X, y = _simulated()
    _, kept, dropped = lg.select_features(X, y)
    assert kept == ["a", "b"]
    assert dropped == []
