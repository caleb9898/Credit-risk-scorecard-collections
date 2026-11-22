import numpy as np
import pandas as pd
import pytest

from creditrisk.features import diagnostics as dg


@pytest.mark.parametrize(
    ("iv", "band"),
    [
        (0.0, "useless"),
        (0.0199, "useless"),
        (0.02, "weak"),
        (0.0999, "weak"),
        (0.1, "medium"),
        (0.3, "strong"),
        (0.5, "strong"),
        (0.5001, "suspicious"),
    ],
)
def test_iv_band_thresholds(iv, band):
    assert dg.iv_band(iv) == band


def test_correlated_pairs_reports_only_pairs_above_the_threshold():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    df = pd.DataFrame({"a": a, "b": a + rng.normal(scale=0.1, size=500), "c": rng.normal(size=500)})
    pairs = dg.correlated_pairs(df, threshold=0.7)
    assert list(map(tuple, pairs[["feature_1", "feature_2"]].to_numpy())) == [("a", "b")]
    assert pairs["correlation"].iloc[0] > 0.9


def test_vif_is_about_one_for_independent_columns_and_large_for_collinear_ones():
    rng = np.random.default_rng(0)
    a = rng.normal(size=2000)
    independent = pd.DataFrame({"a": a, "c": rng.normal(size=2000)})
    collinear = independent.assign(b=a + rng.normal(scale=0.1, size=2000))
    assert dg.vif(independent).max() < 1.1
    assert dg.vif(collinear)["a"] > 50
    assert dg.vif(collinear)["c"] < 1.1


def test_preselect_known_answer():
    rng = np.random.default_rng(0)
    a = rng.normal(size=1000)
    woe_values = pd.DataFrame(
        {
            "a": a,
            "b": a + rng.normal(scale=0.2, size=1000),  # correlated with a, lower IV
            "c": rng.normal(size=1000),  # IV too low
            "d": rng.normal(size=1000),
        }
    )
    iv = pd.Series({"a": 0.4, "b": 0.3, "c": 0.01, "d": 0.2})
    out = dg.preselect(iv, woe_values, min_iv=0.02, max_correlation=0.7)
    assert out.loc["a", "kept"] and out.loc["d", "kept"]
    assert not out.loc["b", "kept"] and out.loc["b", "reason"] == "correlated with a"
    assert not out.loc["c", "kept"] and out.loc["c", "reason"] == "IV below 0.02"
    assert list(out.index) == ["a", "b", "d", "c"]  # ordered by IV


def test_fold_stability_on_a_strong_signal():
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-(1.5 * x - 1)))).astype(int))
    X = pd.DataFrame({"x": x, "noise": noise})
    specs = {
        "x": {"dtype": "numerical", "monotonic_trend": "ascending"},
        "noise": {"dtype": "numerical", "monotonic_trend": None},
    }
    folds = [(np.arange(n) % 3 != k, np.arange(n) % 3 == k) for k in range(3)]
    folds = [(np.flatnonzero(t), np.flatnonzero(v)) for t, v in folds]
    out = dg.fold_stability(X, y, folds, specs)
    assert list(out.index) == ["x", "noise"]
    assert (out["n_folds"] == 3).all()
    assert out.loc["x", "woe_rank_corr_min"] > 0.9  # the risk ordering holds out of fold
    assert out.loc["x", "iv_valid_min"] > 0.3
    assert out.loc["noise", "iv_valid_mean"] < out.loc["noise", "iv_fit_mean"]  # optimism
    assert out.loc["noise", "woe_rank_corr_mean"] < out.loc["x", "woe_rank_corr_mean"]


def test_fold_stability_chance_level():
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.normal(size=n)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-(1.5 * x - 1)))).astype(int))
    X = pd.DataFrame({"x": x, "noise": rng.normal(size=n)})
    specs = {
        "x": {"dtype": "numerical", "monotonic_trend": "ascending"},
        "noise": {"dtype": "numerical", "monotonic_trend": None},
    }
    positions = np.arange(n)
    folds = [
        (np.flatnonzero(positions % 3 != k), np.flatnonzero(positions % 3 == k)) for k in range(3)
    ]
    out = dg.fold_stability(X, y, folds, specs, n_permutations=50, seed=1)
    assert out.loc["x", "n_folds_iv_valid_above_null_p95"] == 3
    assert out.loc["noise", "n_folds_iv_valid_above_null_p95"] <= 1
    # Shuffling gives a positive IV (the small-sample bias), and the noise feature's own
    # validation IV is of the same size: it is no better than chance.
    assert out.loc["noise", "iv_null_mean"] > 0
    assert out.loc["noise", "iv_valid_mean"] < 2 * out.loc["noise", "iv_null_mean"]
