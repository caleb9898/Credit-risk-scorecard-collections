import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from creditrisk.validation.metrics import (
    auc,
    bootstrap_ci,
    calibration_summary,
    confusion_at_threshold,
    gini,
    ks_statistic,
    psi,
)


@pytest.fixture
def separable():
    y = np.array([0] * 50 + [1] * 50)
    s = np.concatenate([np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    return y, s


@pytest.fixture
def noisy():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.2, size=5000)
    s = rng.normal(loc=y * 1.0, scale=1.0)
    return y, s


def test_perfect_separation_gives_ks_and_gini_one(separable):
    y, s = separable
    assert ks_statistic(y, s) == pytest.approx(1.0)
    assert gini(y, s) == pytest.approx(1.0)
    assert auc(y, s) == pytest.approx(1.0)


def test_random_score_is_near_chance():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, size=20000)
    s = rng.random(20000)
    assert auc(y, s) == pytest.approx(0.5, abs=0.02)
    assert abs(gini(y, s)) < 0.04
    assert ks_statistic(y, s) < 0.03


def test_gini_equals_two_auc_minus_one(noisy):
    y, s = noisy
    assert gini(y, s) == pytest.approx(2 * roc_auc_score(y, s) - 1)


def test_inverted_score_flips_auc(noisy):
    y, s = noisy
    assert auc(y, s) + auc(y, -s) == pytest.approx(1.0)


def test_ks_known_value():
    # events: scores 3,4 ; non-events: scores 1,2 -> complete separation, KS = 1
    assert ks_statistic([0, 0, 1, 1], [1, 2, 3, 4]) == pytest.approx(1.0)
    # one overlap: events 2,4 ; non-events 1,3 -> max CDF gap is 0.5
    assert ks_statistic([0, 1, 0, 1], [1, 2, 3, 4]) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "y,s",
    [
        ([0, 0, 0], [0.1, 0.2, 0.3]),  # single class
        ([0, 2, 1], [0.1, 0.2, 0.3]),  # non-binary
        ([0, 1], [0.1, 0.2, 0.3]),  # length mismatch
        ([0, 1], [0.1, np.nan]),  # NaN
    ],
)
def test_invalid_inputs_raise(y, s):
    with pytest.raises(ValueError):
        ks_statistic(y, s)


def test_psi_identical_is_zero():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_detects_shift_and_is_nonnegative():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=20000)
    same = rng.normal(size=20000)
    shifted = rng.normal(loc=1.0, size=20000)
    assert 0 <= psi(ref, same) < 0.02
    assert psi(ref, shifted) > 0.25


def test_psi_handles_discrete_scores_with_ties():
    ref = np.repeat([1, 2, 3], 1000)
    assert psi(ref, ref) == pytest.approx(0.0, abs=1e-9)
    assert psi(ref, np.repeat([1, 2, 3], [100, 100, 1800])) > 0.25


def test_psi_constant_reference():
    assert psi(np.ones(10), np.ones(10)) == 0.0
    assert psi(np.ones(10), np.arange(10)) == float("inf")


def test_confusion_at_threshold_counts():
    y = [1, 1, 0, 0, 1, 0]
    s = [0.9, 0.4, 0.8, 0.1, 0.7, 0.2]
    c = confusion_at_threshold(y, s, 0.5)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 1, 2)
    assert c["recall"] == pytest.approx(2 / 3)
    assert c["precision"] == pytest.approx(2 / 3)
    assert c["flag_rate"] == pytest.approx(3 / 6)


def test_calibration_of_a_calibrated_model():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.02, 0.6, size=60000)
    y = rng.binomial(1, p)
    cal = calibration_summary(y, p)
    assert cal["calibration_slope"] == pytest.approx(1.0, abs=0.05)
    assert cal["calibration_intercept"] == pytest.approx(0.0, abs=0.05)


def test_calibration_detects_overconfidence():
    rng = np.random.default_rng(4)
    true_p = rng.uniform(0.1, 0.5, size=60000)
    y = rng.binomial(1, true_p)
    logit = np.log(true_p / (1 - true_p))
    overconfident = 1 / (1 + np.exp(-2.5 * logit))  # stretched predictions
    assert calibration_summary(y, overconfident)["calibration_slope"] < 0.6


def test_calibration_rejects_non_probabilities():
    with pytest.raises(ValueError):
        calibration_summary([0, 1], [0.2, 1.4])


def test_bootstrap_ci_contains_point_and_is_reproducible(noisy):
    y, s = noisy
    p1, lo1, hi1 = bootstrap_ci(auc, y, s, n_boot=200, seed=7)
    p2, lo2, hi2 = bootstrap_ci(auc, y, s, n_boot=200, seed=7)
    assert (p1, lo1, hi1) == (p2, lo2, hi2)
    assert lo1 <= p1 <= hi1


def test_bootstrap_ci_narrows_with_sample_size():
    rng = np.random.default_rng(5)

    def draw(n):
        y = rng.binomial(1, 0.25, size=n)
        return y, rng.normal(loc=y * 0.8, scale=1.0)

    _, lo_s, hi_s = bootstrap_ci(auc, *draw(400), n_boot=300, seed=1)
    _, lo_l, hi_l = bootstrap_ci(auc, *draw(16000), n_boot=300, seed=1)
    assert (hi_l - lo_l) < (hi_s - lo_s)
