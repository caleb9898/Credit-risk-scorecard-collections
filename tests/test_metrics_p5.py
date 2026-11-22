"""Metrics added for the PD model: PR-AUC, calibration by decile and paired bootstrap."""

import numpy as np
import pytest

from creditrisk.validation.metrics import (
    auc,
    calibration_table,
    paired_bootstrap_diff,
    pr_auc,
)


def test_pr_auc_known_answer():
    # Ranked scores: 0.9 (default), 0.8 (no), 0.7 (default), 0.1 (no).
    # Average precision = 0.5 * 1 + 0.5 * 2/3, worked out by hand.
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1])
    assert pr_auc(y, s) == pytest.approx(0.5 + 0.5 * 2 / 3)


def test_pr_auc_of_a_random_score_is_close_to_the_event_rate():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.2, size=20_000)
    assert pr_auc(y, rng.random(20_000)) == pytest.approx(y.mean(), abs=0.01)


def test_calibration_table_groups_by_predicted_pd():
    p = np.arange(1, 21) / 21  # 20 distinct predictions
    y = np.array([0, 1] * 10)
    table = calibration_table(y, p, n_groups=10)
    assert len(table) == 10
    assert (table["n"] == 2).all()
    assert table["mean_pd"].is_monotonic_increasing
    assert table["mean_pd"].iloc[0] == pytest.approx((1 / 21 + 2 / 21) / 2)
    assert (table["default_rate"] == 0.5).all()
    assert table["n"].sum() == 20


def test_paired_bootstrap_of_identical_scores_is_zero():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, size=500)
    s = rng.random(500)
    diff, low, high = paired_bootstrap_diff(auc, y, s, s, n_boot=200, seed=1)
    assert (diff, low, high) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_detects_a_better_score():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, size=3000)
    weak = rng.normal(loc=y * 0.3)
    strong = rng.normal(loc=y * 1.5)
    diff, low, high = paired_bootstrap_diff(auc, y, strong, weak, n_boot=300, seed=1)
    assert diff == pytest.approx(auc(y, strong) - auc(y, weak))
    assert 0 < low < diff < high
