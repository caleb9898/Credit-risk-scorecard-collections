"""Model-validation metrics for binary credit-risk scores.

Convention used everywhere: ``y_true`` is 1 for the adverse event (default) and ``y_score`` is
a risk score where HIGHER means MORE risk (e.g. a predicted PD). Scorecard points, where higher
means safer, must be negated before being passed to these functions.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ArrayLike = np.ndarray | list | tuple


def _prepare(y_true: ArrayLike, y_score: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    s = np.asarray(y_score, dtype=float)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("y_true and y_score must be 1-D arrays of the same length")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must be binary {0, 1}")
    if len(np.unique(y)) < 2:
        raise ValueError("y_true must contain both classes")
    if np.isnan(s).any():
        raise ValueError("y_score contains NaN")
    return y.astype(int), s


def auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """ROC AUC."""
    y, s = _prepare(y_true, y_score)
    return float(roc_auc_score(y, s))


def gini(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Gini coefficient (accuracy ratio) = 2 * AUC - 1."""
    return 2.0 * auc(y_true, y_score) - 1.0


def pr_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the precision-recall curve (average precision).

    A score with no information gets about the event rate (0.22 here), not 0.5.
    """
    y, s = _prepare(y_true, y_score)
    return float(average_precision_score(y, s))


def ks_statistic(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Kolmogorov-Smirnov statistic: max gap between the score CDFs of events and non-events."""
    y, s = _prepare(y_true, y_score)
    return float(stats.ks_2samp(s[y == 1], s[y == 0]).statistic)


def psi(
    expected: ArrayLike,
    actual: ArrayLike,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Population Stability Index of ``actual`` against the ``expected`` (reference) sample.

    Bin edges are the reference sample's quantiles (duplicates collapsed, open-ended outer bins),
    so the result does not depend on the actual sample. Bin shares are floored at ``eps`` to
    keep the logarithm finite. Rule of thumb: <0.10 stable, 0.10-0.25 moderate shift, >0.25 major.
    """
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    if e.size == 0 or a.size == 0:
        raise ValueError("expected and actual must be non-empty")
    if np.isnan(e).any() or np.isnan(a).any():
        raise ValueError("inputs contain NaN")

    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:  # constant reference sample
        return 0.0 if np.all(a == e[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf

    e_share = np.histogram(e, bins=edges)[0] / e.size
    a_share = np.histogram(a, bins=edges)[0] / a.size
    e_share = np.clip(e_share, eps, None)
    a_share = np.clip(a_share, eps, None)
    return float(np.sum((a_share - e_share) * np.log(a_share / e_share)))


def confusion_at_threshold(
    y_true: ArrayLike, y_score: ArrayLike, threshold: float
) -> dict[str, float]:
    """Operating-point metrics when flagging ``y_score >= threshold`` as high risk."""
    y, s = _prepare(y_true, y_score)
    flag = s >= threshold
    tp = int(np.sum(flag & (y == 1)))
    fp = int(np.sum(flag & (y == 0)))
    fn = int(np.sum(~flag & (y == 1)))
    tn = int(np.sum(~flag & (y == 0)))
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": tp / (tp + fn),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "false_positive_rate": fp / (fp + tn),
        "flag_rate": (tp + fp) / y.size,
    }


def calibration_summary(y_true: ArrayLike, p_pred: ArrayLike) -> dict[str, float]:
    """Brier score plus calibration intercept and slope.

    Slope and intercept come from regressing the outcome on logit(p). A well-calibrated model has
    slope ~1 and intercept ~0; slope < 1 signals over-confident predictions.
    """
    y, p = _prepare(y_true, p_pred)
    if p.min() < 0 or p.max() > 1:
        raise ValueError("p_pred must be probabilities in [0, 1]")
    p_clipped = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p_clipped / (1 - p_clipped))
    fit = sm.Logit(y, sm.add_constant(logit)).fit(disp=0)  # unpenalised maximum likelihood
    return {
        "brier": float(brier_score_loss(y, p)),
        "calibration_intercept": float(fit.params[0]),
        "calibration_slope": float(fit.params[1]),
    }


def bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true: ArrayLike,
    y_score: ArrayLike,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Stratified percentile bootstrap CI. Returns (point_estimate, lower, upper).

    Resampling is done within each class so every replicate keeps the original event rate.
    """
    y, s = _prepare(y_true, y_score)
    rng = np.random.default_rng(seed)
    idx_pos = np.flatnonzero(y == 1)
    idx_neg = np.flatnonzero(y == 0)
    point = float(metric(y, s))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(idx_pos, size=idx_pos.size, replace=True),
                rng.choice(idx_neg, size=idx_neg.size, replace=True),
            ]
        )
        draws[b] = metric(y[idx], s[idx])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def calibration_table(y_true: ArrayLike, p_pred: ArrayLike, n_groups: int = 10) -> pd.DataFrame:
    """Mean predicted PD and observed default rate in groups of equal size, by predicted PD.

    Rows are sorted by prediction (ties broken by position) and cut into ``n_groups`` groups, so
    every group has the same number of rows up to one. This is the table behind a reliability plot.
    """
    y, p = _prepare(y_true, p_pred)
    order = pd.Series(p).rank(method="first")
    group = pd.qcut(order, n_groups, labels=False) + 1
    data = pd.DataFrame({"group": group, "y": y, "p": p})
    table = data.groupby("group").agg(
        n=("y", "size"), mean_pd=("p", "mean"), default_rate=("y", "mean")
    )
    return table.reset_index()


def paired_bootstrap_diff(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true: ArrayLike,
    score_a: ArrayLike,
    score_b: ArrayLike,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Difference metric(a) - metric(b) on the same rows, with a paired bootstrap CI.

    Both scores are recomputed on the same resampled rows in every replicate, so the interval
    reflects the difference itself and not the uncertainty of each metric separately. Resampling
    is stratified by class, as in ``bootstrap_ci``. Returns (difference, lower, upper).
    """
    y, a = _prepare(y_true, score_a)
    _, b = _prepare(y_true, score_b)
    rng = np.random.default_rng(seed)
    idx_pos = np.flatnonzero(y == 1)
    idx_neg = np.flatnonzero(y == 0)
    point = float(metric(y, a) - metric(y, b))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(idx_pos, size=idx_pos.size, replace=True),
                rng.choice(idx_neg, size=idx_neg.size, replace=True),
            ]
        )
        draws[i] = metric(y[idx], a[idx]) - metric(y[idx], b[idx])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)
