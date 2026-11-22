"""Discrimination, calibration and operating points of a set of predicted PDs (ADR-012).

Used both for the out-of-fold predictions on train and for the final evaluation on test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditrisk.validation.metrics import (
    auc,
    bootstrap_ci,
    calibration_summary,
    calibration_table,
    confusion_at_threshold,
    gini,
    ks_statistic,
    pr_auc,
)

METRICS = {"auc": auc, "gini": gini, "ks": ks_statistic, "pr_auc": pr_auc}


def operating_points(y, p, flag_rates=(0.1, 0.2, 0.3), thresholds=(0.5,)) -> pd.DataFrame:
    """Recall, precision and flag rate at chosen operating points.

    - ``flag rates``: flag the given share of accounts with the highest PD (the threshold is the
      PD of the last account flagged; ties at that PD are flagged too).
    - ``thresholds``: flag every account with PD at or above the value.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    ranked = np.sort(p)[::-1]
    points = {}
    for rate in flag_rates:
        k = max(1, int(round(rate * len(p))))
        points[f"flag {rate:.0%}"] = ranked[k - 1]
    for threshold in thresholds:
        points[f"PD >= {threshold:g}"] = threshold
    rows = {name: confusion_at_threshold(y, p, t) for name, t in points.items()}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "point"
    return table


def _records(table: pd.DataFrame) -> list[dict]:
    """Rows of a table as JSON-friendly dicts, writing missing values as null."""
    return table.astype(object).where(table.notna(), None).to_dict(orient="records")


def evaluate_predictions(
    y,
    p,
    n_boot: int = 1000,
    seed: int = 0,
    flag_rates=(0.1, 0.2, 0.3),
    thresholds=(0.5,),
) -> dict:
    """All the validation measures of ``docs/RIGOR_STANDARDS.md`` section 5, as a JSON-ready dict.

    AUC, Gini, KS and PR-AUC come with a 95% stratified bootstrap interval of ``n_boot``
    replicates; calibration comes as Brier, intercept and slope, and as a table by decile.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    out = {"n_rows": int(len(y)), "n_defaults": int(y.sum()), "default_rate": float(y.mean())}
    for name, metric in METRICS.items():
        estimate, low, high = bootstrap_ci(metric, y, p, n_boot=n_boot, seed=seed)
        out[name] = {"estimate": estimate, "low": low, "high": high}
    out["mean_pd"] = float(p.mean())
    out["calibration"] = calibration_summary(y, p)
    out["calibration_by_decile"] = _records(calibration_table(y, p))
    points = operating_points(y, p, flag_rates, thresholds).reset_index()
    out["operating_points"] = _records(points)
    return out
