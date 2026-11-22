"""Weight-of-evidence binning (ADR-005, ADR-011).

``optbinning`` finds the bin edges (or the groups of categories) under the constraints of each
feature: minimum bin size and, where declared, a monotonic default rate. Assigning rows to bins
and computing WoE and IV is done here, in plain pandas, so that the calculation is visible and
can be checked against optbinning's own IV.

Convention (the usual one in credit scoring):

    WoE(bin) = ln( share of non-defaults in the bin / share of defaults in the bin )

so a riskier-than-average bin has a negative WoE. IV = sum over bins of
(share of non-defaults - share of defaults) * WoE.

Missing values get their own bin. A category or a missing value that was not seen when fitting
gets WoE 0, the value of an average-risk bin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from optbinning import OptimalBinning
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

MISSING_BIN = "missing"


def woe_table(bins, y, smoothing: float = 0.0) -> pd.DataFrame:
    """Counts, default rate, WoE and IV contribution of each bin.

    ``bins`` holds the bin label of every row and ``y`` the target (1 = default). ``smoothing`` is
    added to the default and non-default counts of every bin; 0 gives the plain formula (and an
    infinite WoE for a bin without defaults or without non-defaults). The rows of the table are in
    order of first appearance.
    """
    data = pd.DataFrame({"bin": np.asarray(bins, dtype=object), "y": np.asarray(y)})
    grouped = data.groupby("bin", sort=False)["y"]
    n = grouped.size()
    events = grouped.sum()
    non_events = n - events
    share_events = (events + smoothing) / (events + smoothing).sum()
    share_non_events = (non_events + smoothing) / (non_events + smoothing).sum()
    with np.errstate(divide="ignore"):
        woe = np.log(share_non_events / share_events)
    table = pd.DataFrame(
        {
            "n": n,
            "events": events,
            "event_rate": events / n,
            "woe": woe,
            "iv": (share_non_events - share_events) * woe,
        }
    )
    table.index.name = "bin"
    return table


def information_value(bins, y, smoothing: float = 0.0) -> float:
    """Information value of a binned feature (sum of the IV contributions of its bins)."""
    return float(woe_table(bins, y, smoothing)["iv"].sum())


def _interval_labels(edges: np.ndarray) -> list[str]:
    """Labels for the bins defined by ``edges``: (-inf, e1), [e1, e2), ..., [ek, inf)."""
    bounds = [-np.inf, *edges, np.inf]
    labels = []
    for low, high in zip(bounds[:-1], bounds[1:], strict=True):
        opening = "(" if np.isinf(low) else "["
        labels.append(f"{opening}{low:.6g}, {high:.6g})")
    return labels


class WoeBinner(BaseEstimator, TransformerMixin):
    """Fit one binning per feature and replace every value with the WoE of its bin.

    Parameters
    ----------
    specs:
        ``{feature: {"dtype": "numerical" | "categorical", "monotonic_trend": ...}}``, where the
        trend is the direction of the default rate ("ascending", "descending" or None).
    min_bin_size:
        Smallest share of the fitting rows a bin may hold (missing values excepted).
    """

    def __init__(self, specs: dict[str, dict], min_bin_size: float = 0.05):
        self.specs = specs
        self.min_bin_size = min_bin_size

    def fit(self, X: pd.DataFrame, y) -> WoeBinner:
        y = np.asarray(y)
        self.bins_ = {}
        self.woe_ = {}
        self.tables_ = {}
        self.optbinning_iv_ = {}
        for name, spec in self.specs.items():
            values = X[name]
            binning = OptimalBinning(
                name=name,
                dtype=spec["dtype"],
                monotonic_trend=spec["monotonic_trend"],
                min_bin_size=self.min_bin_size,
            )
            binning.fit(self._prepare(values, spec), y)
            self.optbinning_iv_[name] = float(binning.binning_table.build().loc["Totals", "IV"])

            if spec["dtype"] == "numerical":
                edges = np.asarray(binning.splits, dtype=float)
                self.bins_[name] = {"edges": edges, "labels": _interval_labels(edges)}
            else:
                groups = [[str(c) for c in group] for group in binning.splits]
                self.bins_[name] = {
                    "groups": {c: ", ".join(group) for group in groups for c in group},
                    "labels": [", ".join(group) for group in groups],
                }
            table = woe_table(self._labels(name, values), y)
            # Bins in their natural order, missing last; empty bins are left out.
            order = [b for b in [*self.bins_[name]["labels"], MISSING_BIN] if b in table.index]
            self.tables_[name] = table.loc[order]
            self.woe_[name] = table["woe"].to_dict()
        return self

    @staticmethod
    def _prepare(values: pd.Series, spec: dict) -> np.ndarray:
        """Values in the form optbinning expects: floats, or strings with NaN kept as missing."""
        if spec["dtype"] == "numerical":
            return values.to_numpy(dtype=float)
        return values.map(str, na_action="ignore").to_numpy(dtype=object)

    def _labels(self, name: str, values: pd.Series) -> pd.Series:
        """Bin label of every value (None for a category not seen when fitting)."""
        bins = self.bins_[name]
        if self.specs[name]["dtype"] == "numerical":
            numbers = values.to_numpy(dtype=float)
            positions = np.searchsorted(bins["edges"], numbers, side="right")
            labels = np.asarray(bins["labels"], dtype=object)[positions]
        else:
            labels = values.map(str, na_action="ignore").map(bins["groups"]).to_numpy(dtype=object)
        labels = np.where(values.isna().to_numpy(), MISSING_BIN, labels)
        return pd.Series(labels, index=values.index, name=name)

    def assign_bins(self, X: pd.DataFrame) -> pd.DataFrame:
        """Bin label of every value of every feature, with the index of ``X``."""
        check_is_fitted(self, "bins_")
        return pd.DataFrame({name: self._labels(name, X[name]) for name in self.specs})

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace every value with the WoE of its bin (0 for bins not seen when fitting)."""
        bins = self.assign_bins(X)
        return pd.DataFrame(
            {name: bins[name].map(self.woe_[name]).fillna(0.0).astype(float) for name in self.specs}
        )

    def binning_table(self, name: str) -> pd.DataFrame:
        """WoE table of one feature on the fitting rows, bins in order (missing last)."""
        check_is_fitted(self, "bins_")
        return self.tables_[name]

    def iv_table(self) -> pd.DataFrame:
        """IV and number of bins of every feature, on the fitting rows."""
        check_is_fitted(self, "bins_")
        rows = {
            name: {
                "iv": float(table["iv"].sum()),
                "n_bins": int((table.index != MISSING_BIN).sum()),
                "has_missing_bin": bool(MISSING_BIN in table.index),
            }
            for name, table in self.tables_.items()
        }
        table = pd.DataFrame.from_dict(rows, orient="index")
        table.index.name = "feature"
        return table
