"""The whole PD model as one chain: binning -> preselection -> selection -> logistic regression.

Every step that learns from data is inside ``WoeLogitModel.fit``, so ``cross_validate`` can refit
the full chain in each fold and measure it on rows that none of its steps has seen (ADR-012).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from creditrisk.features.diagnostics import preselect
from creditrisk.features.woe import WoeBinner
from creditrisk.models.logistic import select_features
from creditrisk.validation.metrics import auc, ks_statistic


class WoeLogitModel(BaseEstimator):
    """WoE binning, IV/correlation preselection and a logistic regression with sign checks.

    Parameters are the ones of ADR-011 and ADR-012: minimum bin size 5%, IV at least 0.02, WoE
    correlation at most 0.7 and p-values at most 0.05.
    """

    def __init__(
        self,
        specs: dict[str, dict],
        min_bin_size: float = 0.05,
        min_iv: float = 0.02,
        max_correlation: float = 0.7,
        max_p: float = 0.05,
    ):
        self.specs = specs
        self.min_bin_size = min_bin_size
        self.min_iv = min_iv
        self.max_correlation = max_correlation
        self.max_p = max_p

    def fit(self, X: pd.DataFrame, y) -> WoeLogitModel:
        self.binner_ = WoeBinner(self.specs, min_bin_size=self.min_bin_size).fit(X, y)
        woe = self.binner_.transform(X)
        iv = self.binner_.iv_table()["iv"]
        self.preselection_ = preselect(iv, woe, self.min_iv, self.max_correlation)
        shortlist = list(self.preselection_.index[self.preselection_["kept"]])
        self.result_, self.features_, self.dropped_ = select_features(woe[shortlist], y, self.max_p)
        return self

    def predict_pd(self, X: pd.DataFrame) -> pd.Series:
        """Predicted probability of default for every row of ``X``."""
        check_is_fitted(self, "result_")
        woe = self.binner_.transform(X)[self.features_]
        p = self.result_.predict(sm.add_constant(woe, has_constant="add"))
        return pd.Series(np.asarray(p), index=X.index, name="pd")

    def selection_summary(self) -> pd.DataFrame:
        """IV of every candidate and what happened to it: in the model, or why it was left out."""
        check_is_fitted(self, "result_")
        status = self.preselection_["reason"].replace({"kept": "in model"})
        for name, reason in self.dropped_:
            status[name] = f"logistic: {reason}"
        return pd.DataFrame({"iv": self.preselection_["iv"], "status": status})


def cross_validate(
    specs: dict[str, dict],
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_folds: int,
    **model_params,
) -> dict:
    """Refit the whole model in every fold and predict the rows it left out.

    ``folds`` holds (fitting rows, validation rows) as positions in ``X``, ``n_folds`` folds per
    repeat, in the order produced by ``creditrisk.data.split.make_cv``. Returns:

    - ``oof``: out-of-fold predicted PD, one column per repeat (each row predicted once per repeat);
    - ``folds``: AUC, KS and selected variables of every fold;
    - ``selection_counts``: in how many folds each candidate ended up in the model.
    """
    n_repeats = len(folds) // n_folds
    oof = pd.DataFrame(np.nan, index=X.index, columns=range(1, n_repeats + 1))
    per_fold = []
    counts: Counter[str] = Counter()
    for i, (fit_rows, valid_rows) in enumerate(folds):
        repeat = i // n_folds + 1
        model = WoeLogitModel(specs, **model_params).fit(X.iloc[fit_rows], y.iloc[fit_rows])
        p = model.predict_pd(X.iloc[valid_rows])
        oof.iloc[valid_rows, repeat - 1] = p.to_numpy()
        y_valid = y.iloc[valid_rows]
        per_fold.append(
            {
                "fold": i + 1,
                "repeat": repeat,
                "auc": auc(y_valid, p),
                "ks": ks_statistic(y_valid, p),
                "features": list(model.features_),
            }
        )
        counts.update(model.features_)
    selection_counts = pd.Series(counts, dtype=int).reindex(list(specs), fill_value=0)
    return {"oof": oof, "folds": per_fold, "selection_counts": selection_counts}
