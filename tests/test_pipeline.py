import numpy as np
import pandas as pd
import pytest

from creditrisk.models import pipeline as pl
from creditrisk.validation.metrics import auc

SPECS = {
    "x": {"dtype": "numerical", "monotonic_trend": "ascending"},
    "z": {"dtype": "numerical", "monotonic_trend": "descending"},
    "noise": {"dtype": "numerical", "monotonic_trend": None},
    "cat": {"dtype": "categorical", "monotonic_trend": None},
}


def _data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "z": rng.uniform(size=n),
            "noise": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
        },
        index=np.arange(n) * 3,  # a non-contiguous index, like the training rows
    )
    logit = -1.2 + 1.0 * X["x"] - 1.0 * X["z"] + np.where(X["cat"] == "a", 0.8, 0.0)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int), index=X.index)
    return X, y


def _folds(n, k=4, repeats=2):
    rng = np.random.default_rng(0)
    folds = []
    for _ in range(repeats):
        order = rng.permutation(n)
        for part in np.array_split(order, k):
            folds.append((np.setdiff1d(np.arange(n), part), np.sort(part)))
    return folds


def test_model_predicts_probabilities_for_every_row():
    X, y = _data()
    model = pl.WoeLogitModel(SPECS).fit(X, y)
    pd_hat = model.predict_pd(X)
    assert list(pd_hat.index) == list(X.index)
    assert ((pd_hat > 0) & (pd_hat < 1)).all()
    assert auc(y, pd_hat) > 0.7
    assert set(model.features_) <= set(SPECS)
    assert "x" in model.features_


def test_model_records_why_each_candidate_was_left_out():
    X, y = _data()
    model = pl.WoeLogitModel(SPECS).fit(X, y)
    reasons = model.selection_summary()
    assert set(reasons.index) == set(SPECS)
    assert (reasons.loc[model.features_, "status"] == "in model").all()


def test_predict_before_fit_raises():
    X, _ = _data(n=200)
    with pytest.raises(Exception, match="not fitted"):
        pl.WoeLogitModel(SPECS).predict_pd(X)


def test_cross_validation_predicts_every_row_once_per_repeat():
    X, y = _data()
    folds = _folds(len(X), k=4, repeats=2)
    cv = pl.cross_validate(SPECS, X, y, folds, n_folds=4)
    oof = cv["oof"]
    assert list(oof.columns) == [1, 2]
    assert list(oof.index) == list(X.index)
    assert oof.notna().all().all()
    assert len(cv["folds"]) == 8
    assert set(cv["selection_counts"].index) == set(SPECS)
    assert cv["selection_counts"].max() <= 8


def test_cross_validation_never_fits_on_validation_rows(monkeypatch):
    # Spy on every binning fit: in each fold it must see only that fold's fitting rows.
    seen = []
    original_fit = pl.WoeBinner.fit

    def spy_fit(self, X, y):
        seen.append(set(X.index))
        return original_fit(self, X, y)

    monkeypatch.setattr(pl.WoeBinner, "fit", spy_fit)
    X, y = _data()
    folds = _folds(len(X), k=4, repeats=1)
    pl.cross_validate(SPECS, X, y, folds, n_folds=4)
    assert len(seen) == 4
    for fitted_rows, (_, valid) in zip(seen, folds, strict=True):
        assert fitted_rows.isdisjoint(set(X.index[valid]))
