import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from creditrisk.features import woe


def _risk_data(n=4000, seed=0):
    """Synthetic data where default risk rises with x, falls with z and depends on a category."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.uniform(0, 1, size=n)
    cat = rng.choice(["a", "b", "c"], size=n)
    logit = -1.2 + 0.9 * x - 1.0 * z + np.where(cat == "a", 0.8, 0.0)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    z[:300] = np.nan  # a missing group with its own risk
    y[:300] = (rng.random(300) < 0.6).astype(int)
    X = pd.DataFrame({"x": x, "z": z, "cat": cat})
    return X, pd.Series(y, name="DEFAULT")


SPECS = {
    "x": {"dtype": "numerical", "monotonic_trend": "ascending"},
    "z": {"dtype": "numerical", "monotonic_trend": "descending"},
    "cat": {"dtype": "categorical", "monotonic_trend": None},
}


# --- woe_table: the independent calculation -----------------------------------------------------


def test_woe_table_known_answer():
    # Bin A: 10 defaults, 90 non-defaults; bin B: 30 and 70. Worked out by hand:
    # WoE = ln(share of non-defaults / share of defaults) -> A: ln(0.5625/0.25), B: ln(0.4375/0.75)
    bins = np.array(["A"] * 100 + ["B"] * 100)
    y = np.array([1] * 10 + [0] * 90 + [1] * 30 + [0] * 70)
    table = woe.woe_table(bins, y)
    assert table.loc["A", "woe"] == pytest.approx(0.8109302, abs=1e-6)
    assert table.loc["B", "woe"] == pytest.approx(-0.5389965, abs=1e-6)
    assert table["iv"].sum() == pytest.approx(0.4218521, abs=1e-6)
    assert woe.information_value(bins, y) == pytest.approx(0.4218521, abs=1e-6)


def test_woe_is_negative_for_the_riskier_bin():
    bins = np.array(["low"] * 100 + ["high"] * 100)
    y = np.array([1] * 5 + [0] * 95 + [1] * 50 + [0] * 50)
    table = woe.woe_table(bins, y)
    assert table.loc["high", "woe"] < 0 < table.loc["low", "woe"]


def test_woe_table_smoothing_known_answer():
    # Bin A: 0 defaults, 50 non-defaults; bin B: 25 and 25. With 0.5 added to every count:
    # A: ln((50.5/76) / (0.5/26)), B: ln((25.5/76) / (25.5/26)), worked out by hand.
    bins = np.array(["A"] * 50 + ["B"] * 50)
    y = np.array([0] * 50 + [1] * 25 + [0] * 25)
    table = woe.woe_table(bins, y, smoothing=0.5)
    assert table.loc["A", "woe"] == pytest.approx(3.5425, abs=1e-4)
    assert table.loc["B", "woe"] == pytest.approx(-1.0726, abs=1e-4)


def test_fit_refuses_a_bin_with_an_infinite_woe():
    X, y = _risk_data()
    y = y.copy()
    y[X["z"].isna().to_numpy()] = 1  # every missing value is a default
    with pytest.raises(ValueError, match="infinite"):
        woe.WoeBinner(SPECS).fit(X, y)


def test_woe_table_smoothing_handles_a_bin_without_defaults():
    bins = np.array(["A"] * 50 + ["B"] * 50)
    y = np.array([0] * 50 + [1] * 25 + [0] * 25)
    assert np.isinf(woe.woe_table(bins, y).loc["A", "woe"])
    assert np.isfinite(woe.woe_table(bins, y, smoothing=0.5).loc["A", "woe"])


# --- WoeBinner ---------------------------------------------------------------------------------


def test_transform_before_fit_raises():
    X, _ = _risk_data(n=500)
    with pytest.raises(NotFittedError):
        woe.WoeBinner(SPECS).transform(X)


def test_iv_matches_the_independent_calculation():
    # Cross-check: optbinning's own IV equals ours on the same bins.
    X, y = _risk_data()
    binner = woe.WoeBinner(SPECS).fit(X, y)
    bins = binner.assign_bins(X)
    for name in SPECS:
        ours = woe.information_value(bins[name], y)
        assert binner.optbinning_iv_[name] == pytest.approx(ours, rel=1e-6), name
        assert binner.iv_table().loc[name, "iv"] == pytest.approx(ours, rel=1e-6), name


def test_bins_respect_the_minimum_size_and_the_declared_trend():
    X, y = _risk_data()
    binner = woe.WoeBinner(SPECS, min_bin_size=0.05).fit(X, y)
    for name in SPECS:
        table = binner.binning_table(name)
        regular = table[table.index != woe.MISSING_BIN]
        assert (regular["n"] >= 0.05 * len(X)).all(), name
    x_rates = binner.binning_table("x").drop(woe.MISSING_BIN, errors="ignore")["event_rate"]
    z_rates = binner.binning_table("z").drop(woe.MISSING_BIN, errors="ignore")["event_rate"]
    assert x_rates.is_monotonic_increasing
    assert z_rates.is_monotonic_decreasing


def test_missing_values_get_their_own_bin_and_woe():
    X, y = _risk_data()
    binner = woe.WoeBinner(SPECS).fit(X, y)
    table = binner.binning_table("z")
    assert table.loc[woe.MISSING_BIN, "n"] == 300
    out = binner.transform(X)
    missing_rows = X["z"].isna()
    assert out.loc[missing_rows, "z"].nunique() == 1
    assert out.loc[missing_rows, "z"].iloc[0] == pytest.approx(table.loc[woe.MISSING_BIN, "woe"])
    assert out.loc[missing_rows, "z"].iloc[0] < 0  # the missing group is riskier than average


def test_transform_keeps_index_and_gives_finite_values():
    X, y = _risk_data()
    X.index = X.index + 1000
    out = woe.WoeBinner(SPECS).fit(X, y).transform(X)
    assert list(out.index) == list(X.index)
    assert list(out.columns) == list(SPECS)
    assert np.isfinite(out.to_numpy()).all()


def test_values_outside_training_range_and_unseen_categories():
    X, y = _risk_data()
    binner = woe.WoeBinner(SPECS).fit(X, y)
    new = pd.DataFrame({"x": [1e6, -1e6], "z": [0.5, 0.5], "cat": ["a", "zzz"]})
    out = binner.transform(new)
    x_table = binner.binning_table("x").drop(woe.MISSING_BIN, errors="ignore")
    assert out.loc[0, "x"] == pytest.approx(x_table["woe"].iloc[-1])  # last bin
    assert out.loc[1, "x"] == pytest.approx(x_table["woe"].iloc[0])  # first bin
    assert out.loc[1, "cat"] == 0.0  # unseen category: neutral WoE, i.e. average risk


def test_missing_in_new_data_without_missing_in_training_is_neutral():
    X, y = _risk_data()
    binner = woe.WoeBinner(SPECS).fit(X, y)
    new = pd.DataFrame({"x": [np.nan], "z": [0.5], "cat": ["a"]})
    assert binner.transform(new).loc[0, "x"] == 0.0


def test_fit_uses_only_the_rows_it_is_given():
    # Refitting on the same rows gives the same bins, whatever other rows look like.
    X, y = _risk_data()
    first = woe.WoeBinner(SPECS).fit(X.iloc[:2000], y.iloc[:2000])
    X_changed = X.copy()
    X_changed.iloc[2000:] = X_changed.iloc[2000:].sample(frac=1, random_state=1).to_numpy()
    second = woe.WoeBinner(SPECS).fit(X_changed.iloc[:2000], y.iloc[:2000])
    assert first.iv_table().equals(second.iv_table())
