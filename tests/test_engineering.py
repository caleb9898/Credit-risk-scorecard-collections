import numpy as np
import pandas as pd
import pytest

from creditrisk.data import schema as sc
from creditrisk.features import engineering as fe


def _frame(rows):
    """Canonical frame from partial row dicts; unspecified columns get harmless defaults."""
    defaults = dict.fromkeys(sc.CANONICAL_COLUMNS, 0)
    defaults.update({"LIMIT_BAL": 1000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30})
    defaults.update(dict.fromkeys(sc.PAY_COLS, -1))
    return pd.DataFrame([{**defaults, **row} for row in rows])[sc.CANONICAL_COLUMNS]


def test_output_has_every_declared_feature_and_nothing_else():
    out = fe.build_features(_frame([{}, {}]))
    assert list(out.columns) == list(fe.FEATURE_SPECS)
    assert "DEFAULT" not in out and "SEX" not in out and "MARRIAGE" not in out


def test_output_keeps_the_row_index():
    df = _frame([{}, {}, {}])
    df.index = [7, 3, 11]  # e.g. the training rows, indexed by position in the raw file
    assert list(fe.build_features(df).index) == [7, 3, 11]


def test_utilisation_floors_negative_bills_at_zero():
    df = _frame([{"BILL_AMT1": -50, "BILL_AMT2": 500, "BILL_AMT3": 1500, "BILL_AMT6": 200}])
    out = fe.build_features(df).iloc[0]
    assert out["UTIL_1"] == 0.0
    assert out["UTIL_2"] == 0.5
    assert out["UTIL_3"] == 1.5  # over the limit is kept
    assert out["UTIL_MAX"] == 1.5
    assert out["UTIL_MEAN"] == pytest.approx((0 + 0.5 + 1.5 + 0 + 0 + 0.2) / 6)
    assert out["UTIL_TREND"] == pytest.approx(0.0 - 0.2)  # September minus April


def test_payment_ratio_uses_the_previous_statement():
    df = _frame(
        [
            {"PAY_AMT1": 250, "BILL_AMT2": 500},  # paid half of the August statement
            {"PAY_AMT1": 1000, "BILL_AMT2": 500},  # paid twice the statement
        ]
    )
    out = fe.build_features(df)
    assert out["PAY_RATIO_1"].tolist() == [0.5, 2.0]


def test_payment_ratio_is_missing_when_there_was_nothing_to_pay():
    df = _frame([{"PAY_AMT1": 100, "BILL_AMT2": 0, "PAY_AMT2": 0, "BILL_AMT3": -40}])
    out = fe.build_features(df).iloc[0]
    assert np.isnan(out["PAY_RATIO_1"])
    assert np.isnan(out["PAY_RATIO_2"])


def test_payment_ratio_mean_ignores_months_with_nothing_to_pay():
    df = _frame(
        [
            {"BILL_AMT2": 100, "PAY_AMT1": 100, "BILL_AMT3": 200, "PAY_AMT2": 0},
            {},  # nothing billed at all
        ]
    )
    out = fe.build_features(df)
    assert out.loc[0, "PAY_RATIO_MEAN"] == pytest.approx(0.5)  # (1.0 + 0.0) / 2
    assert np.isnan(out.loc[1, "PAY_RATIO_MEAN"])


def test_delinquency_history_uses_august_to_april_only():
    df = _frame(
        [
            {"PAY_1": 5, "PAY_2": 2, "PAY_4": 1},  # September is ignored here
            {"PAY_1": 3},  # delayed only in September
            {"PAY_5": 1, "PAY_6": 4},
            {"PAY_2": -2, "PAY_3": 0},  # undocumented codes are not delays
        ]
    )
    out = fe.build_features(df)
    assert out["N_DELAY_MONTHS"].tolist() == [2, 0, 2, 0]
    assert out["MAX_DELAY"].tolist() == [2, 0, 4, 0]
    assert out.loc[0, "MONTHS_SINCE_DELAY"] == 1  # August
    assert np.isnan(out.loc[1, "MONTHS_SINCE_DELAY"])  # no delay from August to April
    assert out.loc[2, "MONTHS_SINCE_DELAY"] == 4  # May
    assert np.isnan(out.loc[3, "MONTHS_SINCE_DELAY"])


def test_status_codes_are_grouped_without_mixing_meanings():
    df = _frame([{"PAY_1": code, "PAY_2": code} for code in [-2, -1, 0, 1, 2, 3, 8]])
    out = fe.build_features(df)
    expected = ["-2", "-1", "0", "1", "2", "3+", "3+"]
    assert out["STATUS_SEP"].tolist() == expected
    assert out["STATUS_AUG"].tolist() == expected


def test_inactive_flag():
    df = _frame([{}, {"PAY_AMT4": 10}, {"BILL_AMT6": -1}])
    assert fe.build_features(df)["INACTIVE"].tolist() == [1, 0, 0]


def test_basic_columns_pass_through():
    df = _frame([{"LIMIT_BAL": 50_000, "AGE": 41, "EDUCATION": 3}])
    out = fe.build_features(df).iloc[0]
    assert (out["LIMIT_BAL"], out["AGE"], out["EDUCATION"]) == (50_000, 41, "3")


def test_every_spec_has_a_valid_dtype_and_trend():
    trends = {"ascending", "descending", None}
    for name, spec in fe.FEATURE_SPECS.items():
        assert spec["dtype"] in {"numerical", "categorical"}, name
        assert spec["monotonic_trend"] in trends, name
        if spec["dtype"] == "categorical":
            assert spec["monotonic_trend"] is None, name
