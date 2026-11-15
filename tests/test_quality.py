import pandas as pd
import pytest

from creditrisk.data import quality as q
from creditrisk.data import schema as sc
from creditrisk.reporting import save_metrics


def _frame(rows):
    """Build a canonical frame from partial row dicts; unspecified columns get harmless defaults."""
    defaults = dict.fromkeys(sc.CANONICAL_COLUMNS, 0)
    defaults.update({"LIMIT_BAL": 100_000, "SEX": 1, "EDUCATION": 1, "MARRIAGE": 1, "AGE": 30})
    defaults.update(dict.fromkeys(sc.PAY_COLS, -1))
    return pd.DataFrame([{**defaults, **row} for row in rows])[sc.CANONICAL_COLUMNS]


def test_target_summary_counts_and_rate():
    df = _frame([{"DEFAULT": 1}, {"DEFAULT": 0}, {"DEFAULT": 0}, {"DEFAULT": 1}])
    assert q.target_summary(df) == {"n_rows": 4, "n_defaults": 2, "default_rate": 0.5}


def test_missing_counts_per_column():
    df = _frame([{}, {}, {}]).astype({"AGE": "float"})
    df.loc[0, "AGE"] = None
    counts = q.missing_counts(df)
    assert counts["AGE"] == 1
    assert sum(counts.values()) == 1


def test_code_counts_flags_undocumented_values():
    df = _frame([{"EDUCATION": 1}, {"EDUCATION": 5}, {"EDUCATION": 5}, {"EDUCATION": 0}])
    out = q.code_counts(df, "EDUCATION", q.DOCUMENTED_CODES["EDUCATION"])
    assert out["value"].tolist() == [0, 1, 5]
    assert out["count"].tolist() == [1, 1, 2]
    assert out["share"].tolist() == [0.25, 0.25, 0.5]
    assert out["documented"].tolist() == [False, True, False]


def test_pay_code_counts_one_row_per_code_and_month():
    df = _frame([{"PAY_1": -2, "PAY_2": 0}, {"PAY_1": 2, "PAY_2": 0}])
    out = q.pay_code_counts(df)
    assert out.loc[-2, "PAY_1"] == 1
    assert out.loc[2, "PAY_1"] == 1
    assert out.loc[0, "PAY_2"] == 2
    assert out.loc[-2, "PAY_2"] == 0  # absent code is a zero, not a missing value
    assert out.loc[-1, "PAY_3"] == 2
    assert out["documented"].to_dict() == {-2: False, -1: True, 0: False, 2: True}


def test_balance_anomalies_by_month():
    df = _frame(
        [
            {"LIMIT_BAL": 1000, "BILL_AMT1": -50, "BILL_AMT2": 1500},
            {"LIMIT_BAL": 1000, "BILL_AMT1": 1000, "BILL_AMT2": 0},
            {"LIMIT_BAL": 1000, "BILL_AMT1": 1001, "BILL_AMT2": -1},
        ]
    )
    out = q.balance_anomalies(df).set_index("month")
    assert out.loc[1, "n_negative_bill"] == 1
    assert out.loc[1, "n_bill_over_limit"] == 1  # equal to the limit is not over it
    assert out.loc[1, "min_bill"] == -50
    assert out.loc[2, "n_negative_bill"] == 1
    assert out.loc[2, "n_bill_over_limit"] == 1
    assert out.loc[3, "n_negative_bill"] == 0
    assert out["n_negative_payment"].sum() == 0


def test_duplicate_summary_counts_repeats():
    active = {"BILL_AMT1": 500, "PAY_AMT1": 100}
    df = _frame([active, active, active, {"AGE": 40}, {"AGE": 40}, {"AGE": 50}])
    out = q.duplicate_summary(df)
    assert out["n_duplicate_rows"] == 3  # two extra copies of `active`, one extra AGE=40
    assert out["n_rows_in_duplicate_groups"] == 5
    assert out["n_duplicate_rows_all_amounts_zero"] == 1  # the AGE=40 copy has no activity


def test_duplicate_summary_separates_conflicting_targets():
    df = _frame([{"DEFAULT": 0}, {"DEFAULT": 1}])
    out = q.duplicate_summary(df)
    assert out["n_duplicate_rows"] == 0
    assert out["n_duplicate_rows_ignoring_target"] == 1


def test_duplicate_summary_without_duplicates():
    df = _frame([{"AGE": 30}, {"AGE": 31}])
    out = q.duplicate_summary(df)
    assert out["n_duplicate_rows"] == 0
    assert out["n_duplicate_rows_all_amounts_zero"] == 0


def test_distribution_summary_known_quantiles():
    df = _frame([{"AGE": a} for a in range(1, 102)])  # 1..101, median 51
    out = q.distribution_summary(df, ["AGE"]).loc["AGE"]
    assert out["min"] == 1
    assert out["p50"] == 51
    assert out["max"] == 101
    assert out["p01"] == 2
    assert out["p99"] == 100
    assert out["p999"] == pytest.approx(100.9)


def test_distribution_summary_constant_column():
    df = _frame([{}, {}, {}])
    out = q.distribution_summary(df, ["AGE"]).loc["AGE"]
    assert out["min"] == out["p50"] == out["max"] == 30


def test_payment_timing_detects_payment_of_previous_statement():
    # By construction the payment in month k equals the statement of month k+1 (the older one).
    df = _frame(
        [
            {"BILL_AMT1": 900, "BILL_AMT2": 700, "PAY_AMT1": 700},
            {"BILL_AMT1": 300, "BILL_AMT2": 400, "PAY_AMT1": 400},
            {"BILL_AMT1": 250, "BILL_AMT2": 800, "PAY_AMT1": 250},  # matches the same month
            {"BILL_AMT1": 100, "BILL_AMT2": 200, "PAY_AMT1": 50},  # partial payment, no match
            {"BILL_AMT1": 100, "BILL_AMT2": 200, "PAY_AMT1": 0},  # no payment: not counted
        ]
    )
    out = q.payment_timing(df).set_index("month")
    assert list(out.index) == [1, 2, 3, 4, 5]  # month 6 has no earlier statement
    assert out.loc[1, "n_paying"] == 4
    assert out.loc[1, "share_equals_previous_bill"] == 0.5
    assert out.loc[1, "share_equals_same_month_bill"] == 0.25
    assert out.loc[2, "n_paying"] == 0


def test_pay_status_profile_same_month_pairing():
    # Status of month 1 against the statement of month 2 and the payment of month 1.
    df = _frame(
        [
            {"PAY_1": -1, "BILL_AMT2": 700, "PAY_AMT1": 700},  # paid exactly in full
            {"PAY_1": -1, "BILL_AMT2": 700, "PAY_AMT1": 100},  # paid part of it
            {"PAY_1": -1, "BILL_AMT2": 0, "PAY_AMT1": 50},  # nothing to pay: not in the share
            {"PAY_1": -2, "BILL_AMT2": 0, "PAY_AMT1": 0},
        ]
    )
    out = q.pay_status_profile(df, payment="same_month", months=[1])
    assert out.loc[-1, "n"] == 3
    assert out.loc[-1, "share_statement_not_positive"] == 1 / 3
    assert out.loc[-1, "share_no_payment"] == 0.0
    assert out.loc[-1, "share_paid_in_full"] == 0.5  # 1 of the 2 positive statements
    assert out.loc[-2, "share_statement_not_positive"] == 1.0
    assert out.loc[-2, "share_no_payment"] == 1.0
    assert pd.isna(out.loc[-2, "share_paid_in_full"])  # no positive statement to pay


def test_pay_status_profile_following_month_pairing():
    # Status of month 2 against the statement of month 2 and the payment of month 1.
    df = _frame(
        [
            {"PAY_2": -1, "BILL_AMT2": 700, "PAY_AMT1": 900},  # paid more than the statement
            {"PAY_2": 2, "BILL_AMT2": 700, "PAY_AMT1": 0},
            {"PAY_2": 2, "BILL_AMT2": 700, "PAY_AMT1": 699},  # one dollar short is not in full
        ]
    )
    out = q.pay_status_profile(df, payment="following_month", months=[2])
    assert out.loc[-1, "share_paid_in_full"] == 1.0
    assert out.loc[2, "n"] == 2
    assert out.loc[2, "share_paid_in_full"] == 0.0
    assert out.loc[2, "share_no_payment"] == 0.5


def test_pay_status_profile_pools_the_months_each_pairing_allows():
    df = _frame([{f"PAY_{k}": 2 for k in range(1, 7)}])
    assert q.pay_status_profile(df, payment="same_month").loc[2, "n"] == 5  # months 1..5
    assert q.pay_status_profile(df, payment="following_month").loc[2, "n"] == 5  # months 2..6


def test_pay_status_profile_rejects_impossible_requests():
    df = _frame([{}])
    with pytest.raises(ValueError, match="months"):
        q.pay_status_profile(df, payment="following_month", months=[1])  # no October payment
    with pytest.raises(ValueError, match="payment"):
        q.pay_status_profile(df, payment="next_year")


def test_payment_timing_missing_share_and_overlap():
    df = _frame([{"BILL_AMT1": 500, "BILL_AMT2": 500, "PAY_AMT1": 500}])
    out = q.payment_timing(df).set_index("month")
    assert out.loc[1, "share_equals_both"] == 1.0  # both statements are the same amount
    assert out.loc[2, "n_paying"] == 0
    assert pd.isna(out.loc[2, "share_equals_previous_bill"])


def test_implied_purchases_flags_the_wrong_convention():
    # Both readings possible: 300 - 1000 + 900 = 200 and 800 - 300 + 900 = 1400.
    ok = {"BILL_AMT3": 1000, "BILL_AMT2": 300, "BILL_AMT1": 800, "PAY_AMT2": 900}
    # "Paid between statements 2 and 1" is impossible: 100 - 300 + 100 = -100.
    bad_same = {"BILL_AMT3": 400, "BILL_AMT2": 300, "BILL_AMT1": 100, "PAY_AMT2": 100}
    # "Paid between statements 3 and 2" is impossible: 300 - 1000 + 100 = -600.
    bad_previous = {"BILL_AMT3": 1000, "BILL_AMT2": 300, "BILL_AMT1": 800, "PAY_AMT2": 100}
    df = _frame([ok, bad_same, bad_previous, ok])
    out = q.implied_purchases(df).set_index("month")
    assert list(out.index) == [2, 3, 4, 5]
    assert out.loc[2, "share_negative_if_settles_previous_bill"] == 0.25
    assert out.loc[2, "share_negative_if_settles_same_month_bill"] == 0.25


def test_inactive_account_summary():
    df = _frame(
        [
            {"PAY_1": 1},  # no activity, yet a delay code in September
            {"PAY_1": -2},  # no activity
            {"PAY_AMT3": 10},  # no bills, one payment
            {"BILL_AMT1": 5},
        ]
    )
    assert q.inactive_account_summary(df) == {
        "n_all_bills_zero": 3,
        "n_all_amounts_zero": 2,
        "n_all_amounts_zero_with_delay_in_pay_1": 1,
    }


def test_status_without_bill_counts_delay_codes_on_empty_statements():
    df = _frame(
        [
            {"BILL_AMT1": 0, "PAY_1": 1},
            {"BILL_AMT1": -20, "PAY_1": -2},
            {"BILL_AMT1": 0, "PAY_1": 0},
            {"BILL_AMT1": 100, "PAY_1": 2},  # something is owed: not counted
        ]
    )
    out = q.status_without_bill(df).set_index("month")
    assert out.loc[1, "n_bill_not_positive"] == 3
    assert out.loc[1, "n_with_delay_code"] == 1
    assert out.loc[1, "n_with_code_minus_2"] == 1


def test_undocumented_pay_share():
    # 2 rows x 6 months = 12 account-months; three of them carry -2 or 0.
    df = _frame([{"PAY_1": -2, "PAY_2": 0}, {"PAY_1": 0, "PAY_2": 3}])
    assert q.undocumented_pay_share(df) == 3 / 12


def test_balance_anomalies_counts_negative_payments():
    df = _frame([{"PAY_AMT2": -5}, {"PAY_AMT2": 5}])
    out = q.balance_anomalies(df).set_index("month")
    assert out.loc[2, "n_negative_payment"] == 1
    assert out.loc[1, "n_negative_payment"] == 0


def test_duplicate_summary_counts_twins_on_candidate_features():
    # Same candidate features, different sensitive attribute: not a duplicate row, but a twin.
    df = _frame([{"SEX": 1}, {"SEX": 2}, {"AGE": 55}])
    out = q.duplicate_summary(df)
    assert out["n_duplicate_rows"] == 0
    assert out["n_rows_sharing_candidate_features"] == 2


def test_recoded_row_counts():
    df = _frame([{"EDUCATION": 5}, {"EDUCATION": 0, "MARRIAGE": 0}, {"EDUCATION": 2}])
    assert q.recoded_row_counts(df) == {"EDUCATION": 2, "MARRIAGE": 1}


def test_report_can_be_saved_even_with_missing_shares(tmp_path):
    # One inactive account: nobody pays, so several shares have an empty denominator.
    report = q.data_quality_report(_frame([{}]))
    assert report["payment_timing"][0]["share_equals_previous_bill"] is None
    assert report["n_candidate_feature_columns"] == 21
    path = save_metrics("demo", report, "abc", metrics_dir=tmp_path)
    assert path.exists()
