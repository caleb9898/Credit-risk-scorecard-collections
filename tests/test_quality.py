import pandas as pd

from creditrisk.data import quality as q
from creditrisk.data import schema as sc


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


def test_pay_status_profile_shares_by_code():
    df = _frame(
        [
            # status -1 in month 1: previous statement (700) paid in full
            {"PAY_1": -1, "BILL_AMT1": 900, "BILL_AMT2": 700, "PAY_AMT1": 700},
            # status -1 in month 1: paid less than the previous statement
            {"PAY_1": -1, "BILL_AMT1": 900, "BILL_AMT2": 700, "PAY_AMT1": 100},
            # status -2 in month 1: nothing billed, nothing paid
            {"PAY_1": -2, "BILL_AMT1": 0, "BILL_AMT2": 0, "PAY_AMT1": 0},
        ]
    )
    out = q.pay_status_profile(df, months=[1])
    assert out.loc[-1, "n"] == 2
    assert out.loc[-1, "share_paid_previous_bill_in_full"] == 0.5
    assert out.loc[-1, "share_no_payment"] == 0.0
    assert out.loc[-2, "n"] == 1
    assert out.loc[-2, "share_bill_not_positive"] == 1.0
    assert out.loc[-2, "share_previous_bill_not_positive"] == 1.0
    assert out.loc[-2, "share_no_payment"] == 1.0
    assert out.loc[-2, "share_paid_previous_bill_in_full"] == 0.0


def test_pay_status_profile_pools_months():
    df = _frame([{"PAY_1": 2, "PAY_2": 2, "PAY_3": 2, "PAY_4": 2, "PAY_5": 2}])
    out = q.pay_status_profile(df)
    assert out.loc[2, "n"] == 5  # months 1..5 pooled; month 6 has no earlier statement
