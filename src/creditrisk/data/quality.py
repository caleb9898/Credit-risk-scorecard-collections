"""Data-quality checks for the UCI credit card default data.

Every function takes the standardised frame from ``creditrisk.data.schema`` and returns a plain
count or a small table. Nothing here changes the data.

The target is used in two places only: the overall default rate, and the count of identical rows
(with and without the target column). Relationships between features and the target are studied
later, on training data.

Month indexing follows the project convention: month 1 = September 2005 ... month 6 = April 2005,
so month ``k + 1`` is the month *before* month ``k`` and month ``k - 1`` is the month *after* it.
Amounts are in NT dollars.
"""

from __future__ import annotations

import pandas as pd

from creditrisk.data.clean import RECODES, recode_undocumented_codes
from creditrisk.data.schema import (
    BILL_COLS,
    CANDIDATE_FEATURE_COLS,
    DOCUMENTED_CODES,
    N_MONTHS,
    PAY_COLS,
    PAYAMT_COLS,
    TARGET,
)

AMOUNT_COLS = BILL_COLS + PAYAMT_COLS


def target_summary(df: pd.DataFrame) -> dict[str, float]:
    """Number of rows, number of defaults and default rate (a proportion, not a percentage)."""
    return {
        "n_rows": int(len(df)),
        "n_defaults": int(df[TARGET].sum()),
        "default_rate": float(df[TARGET].mean()),
    }


def missing_counts(df: pd.DataFrame) -> dict[str, int]:
    """Number of null values in each column."""
    return {col: int(n) for col, n in df.isna().sum().items()}


def code_counts(df: pd.DataFrame, column: str, documented: set[int]) -> pd.DataFrame:
    """Frequency of each value of a coded column, flagging values missing from the documentation.

    Returns one row per observed value with columns ``value``, ``count``, ``share``, ``documented``.
    """
    counts = df[column].value_counts().sort_index()
    return pd.DataFrame(
        {
            "value": counts.index.to_numpy(),
            "count": counts.to_numpy(),
            "share": counts.to_numpy() / len(df),
            "documented": [value in documented for value in counts.index],
        }
    )


def pay_code_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Number of accounts with each repayment-status code, per month.

    Returns a table indexed by code with one column per ``PAY_k`` and a ``documented`` flag.
    """
    table = pd.DataFrame({col: df[col].value_counts() for col in PAY_COLS})
    table = table.fillna(0).astype(int).sort_index()
    table.index.name = "code"
    table["documented"] = [code in DOCUMENTED_CODES["PAY"] for code in table.index]
    return table


def undocumented_pay_share(df: pd.DataFrame) -> float:
    """Share of account-months (all six months) whose ``PAY_*`` code is not documented."""
    documented = df[PAY_COLS].isin(DOCUMENTED_CODES["PAY"])
    return float(1 - documented.to_numpy().mean())


def balance_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Per month: negative bills, bills above the credit limit and negative payments.

    The file does not define a negative bill; it is read here as a credit balance (the bank owes
    the client). A bill above ``LIMIT_BAL`` is read as an account over its limit. Both are counted
    so that their treatment is a recorded decision.
    """
    rows = []
    for k in range(1, N_MONTHS + 1):
        bill = df[f"BILL_AMT{k}"]
        rows.append(
            {
                "month": k,
                "n_negative_bill": int((bill < 0).sum()),
                "min_bill": float(bill.min()),
                "n_bill_over_limit": int((bill > df["LIMIT_BAL"]).sum()),
                "n_negative_payment": int((df[f"PAY_AMT{k}"] < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def inactive_account_summary(df: pd.DataFrame) -> dict[str, int]:
    """Count accounts with no activity at all in the six months.

    - ``n_all_bills_zero``: every bill is zero.
    - ``n_all_amounts_zero``: every bill and every payment is zero.
    - ``n_all_amounts_zero_with_delay_in_pay_1``: of those, accounts whose September status
      ``PAY_1`` is a delay code (1 or more), although nothing was ever billed.
    """
    no_bills = (df[BILL_COLS] == 0).all(axis=1)
    no_activity = (df[AMOUNT_COLS] == 0).all(axis=1)
    return {
        "n_all_bills_zero": int(no_bills.sum()),
        "n_all_amounts_zero": int(no_activity.sum()),
        "n_all_amounts_zero_with_delay_in_pay_1": int((no_activity & (df["PAY_1"] >= 1)).sum()),
    }


def status_without_bill(df: pd.DataFrame) -> pd.DataFrame:
    """Per month, the status given to accounts whose bill of that month is zero or negative.

    An account with nothing billed has nothing to be late on, so a delay code (1 or more) there
    is not an observable delinquency. Columns: ``n_bill_not_positive``, and of those the number
    with a delay code and the number with code -2.
    """
    rows = []
    for k in range(1, N_MONTHS + 1):
        nothing_owed = df[f"BILL_AMT{k}"] <= 0
        status = df[f"PAY_{k}"]
        rows.append(
            {
                "month": k,
                "n_bill_not_positive": int(nothing_owed.sum()),
                "n_with_delay_code": int((nothing_owed & (status >= 1)).sum()),
                "n_with_code_minus_2": int((nothing_owed & (status == -2)).sum()),
            }
        )
    return pd.DataFrame(rows)


def duplicate_summary(df: pd.DataFrame) -> dict[str, int]:
    """Count rows that are exact copies of an earlier row.

    The loader drops the raw ``ID`` column, which is a row number and is not documented as a
    client key, so identical rows are not proof of a repeated client.

    - ``n_duplicate_rows``: rows identical to an earlier row in every column.
    - ``n_rows_in_duplicate_groups``: those rows plus the first occurrence of each.
    - ``n_duplicate_rows_all_amounts_zero``: duplicate rows with every bill and payment at zero.
    - ``n_duplicate_rows_ignoring_target``: rows identical to an earlier row in every feature,
      whatever the target says.
    - ``n_rows_sharing_candidate_features``: rows that have at least one twin on the candidate
      model features (sensitive attributes and target ignored). These twins can end up on both
      sides of a train/test split.
    """
    is_duplicate = df.duplicated()
    no_activity = (df[AMOUNT_COLS] == 0).all(axis=1)
    features = [col for col in df.columns if col != TARGET]
    return {
        "n_duplicate_rows": int(is_duplicate.sum()),
        "n_rows_in_duplicate_groups": int(df.duplicated(keep=False).sum()),
        "n_duplicate_rows_all_amounts_zero": int((is_duplicate & no_activity).sum()),
        "n_duplicate_rows_ignoring_target": int(df.duplicated(subset=features).sum()),
        "n_rows_sharing_candidate_features": int(
            df.duplicated(subset=CANDIDATE_FEATURE_COLS, keep=False).sum()
        ),
    }


def distribution_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Minimum, selected percentiles and maximum of each column, to show how heavy the tails are.

    Descriptive only: thresholds for any later capping or binning are fitted on training data.
    """
    quantiles = {"min": 0.0, "p01": 0.01, "p50": 0.50, "p99": 0.99, "p999": 0.999, "max": 1.0}
    table = df[columns].quantile(list(quantiles.values())).T
    table.columns = list(quantiles.keys())
    return table


def payment_timing(df: pd.DataFrame) -> pd.DataFrame:
    """Check which statement a payment settles: the previous month's or the same month's.

    For each month ``k`` (1..5) and the accounts that paid something in that month, compute the
    share whose payment ``PAY_AMTk`` is exactly equal to

    - the previous statement ``BILL_AMT{k+1}`` (payment in month k settles last month's bill),
    - the same-month statement ``BILL_AMTk``,
    - both at once (the two statements are the same amount, so the match says nothing).

    Exact equality only happens when a client pays a statement in full, so the shares show which
    statement those full payments refer to. Month 6 is skipped: it has no earlier statement.
    A share is missing when nobody paid in that month.
    """
    rows = []
    for k in range(1, N_MONTHS):
        payment = df[f"PAY_AMT{k}"]
        paying = payment > 0
        n_paying = int(paying.sum())
        equals_previous = paying & (payment == df[f"BILL_AMT{k + 1}"])
        equals_same = paying & (payment == df[f"BILL_AMT{k}"])
        counts = {
            "share_equals_previous_bill": int(equals_previous.sum()),
            "share_equals_same_month_bill": int(equals_same.sum()),
            "share_equals_both": int((equals_previous & equals_same).sum()),
        }
        shares = {name: n / n_paying if n_paying else None for name, n in counts.items()}
        rows.append({"month": k, "n_paying": n_paying, **shares})
    return pd.DataFrame(rows)


def implied_purchases(df: pd.DataFrame) -> pd.DataFrame:
    """Check the payment-timing convention on every account, not only on those paying in full.

    A statement is the earlier statement, minus what was paid, plus new purchases and charges.
    So the purchases implied by two consecutive statements and the payment between them should
    not be negative. For each month ``k`` (2..5) the table gives the share of accounts with
    negative implied purchases under each reading of ``PAY_AMTk``:

    - it is paid between statements k+1 and k:  ``BILL_AMTk - BILL_AMT{k+1} + PAY_AMTk``
    - it is paid between statements k and k-1:  ``BILL_AMT{k-1} - BILL_AMTk + PAY_AMTk``

    The reading that produces fewer impossible (negative) values fits the data better. Refunds
    can make true purchases negative, so a small share is expected under the right reading.
    """
    rows = []
    for k in range(2, N_MONTHS):
        payment = df[f"PAY_AMT{k}"]
        if_previous = df[f"BILL_AMT{k}"] - df[f"BILL_AMT{k + 1}"] + payment
        if_same = df[f"BILL_AMT{k - 1}"] - df[f"BILL_AMT{k}"] + payment
        rows.append(
            {
                "month": k,
                "share_negative_if_settles_previous_bill": float((if_previous < 0).mean()),
                "share_negative_if_settles_same_month_bill": float((if_same < 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def pay_status_profile(
    df: pd.DataFrame, payment: str = "following_month", months: list[int] | None = None
) -> pd.DataFrame:
    """Describe how statements were repaid under each repayment-status code.

    The file does not say which statement and which payment the status of month ``k`` refers
    to, so two pairings can be compared:

    - ``payment="following_month"``: statement ``BILL_AMTk`` and the payment made the month
      after, ``PAY_AMT{k-1}`` (months 2..6; September has no following payment in the file).
    - ``payment="same_month"``: statement ``BILL_AMT{k+1}`` and the payment made in month k,
      ``PAY_AMTk`` (months 1..5; April has no earlier statement in the file).

    Account-months are pooled over ``months``. For each code the table gives

    - ``n``: number of account-months,
    - ``share_statement_not_positive``: the statement is zero or negative (nothing to pay),
    - ``share_no_payment``: nothing was paid,
    - ``share_paid_in_full``: among account-months with a positive statement, the share where
      the payment covered it completely (missing if there is no positive statement).

    This is evidence about what the codes mean. It does not prove what the bank meant by them.
    """
    if payment == "following_month":
        valid_months = list(range(2, N_MONTHS + 1))
    elif payment == "same_month":
        valid_months = list(range(1, N_MONTHS))
    else:
        raise ValueError("payment must be 'following_month' or 'same_month'")
    months = valid_months if months is None else months
    if not set(months) <= set(valid_months):
        raise ValueError(f"months must be within {valid_months} for payment='{payment}'")

    parts = []
    for k in months:
        if payment == "following_month":
            statement, paid = df[f"BILL_AMT{k}"], df[f"PAY_AMT{k - 1}"]
        else:
            statement, paid = df[f"BILL_AMT{k + 1}"], df[f"PAY_AMT{k}"]
        positive = statement > 0
        parts.append(
            pd.DataFrame(
                {
                    "code": df[f"PAY_{k}"],
                    "not_positive": ~positive,
                    "no_payment": paid == 0,
                    "positive": positive,
                    "paid_in_full": positive & (paid >= statement),
                }
            )
        )
    grouped = pd.concat(parts, ignore_index=True).groupby("code")
    n_positive = grouped["positive"].sum()
    return pd.DataFrame(
        {
            "n": grouped.size(),
            "share_statement_not_positive": grouped["not_positive"].mean(),
            "share_no_payment": grouped["no_payment"].mean(),
            # where() leaves the share missing when a code has no positive statement
            "share_paid_in_full": grouped["paid_in_full"].sum() / n_positive.where(n_positive > 0),
        }
    )


def recoded_row_counts(df: pd.DataFrame) -> dict[str, int]:
    """Number of rows changed in each column by the fixed recoding of ADR-008."""
    recoded = recode_undocumented_codes(df)
    return {column: int((df[column] != recoded[column]).sum()) for column in RECODES}


def _records(table: pd.DataFrame) -> list[dict]:
    """Turn a table into JSON-friendly records, writing missing values as null."""
    as_objects = table.astype(object).where(table.notna(), None)
    return as_objects.to_dict(orient="records")


def data_quality_report(df: pd.DataFrame) -> dict:
    """Run every check on the raw frame and collect the results in one JSON-friendly dictionary."""
    coded = {
        column: _records(code_counts(df, column, DOCUMENTED_CODES[column]))
        for column in ["SEX", "EDUCATION", "MARRIAGE"]
    }
    distributions = distribution_summary(df, ["LIMIT_BAL", "AGE"] + AMOUNT_COLS)
    # Both pairings on the months they share (2..5), so that they can be compared like for like,
    # plus September on its own: its following payment (October) is not in the file.
    shared_months = list(range(2, N_MONTHS))
    profiles = {
        "following_month": pay_status_profile(df, "following_month", shared_months),
        "same_month": pay_status_profile(df, "same_month", shared_months),
        "september_same_month": pay_status_profile(df, "same_month", [1]),
    }
    profiles = {name: _records(table.reset_index()) for name, table in profiles.items()}
    return {
        "target": target_summary(df),
        "missing_values_total": int(sum(missing_counts(df).values())),
        "code_counts": coded,
        "recoded_rows": recoded_row_counts(df),
        "pay_code_counts": _records(pay_code_counts(df).reset_index()),
        "share_undocumented_pay_codes": undocumented_pay_share(df),
        "pay_status_profile": profiles,
        "status_without_bill": _records(status_without_bill(df)),
        "balance_anomalies": _records(balance_anomalies(df)),
        "inactive_accounts": inactive_account_summary(df),
        "duplicates": duplicate_summary(df),
        "distributions": _records(distributions.rename_axis("column").reset_index()),
        "payment_timing": _records(payment_timing(df)),
        "implied_purchases": _records(implied_purchases(df)),
        "n_candidate_feature_columns": len(CANDIDATE_FEATURE_COLS),
    }
