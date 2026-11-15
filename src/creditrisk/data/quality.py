"""Data-quality checks for the UCI credit card default data.

Every function takes the standardised frame from ``creditrisk.data.schema`` and returns a plain
count or a small table. Nothing here changes the data, and the target is used only for the overall
default rate: relationships between features and the target are studied later, on training data.

Month indexing follows the project convention: month 1 = September 2005 ... month 6 = April 2005,
so month ``k + 1`` is the month *before* month ``k``. Amounts are in NT dollars.
"""

from __future__ import annotations

import pandas as pd

from creditrisk.data.schema import BILL_COLS, N_MONTHS, PAY_COLS, PAYAMT_COLS, TARGET

# Values described in the dataset documentation. Anything else is an undocumented code.
DOCUMENTED_CODES: dict[str, set[int]] = {
    "SEX": {1, 2},
    "EDUCATION": {1, 2, 3, 4},
    "MARRIAGE": {1, 2, 3},
    "PAY": {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9},
}

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


def balance_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Per month: negative bills, bills above the credit limit and negative payments.

    A negative bill is a credit balance (the bank owes the client). A bill above ``LIMIT_BAL``
    means the account is over its limit. Both are plausible in real card data; they are counted
    here so that their treatment is a recorded decision.
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


def duplicate_summary(df: pd.DataFrame) -> dict[str, int]:
    """Count rows that are exact copies of an earlier row.

    The file has no client identifier, so identical rows are not proof of a repeated client: two
    inactive accounts with the same limit and demographics look exactly the same.

    - ``n_duplicate_rows``: rows identical to an earlier row in every column.
    - ``n_rows_in_duplicate_groups``: those rows plus the first occurrence of each.
    - ``n_duplicate_rows_all_amounts_zero``: duplicate rows with every bill and payment at zero.
    - ``n_duplicate_rows_ignoring_target``: rows identical to an earlier row in every feature,
      whatever the target says.
    """
    is_duplicate = df.duplicated()
    no_activity = (df[AMOUNT_COLS] == 0).all(axis=1)
    features = [col for col in df.columns if col != TARGET]
    return {
        "n_duplicate_rows": int(is_duplicate.sum()),
        "n_rows_in_duplicate_groups": int(df.duplicated(keep=False).sum()),
        "n_duplicate_rows_all_amounts_zero": int((is_duplicate & no_activity).sum()),
        "n_duplicate_rows_ignoring_target": int(df.duplicated(subset=features).sum()),
    }


def distribution_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Minimum, selected percentiles and maximum of each column, to show how heavy the tails are."""
    quantiles = {"min": 0.0, "p01": 0.01, "p50": 0.50, "p99": 0.99, "p999": 0.999, "max": 1.0}
    table = df[columns].quantile(list(quantiles.values())).T
    table.columns = list(quantiles.keys())
    return table


def payment_timing(df: pd.DataFrame) -> pd.DataFrame:
    """Check which statement a payment settles: the previous month's or the same month's.

    For each month ``k`` (1..5) and the accounts that paid something in that month, compute the
    share whose payment ``PAY_AMTk`` is exactly equal to

    - the previous statement ``BILL_AMT{k+1}`` (payment in month k settles last month's bill), or
    - the same-month statement ``BILL_AMTk``.

    Exact equality only happens when a client pays a statement in full, so the two shares show
    which statement those full payments refer to. Month 6 is skipped: it has no earlier statement.
    A share is missing (NaN) when nobody paid in that month.
    """
    rows = []
    for k in range(1, N_MONTHS):
        payment = df[f"PAY_AMT{k}"]
        paying = payment > 0
        n_paying = int(paying.sum())
        equals_previous = int((paying & (payment == df[f"BILL_AMT{k + 1}"])).sum())
        equals_same = int((paying & (payment == df[f"BILL_AMT{k}"])).sum())
        rows.append(
            {
                "month": k,
                "n_paying": n_paying,
                "share_equals_previous_bill": equals_previous / n_paying if n_paying else None,
                "share_equals_same_month_bill": equals_same / n_paying if n_paying else None,
            }
        )
    return pd.DataFrame(rows)


def pay_status_profile(df: pd.DataFrame, months: list[int] | None = None) -> pd.DataFrame:
    """Describe what bills and payments look like under each repayment-status code.

    Account-months are pooled over ``months`` (default 1..5; month 6 has no earlier statement).
    For each code of ``PAY_k`` the table gives the number of account-months and the share where

    - ``share_bill_not_positive``: the statement of the same month is zero or negative,
    - ``share_previous_bill_not_positive``: the previous statement is zero or negative,
    - ``share_no_payment``: nothing was paid in the month,
    - ``share_paid_previous_bill_in_full``: the previous statement was positive and the payment
      in the month covered it completely.

    This is evidence for (or against) the usual reading of the undocumented codes -2 and 0. It
    does not prove what the bank meant by them.
    """
    months = list(range(1, N_MONTHS)) if months is None else months
    parts = []
    for k in months:
        bill = df[f"BILL_AMT{k}"]
        previous_bill = df[f"BILL_AMT{k + 1}"]
        payment = df[f"PAY_AMT{k}"]
        parts.append(
            pd.DataFrame(
                {
                    "code": df[f"PAY_{k}"],
                    "share_bill_not_positive": bill <= 0,
                    "share_previous_bill_not_positive": previous_bill <= 0,
                    "share_no_payment": payment == 0,
                    "share_paid_previous_bill_in_full": (previous_bill > 0)
                    & (payment >= previous_bill),
                }
            )
        )
    pooled = pd.concat(parts, ignore_index=True)
    grouped = pooled.groupby("code")
    table = grouped.mean()
    table.insert(0, "n", grouped.size())
    return table


def _records(table: pd.DataFrame) -> list[dict]:
    """Turn a table into JSON-friendly records, writing missing values as null."""
    as_objects = table.astype(object).where(table.notna(), None)
    return as_objects.to_dict(orient="records")


def data_quality_report(df: pd.DataFrame) -> dict:
    """Run every check and collect the results in one JSON-friendly dictionary."""
    coded = {
        column: _records(code_counts(df, column, DOCUMENTED_CODES[column]))
        for column in ["SEX", "EDUCATION", "MARRIAGE"]
    }
    distributions = distribution_summary(df, ["LIMIT_BAL", "AGE"] + AMOUNT_COLS)
    return {
        "target": target_summary(df),
        "missing_values_total": int(sum(missing_counts(df).values())),
        "code_counts": coded,
        "pay_code_counts": _records(pay_code_counts(df).reset_index()),
        "balance_anomalies": _records(balance_anomalies(df)),
        "duplicates": duplicate_summary(df),
        "distributions": _records(distributions.rename_axis("column").reset_index()),
        "payment_timing": _records(payment_timing(df)),
        "pay_status_profile": _records(pay_status_profile(df).reset_index()),
    }
