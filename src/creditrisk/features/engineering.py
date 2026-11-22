"""Candidate features built row by row from the cleaned data (ADR-011).

Nothing here is fitted to the data: every feature of a client depends only on that client's own
columns. The functions can therefore be applied to any subset of rows without leakage.

Conventions (see ADR-008 and ADR-009):
- Month 1 = September 2005 ... month 6 = April 2005. Amounts are in NT dollars.
- A negative bill is read as a credit balance and counts as zero utilisation.
- ``PAY_AMTk`` pays the statement ``BILL_AMT{k+1}``.
- The September status ``PAY_1`` is recorded on a different basis from the earlier months, so it
  is kept as a feature of its own and left out of the delinquency-history features.
- Status codes -2, -1 and 0 are not delays; codes 3 or more are grouped as "3+".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditrisk.data.schema import BILL_COLS, N_MONTHS, PAYAMT_COLS

# Months with a consistently coded repayment status: August (2) to April (6).
HISTORY_MONTHS = list(range(2, N_MONTHS + 1))

# Every candidate feature with its binning settings. ``monotonic_trend`` is the direction of the
# default rate as the feature grows, declared from business logic before looking at the data
# (None = no constraint). It is the order of the columns returned by ``build_features``.
_ASC = {"dtype": "numerical", "monotonic_trend": "ascending"}
_DESC = {"dtype": "numerical", "monotonic_trend": "descending"}
_FREE = {"dtype": "numerical", "monotonic_trend": None}
_CAT = {"dtype": "categorical", "monotonic_trend": None}

FEATURE_SPECS: dict[str, dict] = {
    # Higher limits are granted to clients the bank already trusts.
    "LIMIT_BAL": _DESC,
    # Age is allowed any shape (risk is often higher at both ends).
    "AGE": _FREE,
    "EDUCATION": _CAT,
    # A more heavily used credit line signals financial strain.
    **{f"UTIL_{k}": _ASC for k in range(1, N_MONTHS + 1)},
    "UTIL_MEAN": _ASC,
    "UTIL_MAX": _ASC,
    # Utilisation rising from April to September.
    "UTIL_TREND": _ASC,
    # Paying a larger share of the statement signals capacity to pay.
    **{f"PAY_RATIO_{k}": _DESC for k in range(1, N_MONTHS)},
    "PAY_RATIO_MEAN": _DESC,
    # More, deeper and more recent delays mean more risk.
    "N_DELAY_MONTHS": _ASC,
    "MAX_DELAY": _ASC,
    "MONTHS_SINCE_DELAY": _DESC,
    "STATUS_SEP": _CAT,
    "STATUS_AUG": _CAT,
    "INACTIVE": _CAT,
}


def utilisation(df: pd.DataFrame) -> pd.DataFrame:
    """Bill divided by the credit limit for each month (``UTIL_1`` .. ``UTIL_6``).

    Negative bills are floored at zero; bills above the limit give values above 1 and are kept.
    """
    bills = df[BILL_COLS].clip(lower=0).to_numpy()
    util = bills / df[["LIMIT_BAL"]].to_numpy()
    return pd.DataFrame(util, index=df.index, columns=[f"UTIL_{k}" for k in range(1, N_MONTHS + 1)])


def payment_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Payment of month k divided by the statement it pays, ``BILL_AMT{k+1}`` (k = 1..5).

    The ratio is missing (NaN) when that statement is zero or negative: there was nothing to pay.
    That group gets its own bin later, so it is not mixed with clients who paid nothing.
    """
    ratios = {}
    for k in range(1, N_MONTHS):
        statement = df[f"BILL_AMT{k + 1}"]
        ratios[f"PAY_RATIO_{k}"] = (df[f"PAY_AMT{k}"] / statement).where(statement > 0)
    return pd.DataFrame(ratios, index=df.index)


def status_group(status: pd.Series) -> pd.Series:
    """Repayment status as a category: -2, -1, 0, 1, 2 and "3+" (three or more months late)."""
    return status.clip(upper=3).astype(int).astype(str).replace({"3": "3+"})


def delinquency_history(df: pd.DataFrame) -> pd.DataFrame:
    """Delays recorded from August to April (``PAY_2`` .. ``PAY_6``).

    - ``N_DELAY_MONTHS``: months with a delay (status 1 or more).
    - ``MAX_DELAY``: worst delay in months (0 if none; codes -2, -1, 0 record no delay).
    - ``MONTHS_SINCE_DELAY``: months between September and the latest delay (1 = August); missing
      when there was no delay in those months.
    """
    status = df[[f"PAY_{k}" for k in HISTORY_MONTHS]].to_numpy()
    delayed = status >= 1
    any_delay = delayed.any(axis=1)
    latest = np.where(any_delay, delayed.argmax(axis=1) + 1, np.nan)  # columns start at August
    return pd.DataFrame(
        {
            "N_DELAY_MONTHS": delayed.sum(axis=1),
            "MAX_DELAY": status.clip(min=0).max(axis=1),
            "MONTHS_SINCE_DELAY": latest,
        },
        index=df.index,
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """All candidate features, in the order of ``FEATURE_SPECS``, with the index of ``df``.

    ``df`` is the cleaned canonical frame (any subset of rows, e.g. the training part). The
    target and the sensitive attributes ``SEX`` and ``MARRIAGE`` are not included (ADR-002).
    """
    util = utilisation(df)
    ratios = payment_ratios(df)
    amounts_zero = (df[BILL_COLS + PAYAMT_COLS] == 0).all(axis=1)
    features = pd.concat(
        [
            df[["LIMIT_BAL", "AGE"]],
            df["EDUCATION"].astype(str),
            util,
            pd.DataFrame(
                {
                    "UTIL_MEAN": util.mean(axis=1),
                    "UTIL_MAX": util.max(axis=1),
                    "UTIL_TREND": util["UTIL_1"] - util[f"UTIL_{N_MONTHS}"],
                },
                index=df.index,
            ),
            ratios,
            ratios.mean(axis=1).rename("PAY_RATIO_MEAN"),  # skips months with nothing to pay
            delinquency_history(df),
            status_group(df["PAY_1"]).rename("STATUS_SEP"),
            status_group(df["PAY_2"]).rename("STATUS_AUG"),
            amounts_zero.astype(int).rename("INACTIVE"),
        ],
        axis=1,
    )
    return features[list(FEATURE_SPECS)]
