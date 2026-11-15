"""Schema, loading and validation for the UCI "Default of Credit Card Clients" dataset.

Canonical column convention used across this project (month index 1 = most recent):

    PAY_1 .. PAY_6      repayment status, 1 = September 2005 ... 6 = April 2005
    BILL_AMT1 .. 6      bill statement amount, same month indexing
    PAY_AMT1 .. 6       amount paid in the month, same month indexing
    DEFAULT             1 if the client defaulted in October 2005 (the target)

The raw files circulate in two naming variants (UCI "X1..X23, Y" and the Kaggle-style
"PAY_0, PAY_2, ..."). ``standardize_columns`` maps both to the canonical names above.
Nothing here modifies values; it only renames columns and checks them.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

N_MONTHS = 6
PAY_COLS = [f"PAY_{i}" for i in range(1, N_MONTHS + 1)]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, N_MONTHS + 1)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, N_MONTHS + 1)]
DEMOGRAPHIC_COLS = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
TARGET = "DEFAULT"
CANONICAL_COLUMNS = DEMOGRAPHIC_COLS + PAY_COLS + BILL_COLS + PAYAMT_COLS + [TARGET]

# Calendar month (2005) behind each month index, for tables and figures.
MONTH_LABELS = {1: "Sep", 2: "Aug", 3: "Jul", 4: "Jun", 5: "May", 6: "Apr"}

# Sensitive attributes are never model inputs (ADR-002); every other non-target column is a
# candidate feature.
SENSITIVE_COLS = ["SEX", "MARRIAGE"]
CANDIDATE_FEATURE_COLS = [c for c in CANONICAL_COLUMNS if c not in SENSITIVE_COLS + [TARGET]]

# Values described in the dataset documentation. Anything else is an undocumented code.
DOCUMENTED_CODES: dict[str, set[int]] = {
    "SEX": {1, 2},
    "EDUCATION": {1, 2, 3, 4},
    "MARRIAGE": {1, 2, 3},
    "PAY": {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9},
}

# UCI layout: X1..X5 demographics, X6..X11 PAY, X12..X17 BILL_AMT, X18..X23 PAY_AMT, Y target.
_UCI_RENAMES = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    **{f"X{6 + i}": f"PAY_{i + 1}" for i in range(N_MONTHS)},
    **{f"X{12 + i}": f"BILL_AMT{i + 1}" for i in range(N_MONTHS)},
    **{f"X{18 + i}": f"PAY_AMT{i + 1}" for i in range(N_MONTHS)},
    "Y": TARGET,
}

# Kaggle-style / descriptive layout: the September status column is called PAY_0 (no PAY_1).
_NAMED_RENAMES = {
    "PAY_0": "PAY_1",
    "default payment next month": TARGET,
    "default.payment.next.month": TARGET,
    "default_payment_next_month": TARGET,
}

# Size and event count of the file this project is built on (30,000 rows is also the size
# given by the dataset documentation). A mismatch means a different file or a corrupted copy:
# investigate, never silently adjust.
EXPECTED_N_ROWS = 30_000
EXPECTED_DEFAULT_COUNT = 6_636


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, used to pin the exact raw data version."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with canonical column names, ID dropped, and canonical column order.

    Raises ValueError listing missing/unexpected columns if the layout is not recognised.
    """
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    renames = {**_UCI_RENAMES, **_NAMED_RENAMES}
    out = out.rename(columns={c: renames[c] for c in out.columns if c in renames})
    out = out.drop(columns=[c for c in out.columns if c.upper() == "ID"])

    missing = [c for c in CANONICAL_COLUMNS if c not in out.columns]
    extra = [c for c in out.columns if c not in CANONICAL_COLUMNS]
    if missing or extra:
        raise ValueError(
            f"Unrecognised schema. Missing columns: {missing}. Unexpected columns: {extra}."
        )
    return out[CANONICAL_COLUMNS]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw file (.csv, .xls or .xlsx) and standardise column names.

    Excel files from UCI carry a label row (X1..X23, Y) above the descriptive header, so the
    header is read from the second row.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xls", ".xlsx"}:
        df = pd.read_excel(path, header=1)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return standardize_columns(df)


def validate_raw(
    df: pd.DataFrame,
    expected_rows: int | None = EXPECTED_N_ROWS,
    expected_defaults: int | None = EXPECTED_DEFAULT_COUNT,
) -> dict[str, object]:
    """Validate a standardised frame. Raises ValueError on any violation; returns a summary.

    Checks: canonical columns, numeric dtypes, no nulls, binary target, expected row and event
    counts (pass ``None`` to skip, e.g. for synthetic test data).
    """
    problems: list[str] = []

    if list(df.columns) != CANONICAL_COLUMNS:
        problems.append("columns do not match canonical schema; run standardize_columns first")
    else:
        non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            problems.append(f"non-numeric columns: {non_numeric}")
        n_null = int(df.isna().sum().sum())
        if n_null:
            problems.append(f"{n_null} null values found")
        if not set(df[TARGET].unique()) <= {0, 1}:
            problems.append("target must be binary {0, 1}")

    if expected_rows is not None and len(df) != expected_rows:
        problems.append(f"expected {expected_rows} rows, found {len(df)}")
    if expected_defaults is not None and not problems:
        n_def = int(df[TARGET].sum())
        if n_def != expected_defaults:
            problems.append(f"expected {expected_defaults} defaults, found {n_def}")

    if problems:
        raise ValueError("Raw data validation failed: " + "; ".join(problems))

    return {
        "n_rows": int(len(df)),
        "n_defaults": int(df[TARGET].sum()),
        "default_rate": float(df[TARGET].mean()),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m creditrisk.data.schema <path-to-raw-file>``."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m creditrisk.data.schema <path-to-raw-file>", file=sys.stderr)
        return 2
    path = Path(args[0])
    print(f"file:    {path}")
    print(f"sha256:  {sha256_file(path)}")
    summary = validate_raw(load_raw(path))
    print(f"rows:    {summary['n_rows']}")
    print(f"defaults:{summary['n_defaults']} (rate {summary['default_rate']:.4f})")
    print("schema validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
