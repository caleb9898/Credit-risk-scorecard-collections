import numpy as np
import pandas as pd
import pytest

from creditrisk.data import schema as sc


def _synthetic(n=200, seed=0, layout="canonical"):
    rng = np.random.default_rng(seed)
    data = {
        "LIMIT_BAL": rng.integers(10_000, 500_000, n),
        "SEX": rng.integers(1, 3, n),
        "EDUCATION": rng.integers(1, 5, n),
        "MARRIAGE": rng.integers(1, 4, n),
        "AGE": rng.integers(21, 70, n),
    }
    for col in sc.PAY_COLS:
        data[col] = rng.integers(-1, 4, n)
    for col in sc.BILL_COLS + sc.PAYAMT_COLS:
        data[col] = rng.integers(0, 100_000, n)
    data[sc.TARGET] = rng.binomial(1, 0.22, n)
    df = pd.DataFrame(data)
    if layout == "uci":
        inverse = {v: k for k, v in sc._UCI_RENAMES.items()}
        df = df.rename(columns=inverse)
    elif layout == "named":
        df = df.rename(columns={"PAY_1": "PAY_0", sc.TARGET: "default.payment.next.month"})
        df.insert(0, "ID", np.arange(1, len(df) + 1))
    return df


def test_uci_mapping_is_complete_and_chronological():
    assert len(sc._UCI_RENAMES) == 24
    assert sc._UCI_RENAMES["X6"] == "PAY_1"  # September
    assert sc._UCI_RENAMES["X11"] == "PAY_6"  # April
    assert sc._UCI_RENAMES["X12"] == "BILL_AMT1"
    assert sc._UCI_RENAMES["X23"] == "PAY_AMT6"
    assert sc._UCI_RENAMES["Y"] == "DEFAULT"


@pytest.mark.parametrize("layout", ["canonical", "uci", "named"])
def test_standardize_columns_handles_all_layouts(layout):
    raw = _synthetic(layout=layout)
    out = sc.standardize_columns(raw)
    assert list(out.columns) == sc.CANONICAL_COLUMNS


def test_standardize_preserves_values():
    raw = _synthetic(layout="named")
    out = sc.standardize_columns(raw)
    assert (out["PAY_1"].to_numpy() == raw["PAY_0"].to_numpy()).all()
    assert (out["DEFAULT"].to_numpy() == raw["default.payment.next.month"].to_numpy()).all()


def test_standardize_rejects_unknown_schema():
    bad = _synthetic().drop(columns=["AGE"]).assign(MYSTERY=1)
    with pytest.raises(ValueError, match="Missing columns"):
        sc.standardize_columns(bad)


def test_validate_raw_ok_and_summary():
    df = _synthetic()
    summary = sc.validate_raw(df, expected_rows=None, expected_defaults=None)
    assert summary["n_rows"] == len(df)
    assert summary["n_defaults"] == int(df["DEFAULT"].sum())


def test_validate_raw_flags_nulls_nonbinary_and_counts():
    df = _synthetic()
    with pytest.raises(ValueError, match="null"):
        sc.validate_raw(df.assign(AGE=np.where(df.index == 0, np.nan, df["AGE"])), None, None)
    with pytest.raises(ValueError, match="binary"):
        sc.validate_raw(df.assign(DEFAULT=2), None, None)
    with pytest.raises(ValueError, match="expected 30000 rows"):
        sc.validate_raw(df)  # defaults apply: wrong size


def test_load_raw_csv_roundtrip_and_hash(tmp_path):
    df = _synthetic(layout="named")
    path = tmp_path / "credit.csv"
    df.to_csv(path, index=False)
    loaded = sc.load_raw(path)
    assert list(loaded.columns) == sc.CANONICAL_COLUMNS
    assert len(loaded) == len(df)
    h1 = sc.sha256_file(path)
    assert h1 == sc.sha256_file(path) and len(h1) == 64
    path.write_text(path.read_text() + "\n")
    assert sc.sha256_file(path) != h1


def test_load_raw_rejects_unknown_extension(tmp_path):
    p = tmp_path / "data.parquet"
    p.write_bytes(b"x")
    with pytest.raises(ValueError, match="Unsupported"):
        sc.load_raw(p)
