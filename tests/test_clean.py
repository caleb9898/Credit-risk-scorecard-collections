import pandas as pd
import pytest

from creditrisk.config import load_config
from creditrisk.data import clean
from creditrisk.data import schema as sc
from creditrisk.data.load import load_project_data


def _frame(education, marriage):
    df = pd.DataFrame(0, index=range(len(education)), columns=sc.CANONICAL_COLUMNS)
    df["EDUCATION"] = education
    df["MARRIAGE"] = marriage
    df["PAY_1"] = [-2, 0, 1, 2, -1, 0, 0][: len(education)]
    return df


def test_recode_groups_undocumented_codes_into_others():
    df = _frame(education=[0, 1, 2, 3, 4, 5, 6], marriage=[0, 1, 2, 3, 1, 2, 3])
    out = clean.recode_undocumented_codes(df)
    assert out["EDUCATION"].tolist() == [4, 1, 2, 3, 4, 4, 4]
    assert out["MARRIAGE"].tolist() == [3, 1, 2, 3, 1, 2, 3]


def test_recode_leaves_everything_else_untouched():
    df = _frame(education=[0, 1, 5], marriage=[0, 1, 2])
    original = df.copy()
    out = clean.recode_undocumented_codes(df)
    pd.testing.assert_frame_equal(df, original)  # the input is not modified
    assert len(out) == len(df)
    other = [c for c in df.columns if c not in ("EDUCATION", "MARRIAGE")]
    pd.testing.assert_frame_equal(out[other], df[other])
    assert out["PAY_1"].tolist() == [-2, 0, 1]  # PAY codes -2 and 0 are kept as they are


def test_recode_is_a_no_op_on_documented_codes():
    df = _frame(education=[1, 2, 3, 4], marriage=[1, 2, 3, 1])
    pd.testing.assert_frame_equal(clean.recode_undocumented_codes(df), df)


@pytest.mark.parametrize("column", ["EDUCATION", "MARRIAGE"])
def test_recode_rejects_codes_never_seen_before(column):
    df = _frame(education=[1, 2], marriage=[1, 2])
    df.loc[0, column] = 9
    with pytest.raises(ValueError, match=column):
        clean.recode_undocumented_codes(df)


def test_load_project_data_checks_hash_and_schema():
    df, sha = load_project_data()
    cfg = load_config("project")
    assert sha == cfg["data"]["raw_sha256"]
    assert list(df.columns) == sc.CANONICAL_COLUMNS
    assert len(df) == sc.EXPECTED_N_ROWS


def test_load_project_data_rejects_a_different_file():
    cfg = load_config("project")
    cfg["data"]["raw_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        load_project_data(cfg)
