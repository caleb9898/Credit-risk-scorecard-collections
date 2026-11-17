import numpy as np
import pandas as pd
import pytest

from creditrisk.config import load_config
from creditrisk.data import schema as sc
from creditrisk.data import split as sp
from creditrisk.data.load import load_project_data


def _target(n=1000, rate=0.2, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.binomial(1, rate, n), name=sc.TARGET)


def _frame(rows):
    """Canonical frame from partial row dicts; unspecified columns are zero."""
    defaults = dict.fromkeys(sc.CANONICAL_COLUMNS, 0)
    return pd.DataFrame([{**defaults, **row} for row in rows])[sc.CANONICAL_COLUMNS]


# --- hold-out -------------------------------------------------------------------------------


def test_holdout_has_expected_sizes_and_no_overlap():
    y = _target()
    labels = sp.make_holdout(y, test_size=0.2, seed=1)
    assert list(labels.index) == list(range(len(y)))
    assert set(labels.unique()) == {"train", "test"}
    assert (labels == "test").sum() == 200
    assert (labels == "train").sum() == 800
    train_idx = set(labels.index[labels == "train"])
    test_idx = set(labels.index[labels == "test"])
    assert train_idx.isdisjoint(test_idx)
    assert train_idx | test_idx == set(range(len(y)))


def test_holdout_is_reproducible_and_depends_on_the_seed():
    y = _target()
    first = sp.make_holdout(y, 0.2, seed=1)
    assert first.equals(sp.make_holdout(y, 0.2, seed=1))
    assert not first.equals(sp.make_holdout(y, 0.2, seed=2))


def test_holdout_keeps_the_default_rate_in_both_parts():
    y = _target(n=5000, rate=0.22)
    labels = sp.make_holdout(y, 0.2, seed=1)
    train_rate = y[labels == "train"].mean()
    test_rate = y[labels == "test"].mean()
    # Stratification can only miss by a fraction of one row in the smaller part.
    assert abs(train_rate - test_rate) < 1 / (labels == "test").sum()


def test_save_and_load_roundtrip(tmp_path):
    labels = sp.make_holdout(_target(), 0.2, seed=1)
    path = tmp_path / "holdout_index.csv"
    sp.save_holdout(labels, path)
    assert sp.load_holdout(path, n_rows=len(labels)).equals(labels)


def test_load_holdout_rejects_a_file_for_other_data(tmp_path):
    labels = sp.make_holdout(_target(n=100), 0.2, seed=1)
    path = tmp_path / "holdout_index.csv"
    sp.save_holdout(labels, path)
    with pytest.raises(ValueError, match="rows"):
        sp.load_holdout(path, n_rows=101)


def test_load_holdout_rejects_unknown_labels(tmp_path):
    path = tmp_path / "holdout_index.csv"
    pd.DataFrame({"row": [0, 1], "split": ["train", "validation"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="labels"):
        sp.load_holdout(path, n_rows=2)


def test_labels_hash_changes_when_one_row_moves():
    labels = sp.make_holdout(_target(), 0.2, seed=1)
    moved = labels.copy()
    moved.iloc[0] = "test" if moved.iloc[0] == "train" else "train"
    assert len(sp.labels_sha256(labels)) == 64
    assert sp.labels_sha256(labels) == sp.labels_sha256(labels.copy())
    assert sp.labels_sha256(labels) != sp.labels_sha256(moved)


# --- cross-validation -----------------------------------------------------------------------


def test_cv_uses_each_row_once_per_repeat_and_stays_stratified():
    y = _target(n=1000, rate=0.2)
    cv = sp.make_cv(n_folds=5, n_repeats=3, seed=1)
    times_in_validation = np.zeros(len(y), dtype=int)
    folds = list(cv.split(np.zeros(len(y)), y))
    assert len(folds) == 15
    for train_idx, valid_idx in folds:
        assert set(train_idx).isdisjoint(valid_idx)
        assert len(train_idx) + len(valid_idx) == len(y)
        assert abs(y.iloc[valid_idx].mean() - y.mean()) < 0.01
        times_in_validation[valid_idx] += 1
    assert (times_in_validation == 3).all()  # once in each of the three repeats


def test_cv_is_reproducible():
    y = _target()
    first = [v.tolist() for _, v in sp.make_cv(5, 3, seed=1).split(np.zeros(len(y)), y)]
    second = [v.tolist() for _, v in sp.make_cv(5, 3, seed=1).split(np.zeros(len(y)), y)]
    assert first == second


def test_cv_defaults_come_from_the_project_config():
    cfg = load_config("project")
    cv = sp.make_cv()
    assert cv.get_n_splits() == cfg["split"]["cv_folds"] * cfg["split"]["cv_repeats"]


# --- twins across the split -----------------------------------------------------------------


def test_twins_across_split_known_answer():
    df = _frame(
        [
            {"AGE": 30, "SEX": 1},  # 0 train
            {"AGE": 30, "SEX": 2},  # 1 test: same candidate features as row 0 (SEX is ignored)
            {"AGE": 30, "SEX": 1},  # 2 test: another twin of row 0
            {"AGE": 40},  # 3 train: twin of row 4, but both are in train
            {"AGE": 40},  # 4 train
            {"AGE": 50},  # 5 test: no twin anywhere
            {"AGE": 60},  # 6 test: twin of row 7, but both are in test
            {"AGE": 60},  # 7 test
        ]
    )
    labels = pd.Series(["train", "test", "test", "train", "train", "test", "test", "test"])
    assert sp.twins_across_split(df, labels) == {
        "n_test_rows_with_twin_in_train": 2,
        "n_train_rows_with_twin_in_test": 1,
    }


def test_twins_across_split_without_twins():
    df = _frame([{"AGE": 30}, {"AGE": 31}, {"AGE": 32}])
    labels = pd.Series(["train", "test", "train"])
    assert sp.twins_across_split(df, labels) == {
        "n_test_rows_with_twin_in_train": 0,
        "n_train_rows_with_twin_in_test": 0,
    }


# --- the split saved in the repository --------------------------------------------------------


def test_saved_holdout_matches_the_seed_and_the_data():
    # The committed index must be exactly what the project seed produces on the project data.
    cfg = load_config("project")
    raw, _ = load_project_data()
    saved = sp.load_holdout(n_rows=len(raw))
    expected = sp.make_holdout(raw[sc.TARGET], cfg["split"]["test_size"], cfg["seed"])
    assert saved.equals(expected)


def test_train_and_test_do_not_overlap_and_cover_the_file():
    train = sp.load_train()
    test = sp.load_test()
    assert len(train) == 24_000
    assert len(test) == 6_000
    assert set(train.index).isdisjoint(test.index)
    assert set(train.index) | set(test.index) == set(range(sc.EXPECTED_N_ROWS))
    assert list(train.columns) == sc.CANONICAL_COLUMNS


def test_train_and_test_carry_the_fixed_recoding():
    for part in (sp.load_train(), sp.load_test()):
        assert set(part["EDUCATION"].unique()) <= sc.DOCUMENTED_CODES["EDUCATION"]
        assert set(part["MARRIAGE"].unique()) <= sc.DOCUMENTED_CODES["MARRIAGE"]


def test_split_report_is_consistent():
    raw, _ = load_project_data()
    report = sp.split_report(raw, sp.load_holdout(n_rows=len(raw)))
    assert report["train"]["n_rows"] + report["test"]["n_rows"] == len(raw)
    assert report["train"]["n_defaults"] + report["test"]["n_defaults"] == sc.EXPECTED_DEFAULT_COUNT
    low, high = report["test"]["default_rate_ci95"]
    assert low < report["test"]["default_rate"] < high
    assert report["cv"]["n_folds_total"] == 15
    assert report["cv"]["validation_rows_min"] <= report["cv"]["validation_rows_max"]
