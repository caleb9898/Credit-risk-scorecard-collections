"""Hold-out split and cross-validation design (ADR-003, ADR-010).

The hold-out is created once, from the project seed, and saved in ``data/splits/holdout_index.csv``
with one line per row of the raw file: its position (0 = first data row) and ``train`` or ``test``.
Later phases never create the split again. They call ``load_train`` for all their work and
``load_test`` only for the final evaluation of a finished model.

Run ``python -m creditrisk.data.split`` to create the index file (if it does not exist yet) and
write ``reports/metrics/split.json``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from statsmodels.stats.proportion import proportion_confint

from creditrisk.config import REPO_ROOT, load_config
from creditrisk.data.clean import recode_undocumented_codes
from creditrisk.data.load import load_project_data
from creditrisk.data.schema import CANDIDATE_FEATURE_COLS, TARGET
from creditrisk.reporting import save_metrics

INDEX_PATH = REPO_ROOT / "data" / "splits" / "holdout_index.csv"
LABELS = {"train", "test"}


def make_holdout(target: pd.Series, test_size: float, seed: int) -> pd.Series:
    """Stratified hold-out: label every row ``train`` or ``test``.

    The split is stratified on the target, so both parts keep the same default rate. Returns a
    Series named ``split`` indexed by row position (0 .. n-1).
    """
    positions = np.arange(len(target))
    _, test_positions = train_test_split(
        positions, test_size=test_size, stratify=target.to_numpy(), random_state=seed
    )
    labels = pd.Series("train", index=pd.RangeIndex(len(target)), name="split")
    labels.iloc[test_positions] = "test"
    return labels


def save_holdout(labels: pd.Series, path: Path = INDEX_PATH) -> Path:
    """Write the labels as a two-column CSV (``row``, ``split``); return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.rename_axis("row").to_csv(path, lineterminator="\n")
    return path


def load_holdout(path: Path = INDEX_PATH, n_rows: int | None = None) -> pd.Series:
    """Read the saved labels and check that they fit the data.

    Raises ValueError if the file does not list every row position exactly once, if it has labels
    other than ``train``/``test``, or if ``n_rows`` is given and differs from its length.
    """
    labels = pd.read_csv(path).set_index("row")["split"]
    if n_rows is not None and len(labels) != n_rows:
        raise ValueError(f"{Path(path).name} has {len(labels)} rows, the data has {n_rows}")
    if list(labels.index) != list(range(len(labels))):
        raise ValueError(f"{Path(path).name} must list rows 0..n-1 in order, once each")
    if not set(labels.unique()) <= LABELS:
        raise ValueError(f"{Path(path).name} has labels other than {sorted(LABELS)}")
    labels.index = pd.RangeIndex(len(labels))
    return labels


def labels_sha256(labels: pd.Series) -> str:
    """SHA-256 of the sequence of labels, to pin the exact split in the saved metrics.

    Computed from the labels, not from the file, so it does not depend on line endings.
    """
    return hashlib.sha256("\n".join(labels).encode("utf-8")).hexdigest()


def _load_part(label: str) -> pd.DataFrame:
    """Rows of the cleaned project data with the given label; the index is the row position."""
    raw, _ = load_project_data()
    labels = load_holdout(n_rows=len(raw))
    return recode_undocumented_codes(raw)[(labels == label).to_numpy()]


def load_train() -> pd.DataFrame:
    """Training part (80%) with the fixed recoding of ADR-008 applied. Use this everywhere."""
    return _load_part("train")


def load_test() -> pd.DataFrame:
    """Hold-out part (20%) with the fixed recoding applied.

    Only for the final evaluation of a finished model: once per model, and nothing is changed
    afterwards because of what it shows.
    """
    return _load_part("test")


def make_cv(
    n_folds: int | None = None, n_repeats: int | None = None, seed: int | None = None
) -> RepeatedStratifiedKFold:
    """Repeated stratified k-fold for model selection on the training part.

    Defaults (5 folds, 3 repeats, project seed) come from ``config/project.yaml``.
    """
    config = load_config("project")
    return RepeatedStratifiedKFold(
        n_splits=config["split"]["cv_folds"] if n_folds is None else n_folds,
        n_repeats=config["split"]["cv_repeats"] if n_repeats is None else n_repeats,
        random_state=config["seed"] if seed is None else seed,
    )


def twins_across_split(df: pd.DataFrame, labels: pd.Series) -> dict[str, int]:
    """Count rows that have a twin on the other side of the split.

    Two rows are twins when they are identical on every candidate model feature. A test row with
    a twin in train is not a fully independent test case, so the number is reported (ADR-008).
    """
    features = df[CANDIDATE_FEATURE_COLS]
    is_test = (labels == "test").to_numpy()
    train_profiles = features[~is_test].drop_duplicates()
    test_profiles = features[is_test].drop_duplicates()

    def count_matches(rows: pd.DataFrame, profiles: pd.DataFrame) -> int:
        merged = rows.merge(profiles, on=CANDIDATE_FEATURE_COLS, how="left", indicator=True)
        return int((merged["_merge"] == "both").sum())

    return {
        "n_test_rows_with_twin_in_train": count_matches(features[is_test], train_profiles),
        "n_train_rows_with_twin_in_test": count_matches(features[~is_test], test_profiles),
    }


def _part_summary(target: pd.Series) -> dict:
    """Rows, defaults, default rate and its 95% Wilson interval for one part of the split."""
    n_rows, n_defaults = int(len(target)), int(target.sum())
    low, high = proportion_confint(n_defaults, n_rows, alpha=0.05, method="wilson")
    return {
        "n_rows": n_rows,
        "n_defaults": n_defaults,
        "default_rate": n_defaults / n_rows,
        "default_rate_ci95": [float(low), float(high)],
    }


def split_report(raw: pd.DataFrame, labels: pd.Series) -> dict:
    """Describe the hold-out and the cross-validation folds in one JSON-friendly dictionary."""
    config = load_config("project")
    is_test = (labels == "test").to_numpy()
    train_target = raw.loc[~is_test, TARGET]
    train = _part_summary(train_target)
    test = _part_summary(raw.loc[is_test, TARGET])

    folds = list(make_cv().split(np.zeros(len(train_target)), train_target))
    validation_rows = [len(valid) for _, valid in folds]
    validation_rates = [float(train_target.iloc[valid].mean()) for _, valid in folds]

    return {
        "test_size": config["split"]["test_size"],
        "labels_sha256": labels_sha256(labels),
        "train": train,
        "test": test,
        "default_rate_difference_train_minus_test": train["default_rate"] - test["default_rate"],
        "twins": twins_across_split(recode_undocumented_codes(raw), labels),
        "cv": {
            "n_folds": config["split"]["cv_folds"],
            "n_repeats": config["split"]["cv_repeats"],
            "n_folds_total": len(folds),
            "validation_rows_min": min(validation_rows),
            "validation_rows_max": max(validation_rows),
            "validation_default_rate_min": min(validation_rates),
            "validation_default_rate_max": max(validation_rates),
        },
    }


def main() -> int:
    """Create the hold-out index if it does not exist, check it, and save ``split.json``."""
    config = load_config("project")
    raw, data_sha256 = load_project_data()
    expected = make_holdout(raw[TARGET], config["split"]["test_size"], config["seed"])

    if INDEX_PATH.exists():
        # The hold-out is created once. An existing file is checked, never replaced.
        if not load_holdout(n_rows=len(raw)).equals(expected):
            raise ValueError(
                f"{INDEX_PATH.name} does not match the split given by the project seed. "
                "Find out what changed (seed, data or library version) before going on."
            )
        print(f"hold-out index: {INDEX_PATH.name} already exists and matches the seed")
    else:
        save_holdout(expected)
        print(f"hold-out index: created {INDEX_PATH.name}")

    report = split_report(raw, expected)
    path = save_metrics("split", report, data_sha256)
    print(f"train: {report['train']['n_rows']} rows, rate {report['train']['default_rate']:.4f}")
    print(f"test:  {report['test']['n_rows']} rows, rate {report['test']['default_rate']:.4f}")
    print(f"saved: {path.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
