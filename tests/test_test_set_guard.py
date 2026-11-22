"""Guard for the hold-out: code that can see test rows must be on a short, deliberate list.

``load_test`` returns the hold-out, and ``load_project_data`` returns all 30,000 rows, test rows
included. Anything fitted on either one would leak. This test fails when a new source file or
notebook calls them, so that the call is noticed and, if it is legitimate, added here on purpose.
"""

import json

from creditrisk.config import REPO_ROOT

# Files allowed to load the full data: the loader itself, the split code, and the data-quality
# checks that ran before the split existed (they use the target only for the overall rate).
MAY_LOAD_ALL_ROWS = {
    "src/creditrisk/data/load.py",
    "src/creditrisk/data/split.py",
    "notebooks/01_eda.ipynb",
}
# Files allowed to load the hold-out. Add the final-evaluation code here when it exists.
MAY_LOAD_TEST = {
    "src/creditrisk/data/split.py",
    "src/creditrisk/models/final_evaluation.py",
}


def _source_code():
    """Yield (relative path, code) for every library module and every notebook."""
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        yield path.relative_to(REPO_ROOT).as_posix(), path.read_text(encoding="utf-8")
    for path in sorted((REPO_ROOT / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        code = "\n".join("".join(cell["source"]) for cell in cells)
        yield path.relative_to(REPO_ROOT).as_posix(), code


def _files_using(name):
    return {path for path, code in _source_code() if name in code}


def test_only_listed_files_load_the_hold_out():
    assert _files_using("load_test") <= MAY_LOAD_TEST
    assert _files_using("_load_part") <= MAY_LOAD_TEST


def test_only_listed_files_read_the_split_or_the_raw_file_directly():
    allowed = MAY_LOAD_ALL_ROWS | {"src/creditrisk/data/schema.py"}
    assert _files_using("holdout_index") <= allowed
    assert _files_using("load_raw") <= allowed


# Files allowed to run the one-time evaluation of the frozen models on the test set.
MAY_RUN_FINAL_EVALUATION = {
    "src/creditrisk/models/final_evaluation.py",
    "notebooks/03_pd_model_logistic.ipynb",
}


def test_only_listed_files_run_the_final_evaluation():
    assert _files_using("evaluate_on_test") <= MAY_RUN_FINAL_EVALUATION
    assert _files_using("final_evaluation") <= MAY_RUN_FINAL_EVALUATION


def test_only_listed_files_load_all_rows():
    assert _files_using("load_project_data") <= MAY_LOAD_ALL_ROWS


def test_the_guard_reads_the_notebooks():
    # If this fails the scan is not looking at notebook code, and the guard would be blind.
    assert "notebooks/01_eda.ipynb" in _files_using("load_project_data")
