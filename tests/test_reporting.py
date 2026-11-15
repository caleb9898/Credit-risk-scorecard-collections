import json

import numpy as np
import pytest

from creditrisk import reporting
from creditrisk.config import load_config


def test_save_metrics_adds_provenance(tmp_path):
    path = reporting.save_metrics("demo", {"auc": 0.5}, data_sha256="abc", metrics_dir=tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "demo.json"
    assert saved["metrics"] == {"auc": 0.5}
    prov = saved["provenance"]
    assert prov["seed"] == load_config("project")["seed"]
    assert prov["data_sha256"] == "abc"
    assert isinstance(prov["git_sha"], str) and prov["git_sha"]
    assert isinstance(prov["uncommitted_code_changes"], bool)


def test_save_metrics_is_deterministic(tmp_path):
    metrics = {"b": 2, "a": {"z": 1.5, "y": [1, 2]}}
    first = reporting.save_metrics("demo", metrics, "abc", metrics_dir=tmp_path).read_bytes()
    second = reporting.save_metrics("demo", metrics, "abc", metrics_dir=tmp_path).read_bytes()
    assert first == second


def test_save_metrics_converts_numpy_values(tmp_path):
    metrics = {"n": np.int64(3), "rate": np.float64(0.25), "flag": np.bool_(True)}
    path = reporting.save_metrics("demo", metrics, "abc", metrics_dir=tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    assert saved == {"n": 3, "rate": 0.25, "flag": True}


def test_save_metrics_rejects_nan(tmp_path):
    # NaN is not valid JSON and usually hides an empty denominator: fail loudly.
    with pytest.raises(ValueError):
        reporting.save_metrics("demo", {"rate": float("nan")}, "abc", metrics_dir=tmp_path)


def test_provenance_flags_uncommitted_code(monkeypatch):
    answers = {"rev-parse": "abc123", "status": " M src/creditrisk/reporting.py"}
    monkeypatch.setattr(reporting, "_git", lambda *args: answers[args[0]])
    prov = reporting.provenance("hash")
    assert prov["git_sha"] == "abc123"
    assert prov["uncommitted_code_changes"] is True


def test_provenance_with_clean_code(monkeypatch):
    answers = {"rev-parse": "abc123", "status": ""}
    monkeypatch.setattr(reporting, "_git", lambda *args: answers[args[0]])
    assert reporting.provenance("hash")["uncommitted_code_changes"] is False


def test_provenance_without_git(monkeypatch):
    # No git available: the commit is unknown, so the numbers cannot be called reproducible.
    monkeypatch.setattr(reporting, "_git", lambda *args: None)
    prov = reporting.provenance("hash")
    assert prov["git_sha"] == "unknown"
    assert prov["uncommitted_code_changes"] is True
