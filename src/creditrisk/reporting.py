"""Save metrics to ``reports/metrics/`` together with the information needed to trace them.

Every number quoted in the README, the notebooks or the docs must come from one of these files.
Each file records the git commit of the code, the project seed, the SHA-256 of the raw data and
the versions of Python and of the main libraries.
No timestamp is written, so running the same code on the same data gives the same file.
"""

from __future__ import annotations

import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from creditrisk.config import REPO_ROOT, load_config

METRICS_DIR = REPO_ROOT / "reports" / "metrics"

# Libraries whose version can change a result (random splits, solvers, numerical defaults).
TRACKED_LIBRARIES = ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "optbinning"]


def _git(*args: str) -> str | None:
    """Run a git command in the repository; return its output, or None if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def library_versions() -> dict[str, str]:
    """Version of Python and of each tracked library ("not installed" if it is missing)."""
    versions = {"python": platform.python_version()}
    for name in TRACKED_LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def provenance(data_sha256: str) -> dict[str, Any]:
    """Git commit, uncommitted-code flag, seed, data hash and library versions for the run.

    ``uncommitted_code_changes`` is True when files under ``src/`` or ``config/`` differ from the
    recorded commit, which means the commit alone does not reproduce the numbers.
    """
    status = _git("status", "--porcelain", "--", "src", "config")
    return {
        "git_sha": _git("rev-parse", "HEAD") or "unknown",
        "uncommitted_code_changes": bool(status) if status is not None else True,
        "seed": load_config("project")["seed"],
        "data_sha256": data_sha256,
        "library_versions": library_versions(),
    }


def _to_builtin(value: Any) -> Any:
    """Convert numpy scalars to plain Python types so that json can write them."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot write a value of type {type(value).__name__} to JSON")


def save_metrics(
    name: str,
    metrics: dict[str, Any],
    data_sha256: str,
    metrics_dir: Path | None = None,
) -> Path:
    """Write ``<metrics_dir>/<name>.json`` with the metrics and their provenance; return the path.

    Raises ValueError if a metric is NaN or infinite: that is not valid JSON and usually means an
    empty denominator that should be handled where the metric is computed.
    """
    folder = METRICS_DIR if metrics_dir is None else Path(metrics_dir)
    folder.mkdir(parents=True, exist_ok=True)
    payload = {"provenance": provenance(data_sha256), "metrics": metrics}
    text = json.dumps(payload, indent=2, sort_keys=True, default=_to_builtin, allow_nan=False)
    path = folder / f"{name}.json"
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
    return path
