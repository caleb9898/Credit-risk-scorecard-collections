"""Load the project's raw data file, making sure it is the exact file the project was built on."""

from __future__ import annotations

from typing import Any

import pandas as pd

from creditrisk.config import REPO_ROOT, load_config
from creditrisk.data.schema import load_raw, sha256_file, validate_raw


def load_project_data(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, str]:
    """Load and validate the raw file named in ``config/project.yaml``.

    Returns the standardised frame (values untouched) and the file's SHA-256, which is saved
    next to every metric. Raises ValueError if the hash differs from the one in the config.
    """
    config = load_config("project") if config is None else config
    path = REPO_ROOT / "data" / "raw" / config["data"]["raw_file"]
    sha256 = sha256_file(path)
    if sha256 != config["data"]["raw_sha256"]:
        raise ValueError(
            f"SHA-256 of {path.name} is {sha256}, expected {config['data']['raw_sha256']}. "
            "This is not the file the project was built on."
        )
    df = load_raw(path)
    validate_raw(df)
    return df, sha256
