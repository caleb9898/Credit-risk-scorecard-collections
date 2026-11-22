"""Which candidate features go forward to the model, and the Phase 4 report (ADR-009, ADR-011).

Everything here is computed on the training part only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditrisk.data.schema import N_MONTHS, TARGET
from creditrisk.data.split import make_cv
from creditrisk.features.diagnostics import (
    correlated_pairs,
    fold_stability,
    iv_band,
    preselect,
    vif,
)
from creditrisk.features.engineering import FEATURE_SPECS, build_features, status_group
from creditrisk.features.woe import WoeBinner

# Features kept out of the champion model, with the decision behind each one.
EXCLUDED_FROM_CHAMPION = {
    "STATUS_SEP": "ADR-009: possible information from the outcome month; sensitivity model only",
    "AGE": "ADR-011: IV close to 0.02, about half of it chance; unstable across folds",
    "EDUCATION": "ADR-011: real but weak signal (IV 0.022); not enough to justify it (ADR-002)",
}
# Added back, on top of the champion's shortlist, only in the sensitivity model of ADR-009.
SENSITIVITY_FEATURES = ["STATUS_SEP"]

MIN_IV = 0.02
MAX_CORRELATION = 0.7
VIF_FLAG = 5.0
IV_ALARM = 0.5
N_PERMUTATIONS = 100  # shuffles per fold and feature, for the chance level of the validation IV


def champion_specs() -> dict[str, dict]:
    """Binning settings of the candidate features allowed in the champion model."""
    return {k: v for k, v in FEATURE_SPECS.items() if k not in EXCLUDED_FROM_CHAMPION}


def status_iv_by_month(train: pd.DataFrame) -> pd.Series:
    """IV of the grouped repayment status of each month (1 = September ... 6 = April)."""
    names = {f"PAY_{k}": k for k in range(1, N_MONTHS + 1)}
    status = pd.DataFrame({name: status_group(train[name]) for name in names})
    specs = dict.fromkeys(names, {"dtype": "categorical", "monotonic_trend": None})
    binner = WoeBinner(specs).fit(status, train[TARGET])
    return binner.iv_table()["iv"].rename(index=names).rename("iv")


def _records(table: pd.DataFrame) -> list[dict]:
    """Rows of a table as JSON-friendly dicts, writing missing values as null."""
    return table.astype(object).where(table.notna(), None).to_dict(orient="records")


def _bins_records(binner: WoeBinner, name: str) -> list[dict]:
    table = binner.binning_table(name).reset_index()
    return _records(table[["bin", "n", "events", "event_rate", "woe", "iv"]])


def woe_iv_report(train: pd.DataFrame) -> dict:
    """Binning, IV, stability, preselection and the ``PAY_1`` check, as a JSON-friendly dict."""
    X = build_features(train)
    y = train[TARGET]

    binner = WoeBinner(FEATURE_SPECS).fit(X, y)
    iv = binner.iv_table()
    iv["band"] = iv["iv"].map(iv_band)
    iv["optbinning_iv"] = pd.Series(binner.optbinning_iv_)
    iv["in_champion_candidates"] = ~iv.index.isin(list(EXCLUDED_FROM_CHAMPION))

    folds = list(make_cv().split(np.zeros(len(y)), y))
    stability = fold_stability(
        X, y, folds, FEATURE_SPECS, min_iv=MIN_IV, n_permutations=N_PERMUTATIONS, seed=0
    )

    candidates = list(champion_specs())
    woe_values = binner.transform(X)
    shortlist = preselect(iv.loc[candidates, "iv"], woe_values[candidates], MIN_IV, MAX_CORRELATION)
    kept = list(shortlist.index[shortlist["kept"]])
    vif_kept = vif(woe_values[kept])

    by_month = status_iv_by_month(train)
    return {
        "n_train_rows": int(len(train)),
        "n_train_inactive": int(X["INACTIVE"].sum()),
        "settings": {
            "min_bin_size": binner.min_bin_size,
            "min_iv": MIN_IV,
            "max_correlation": MAX_CORRELATION,
            "vif_flag": VIF_FLAG,
            "iv_alarm": IV_ALARM,
            "n_permutations": N_PERMUTATIONS,
            "monotonic_trend": {k: v["monotonic_trend"] for k, v in FEATURE_SPECS.items()},
        },
        "iv_table": _records(iv.reset_index()),
        "bins": {name: _bins_records(binner, name) for name in FEATURE_SPECS},
        "fold_stability": _records(stability.reset_index()),
        "excluded_from_champion": EXCLUDED_FROM_CHAMPION,
        "preselection": _records(shortlist.reset_index()),
        "shortlist": kept,
        "sensitivity_features": SENSITIVITY_FEATURES,
        "correlated_pairs_all_candidates": _records(
            correlated_pairs(woe_values[candidates], MAX_CORRELATION)
        ),
        "vif_shortlist": vif_kept.to_dict(),
        "n_shortlist_vif_above_flag": int((vif_kept > VIF_FLAG).sum()),
        "pay_1_check": {
            "feature": "STATUS_SEP",
            "iv": float(iv.loc["STATUS_SEP", "iv"]),
            "alarm": bool(iv.loc["STATUS_SEP", "iv"] > IV_ALARM),
            "status_iv_by_month": {int(k): float(v) for k, v in by_month.items()},
        },
        "features_above_iv_alarm": list(iv.index[iv["iv"] > IV_ALARM]),
    }
