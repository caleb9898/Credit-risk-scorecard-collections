"""Checks on binned features: IV bands, correlation, VIF, preselection and stability across folds.

Thresholds are the ones declared in ``docs/RIGOR_STANDARDS.md`` and ADR-011, fixed before looking
at the results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

from creditrisk.features.woe import WoeBinner, information_value, woe_table

# Upper limit of each IV band; above the last one an IV is suspicious (check for leakage).
IV_BANDS = [(0.02, "useless"), (0.1, "weak"), (0.3, "medium"), (0.5, "strong")]


def iv_band(iv: float) -> str:
    """Usual reading of an IV: <0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong,
    >0.5 suspicious. A band includes its lower limit; 0.5 itself is still "strong"."""
    for upper, band in IV_BANDS[:-1]:
        if iv < upper:
            return band
    return "strong" if iv <= IV_BANDS[-1][0] else "suspicious"


def correlated_pairs(values: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Pairs of columns whose Pearson correlation is above ``threshold`` in absolute value."""
    corr = values.corr()
    rows = [
        {"feature_1": a, "feature_2": b, "correlation": float(corr.loc[a, b])}
        for i, a in enumerate(corr.columns)
        for b in corr.columns[i + 1 :]
        if abs(corr.loc[a, b]) > threshold
    ]
    pairs = pd.DataFrame(rows, columns=["feature_1", "feature_2", "correlation"])
    return pairs.sort_values("correlation", key=abs, ascending=False, ignore_index=True)


def vif(values: pd.DataFrame) -> pd.Series:
    """Variance inflation factor of every column, with an intercept in the regression."""
    design = add_constant(values, has_constant="add").to_numpy(dtype=float)
    return pd.Series(
        [variance_inflation_factor(design, i + 1) for i in range(values.shape[1])],
        index=values.columns,
        name="vif",
    )


def preselect(
    iv: pd.Series,
    woe_values: pd.DataFrame,
    min_iv: float = 0.02,
    max_correlation: float = 0.7,
) -> pd.DataFrame:
    """Shortlist features: IV at least ``min_iv`` and no strong correlation with a stronger one.

    Features are visited from the highest IV down. A feature is kept unless its IV is below
    ``min_iv`` or its WoE correlates above ``max_correlation`` (in absolute value) with a feature
    already kept. Returns one row per feature, ordered by IV, with ``kept`` and ``reason``.
    """
    corr = woe_values.corr()
    kept: list[str] = []
    rows = []
    for name in iv.sort_values(ascending=False).index:
        if iv[name] < min_iv:
            reason = f"IV below {min_iv:g}"
        else:
            partner = next((k for k in kept if abs(corr.loc[name, k]) > max_correlation), None)
            reason = f"correlated with {partner}" if partner else "kept"
        if reason == "kept":
            kept.append(name)
        rows.append(
            {"feature": name, "iv": float(iv[name]), "kept": reason == "kept", "reason": reason}
        )
    return pd.DataFrame(rows).set_index("feature")


def fold_stability(
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    specs: dict[str, dict],
    min_bin_size: float = 0.05,
    min_iv: float = 0.02,
    n_permutations: int = 0,
    seed: int = 0,
) -> pd.DataFrame:
    """Refit the binning in every fold and check it on the rows left out of that fold.

    ``folds`` holds (fitting rows, validation rows) as positions in ``X``. For every feature:

    - ``iv_fit_mean``: IV on the fitting rows, averaged over folds;
    - ``iv_valid_mean``, ``iv_valid_min``, ``iv_valid_max``: IV of the same bins on the validation
      rows (0.5 is added to every count there, so a small bin without defaults stays finite);
    - ``n_folds_iv_valid_at_least_min``: folds where that validation IV reaches ``min_iv``;
    - ``woe_rank_corr_mean``, ``woe_rank_corr_min``: Spearman correlation between the WoE of the
      bins on the fitting rows and on the validation rows. 1 means the risk ordering of the bins
      learned in the fold holds on the rows it did not see. Missing when a fold has a single bin.
    - ``n_bins_min``, ``n_bins_max``: number of bins found across folds.

    The validation IV is not compared like for like with the fitting IV: it is smoothed, and it
    leaves out validation rows whose category was not seen when fitting. With small samples it is
    also biased upwards, more so the more bins a feature has. ``n_permutations`` > 0 measures that
    bias: the validation target is shuffled that many times, keeping the same bins, which gives
    the IV that pure chance would produce. Extra columns then:

    - ``iv_null_mean``: mean IV under shuffling, averaged over folds;
    - ``n_folds_iv_valid_above_null_p95``: folds where the validation IV is above the 95th
      percentile of its own shuffled IVs, i.e. clearly more than chance.

    With two bins the rank correlation can only be +1 or -1, so it says little.
    """
    rng = np.random.default_rng(seed)
    y = pd.Series(np.asarray(y), index=X.index)
    records = {name: [] for name in specs}
    for fit_rows, valid_rows in folds:
        X_fit, y_fit = X.iloc[fit_rows], y.iloc[fit_rows]
        X_valid, y_valid = X.iloc[valid_rows], y.iloc[valid_rows]
        binner = WoeBinner(specs, min_bin_size=min_bin_size).fit(X_fit, y_fit)
        fit_iv = binner.iv_table()
        valid_bins = binner.assign_bins(X_valid)
        for name in specs:
            valid_table = woe_table(valid_bins[name], y_valid, smoothing=0.5)
            fit_woe = pd.Series(binner.woe_[name]).reindex(valid_table.index)
            seen = fit_woe.notna()
            rank_corr = (
                fit_woe[seen].corr(valid_table.loc[seen, "woe"], method="spearman")
                if seen.sum() >= 2
                else np.nan
            )
            record = {
                "iv_fit": fit_iv.loc[name, "iv"],
                "iv_valid": float(valid_table["iv"].sum()),
                "rank_corr": float(rank_corr),
                "n_bins": int(fit_iv.loc[name, "n_bins"]),
            }
            if n_permutations:
                shuffled = [
                    information_value(valid_bins[name], rng.permutation(y_valid.to_numpy()), 0.5)
                    for _ in range(n_permutations)
                ]
                record["iv_null_mean"] = float(np.mean(shuffled))
                record["above_null_p95"] = record["iv_valid"] > float(np.quantile(shuffled, 0.95))
            records[name].append(record)
    rows = {}
    for name, per_fold in records.items():
        folds_df = pd.DataFrame(per_fold)
        rows[name] = {
            "n_folds": len(folds_df),
            "iv_fit_mean": folds_df["iv_fit"].mean(),
            "iv_valid_mean": folds_df["iv_valid"].mean(),
            "iv_valid_min": folds_df["iv_valid"].min(),
            "iv_valid_max": folds_df["iv_valid"].max(),
            "n_folds_iv_valid_at_least_min": int((folds_df["iv_valid"] >= min_iv).sum()),
            "woe_rank_corr_mean": folds_df["rank_corr"].mean(),
            "woe_rank_corr_min": folds_df["rank_corr"].min(),
            "n_bins_min": int(folds_df["n_bins"].min()),
            "n_bins_max": int(folds_df["n_bins"].max()),
        }
        if n_permutations:
            rows[name]["iv_null_mean"] = folds_df["iv_null_mean"].mean()
            rows[name]["n_folds_iv_valid_above_null_p95"] = int(folds_df["above_null_p95"].sum())
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "feature"
    return table
