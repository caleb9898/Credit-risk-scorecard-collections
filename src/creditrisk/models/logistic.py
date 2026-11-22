"""Logistic regression on WoE features, fitted with statsmodels (ADR-012).

Sign convention: WoE = ln(share of non-defaults / share of defaults), so a higher WoE means a
safer bin. In a model of the probability of default every coefficient must therefore be
negative. A positive coefficient is a "wrong sign": it would say that a safer bin is riskier,
which usually means the variable is overlapping with others, and it cannot be defended.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import BinaryResultsWrapper


def fit_logit(X: pd.DataFrame, y) -> BinaryResultsWrapper:
    """Unpenalised maximum-likelihood logistic regression with an intercept named ``const``."""
    design = sm.add_constant(X, has_constant="add")
    return sm.Logit(np.asarray(y), design).fit(disp=0)


def coefficient_table(result: BinaryResultsWrapper) -> pd.DataFrame:
    """Estimate, standard error, z, p-value and 95% confidence interval of every coefficient."""
    ci = result.conf_int(alpha=0.05)
    table = pd.DataFrame(
        {
            "coef": result.params,
            "std_err": result.bse,
            "z": result.tvalues,
            "p_value": result.pvalues,
            "ci_low": ci[0],
            "ci_high": ci[1],
        }
    )
    table.index.name = "term"
    return table


def select_features(
    X: pd.DataFrame, y, max_p: float = 0.05
) -> tuple[BinaryResultsWrapper, list[str], list[tuple[str, str]]]:
    """Backward elimination, one variable at a time, until every coefficient is defensible.

    At each step the model is refitted and one variable is removed:
    1. if any coefficient has the wrong (positive) sign, the one of them with the largest p-value;
    2. otherwise, if any p-value is above ``max_p``, the variable with the largest p-value.

    Returns the final fit, the kept variables (in their original order) and the removed ones
    with the reason, in the order they were removed.
    """
    features = list(X.columns)
    dropped: list[tuple[str, str]] = []
    while features:
        result = fit_logit(X[features], y)
        coefs = result.params.drop("const")
        p_values = result.pvalues.drop("const")
        wrong_sign = coefs.index[coefs > 0]
        if len(wrong_sign):
            name, reason = p_values[wrong_sign].idxmax(), "wrong sign"
        elif p_values.max() > max_p:
            name, reason = p_values.idxmax(), f"p-value above {max_p:g}"
        else:
            return result, features, dropped
        features.remove(name)
        dropped.append((name, reason))
    raise ValueError("No variable survived the selection")
