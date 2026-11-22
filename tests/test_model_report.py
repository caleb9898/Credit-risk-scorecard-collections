"""Phase 5 report and final-evaluation code, run on training rows only.

The final evaluation is tested with ``load_test`` replaced by a slice of the training data, so the
real test set is never read here.
"""

import numpy as np
import pytest

from creditrisk.data.split import load_train
from creditrisk.features.engineering import build_features
from creditrisk.models import final_evaluation as fe
from creditrisk.models import report as rp
from creditrisk.models.pipeline import WoeLogitModel


@pytest.fixture(scope="module")
def train():
    return load_train()


def test_model_specs_follow_adr_009_and_011():
    specs = rp.model_specs()
    assert "STATUS_SEP" not in specs["champion"]
    assert "AGE" not in specs["champion"] and "EDUCATION" not in specs["champion"]
    assert set(specs["sensitivity_with_status_sep"]) == set(specs["champion"]) | {"STATUS_SEP"}


def test_train_report_structure(train):
    report, models = rp.train_report(train.iloc[:6000], n_boot=20)
    assert set(models) == {"champion", "sensitivity_with_status_sep"}
    champion = report["models"]["champion"]
    assert "STATUS_SEP" not in champion["final_features"]
    assert champion["cv"]["n_folds"] == 15
    assert set(champion["cv"]["auc_by_repeat"]) == {1, 2, 3}
    coefs = {row["term"]: row["coef"] for row in champion["coefficients"]}
    assert all(coefs[f] < 0 for f in champion["final_features"])  # WoE convention: all negative
    assert report["operating_thresholds"][1] == 0.5


def test_final_evaluation_on_a_stand_in_test_set(train, monkeypatch):
    fit_rows, stand_in = train.iloc[:12000], train.iloc[12000:15000]
    monkeypatch.setattr(fe, "load_train", lambda: fit_rows)
    monkeypatch.setattr(fe, "load_test", lambda: stand_in)
    specs = rp.model_specs()
    X, y = build_features(fit_rows), fit_rows["DEFAULT"]
    models = {name: WoeLogitModel(s).fit(X, y) for name, s in specs.items()}
    out = fe.evaluate_on_test(
        models,
        thresholds=(0.22, 0.5),
        compare=("sensitivity_with_status_sep", "champion"),
        n_boot=20,
    )
    assert out["n_test_rows"] == 3000
    assert 0 <= out["n_test_rows_with_twin_in_train"] < 3000
    for name in specs:
        both = out["models"][name]
        assert both["all_rows"]["n_rows"] == 3000
        expected = 3000 - out["n_test_rows_with_twin_in_train"]
        assert both["without_twins"]["n_rows"] == expected
    diff = out["auc_difference"]
    first = out["models"]["sensitivity_with_status_sep"]["all_rows"]["auc"]["estimate"]
    second = out["models"]["champion"]["all_rows"]["auc"]["estimate"]
    assert diff["estimate"] == pytest.approx(first - second)
    assert np.isfinite([diff["low"], diff["high"]]).all()
