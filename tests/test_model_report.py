"""Phase 5 report and final-evaluation code, run on training rows only.

The final evaluation is tested with ``load_test`` replaced by a slice of the training data and
with a temporary metrics folder, so the real test set and the real metrics are never touched.
"""

import json

import numpy as np
import pytest

from creditrisk.data.split import has_twin_in, load_train
from creditrisk.features.engineering import build_features
from creditrisk.models import final_evaluation as fe
from creditrisk.models import report as rp
from creditrisk.models.evaluation import evaluate_predictions
from creditrisk.models.pipeline import WoeLogitModel
from creditrisk.reporting import save_metrics


@pytest.fixture(scope="module")
def train():
    return load_train()


def test_model_specs_follow_adr_009_and_011():
    specs = rp.model_specs()
    assert "STATUS_SEP" not in specs["champion"]
    assert "AGE" not in specs["champion"] and "EDUCATION" not in specs["champion"]
    assert set(specs["sensitivity_with_status_sep"]) == set(specs["champion"]) | {"STATUS_SEP"}


@pytest.fixture(scope="module")
def small_report(train):
    return rp.train_report(train.iloc[:6000], n_boot=20)


def test_train_report_structure(small_report):
    report, models, oof = small_report
    assert set(oof) == set(models) == {"champion", "sensitivity_with_status_sep"}
    assert oof["champion"].notna().all() and len(oof["champion"]) == 6000
    champion = report["models"]["champion"]
    assert "STATUS_SEP" not in champion["final_features"]
    assert champion["cv"]["n_folds"] == 15
    assert set(champion["cv"]["auc_by_repeat"]) == {1, 2, 3}
    coefs = {row["term"]: row["coef"] for row in champion["coefficients"]}
    assert all(coefs[f] < 0 for f in champion["final_features"])  # WoE convention: all negative
    assert report["operating_thresholds"][1] == 0.5
    assert report["bootstrap_seed"] == 20261004


@pytest.fixture
def frozen_setup(train, tmp_path, monkeypatch):
    """Models fitted on part of train, a frozen CV report for them, and a stand-in test set."""
    fit_rows, stand_in = train.iloc[:12000], train.iloc[12000:15000]
    monkeypatch.setattr(fe, "load_train", lambda: fit_rows)
    monkeypatch.setattr(fe, "load_test", lambda: stand_in)
    report, models, _ = rp.train_report(fit_rows, n_boot=20)
    save_metrics(fe.CV_METRICS, report, "abc", metrics_dir=tmp_path)
    return models, stand_in, fit_rows, tmp_path


def test_final_evaluation_reports_all_rows_and_rows_without_twins(frozen_setup):
    models, stand_in, fit_rows, folder = frozen_setup
    out = fe.evaluate_on_test(
        models, compare=("sensitivity_with_status_sep", "champion"), n_boot=20, metrics_dir=folder
    )
    twin = has_twin_in(stand_in, fit_rows)
    assert out["n_test_rows"] == 3000
    assert out["n_test_rows_with_twin_in_train"] == int(twin.sum())
    # The "without twins" block is exactly the evaluation of the rows without a twin.
    p = models["champion"].predict_pd(build_features(stand_in))
    thresholds = out["models"]["champion"]["without_twins"]["operating_points"]
    expected = evaluate_predictions(
        stand_in["DEFAULT"][~twin],
        p[~twin],
        n_boot=20,
        seed=out["bootstrap_seed"],
        thresholds=[r["threshold"] for r in thresholds if r["point"].startswith("PD")],
    )
    assert out["models"]["champion"]["without_twins"]["auc"] == expected["auc"]
    diff = out["auc_difference"]["all_rows"]
    first = out["models"]["sensitivity_with_status_sep"]["all_rows"]["auc"]["estimate"]
    second = out["models"]["champion"]["all_rows"]["auc"]["estimate"]
    assert diff["estimate"] == pytest.approx(first - second)
    assert np.isfinite([diff["low"], diff["high"]]).all()
    saved = json.loads((folder / "pd_logistic_test.json").read_text(encoding="utf-8"))
    assert saved["metrics"]["frozen_models"]["champion"]["final_features"] == list(
        models["champion"].features_
    )


def test_final_evaluation_refuses_a_model_that_is_not_frozen(frozen_setup):
    models, stand_in, _, folder = frozen_setup
    other = WoeLogitModel(rp.model_specs()["champion"]).fit(
        build_features(stand_in), stand_in["DEFAULT"]
    )
    with pytest.raises(ValueError, match="not the frozen model"):
        fe.evaluate_on_test(
            {**models, "champion": other},
            compare=("sensitivity_with_status_sep", "champion"),
            n_boot=20,
            metrics_dir=folder,
        )


def test_final_evaluation_refuses_a_second_version_of_a_model(frozen_setup):
    models, stand_in, _, folder = frozen_setup
    compare = ("sensitivity_with_status_sep", "champion")
    fe.evaluate_on_test(models, compare=compare, n_boot=20, metrics_dir=folder)
    # Same frozen model again: allowed, it reproduces the same evaluation.
    fe.evaluate_on_test(models, compare=compare, n_boot=20, metrics_dir=folder)
    # Pretend the CV report was redone with another model: the test was already used.
    other = WoeLogitModel(rp.model_specs()["champion"]).fit(
        build_features(stand_in), stand_in["DEFAULT"]
    )
    cv_path = folder / "pd_logistic_cv.json"
    cv = json.loads(cv_path.read_text(encoding="utf-8"))
    cv["metrics"]["models"]["champion"]["final_features"] = list(other.features_)
    cv["metrics"]["models"]["champion"]["coefficients"] = [
        {"term": k, "coef": float(v)} for k, v in other.result_.params.items()
    ]
    cv_path.write_text(json.dumps(cv), encoding="utf-8")
    with pytest.raises(ValueError, match="already evaluated"):
        fe.evaluate_on_test(
            {**models, "champion": other}, compare=compare, n_boot=20, metrics_dir=folder
        )
