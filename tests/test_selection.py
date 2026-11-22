import json

from creditrisk.config import REPO_ROOT
from creditrisk.features import selection as sel
from creditrisk.features.engineering import FEATURE_SPECS


def test_champion_specs_leave_out_the_excluded_features():
    specs = sel.champion_specs()
    assert set(specs).isdisjoint(sel.EXCLUDED_FROM_CHAMPION)
    assert set(specs) | set(sel.EXCLUDED_FROM_CHAMPION) == set(FEATURE_SPECS)


def test_sensitivity_features_are_excluded_from_the_champion():
    assert set(sel.SENSITIVITY_FEATURES) <= set(sel.EXCLUDED_FROM_CHAMPION)


def test_saved_report_respects_the_decisions():
    path = REPO_ROOT / "reports" / "metrics" / "woe_iv.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    shortlist = metrics["shortlist"]
    assert set(shortlist).isdisjoint(sel.EXCLUDED_FROM_CHAMPION)
    iv = {row["feature"]: row["iv"] for row in metrics["iv_table"]}
    assert all(iv[name] >= sel.MIN_IV for name in shortlist)
    # The independent IV and optbinning's IV agree for every feature.
    for row in metrics["iv_table"]:
        assert abs(row["iv"] - row["optbinning_iv"]) < 1e-9, row["feature"]
