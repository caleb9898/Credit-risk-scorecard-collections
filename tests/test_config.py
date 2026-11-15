import pytest

from creditrisk.config import load_config


def test_project_config_has_required_keys():
    cfg = load_config("project")
    assert isinstance(cfg["seed"], int)
    assert 0 < cfg["split"]["test_size"] < 0.5
    assert cfg["split"]["cv_folds"] >= 3


def test_assumptions_are_grids_with_unset_bases():
    cfg = load_config("assumptions")
    assert len(cfg["ead"]["ccf_grid"]) >= 3
    assert len(cfg["lgd"]["lgd_grid"]) >= 3
    # Base values must be chosen and justified deliberately, not shipped pre-filled.
    assert cfg["ead"]["ccf_base"] is None
    assert cfg["lgd"]["lgd_base"] is None


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist")
