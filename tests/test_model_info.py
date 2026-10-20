"""PLAN_V5_1_REVISION.md section 1: every model the study's batch matrix can
produce (and perfect_foresight) must have a dashboard card."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from data_loader import load_model_info  # noqa: E402

from src.config import load_forecast_blends_config, load_study_config  # noqa: E402

_REQUIRED_FIELDS = {"kind", "summary", "strength", "weakness", "learns"}


def test_every_study_model_and_blend_has_a_card():
    info = load_model_info()
    study = load_study_config()
    missing = [name for name in study.optimizer_models if name not in info]
    assert not missing, f"app/model_info.yaml is missing: {missing}"
    assert "perfect_foresight" in info


def test_every_card_has_the_required_fields():
    info = load_model_info()
    for name, card in info.items():
        assert _REQUIRED_FIELDS <= card.keys(), f"{name} is missing one of {_REQUIRED_FIELDS}"


def test_every_forecast_blend_resolves_to_known_models():
    blends = load_forecast_blends_config()
    info = load_model_info()
    for name, per_market in blends.blends.items():
        assert name in info, f"blend '{name}' has no card in app/model_info.yaml"
        for market, model in per_market.items():
            assert model in info, f"blend '{name}' market '{market}' points at unknown model '{model}'"
