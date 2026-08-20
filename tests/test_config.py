import pytest
from pydantic import ValidationError

from medscope.config import Settings


def test_settings_instantiates_with_defaults():
    s = Settings()
    assert s.cnn_weights == "densenet121-res224-all"
    assert s.cnn_prob_threshold == 0.5
    assert s.critical_threshold == 0.3
    assert s.reader_b_mode == "reader"
    assert s.kappa_floor == 0.4
    assert s.disagreement_ceiling == 0.4
    assert s.max_llm_judgments == 12
    assert s.use_real_vlm is False
    assert s.lateral_symmetry_threshold == 0.5


def test_tracing_disabled_with_no_keys():
    assert Settings().tracing_enabled is False


def test_invalid_reader_b_mode_raises():
    with pytest.raises(ValidationError):
        Settings(reader_b_mode="not_a_valid_mode")
