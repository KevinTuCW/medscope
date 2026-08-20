import pytest
from pydantic import ValidationError

from medscope.state import Finding, StudyState


def test_finding_prob_must_be_in_unit_range():
    Finding(
        label="Pneumothorax",
        prob=0.87,
        source="cnn",
        locus={"cx": 0.31, "cy": 0.22, "r": 0.09},
    )

    with pytest.raises(ValidationError):
        Finding(
            label="Pneumothorax",
            prob=1.4,
            source="cnn",
            locus={"cx": 0.31, "cy": 0.22, "r": 0.09},
        )

    with pytest.raises(ValidationError):
        Finding(
            label="Pneumothorax",
            prob=-0.1,
            source="cnn",
            locus={"cx": 0.31, "cy": 0.22, "r": 0.09},
        )


def test_study_state_defaults():
    state = StudyState(study_id="s1")
    assert state.status == "pending"
    assert state.findings == []
    assert state.alerts == []
    assert state.trace_events == []
    assert state.tokens_used == 0


def test_finding_evidence_id_defaults_to_source_label_key():
    finding = Finding(label="Pneumothorax", prob=0.87, source="cnn")
    assert finding.evidence_id == "cnn:Pneumothorax"


def test_finding_evidence_id_explicit_value_is_preserved():
    finding = Finding(
        label="Pneumothorax",
        prob=0.87,
        source="cnn",
        evidence_id="ev-custom-1",
    )
    assert finding.evidence_id == "ev-custom-1"
