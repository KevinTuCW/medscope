"""Tests for medscope.critical -- the highest-stakes module in Phase 1.

Gate G1 requires critical-finding recall == 1.0: missing a pneumothorax
can kill, over-calling one costs a radiologist thirty seconds. That
asymmetry is why `triage()` fires on `Settings.critical_threshold` (0.3),
well below `Settings.cnn_prob_threshold` (0.5) -- a finding that never
makes it into the report can and must still raise a critical alert.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from medscope.config import Settings
from medscope.critical import triage
from medscope.ontology import CRITICAL_LABELS, canonical
from medscope.state import CriticalAlert, Finding

GOLDSET_PATH = Path("data/evals/critical.json")


def _finding(label, prob, source="cnn"):
    return Finding(label=label, prob=prob, source=source, raw_label=label)


@pytest.fixture
def settings():
    return Settings()


def test_alert_fires_above_critical_threshold(settings):
    findings = [_finding("Pneumothorax", 0.6)]

    alerts = triage(findings, settings)

    assert len(alerts) == 1
    assert alerts[0].label == "Pneumothorax"


def test_alert_fires_below_report_threshold_but_above_critical_threshold(settings):
    # THE load-bearing test: 0.35 is below cnn_prob_threshold (0.5), so this
    # finding never makes the report -- but it is above critical_threshold
    # (0.3), so it MUST still raise a critical alert. If someone later
    # "tidies" these two thresholds into one, this test is what catches it.
    assert 0.35 < settings.cnn_prob_threshold
    assert 0.35 >= settings.critical_threshold

    findings = [_finding("Pneumothorax", 0.35)]

    alerts = triage(findings, settings)

    assert len(alerts) == 1
    assert alerts[0].label == "Pneumothorax"
    assert alerts[0].prob == 0.35


def test_below_critical_threshold_does_not_alert(settings):
    findings = [_finding("Pneumothorax", 0.2)]

    alerts = triage(findings, settings)

    assert alerts == []


def test_noncritical_label_never_alerts_even_at_high_prob(settings):
    assert "Cardiomegaly" not in CRITICAL_LABELS
    findings = [_finding("Cardiomegaly", 0.99)]

    alerts = triage(findings, settings)

    assert alerts == []


def test_dedup_both_readers_same_label_yields_one_alert_source_both(settings):
    findings = [
        _finding("Pneumothorax", 0.7, source="cnn"),
        _finding("Pneumothorax", 0.6, source="vlm"),
    ]

    alerts = triage(findings, settings)

    assert len(alerts) == 1
    assert alerts[0].source == "both"
    assert alerts[0].prob == 0.7  # the higher of the two, per the same
    # over-call-is-safe philosophy as the threshold choice itself


def test_single_reader_alert_carries_that_readers_source(settings):
    findings = [_finding("PleuralEffusion", 0.5, source="vlm")]

    alerts = triage(findings, settings)

    assert len(alerts) == 1
    assert alerts[0].source == "vlm"


def test_alert_fields_match_originating_finding(settings):
    before = datetime.now()
    findings = [_finding("Pneumomediastinum", 0.9, source="cnn")]

    alerts = triage(findings, settings, image_ref="study-42/image.png")
    after = datetime.now()

    assert len(alerts) == 1
    alert = alerts[0]
    assert isinstance(alert, CriticalAlert)
    assert alert.label == "Pneumomediastinum"
    assert alert.prob == 0.9
    assert alert.source == "cnn"
    assert alert.image_ref == "study-42/image.png"
    assert before <= alert.detected_at <= after


def test_multiple_distinct_critical_labels_each_get_their_own_alert(settings):
    findings = [
        _finding("Pneumothorax", 0.9, source="cnn"),
        _finding("PleuralEffusion", 0.5, source="cnn"),
        _finding("Cardiomegaly", 0.9, source="cnn"),  # never alerts
    ]

    alerts = triage(findings, settings)

    labels = {a.label for a in alerts}
    assert labels == {"Pneumothorax", "PleuralEffusion"}


def test_empty_findings_list_produces_no_alerts(settings):
    assert triage([], settings) == []


# ---------------------------------------------------------------------------
# Gold set: data/evals/critical.json
# ---------------------------------------------------------------------------


def test_goldset_file_loads_and_is_well_formed():
    data = json.loads(GOLDSET_PATH.read_text())

    assert "metadata" in data
    assert "cases" in data
    assert isinstance(data["cases"], list)
    assert len(data["cases"]) > 0

    required_keys = {"study_id", "image_path", "expected_critical", "source", "note"}
    for case in data["cases"]:
        assert required_keys <= set(case.keys()), case["study_id"]
        assert isinstance(case["expected_critical"], list)
        assert isinstance(case["note"], str) and case["note"]
        assert case["source"] == "openi"


def test_goldset_every_expected_label_is_a_valid_canonical_name():
    data = json.loads(GOLDSET_PATH.read_text())

    for case in data["cases"]:
        for label in case["expected_critical"]:
            assert canonical(label) == label, f"{case['study_id']}: {label!r} is not canonical"
            assert label in CRITICAL_LABELS, f"{case['study_id']}: {label!r} is not a critical label"


def test_goldset_study_ids_are_unique():
    data = json.loads(GOLDSET_PATH.read_text())
    ids = [case["study_id"] for case in data["cases"]]
    assert len(ids) == len(set(ids))


def test_goldset_metadata_documents_pneumomediastinum_coverage():
    # Task 1.7's honest-accounting requirement: the dataset has almost no
    # confirmed pneumomediastinum, and that fact must be visible in the
    # file itself, not just in a report nobody re-reads.
    data = json.loads(GOLDSET_PATH.read_text())
    assert "Pneumomediastinum" in json.dumps(data["metadata"])
