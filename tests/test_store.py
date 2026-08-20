"""Tests for medscope.store -- Task 3.3's run-persistence layer.

`InMemoryRunStore` and `SqliteRunStore` are graded against the same
contract via the `store` fixture below, parametrized over both -- so the
two implementations can't quietly drift apart from each other.

The privacy tests are the load-bearing ones here. `store.py`'s central
design decision is an explicit allowlist of what gets persisted, not
`state.model_dump()` wholesale: `indication`/`history_text` are NOT
guaranteed de-identified for every terminal status a `StudyState` can
reach. A study blocked at intake (`status == "GUARDRAIL_BLOCKED"`) never
reaches the `deid` node at all -- see `graph._route_after_intake`, which
routes straight to `END` on a blocked verdict -- so those two fields can
still hold the exact raw text that arrived with the study. An audit log
is supposed to be the thing that proves PHI didn't leak, not a second
place it can leak from.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from medscope.state import CriticalAlert, Finding, ReportDraft, ReportSentence, StudyState
from medscope.store import InMemoryRunStore, SqliteRunStore


def _state(**overrides) -> StudyState:
    defaults: dict = dict(
        study_id="study-1",
        image_path="data/samples/studies/images/CXR38_IM-1911-1001.png",
        status="DRAFT_READY",
        deid_report={
            "indication": {"patterns_fired": ["name"], "before_hash": "h1", "after_hash": "h2"},
            "history_text": {"patterns_fired": [], "before_hash": "h3", "after_hash": "h3"},
        },
        findings=[Finding(label="Cardiomegaly", prob=0.9, source="cnn")],
        alerts=[
            CriticalAlert(
                label="Pneumothorax",
                prob=0.8,
                source="cnn",
                detected_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                image_ref="data/samples/studies/images/CXR38_IM-1911-1001.png",
            )
        ],
        report=ReportDraft(
            sentences=[ReportSentence(text="Cardiac silhouette enlarged.", section="findings")],
            disclaimer="AI-assisted draft -- requires physician review.",
        ),
    )
    defaults.update(overrides)
    return StudyState(**defaults)


@pytest.fixture(params=["memory", "sqlite"])
def store(request):
    if request.param == "memory":
        return InMemoryRunStore()
    return SqliteRunStore(":memory:")


# ---------------------------------------------------------------------------
# Shared contract
# ---------------------------------------------------------------------------


def test_round_trips_a_state(store):
    state = _state(study_id="s1")
    run_id = store.save(state)
    assert isinstance(run_id, str) and run_id

    fetched = store.get(run_id)
    assert fetched is not None
    assert fetched.study_id == "s1"
    assert fetched.status == "DRAFT_READY"
    assert fetched.findings[0].label == "Cardiomegaly"
    assert fetched.findings[0].prob == pytest.approx(0.9)
    assert fetched.alerts[0].label == "Pneumothorax"
    assert fetched.report is not None
    assert fetched.report.sentences[0].text == "Cardiac silhouette enlarged."


def test_missing_run_id_returns_none(store):
    assert store.get("does-not-exist") is None


def test_list_returns_newest_first(store):
    ids = [store.save(_state(study_id=f"s{i}")) for i in range(3)]
    summaries = store.list(limit=10)
    assert [s["run_id"] for s in summaries] == list(reversed(ids))


def test_list_respects_limit(store):
    for i in range(5):
        store.save(_state(study_id=f"s{i}"))
    summaries = store.list(limit=2)
    assert len(summaries) == 2


def test_list_summary_shape(store):
    store.save(_state(study_id="s1", status="HELD"))
    summary = store.list(limit=1)[0]
    assert set(summary) == {"run_id", "study_id", "status", "created_at", "alert_count"}
    assert summary["study_id"] == "s1"
    assert summary["status"] == "HELD"
    assert summary["alert_count"] == 1
    assert isinstance(summary["created_at"], (int, float))


# ---------------------------------------------------------------------------
# Privacy: no raw pre-de-identification text, ever
# ---------------------------------------------------------------------------


def test_no_raw_text_persisted_for_a_guardrail_blocked_study(store):
    """Reproduces the actual gap: a study blocked at intake never reaches
    `deid`, so its `indication`/`history_text` are still exactly what
    arrived -- unscrubbed. Saving must not let that reach the store.
    """
    raw_indication = "Patient: John Smith, MRN123456, phone 138-1234-5678"
    raw_history = "Referring physician Dr. Jane Doe, seen 2024-01-01."
    state = _state(
        study_id="blocked-1",
        status="GUARDRAIL_BLOCKED",
        indication=raw_indication,
        history_text=raw_history,
        deid_report={},  # deid never ran
        notes=["injection detected in indication (prompt_override)"],
        findings=[],
        alerts=[],
        report=None,
    )
    run_id = store.save(state)
    fetched = store.get(run_id)
    assert fetched is not None

    dumped = json.dumps(fetched.model_dump(mode="json"))
    for leaked in ("John Smith", "MRN123456", "138-1234-5678", "Jane Doe", "2024-01-01"):
        assert leaked not in dumped, f"{leaked!r} leaked into the persisted record"

    # Not partially scrubbed -- left out of the record entirely.
    assert fetched.indication == ""
    assert fetched.history_text == ""


def test_deid_summary_survives_even_though_raw_text_does_not(store):
    """`deid_report` never carries raw text -- `deid.deid_text` returns
    pattern names plus hashes, never the strings themselves -- so unlike
    indication/history_text it belongs in the record. Pins that it
    actually round-trips.
    """
    report = {"indication": {"patterns_fired": ["name", "phone"], "before_hash": "h1", "after_hash": "h2"}}
    state = _state(deid_report=report)
    run_id = store.save(state)
    fetched = store.get(run_id)
    assert fetched.deid_report == report


# ---------------------------------------------------------------------------
# SqliteRunStore specifics
# ---------------------------------------------------------------------------


def test_sqlite_store_uses_parameter_binding_not_string_interpolation():
    """A study_id crafted to look like SQL must be stored and retrieved
    verbatim, not executed -- proof the SQL in store.py binds parameters.
    """
    store = SqliteRunStore(":memory:")
    hostile_id = "x'; DROP TABLE runs; --"
    run_id = store.save(_state(study_id=hostile_id))

    fetched = store.get(run_id)
    assert fetched.study_id == hostile_id
    # If the DROP TABLE had actually executed, this would raise.
    assert store.list(limit=10)


def test_sqlite_memory_path_is_a_real_shared_connection():
    store = SqliteRunStore(":memory:")
    run_id = store.save(_state(study_id="mem-1"))
    # Same store instance, second call -- must see what the first call wrote,
    # proving the connection is held open rather than reopened per call
    # (a fresh `:memory:` connection would be a brand-new empty database).
    assert store.get(run_id) is not None
    assert store.list(limit=10)


# ---------------------------------------------------------------------------
# GET /runs
# ---------------------------------------------------------------------------


def test_runs_endpoint_returns_well_formed_json_and_reflects_a_run():
    from fastapi.testclient import TestClient

    from medscope.app import app

    client = TestClient(app)
    run_response = client.post("/workbench/run", json={"study_id": "1187"})
    assert run_response.status_code == 200

    response = client.get("/runs")
    assert response.status_code == 200
    body = response.json()
    assert "runs" in body
    assert isinstance(body["runs"], list)
    assert body["runs"], "expected at least the run just performed"

    entry = body["runs"][0]
    assert {"run_id", "study_id", "status", "created_at", "alert_count"} <= set(entry)
    assert any(r["study_id"] == "1187" for r in body["runs"])
