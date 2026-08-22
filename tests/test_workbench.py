"""Tests for medscope.workbench -- the radiologist-facing dashboard and its
FastAPI surface (Task 3.1).

`build_dashboard` is the load-bearing function: it is what turns a
completed `StudyState` into the five blocks the workbench renders
(`image`, `dual_read`, `report`, `critical`, `audit`), plus a top-level
`status`/`blocked` pair. Everything else here (the SSE stream, the
endpoints) is plumbing around that one function.

Fakes mirror `tests/test_graph.py`'s -- deliberately not imported from
there (no test file in this repo imports fixtures from another one; see
`tests/test_graph.py`, `tests/test_arbiter.py`, etc., each of which defines
its own). Endpoint- and stream-level tests drive the real sample studies
through `build_sample_deps()` per the task brief; the `build_dashboard`
unit tests use hand-built fakes so the DRAFT_READY/HELD/CRITICAL_ESCALATED
branches are deterministic rather than depending on what a real CNN model
happens to say about a real X-ray.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from medscope.arbiter import OfflineArbiterClient
from medscope.config import Settings
from medscope.graph import GraphDeps, build_graph
from medscope.llm import ModelResponse
from medscope.rag.store import Doc
from medscope.report import OfflineReportClient
from medscope.runner import run_study
from medscope.state import Finding, ReadResult, StudyState

SAMPLE_IMAGE = "data/samples/studies/images/CXR38_IM-1911-1001.png"


# ---------------------------------------------------------------------------
# Fakes -- mirror tests/test_graph.py
# ---------------------------------------------------------------------------


class FakeReaderA:
    def __init__(self, findings: list[Finding]):
        self._findings = findings
        self.calls = 0

    def read(self, image_path) -> ReadResult:
        return self.read_study([image_path])

    def read_study(self, image_paths) -> ReadResult:
        self.calls += 1
        return ReadResult(reader="a", findings=list(self._findings), latency_ms=7)


class FakeVLMClient:
    name = "fake-vlm"

    def __init__(self, payload: list[dict] | None = None):
        self._payload = payload if payload is not None else []
        self.calls = 0

    def chat_with_image(self, prompt: str, image_path, *, system: str | None = None) -> ModelResponse:
        self.calls += 1
        return ModelResponse(text=json.dumps(self._payload, ensure_ascii=False), tokens=3)


class FakeArbiterClient:
    name = "fake-arbiter"

    def __init__(self, verdicts: list[str]):
        self._verdicts = list(verdicts)
        self.calls = 0

    def chat_with_image(self, prompt: str, image_path, *, system: str | None = None) -> ModelResponse:
        verdict = self._verdicts[self.calls] if self.calls < len(self._verdicts) else "UNCERTAIN"
        self.calls += 1
        return ModelResponse(text=json.dumps({"verdict": verdict, "reasoning": f"canned {verdict}"}))


class FakeRetriever:
    def search(self, query: str, k: int = 3) -> list[Doc]:
        return []


class SlowReportClient:
    """Same trick test_graph.py uses to give the critical-bypass ordering an
    actual wall-clock gap to catch a wrongly-sequenced graph."""

    name = "slow-report"

    def __init__(self, delay: float = 0.4):
        self._inner = OfflineReportClient()
        self._delay = delay

    def chat_with_image(self, prompt: str, image_path, *, system: str | None = None) -> ModelResponse:
        import time

        time.sleep(self._delay)
        return self._inner.chat_with_image(prompt, image_path, system=system)


def _finding(label: str, prob: float, source: str = "cnn", locus: dict | None = None) -> Finding:
    return Finding(label=label, prob=prob, source=source, raw_label=label, locus=locus)


def _deps(
    *,
    reader_a_findings: list[Finding] | None = None,
    vlm_payload: list[dict] | None = None,
    arbiter_verdicts: list[str] | None = None,
    report_client=None,
    settings: Settings | None = None,
) -> GraphDeps:
    return GraphDeps(
        settings=settings or Settings(),
        cnn_reader=FakeReaderA(reader_a_findings or []),
        vlm_client=FakeVLMClient(vlm_payload),
        arbiter_client=FakeArbiterClient(arbiter_verdicts or []) if arbiter_verdicts is not None else OfflineArbiterClient(),
        retriever=FakeRetriever(),
        report_client=report_client or OfflineReportClient(),
    )


def _state(image_path: str = SAMPLE_IMAGE, **kwargs) -> StudyState:
    return StudyState(study_id="CXR-test", image_path=image_path, **kwargs)


def _draft_ready_result() -> StudyState:
    deps = _deps(
        reader_a_findings=[_finding("Cardiomegaly", 0.9, locus={"cx": 0.5, "cy": 0.4, "r": 0.2})],
        vlm_payload=[{"label": "Cardiomegaly", "confidence": "certain"}],
    )
    return run_study(_state(), deps)


def _held_result() -> StudyState:
    deps = _deps(
        reader_a_findings=[_finding("Cardiomegaly", 0.7)],
        vlm_payload=[{"label": "Cardiomegaly", "confidence": "possible"}],
        arbiter_verdicts=["UNCERTAIN"],
    )
    return run_study(_state(), deps)


def _critical_escalated_result(*, report_client=None) -> StudyState:
    deps = _deps(
        reader_a_findings=[_finding("Pneumothorax", 0.9, locus={"cx": 0.3, "cy": 0.6, "r": 0.1})],
        vlm_payload=[{"label": "Pneumothorax", "confidence": "certain"}],
        report_client=report_client,
    )
    return run_study(_state(), deps)


# ---------------------------------------------------------------------------
# build_dashboard
# ---------------------------------------------------------------------------


def test_build_dashboard_returns_all_five_blocks_plus_status():
    from medscope.workbench import build_dashboard

    result = _draft_ready_result()
    dashboard = build_dashboard(result)

    assert dashboard["status"] == "DRAFT_READY"
    assert dashboard["blocked"] is False
    for block in ("image", "dual_read", "report", "critical", "audit"):
        assert block in dashboard, f"missing block {block!r}"


def test_image_block_has_reader_loci_normalized_and_reader_b_contributes_none():
    from medscope.workbench import build_dashboard

    result = _draft_ready_result()
    dashboard = build_dashboard(result)

    image = dashboard["image"]
    assert image["path"] == result.image_path
    a_loci = image["readers"]["a"]
    assert len(a_loci) == 1
    locus = a_loci[0]
    assert locus["label"] == "Cardiomegaly"
    assert 0.0 <= locus["cx"] <= 1.0
    assert 0.0 <= locus["cy"] <= 1.0
    assert 0.0 <= locus["r"] <= 1.0
    assert image["readers"]["b"] == []


def test_dual_read_block_shows_agreement_and_kappa():
    from medscope.workbench import build_dashboard

    result = _draft_ready_result()
    dashboard = build_dashboard(result)

    dual_read = dashboard["dual_read"]
    assert dual_read["kappa"] == result.kappa
    rows = {row["label"]: row for row in dual_read["rows"]}
    assert "Cardiomegaly" in rows
    row = rows["Cardiomegaly"]
    assert row["agreed"] is True
    assert row["kind"] is None
    assert row["a_prob"] == pytest.approx(0.9)
    assert row["b_prob"] is not None


def test_dual_read_block_shows_disagreement_and_arbiter_verdict():
    from medscope.workbench import build_dashboard

    result = _held_result()
    dashboard = build_dashboard(result)

    rows = {row["label"]: row for row in dashboard["dual_read"]["rows"]}
    assert "Cardiomegaly" in rows
    row = rows["Cardiomegaly"]
    assert row["agreed"] is False
    assert row["kind"] == "presence"
    assert row["verdict"] == "UNCERTAIN"


def test_blocked_true_for_held_and_critical_escalated_false_for_draft_ready():
    from medscope.workbench import build_dashboard

    draft_ready = build_dashboard(_draft_ready_result())
    assert draft_ready["status"] == "DRAFT_READY"
    assert draft_ready["blocked"] is False

    held = build_dashboard(_held_result())
    assert held["status"] == "HELD"
    assert held["blocked"] is True

    critical = build_dashboard(_critical_escalated_result())
    assert critical["status"] == "CRITICAL_ESCALATED"
    assert critical["blocked"] is True


def test_report_block_evidence_map_covers_every_cited_id():
    from medscope.workbench import build_dashboard

    result = _draft_ready_result()
    dashboard = build_dashboard(result)

    report = dashboard["report"]
    assert report["disclaimer"]
    cited_ids = {eid for s in report["sentences"] for eid in s["evidence_ids"]}
    assert cited_ids, "the happy path should produce at least one cited id"
    for eid in cited_ids:
        assert eid in report["evidence_map"], f"{eid} cited but not resolved"
        resolved = report["evidence_map"][eid]
        assert resolved["label"]
        assert "prob" in resolved
        assert "source" in resolved


def test_report_block_is_empty_but_well_formed_when_no_report_was_written(tmp_path):
    from PIL import Image

    from medscope.workbench import build_dashboard

    bad_image = tmp_path / "blank.png"
    Image.new("L", (10, 10), color=0).save(bad_image)

    deps = _deps()
    bad_state = _state(image_path=str(bad_image))
    result = run_study(bad_state, deps)
    assert result.status == "NEEDS_REPEAT"
    assert result.report is None

    dashboard = build_dashboard(result)
    assert dashboard["report"]["sentences"] == []
    assert dashboard["report"]["evidence_map"] == {}
    assert dashboard["blocked"] is True


def test_critical_block_timeline_preserves_alert_before_report_ordering():
    from medscope.workbench import build_dashboard

    result = _critical_escalated_result(report_client=SlowReportClient(delay=0.4))
    dashboard = build_dashboard(result)

    critical = dashboard["critical"]
    assert len(critical["alerts"]) == 1
    assert critical["alerts"][0]["label"] == "Pneumothorax"
    assert critical["alerts"][0]["detected_at"]

    timeline = critical["timeline"]
    node_order = [e["node"] for e in timeline]
    assert "critical_triage" in node_order
    assert "report_writer" in node_order
    assert node_order.index("critical_triage") < node_order.index("report_writer")
    # timeline is actually ordered by timestamp, not by insertion order
    timestamps = [e["ts"] for e in timeline]
    assert timestamps == sorted(timestamps)


def test_audit_block_carries_deid_qc_and_trace():
    from medscope.workbench import build_dashboard

    result = _draft_ready_result()
    dashboard = build_dashboard(result)

    audit = dashboard["audit"]
    assert audit["deid"] == result.deid_report
    assert audit["qc"] == result.qc
    assert audit["trace"], "per-node trace must be present"
    assert all("node" in e and "ts" in e for e in audit["trace"])


# ---------------------------------------------------------------------------
# sse_events
# ---------------------------------------------------------------------------


def test_sse_events_yields_start_node_and_complete_with_full_dashboard():
    from medscope.workbench import build_dashboard, sse_events

    deps = _deps(
        reader_a_findings=[_finding("Cardiomegaly", 0.9)],
        vlm_payload=[{"label": "Cardiomegaly", "confidence": "certain"}],
    )
    state = _state()

    events = list(sse_events(state, deps))
    assert events, "sse_events produced nothing"

    parsed = []
    for chunk in events:
        lines = [l for l in chunk.strip("\n").split("\n") if l]
        event_line = next(l for l in lines if l.startswith("event:"))
        data_line = next(l for l in lines if l.startswith("data:"))
        parsed.append((event_line.split(":", 1)[1].strip(), json.loads(data_line.split(":", 1)[1].strip())))

    kinds = [k for k, _ in parsed]
    assert kinds[0] == "start"
    assert "node" in kinds
    assert kinds[-1] == "complete"

    complete_payload = parsed[-1][1]
    assert complete_payload["status"] == "DRAFT_READY"
    for block in ("image", "dual_read", "report", "critical", "audit"):
        assert block in complete_payload


# ---------------------------------------------------------------------------
# Endpoints -- driven through the real sample studies via build_sample_deps()
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    from medscope.app import app

    return TestClient(app)


def test_studies_endpoint_lists_the_sample_studies(client):
    response = client.get("/studies")
    assert response.status_code == 200
    body = response.json()
    assert len(body["studies"]) == 3
    ids = {s["study_id"] for s in body["studies"]}
    assert {"38", "797", "1187"} <= ids


def test_workbench_page_is_served(client):
    response = client.get("/workbench")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "医师复核" in response.text or "physician review" in response.text.lower()


def test_workbench_run_and_dashboard_endpoints(client):
    run_response = client.post("/workbench/run", json={"study_id": "797"})
    assert run_response.status_code == 200
    run_body = run_response.json()
    for block in ("image", "dual_read", "report", "critical", "audit"):
        assert block in run_body

    dashboard_response = client.get("/workbench/dashboard", params={"study_id": "797"})
    assert dashboard_response.status_code == 200
    dashboard_body = dashboard_response.json()
    assert dashboard_body["status"] == run_body["status"]


def test_workbench_dashboard_404_before_any_run(client):
    response = client.get("/workbench/dashboard", params={"study_id": "not-run-yet"})
    assert response.status_code == 404


def test_workbench_stream_endpoint_emits_sse(client):
    response = client.get("/workbench/stream", params={"study_id": "38"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "event: start" in response.text
    assert "event: complete" in response.text


def test_arbiter_verdicts_come_from_records_not_from_finding_presence():
    """The audit view must report what the arbiter decided, not a guess.

    Verdicts used to be inferred from whether an arbiter-sourced finding
    survived into the final set. That happened to match `arbiter.py`, but it
    coupled this view to that module's internals — if REJECT ever stopped
    meaning "adds nothing to the final set", the dual-read table would go on
    stating verdicts confidently and wrongly. An audit surface that
    misreports is worse than one that admits it doesn't know.

    Here the stored record says REJECT while an arbiter-sourced finding for
    the same label *is* present — a state the old inference would have read
    as CONFIRM.
    """
    from medscope.state import Disagreement
    from medscope.workbench import build_dashboard

    state = StudyState(
        study_id="s",
        image_path=SAMPLE_IMAGE,
        findings=[Finding(label="Cardiomegaly", prob=0.7, source="arbiter")],
        disagreements=[
            Disagreement(label="Cardiomegaly", a_prob=0.9, b_prob=0.1, kind="presence")
        ],
        arbitration_records=[
            {"label": "Cardiomegaly", "verdict": "REJECT", "reasoning": "canned"}
        ],
    )

    rows = build_dashboard(state)["dual_read"]["rows"]
    row = next(r for r in rows if r["label"] == "Cardiomegaly")
    assert row["verdict"] == "REJECT"
