"""The LangGraph pipeline -- wires every previously-built stage into one
runnable study.

```
intake -> deid -> qc --(fail)--> NEEDS_REPEAT, terminate
                   |
                   +--> reader_a  --+
                   +--> reader_b  --+ (independent)
                                    v
                                  merge
                                    |
                     +--------------+--------------+
                     v                             v
              critical_triage                   arbiter
                     |                             v
                     |                       report_writer
                     |                             v
                     |                       evidence_check
                     |                             v
                     |                       language_guard
                     +--------------+--------------+
                                    v
                              review_queue
```

Three design points carried over from the modules this wires together, and
load-bearing for how the graph itself is built (not just how each node
behaves in isolation):

**QC is a hard gate, not a speed bump.** `qc` routes straight to `END` with
`status="NEEDS_REPEAT"` on a hard failure -- `reader_a`/`reader_b` are never
even scheduled, so neither reader's client is ever called. A cheap gate that
still pays for inference on a bad image is not a gate.

**`critical_triage` and the report chain (`arbiter` -> `report_writer` ->
`evidence_check` -> `language_guard`) are both direct successors of `merge`,
not sequenced one after the other.** LangGraph's synchronous `.invoke()` runs
every node ready in a superstep on its own thread (`langgraph.pregel.
_executor.BackgroundExecutor`), and `critical_triage` sits exactly one
superstep after `merge` while `report_writer` sits at least two (behind
`arbiter`) -- so a critical alert is recorded, and can reach a clinician, well
before a slow report finishes writing, mirroring real hospital
critical-results communication policy. Building this as `merge -> arbiter ->
... -> critical_triage` (report first, alert second) would make every
functional test in this module still pass; only the ordering recorded in
`trace_events` catches it, which is what `tests/test_graph.py::
test_critical_alert_precedes_report_in_trace_even_when_report_is_slow`
exists to check.

**`GraphState`'s `trace_events` and `notes` channels are additive
(`Annotated[..., operator.add]`), not overwrite.** `reader_a`/`reader_b` and
`critical_triage`/`arbiter` write to `trace_events` in the same superstep as
each other; without a reducer, LangGraph raises on the second concurrent
write to the same channel in one superstep (`InvalidUpdateError`) rather than
silently dropping one -- but every node here returns only the events/notes it
is adding this call, not the accumulated list, since the reducer is what
concatenates across supersteps and across concurrent writers. `GraphState` is
deliberately a superset of `StudyState`'s fields (it also carries
`qc_ok`/`guardrail_blocked`/`evidence_retried`, pure graph-routing
bookkeeping) rather than reusing `StudyState` itself as the LangGraph schema:
`StudyState.model_validate` at the end of `runner.run_study` drops those
extra keys for free (pydantic's default `extra="ignore"`), so the routing
state never leaks into the type every other module already depends on.
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Protocol, TypedDict, runtime_checkable

from langgraph.graph import END, START, StateGraph

from medscope.arbiter import ArbiterClient, ArbiterDeps, arbitrate
from medscope.config import Settings
from medscope.critical import triage
from medscope.deid import deid_text
from medscope.evidence import check_evidence, coverage
from medscope.guardrails.input import screen_intake
from medscope.guardrails.output import enforce_output
from medscope.guardrails.process import cap_findings
from medscope.language import REQUIRED_DISCLAIMER, detect_redlines, has_disclaimer, neutralize
from medscope.merge import merge_reads
from medscope.qc import check_quality
from medscope.rag.store import Retriever
from medscope.readers.vlm import VLMClient, read_b
from medscope.report import ReportClient, write_report
from medscope.security.sanitize import detect_injection
from medscope.state import CriticalAlert, Disagreement, Finding, ReadResult, ReportDraft, StudyState

_REQUIRED_SECTIONS = frozenset({"technique", "findings", "impression", "recommendation"})


@runtime_checkable
class ReaderAClient(Protocol):
    """reader_a's shape: `medscope.readers.cnn.CNNReader` satisfies this
    directly. A separate Protocol (rather than importing `CNNReader` here)
    keeps this module -- and anything that imports it -- from paying a torch
    import just to build the graph; tests inject a plain stub with the same
    `.read()` method.
    """

    def read(self, image_path: str | Path) -> ReadResult: ...


@dataclass(frozen=True)
class GraphDeps:
    """Everything the graph's nodes need, injected so the graph stays
    testable without a real model, a real VLM, or a real retriever --
    mirrors `medscope.arbiter.ArbiterDeps` (and wealthwise's `AdvisoryDeps`,
    which that docstring points to).
    """

    settings: Settings
    cnn_reader: ReaderAClient
    vlm_client: VLMClient
    arbiter_client: ArbiterClient
    retriever: Retriever
    report_client: ReportClient
    arbiter_top_k: int = 3


class GraphState(TypedDict, total=False):
    """LangGraph's state schema. A superset of `StudyState`'s fields (see
    module docstring for why) -- `runner.run_study` reconciles the two with
    `StudyState.model_validate` after `.invoke()`.
    """

    study_id: str
    image_path: str
    history_text: str
    indication: str
    deid_report: dict
    qc: dict
    read_a: ReadResult | None
    read_b: ReadResult | None
    findings: list[Finding]
    disagreements: list[Disagreement]
    kappa: float | None
    arbitration_records: list[dict]
    alerts: list[CriticalAlert]
    report: ReportDraft | None
    status: str
    notes: Annotated[list[str], operator.add]
    trace_events: Annotated[list[dict], operator.add]
    budget_spent: int
    tokens_used: int
    # Graph-routing bookkeeping only -- never read by any other module, and
    # dropped by StudyState.model_validate at the end of runner.run_study.
    guardrail_blocked: bool
    qc_ok: bool
    evidence_retried: bool
    # Must be declared even though only `_route_after_evidence_check` reads
    # it: LangGraph silently drops any key a node returns that isn't in the
    # state schema, so an undeclared flag leaves the router reading `None`
    # forever -- the retry branch then becomes unreachable with no error
    # anywhere to point at it.
    needs_retry: bool


def _trace_event(node: str) -> dict:
    return {"node": node, "ts": time.time()}


def _draft_complete(draft: ReportDraft | None) -> bool:
    if draft is None:
        return False
    return _REQUIRED_SECTIONS <= {s.section for s in draft.sentences}


def _apply_evidence_gate(
    draft: ReportDraft, findings: list[Finding], already_retried: bool
) -> tuple[ReportDraft, list[str], bool]:
    """Gate G2 at runtime, applied to whatever `report_writer` returned.

    Returns `(draft_to_keep, new_notes, needs_retry)`. `needs_retry=True`
    means the caller should send the study back to `report_writer` exactly
    once (guarded by `already_retried`, a sentinel so this can never loop
    more than once) rather than shipping a violating draft.

    NOTE on reachability: `report.write_report` already filters every
    citation down to the known-finding set and drops any claim-section
    sentence that loses all its citations in that filtering (see that
    module's `_sentences_from_sections`) -- so in practice a `ReportDraft`
    coming out of `write_report` never fails `check_evidence`, and the
    retry branch below is defense-in-depth against that guarantee ever
    weakening (or a future `ReportClient` bypassing `write_report`
    entirely), not something a conforming `write_report` call can trigger.
    It is exercised directly in `tests/test_graph.py` against a
    hand-built violating `ReportDraft`, since no `ReportClient` stub can
    reach it end-to-end through `write_report`.
    """
    violations = check_evidence(draft, findings)
    if not violations:
        return draft, [], False

    if not already_retried:
        return draft, [], True

    bad_indexes = {v.index for v in violations}
    kept = [s for i, s in enumerate(draft.sentences) if i not in bad_indexes]
    stripped_draft = draft.model_copy(
        update={"sentences": kept, "removed_count": draft.removed_count + len(bad_indexes)}
    )
    note = (
        f"evidence_check: retry did not resolve {len(bad_indexes)} bare/dangling "
        "sentence(s) -- stripped rather than shipped"
    )
    return stripped_draft, [note], False


def build_graph(deps: GraphDeps):
    """Compile the study pipeline over `GraphState`, wired against `deps`."""

    arbiter_deps = ArbiterDeps(client=deps.arbiter_client, retriever=deps.retriever, top_k=deps.arbiter_top_k)

    # -- intake -----------------------------------------------------------
    def _intake(state: GraphState) -> dict:
        """Screen what arrived, via `guardrails.input.screen_intake`.

        Note the asymmetry it encodes: a study with no image is blocked
        (there is nothing to read), but a suspected injection in the history
        is *recorded and read anyway*. Refusing to interpret a real patient's
        film because someone typed something odd into a free-text field is
        the more harmful failure, and the text is already defanged by
        `neutralize_untrusted()` at every prompt boundary.
        """
        trace = [_trace_event("intake")]
        study = StudyState.model_validate(dict(state))
        verdict = screen_intake(study)

        update: dict = {"trace_events": trace, "guardrail_blocked": not verdict.ok}
        if verdict.reasons:
            update["notes"] = list(verdict.reasons)
        if not verdict.ok:
            update["status"] = "GUARDRAIL_BLOCKED"
        return update

    def _route_after_intake(state: GraphState) -> str:
        return END if state.get("guardrail_blocked") else "deid"

    # -- deid ---------------------------------------------------------------
    def _deid(state: GraphState) -> dict:
        scrubbed_indication, report_indication = deid_text(state.get("indication") or "")
        scrubbed_history, report_history = deid_text(state.get("history_text") or "")
        return {
            "indication": scrubbed_indication,
            "history_text": scrubbed_history,
            "deid_report": {"indication": report_indication, "history_text": report_history},
            "trace_events": [_trace_event("deid")],
        }

    # -- qc -------------------------------------------------------------------
    def _qc(state: GraphState) -> dict:
        from PIL import Image

        image = Image.open(state["image_path"])
        result = check_quality(image, deps.settings)
        update = {"qc": result.model_dump(), "trace_events": [_trace_event("qc")], "qc_ok": result.ok}
        if not result.ok:
            update["status"] = "NEEDS_REPEAT"
        return update

    def _route_after_qc(state: GraphState) -> str | list[str]:
        return ["reader_a", "reader_b"] if state.get("qc_ok") else END

    # -- reader_a / reader_b (independent) -----------------------------------
    def _reader_a(state: GraphState) -> dict:
        result = deps.cnn_reader.read(state["image_path"])
        return {"read_a": result, "trace_events": [_trace_event("reader_a")]}

    def _reader_b(state: GraphState) -> dict:
        study = StudyState.model_validate(dict(state))
        result = read_b(study, deps.vlm_client, deps.settings)
        return {"read_b": result, "trace_events": [_trace_event("reader_b")]}

    # -- merge ----------------------------------------------------------------
    def _merge(state: GraphState) -> dict:
        findings, disagreements, kappa = merge_reads(
            state["read_a"], state["read_b"], deps.settings.cnn_prob_threshold, deps.settings.reader_b_mode
        )
        # Process guardrail: deduplicate and drop malformed findings *before*
        # the arbiter, where every surviving disagreement buys an LLM call,
        # and before the report, where a malformed finding would become a
        # malformed citation.
        findings, dropped = cap_findings(findings, deps.settings.max_findings)
        update: dict = {
            "findings": findings,
            "disagreements": disagreements,
            "kappa": kappa,
            "trace_events": [_trace_event("merge")],
        }
        if dropped:
            update["notes"] = [f"process guardrail dropped {dropped} finding(s)"]
        return update

    # -- critical_triage --------------------------------------------------
    def _critical_triage(state: GraphState) -> dict:
        read_a = state.get("read_a")
        read_b_result = state.get("read_b")
        raw_findings = list(read_a.findings if read_a else []) + list(
            read_b_result.findings if read_b_result else []
        )
        alerts = triage(raw_findings, deps.settings, image_ref=state.get("image_path", ""))
        return {"alerts": alerts, "trace_events": [_trace_event("critical_triage")]}

    # -- arbiter --------------------------------------------------------------
    def _arbiter(state: GraphState) -> dict:
        outcome = arbitrate(
            state.get("findings", []),
            state.get("disagreements", []),
            arbiter_deps,
            deps.settings,
            state["image_path"],
        )
        return {
            "findings": outcome.findings,
            "notes": list(outcome.notes),
            "arbitration_records": [r.model_dump(mode="json") for r in outcome.records],
            "budget_spent": outcome.calls_made,
            "trace_events": [_trace_event("arbiter")],
        }

    # -- report_writer ------------------------------------------------------
    def _report_writer(state: GraphState) -> dict:
        study = StudyState.model_validate(dict(state))
        before = len(study.notes)
        draft = write_report(state.get("findings", []), study, deps.report_client, deps.settings)
        new_notes = study.notes[before:]
        return {"report": draft, "notes": new_notes, "trace_events": [_trace_event("report_writer")]}

    # -- evidence_check (gate G2) -- retries report_writer exactly once ------
    def _evidence_check(state: GraphState) -> dict:
        draft = state["report"]
        findings = state.get("findings", [])
        already_retried = state.get("evidence_retried", False)
        trace = [_trace_event("evidence_check")]

        new_draft, new_notes, needs_retry = _apply_evidence_gate(draft, findings, already_retried)

        update: dict = {"trace_events": trace, "needs_retry": needs_retry}
        if needs_retry:
            update["evidence_retried"] = True
        else:
            update["report"] = new_draft
            if new_notes:
                update["notes"] = new_notes
        return update

    def _route_after_evidence_check(state: GraphState) -> str:
        return "report_writer" if state.get("needs_retry") else "language_guard"

    # -- language_guard (gate G3) -- neutralize, don't reject ----------------
    def _language_guard(state: GraphState) -> dict:
        draft = state["report"]
        new_sentences = []
        changed = False
        for sentence in draft.sentences:
            if detect_redlines(sentence.text):
                new_sentences.append(sentence.model_copy(update={"text": neutralize(sentence.text)}))
                changed = True
            else:
                new_sentences.append(sentence)
        new_draft = draft.model_copy(update={"sentences": new_sentences})
        if not has_disclaimer(new_draft):
            new_draft = new_draft.model_copy(update={"disclaimer": REQUIRED_DISCLAIMER})

        notes = ["language_guard: neutralized redlined sentence(s) rather than rejecting the draft"] if changed else []
        return {"report": new_draft, "notes": notes, "trace_events": [_trace_event("language_guard")]}

    # -- review_queue (fan-in: critical_triage + language_guard) -------------
    def _review_queue(state: GraphState) -> dict:
        """Terminal status, decided by the output guardrail.

        The completeness checks deliberately live in
        `guardrails.output.enforce_output` rather than being repeated here:
        gates G2 and G3 are also checked by the eval suite, and two copies of
        the same rule drift apart exactly when it matters. This node decides
        what status is *proposed*; the guardrail only ever downgrades it.
        """
        trace = [_trace_event("review_queue")]

        if state.get("alerts"):
            proposed = "CRITICAL_ESCALATED"
        elif any(f.needs_human for f in state.get("findings", [])):
            proposed = "HELD"
        else:
            proposed = "DRAFT_READY"

        status, reasons = enforce_output(
            state.get("report"), state.get("findings", []), proposed
        )

        update: dict = {"status": status, "trace_events": trace}
        if reasons:
            update["notes"] = [f"output guardrail withheld the draft: {r}" for r in reasons]
        return update

    graph = StateGraph(GraphState)
    graph.add_node("intake", _intake)
    graph.add_node("deid", _deid)
    graph.add_node("qc", _qc)
    graph.add_node("reader_a", _reader_a)
    graph.add_node("reader_b", _reader_b)
    graph.add_node("merge", _merge)
    graph.add_node("critical_triage", _critical_triage)
    graph.add_node("arbiter", _arbiter)
    graph.add_node("report_writer", _report_writer)
    graph.add_node("evidence_check", _evidence_check)
    graph.add_node("language_guard", _language_guard)
    graph.add_node("review_queue", _review_queue)

    graph.add_edge(START, "intake")
    graph.add_conditional_edges("intake", _route_after_intake)
    graph.add_edge("deid", "qc")
    graph.add_conditional_edges("qc", _route_after_qc)
    graph.add_edge("reader_a", "merge")
    graph.add_edge("reader_b", "merge")
    graph.add_edge("merge", "critical_triage")
    graph.add_edge("merge", "arbiter")
    graph.add_edge("arbiter", "report_writer")
    graph.add_conditional_edges("evidence_check", _route_after_evidence_check)
    graph.add_edge("report_writer", "evidence_check")
    graph.add_edge("language_guard", "review_queue")
    graph.add_edge("critical_triage", "review_queue")
    graph.add_edge("review_queue", END)

    return graph.compile()
