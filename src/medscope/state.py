"""Pydantic state models shared across the medscope pipeline.

These are the data contracts between graph nodes (reader_a, reader_b, the
arbiter, critical_triage, and the report writer). Field names and types
here are load-bearing for later tasks (the LangGraph state, the report
generator, and the eval gates in Phase 3), not just for this one.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Finding(BaseModel):
    label: str  # canonical ontology name
    prob: float = Field(ge=0, le=1)
    source: Literal["cnn", "vlm", "arbiter"]
    locus: dict | None = None  # normalized coords cx/cy/r; None if reader gives no localization
    evidence_id: str = ""  # stable id that report sentences cite; auto-derived if left blank
    raw_label: str = ""  # reader's original wording, kept for audit
    notes: list[str] = Field(default_factory=list)  # carries VLM description text in describer mode

    @model_validator(mode="after")
    def _default_evidence_id(self) -> "Finding":
        # An empty evidence_id must be structurally impossible: gate G2 treats
        # a dangling id (one that matches no finding) as a violation, and an
        # empty string would let an unrelated sentence "match" an unrelated
        # empty-id finding, silently passing a gate that should have failed.
        # Deterministic (not a uuid) so the same finding gets the same id
        # across a rerun; (source, label) is unique within a study post-merge.
        if not self.evidence_id:
            self.evidence_id = f"{self.source}:{self.label}"
        return self


class ReadResult(BaseModel):
    reader: Literal["a", "b"]
    findings: list[Finding] = Field(default_factory=list)
    latency_ms: int
    tokens: int = 0
    notes: list[str] = Field(default_factory=list)


class Disagreement(BaseModel):
    label: str
    a_prob: float | None
    b_prob: float | None
    kind: Literal["presence", "magnitude", "unique"]
    # presence = one reader positive, one negative
    # magnitude = both positive but across a threshold
    # unique = only one reader mentioned this label at all


class CriticalAlert(BaseModel):
    label: str
    prob: float
    source: str
    detected_at: datetime
    image_ref: str


class ReportSentence(BaseModel):
    text: str
    section: Literal["technique", "findings", "impression", "recommendation"]
    evidence_ids: list[str] = Field(default_factory=list)


class ReportDraft(BaseModel):
    sentences: list[ReportSentence] = Field(default_factory=list)
    disclaimer: str
    removed_count: int = 0


class StudyState(BaseModel):
    study_id: str
    image_path: str = ""
    history_text: str = ""
    indication: str = ""
    deid_report: dict = Field(default_factory=dict)
    qc: dict = Field(default_factory=dict)
    read_a: ReadResult | None = None
    read_b: ReadResult | None = None
    findings: list[Finding] = Field(default_factory=list)
    disagreements: list[Disagreement] = Field(default_factory=list)
    kappa: float | None = None
    alerts: list[CriticalAlert] = Field(default_factory=list)
    report: ReportDraft | None = None
    status: Literal[
        "pending",
        "NEEDS_REPEAT",
        "CRITICAL_ESCALATED",
        "DRAFT_READY",
        "HELD",
        "GUARDRAIL_BLOCKED",
    ] = "pending"
    notes: list[str] = Field(default_factory=list)
    trace_events: list[dict] = Field(default_factory=list)
    budget_spent: int = 0
    tokens_used: int = 0
