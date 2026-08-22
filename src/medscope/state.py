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
    # Which film of the study this reading came from. Empty for readers that
    # only ever see one image (reader_b, the arbiter). Once reader_a reads
    # every view of a study, `locus` on its own becomes a trap: a Grad-CAM
    # computed on the lateral film and drawn over the frontal one is a
    # confidently wrong overlay. The coordinates only mean anything
    # alongside the image they were computed on.
    image_ref: str = ""
    notes: list[str] = Field(default_factory=list)  # carries VLM description text in describer mode
    # Set by the arbiter (Task 2.4) for a disagreement it could not resolve
    # (verdict UNCERTAIN) or never got to (budget exceeded): the finding is
    # kept in the final set rather than silently dropped, but flagged so a
    # human reviews it. Defaulted so existing readers/merge output is
    # unaffected -- only the arbiter ever sets this True.
    needs_human: bool = False

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


class Description(BaseModel):
    """One reader_b observation when `Settings.reader_b_mode ==
    "describer"`.

    Distinct from `Finding` on purpose: a `Finding` is a positive judgement
    with a probability, from a named source, and gate G2 lets report
    sentences cite a `Finding` as evidence. A demoted reader_b makes no
    judgement -- if its description lived in the findings list in any form
    (even a `Finding` with an unused/zero `prob`), the report writer
    (Task 2.5) could cite it as evidence for a claim it never made. This
    type makes that structurally impossible instead of relying on every
    downstream consumer to remember the convention.

    `label` carries `canonical()`'s output where the description's stated
    label maps onto the shared ontology, or the raw label text otherwise --
    this is what `merge.py`'s describer-mode matching keys on to attach a
    description to the reader_a `Finding` it's about.
    """

    label: str
    text: str
    raw_label: str = ""


class ReadResult(BaseModel):
    reader: Literal["a", "b"]
    findings: list[Finding] = Field(default_factory=list)
    latency_ms: int
    tokens: int = 0
    notes: list[str] = Field(default_factory=list)
    # Populated instead of `findings` when reader_b runs in describer mode
    # -- see `Description`'s docstring for why these are a separate type.
    descriptions: list[Description] = Field(default_factory=list)


class Disagreement(BaseModel):
    label: str
    a_prob: float | None
    b_prob: float | None
    kind: Literal["presence", "magnitude", "unique"]
    #: Whether this label is one the two readers can actually be compared
    #: on (`ontology.COMPARISON_LABELS`). False means one reader named
    #: something the other has no way to speak about -- reader_a calling a
    #: label reader_b never uses, or reader_b reporting "central venous
    #: catheter", which is real information but is not two readers
    #: disagreeing about the same thing. Measured over 40 studies, 95% of
    #: all disagreements were of that second kind; counting them as
    #: conflicts is what drove kappa to -0.041. Still recorded, still
    #: arbitrated -- just after the ones where both readers spoke.
    in_vocabulary: bool = True
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
    #: The film shown in the workbench and handed to the single-image
    #: consumers (reader_b, the arbiter, the report writer). Chosen by
    #: `views.primary_view`, never by filesystem order.
    image_path: str = ""
    #: Every film of the study. reader_a reads all of them -- a radiologist
    #: reads a study, not a file. Left empty by callers that genuinely have
    #: one image; reader_a then falls back to `image_path`, so no consumer
    #: has to keep the two fields in sync by hand.
    image_paths: list[str] = Field(default_factory=list)
    history_text: str = ""
    indication: str = ""
    deid_report: dict = Field(default_factory=dict)
    qc: dict = Field(default_factory=dict)
    read_a: ReadResult | None = None
    read_b: ReadResult | None = None
    findings: list[Finding] = Field(default_factory=list)
    disagreements: list[Disagreement] = Field(default_factory=list)
    #: Agreement over `ontology.COMPARISON_LABELS` -- the labels the two
    #: readers can actually be compared on.
    kappa: float | None = None
    #: The same agreement computed over every mapped label, kept next to
    #: the narrowed one so narrowing can never quietly improve the number.
    kappa_all_labels: float | None = None
    #: One record per disagreement, including ones the budget cap left
    #: unarbitrated. Kept as a fact rather than re-derived downstream: the
    #: workbench used to infer a verdict from whether a finding survived
    #: into the final set, which silently couples the audit view to
    #: `arbiter.py`'s internals and would misreport if those ever change.
    arbitration_records: list[dict] = Field(default_factory=list)
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
