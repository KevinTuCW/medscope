"""Gate G2 — every conclusion sentence must cite real evidence.

The whole product promise is a draft where each claim can be traced back to a
`Finding` carrying a model source, a probability and a location on the image.
This module is where that promise becomes checkable.

Two distinct violations, and the second is the dangerous one:

* ``no_evidence`` — a bare assertion, nothing cited.
* ``dangling_evidence`` — an id that matches no finding. This is worse than a
  bare assertion precisely because it *looks* grounded: a reviewer skimming for
  uncited sentences slides right past it. Note that `Finding` auto-derives a
  non-empty ``evidence_id``, so an empty citation can never coincidentally
  match a real finding — it is always dangling.

`technique` sentences are exempt: "PA and lateral chest radiograph" describes
how the image was acquired, not a claim about the patient. The other three
sections all make claims and are all held to the rule, recommendations
included — advising follow-up *is* a statement about this patient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from medscope.state import Finding, ReportDraft, ReportSentence

#: Sections whose sentences assert something about the patient.
CLAIM_SECTIONS: frozenset[str] = frozenset({"findings", "impression", "recommendation"})

ViolationReason = Literal["no_evidence", "dangling_evidence"]


@dataclass(frozen=True)
class BareAssertion:
    index: int
    section: str
    text: str
    reason: ViolationReason
    dangling_ids: tuple[str, ...] = ()


def _claim_sentences(draft: ReportDraft) -> list[tuple[int, ReportSentence]]:
    return [
        (i, s) for i, s in enumerate(draft.sentences) if s.section in CLAIM_SECTIONS
    ]


def check_evidence(
    draft: ReportDraft, findings: Iterable[Finding]
) -> list[BareAssertion]:
    """Report every claim sentence that is uncited or cites a non-existent finding.

    A sentence citing one real id and one dangling id is *supported* but still
    reports the dangling reference — the bad citation is a real defect even
    though the sentence itself is grounded.
    """
    known = {f.evidence_id for f in findings}
    violations: list[BareAssertion] = []

    for index, sentence in _claim_sentences(draft):
        cited = list(sentence.evidence_ids)
        if not cited:
            violations.append(
                BareAssertion(
                    index=index,
                    section=sentence.section,
                    text=sentence.text,
                    reason="no_evidence",
                )
            )
            continue

        dangling = tuple(eid for eid in cited if eid not in known)
        if dangling:
            violations.append(
                BareAssertion(
                    index=index,
                    section=sentence.section,
                    text=sentence.text,
                    reason="dangling_evidence",
                    dangling_ids=dangling,
                )
            )

    return violations


def coverage(draft: ReportDraft, findings: Iterable[Finding]) -> float:
    """Fraction of claim sentences backed by at least one real finding.

    A draft that makes no claims returns 1.0 rather than 0.0. Returning zero
    would fail G2 on a report that asserts nothing — penalising the safest
    output a system can produce.
    """
    known = {f.evidence_id for f in findings}
    claims = _claim_sentences(draft)
    if not claims:
        return 1.0

    supported = sum(
        1
        for _, sentence in claims
        if any(eid in known for eid in sentence.evidence_ids)
    )
    return supported / len(claims)
