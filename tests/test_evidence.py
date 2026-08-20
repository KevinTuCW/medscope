"""Gate G2: no conclusion sentence without evidence.

Every findings/impression/recommendation sentence must cite at least one real
`Finding.evidence_id`. Two failure shapes count as violations:

* a bare assertion — no citation at all;
* a dangling citation — an id matching no finding. This is the more dangerous
  of the two, because it looks like evidence. A reviewer scanning for
  uncited sentences would pass straight over it.

`technique` is exempt: "PA and lateral chest radiograph" describes the
acquisition, not a claim about the patient.
"""

from medscope.evidence import check_evidence, coverage
from medscope.state import Finding, ReportDraft, ReportSentence


def _finding(label: str = "Cardiomegaly", prob: float = 0.8) -> Finding:
    return Finding(label=label, prob=prob, source="cnn")


def _draft(*sentences: ReportSentence) -> ReportDraft:
    return ReportDraft(sentences=list(sentences), disclaimer="x")


def test_fully_cited_draft_has_no_violations():
    f = _finding()
    draft = _draft(
        ReportSentence(text="心影增大。", section="findings", evidence_ids=[f.evidence_id])
    )
    assert check_evidence(draft, [f]) == []
    assert coverage(draft, [f]) == 1.0


def test_bare_assertion_detected():
    f = _finding()
    draft = _draft(
        ReportSentence(text="心影增大。", section="impression", evidence_ids=[])
    )
    violations = check_evidence(draft, [f])
    assert len(violations) == 1
    assert violations[0].reason == "no_evidence"


def test_dangling_citation_detected():
    """An id pointing at nothing is worse than no id: it fakes provenance."""
    f = _finding()
    draft = _draft(
        ReportSentence(
            text="心影增大。", section="impression", evidence_ids=["cnn:DoesNotExist"]
        )
    )
    violations = check_evidence(draft, [f])
    assert len(violations) == 1
    assert violations[0].reason == "dangling_evidence"


def test_partially_dangling_citation_still_counts_as_supported():
    """One real citation is enough; the dangling one is still reported."""
    f = _finding()
    draft = _draft(
        ReportSentence(
            text="心影增大。",
            section="impression",
            evidence_ids=[f.evidence_id, "cnn:Ghost"],
        )
    )
    violations = check_evidence(draft, [f])
    assert [v.reason for v in violations] == ["dangling_evidence"]
    assert coverage(draft, [f]) == 1.0


def test_technique_section_is_exempt():
    draft = _draft(
        ReportSentence(text="胸部正位，立位摄片。", section="technique", evidence_ids=[])
    )
    assert check_evidence(draft, []) == []


def test_recommendation_is_not_exempt():
    """Recommendations are claims about the patient and need grounding too."""
    draft = _draft(
        ReportSentence(text="建议随访复查。", section="recommendation", evidence_ids=[])
    )
    assert len(check_evidence(draft, [])) == 1


def test_coverage_ratio_counts_only_non_exempt_sentences():
    f = _finding()
    draft = _draft(
        ReportSentence(text="胸部正位片。", section="technique", evidence_ids=[]),
        ReportSentence(text="心影增大。", section="findings", evidence_ids=[f.evidence_id]),
        ReportSentence(text="建议随访。", section="recommendation", evidence_ids=[]),
    )
    assert coverage(draft, [f]) == 0.5


def test_coverage_of_draft_with_no_claims_is_one():
    """A technique-only draft makes no claims, so it is vacuously fully cited.

    Returning 0.0 here would fail gate G2 on a draft that asserts nothing —
    penalising the safest possible output.
    """
    draft = _draft(
        ReportSentence(text="胸部正位片。", section="technique", evidence_ids=[])
    )
    assert coverage(draft, []) == 1.0


def test_empty_string_evidence_id_is_dangling_not_supporting():
    """Guards the Finding.evidence_id invariant from the other side.

    `Finding` auto-derives a non-empty id, so an empty citation can never match
    a real finding — it must be reported rather than silently pairing with one.
    """
    f = _finding()
    draft = _draft(
        ReportSentence(text="心影增大。", section="impression", evidence_ids=[""])
    )
    violations = check_evidence(draft, [f])
    assert violations and violations[0].reason == "dangling_evidence"
