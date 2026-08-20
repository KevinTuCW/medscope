"""Tests for medscope.report -- the report writer sitting under gates G2
(no conclusion sentence without evidence) and G3 (diagnostic-language
redlines).

The central contract under test: every findings/impression/recommendation
sentence the writer emits cites a real `Finding.evidence_id`, and never a
`Description`'s or an unparseable/unknown id. Around that: the four-section
shape, the disclaimer, redline-freeness, a negative study still producing a
grounded report, tolerant parsing of a real model's messy JSON, and the
rule-based fallback that fires when a reply can't be salvaged.

All doubles here are hermetic (no network, no key) -- `_StubClient` returns
canned text like `tests/test_describer_mode.py`'s double; `OfflineReportClient`
is exercised directly for the four "always works, plumbing-only" checks.
"""

import json

import pytest

from medscope.config import Settings
from medscope.evidence import check_evidence, coverage
from medscope.language import REQUIRED_DISCLAIMER, detect_redlines, has_disclaimer
from medscope.llm import ModelResponse
from medscope.report import OfflineReportClient, write_report
from medscope.state import Description, Finding, ReadResult, StudyState

_SECTION_ORDER = ("technique", "findings", "impression", "recommendation")


class _StubClient:
    name = "stub"

    def __init__(self, text: str, tokens: int = 5):
        self._text = text
        self._tokens = tokens

    def chat_with_image(self, prompt, image_path, *, system=None):
        return ModelResponse(text=self._text, tokens=self._tokens)


def _finding(label="Cardiomegaly", prob=0.8, source="cnn", evidence_id=None):
    kwargs = {"label": label, "prob": prob, "source": source}
    if evidence_id is not None:
        kwargs["evidence_id"] = evidence_id
    return Finding(**kwargs)


def _state(image_path="/tmp/fake.png") -> StudyState:
    return StudyState(study_id="CXR1", image_path=image_path)


def _assert_sections_in_order(draft):
    seen_order = []
    for s in draft.sentences:
        if s.section not in seen_order:
            seen_order.append(s.section)
    assert seen_order == list(_SECTION_ORDER)
    assert {s.section for s in draft.sentences} == set(_SECTION_ORDER)


# ---------------------------------------------------------------------------
# OfflineReportClient -- plumbing-only, always exercised offline
# ---------------------------------------------------------------------------


def test_offline_writer_produces_four_sections_in_order():
    findings = [_finding("Cardiomegaly", 0.85), _finding("Nodule", 0.4, source="vlm")]
    draft = write_report(findings, _state(), OfflineReportClient(), Settings())

    _assert_sections_in_order(draft)


def test_offline_writer_fully_cited():
    findings = [_finding("Cardiomegaly", 0.85), _finding("Nodule", 0.4, source="vlm")]
    draft = write_report(findings, _state(), OfflineReportClient(), Settings())

    assert check_evidence(draft, findings) == []
    assert coverage(draft, findings) == 1.0


def test_offline_writer_disclaimer():
    findings = [_finding()]
    draft = write_report(findings, _state(), OfflineReportClient(), Settings())

    assert draft.disclaimer == REQUIRED_DISCLAIMER
    assert has_disclaimer(draft)


def test_offline_writer_no_redlines():
    findings = [_finding("Cardiomegaly", 0.85), _finding("Nodule", 0.4, source="vlm")]
    draft = write_report(findings, _state(), OfflineReportClient(), Settings())

    for sentence in draft.sentences:
        assert detect_redlines(sentence.text) == [], sentence.text


# ---------------------------------------------------------------------------
# Negative study
# ---------------------------------------------------------------------------


def test_negative_study_still_yields_valid_grounded_draft():
    # All sub-threshold -- exactly what merge_reads emits for an agreed
    # both-negative label (see medscope.merge._merge_reader's "else" branch).
    findings = [
        _finding("Cardiomegaly", 0.12),
        _finding("Pneumothorax", 0.05, source="vlm"),
        _finding("Nodule", 0.2),
    ]
    draft = write_report(findings, _state(), OfflineReportClient(), Settings())

    _assert_sections_in_order(draft)
    assert check_evidence(draft, findings) == []
    assert coverage(draft, findings) == 1.0

    impression = [s for s in draft.sentences if s.section == "impression"]
    assert impression
    assert any("未见明显异常" in s.text for s in impression)
    assert all(s.evidence_ids for s in impression)


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------


def test_garbage_model_reply_falls_back_to_rule_based_and_notes_it():
    findings = [_finding("Cardiomegaly", 0.85), _finding("Nodule", 0.3, source="vlm")]
    state = _state()
    client = _StubClient("Sorry, I cannot help with that request today.")

    draft = write_report(findings, state, client, Settings())

    _assert_sections_in_order(draft)
    assert check_evidence(draft, findings) == []
    assert coverage(draft, findings) == 1.0
    assert draft.disclaimer == REQUIRED_DISCLAIMER
    assert any("fallback" in note.lower() for note in state.notes)


def test_incomplete_json_missing_required_section_falls_back():
    """Well-formed JSON, but missing a whole required section (no
    recommendation at all) is not salvageable -- fall back rather than
    ship a three-section draft."""
    findings = [_finding("Cardiomegaly", 0.85)]
    state = _state()
    payload = {
        "technique": [{"text": "胸部正位片。", "evidence_ids": []}],
        "findings": [{"text": "心影增大。", "evidence_ids": [findings[0].evidence_id]}],
        "impression": [{"text": "考虑心影增大。", "evidence_ids": [findings[0].evidence_id]}],
        "recommendation": [],
    }
    client = _StubClient(json.dumps(payload, ensure_ascii=False))

    draft = write_report(findings, state, client, Settings())

    _assert_sections_in_order(draft)
    assert check_evidence(draft, findings) == []
    assert any("fallback" in note.lower() for note in state.notes)


# ---------------------------------------------------------------------------
# Citation filtering -- the structural "descriptions/unknown ids never
# survive" enforcement
# ---------------------------------------------------------------------------


def test_unknown_and_description_citations_do_not_survive():
    findings = [_finding("Cardiomegaly", 0.85, evidence_id="cnn:Cardiomegaly")]
    state = _state()
    # A description reader_b produced -- structurally has no evidence_id at
    # all, so any id a model reply invents to point at it is, by
    # construction, not in the known-evidence set.
    state.read_b = ReadResult(
        reader="b", descriptions=[Description(label="Consolidation", text="斑片影，考虑感染可能")], latency_ms=10
    )
    payload = {
        "technique": [{"text": "胸部正位片。", "evidence_ids": []}],
        "findings": [
            {
                "text": "心影增大，另见斑片影。",
                "evidence_ids": ["cnn:Cardiomegaly", "vlm-desc:Consolidation", "cnn:DoesNotExist"],
            }
        ],
        "impression": [{"text": "考虑心影增大。", "evidence_ids": ["cnn:Cardiomegaly"]}],
        "recommendation": [
            {"text": "建议结合临床随诊。", "evidence_ids": ["cnn:Cardiomegaly"]}
        ],
    }
    client = _StubClient(json.dumps(payload, ensure_ascii=False))

    draft = write_report(findings, state, client, Settings())

    findings_sentences = [s for s in draft.sentences if s.section == "findings"]
    assert len(findings_sentences) == 1
    assert findings_sentences[0].evidence_ids == ["cnn:Cardiomegaly"]
    assert check_evidence(draft, findings) == []
    assert coverage(draft, findings) == 1.0
    # no fallback needed -- the sentence survived with its one real citation
    assert not any("fallback" in note.lower() for note in state.notes)


def test_invalid_citation_is_stripped_but_the_sentence_still_surfaces():
    """The writer filters bad ids; it does not decide the sentence's fate.

    Deleting an uncitable sentence at construction time had a hidden cost: no
    model reply could then produce a draft that failed `check_evidence`, so
    the graph's "retry once, then strip" node became unreachable and the
    writer never got a second chance to cite the claim properly. That policy
    call now lives in `graph._apply_evidence_gate`; see `tests/test_graph.py`.

    What must still hold here is the structural guarantee: the dangling id
    (`vlm-desc:Ghost`, the shape a `Description` citation would take) is gone,
    so it can never masquerade as evidence.
    """
    findings = [_finding("Cardiomegaly", 0.85, evidence_id="cnn:Cardiomegaly")]
    state = _state()
    payload = {
        "technique": [{"text": "胸部正位片。", "evidence_ids": []}],
        "findings": [
            {"text": "心影增大。", "evidence_ids": ["cnn:Cardiomegaly"]},
            {"text": "另见可疑影，来源不明。", "evidence_ids": ["vlm-desc:Ghost"]},
        ],
        "impression": [{"text": "考虑心影增大。", "evidence_ids": ["cnn:Cardiomegaly"]}],
        "recommendation": [
            {"text": "建议结合临床随诊。", "evidence_ids": ["cnn:Cardiomegaly"]}
        ],
    }
    client = _StubClient(json.dumps(payload, ensure_ascii=False))

    draft = write_report(findings, state, client, Settings())

    findings_sentences = [s for s in draft.sentences if s.section == "findings"]
    assert len(findings_sentences) == 2
    ungrounded = findings_sentences[1]
    assert ungrounded.text == "另见可疑影，来源不明。"
    # the invented id did not survive -- that guarantee stays in the writer
    assert ungrounded.evidence_ids == []
    # and the draft is now a genuine G2 violation for the graph to act on
    assert check_evidence(draft, findings) != []


# ---------------------------------------------------------------------------
# Valid citation preserved verbatim
# ---------------------------------------------------------------------------


def test_valid_model_citation_preserved_verbatim():
    findings = [_finding("Pneumothorax", 0.9, source="cnn", evidence_id="cnn:Pneumothorax")]
    state = _state()
    payload = {
        "technique": [{"text": "胸部正位片。", "evidence_ids": []}],
        "findings": [
            {"text": "右侧胸腔见气胸征象。", "evidence_ids": ["cnn:Pneumothorax"]}
        ],
        "impression": [
            {"text": "提示右侧气胸。", "evidence_ids": ["cnn:Pneumothorax"]}
        ],
        "recommendation": [
            {"text": "建议结合临床及时处理。", "evidence_ids": ["cnn:Pneumothorax"]}
        ],
    }
    client = _StubClient(json.dumps(payload, ensure_ascii=False))

    draft = write_report(findings, state, client, Settings())

    findings_sentences = [s for s in draft.sentences if s.section == "findings"]
    assert findings_sentences[0].text == "右侧胸腔见气胸征象。"
    assert findings_sentences[0].evidence_ids == ["cnn:Pneumothorax"]
    assert check_evidence(draft, findings) == []
    assert coverage(draft, findings) == 1.0
    assert not any("fallback" in note.lower() for note in state.notes)


# ---------------------------------------------------------------------------
# Tolerant parsing
# ---------------------------------------------------------------------------


def test_json_wrapped_in_prose_and_code_fence_is_parsed():
    findings = [_finding("Cardiomegaly", 0.85, evidence_id="cnn:Cardiomegaly")]
    state = _state()
    payload = {
        "technique": [{"text": "胸部正位片。", "evidence_ids": []}],
        "findings": [{"text": "心影增大。", "evidence_ids": ["cnn:Cardiomegaly"]}],
        "impression": [{"text": "考虑心影增大。", "evidence_ids": ["cnn:Cardiomegaly"]}],
        "recommendation": [{"text": "建议随诊。", "evidence_ids": ["cnn:Cardiomegaly"]}],
    }
    wrapped = (
        "Here is the report you asked for:\n\n```json\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n```\nLet me know if you need anything else."
    )
    client = _StubClient(wrapped)

    draft = write_report(findings, state, client, Settings())

    _assert_sections_in_order(draft)
    assert check_evidence(draft, findings) == []
    assert not any("fallback" in note.lower() for note in state.notes)


def test_bare_object_instead_of_list_per_section_is_tolerated():
    """A real model sometimes emits a single object where a one-item list
    was expected -- must not be treated as unparseable."""
    findings = [_finding("Cardiomegaly", 0.85, evidence_id="cnn:Cardiomegaly")]
    state = _state()
    payload = {
        "technique": {"text": "胸部正位片。", "evidence_ids": []},
        "findings": {"text": "心影增大。", "evidence_ids": ["cnn:Cardiomegaly"]},
        "impression": {"text": "考虑心影增大。", "evidence_ids": ["cnn:Cardiomegaly"]},
        "recommendation": {"text": "建议随诊。", "evidence_ids": ["cnn:Cardiomegaly"]},
    }
    client = _StubClient(json.dumps(payload, ensure_ascii=False))

    draft = write_report(findings, state, client, Settings())

    _assert_sections_in_order(draft)
    assert check_evidence(draft, findings) == []
    assert not any("fallback" in note.lower() for note in state.notes)
