"""Tests for Task 2.3's describer mode -- the fallback reader_b takes when
it's demoted from peer reader to a plain observation-describer (see
Settings.reader_b_mode, decided by scripts/calibrate_vlm.py against
Settings.kappa_floor / disagreement_ceiling).

Demotion narrows what reader_b is allowed to claim (no presence/absence
judgement, no confidence) -- it must not become a licence to leak reader_a's
output into reader_b's prompt. test_describer_still_independent guards
exactly that.
"""

import json

from medscope.config import Settings
from medscope.llm import ModelResponse
from medscope.merge import merge_reads
from medscope.readers.vlm import build_reader_b_prompt, read_b
from medscope.state import Description, Finding, ReadResult, StudyState

_INDICATION = "Evaluate for infection given fever and cough for three days."
_HISTORY = "患者男，45岁，发热咳嗽三天，既往体健。"


def _reader_settings() -> Settings:
    return Settings(reader_b_mode="reader")


def _describer_settings() -> Settings:
    return Settings(reader_b_mode="describer")


def _state_with_leaked_read_a() -> StudyState:
    read_a = ReadResult(
        reader="a",
        findings=[
            Finding(
                label="Pneumothorax",
                prob=0.87,
                source="cnn",
                locus={"cx": 0.5, "cy": 0.5, "r": 0.1},
                raw_label="Pneumothorax",
            ),
        ],
        latency_ms=120,
    )
    return StudyState(
        study_id="CXR1",
        image_path="/tmp/fake.png",
        indication=_INDICATION,
        history_text=_HISTORY,
        read_a=read_a,
    )


class _StubClient:
    name = "stub"

    def __init__(self, text: str, tokens: int = 5):
        self._text = text
        self._tokens = tokens

    def chat_with_image(self, prompt, image_path, *, system=None):
        return ModelResponse(text=self._text, tokens=self._tokens)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_describer_prompt_forbids_judgement():
    state = _state_with_leaked_read_a()
    prompt = build_reader_b_prompt(state, _describer_settings())

    lowered = prompt.lower()
    assert "describe" in lowered
    # Reader-mode's confidence request must not leak into describer mode --
    # mentioning the word "confidence" to explicitly rule it out is fine,
    # asking for a hedging word is not.
    assert "hedging word" not in lowered
    assert "do not include" in lowered and "confidence" in lowered
    # Must not ask for a presence/absence call.
    assert "present or absent" in lowered or "presence or absence" in lowered


def test_describer_still_independent():
    """Demotion is not a licence to leak -- the describer-mode prompt must
    still contain none of reader_a's output."""
    state = _state_with_leaked_read_a()
    prompt = build_reader_b_prompt(state, _describer_settings())

    assert "Pneumothorax" not in prompt
    assert "0.87" not in prompt
    assert "cnn" not in prompt.lower()


# ---------------------------------------------------------------------------
# read_b in describer mode
# ---------------------------------------------------------------------------


def test_describer_read_b_returns_no_findings():
    text = json.dumps(
        [
            {"label": "Cardiomegaly", "description": "心影明显增大，符合心脏扩大表现"},
            {"label": "纵隔气肿", "description": "可见纵隔气肿"},
        ]
    )
    state = StudyState(study_id="CXR-describer", image_path="/tmp/fake.png")

    result = read_b(state, _StubClient(text), _describer_settings())

    assert result.findings == []
    assert result.descriptions
    assert all(isinstance(d, Description) for d in result.descriptions)
    by_label = {d.label: d.text for d in result.descriptions}
    assert by_label["Cardiomegaly"] == "心影明显增大，符合心脏扩大表现"
    # "纵隔气肿" (pneumomediastinum) canonicalizes -- see ontology.CANONICAL
    # -- so its Description.label carries the canonical form, not the raw
    # Chinese text, mirroring _finding_from_item's reader-mode behaviour.
    assert "Pneumomediastinum" in by_label
    assert by_label["Pneumomediastinum"] == "可见纵隔气肿"


def test_reader_mode_unchanged():
    """Regression guard: peer-mode read_b/prompt behaviour is untouched now
    that both modes share code."""
    state = _state_with_leaked_read_a()
    prompt = build_reader_b_prompt(state, _reader_settings())
    assert "fever and cough" in prompt
    assert "Pneumothorax" not in prompt

    text = json.dumps([{"label": "Cardiomegaly", "confidence": "certain"}])
    plain_state = StudyState(study_id="CXR-reader", image_path="/tmp/fake.png")
    result = read_b(plain_state, _StubClient(text), _reader_settings())

    assert len(result.findings) == 1
    assert result.findings[0].label == "Cardiomegaly"
    assert result.findings[0].prob == 0.9


# ---------------------------------------------------------------------------
# merge_reads, mode="describer" (see tests/test_merge.py for the base cases;
# these three pin the contract described in Task 2.3's spec). Fixture-built
# read_b results below, matching test_merge.py's style -- exercised for real
# (no hand-built ReadResult) in test_describer_end_to_end_composition below,
# since fixture-only coverage is exactly what let Task 1.6's contract
# mismatch (Finding-shaped read_b vs. a real read_b() that never produced
# one) go unnoticed.
# ---------------------------------------------------------------------------


def _finding(label, prob, source="cnn", locus=None, raw_label=None, notes=None):
    return Finding(
        label=label,
        prob=prob,
        source=source,
        locus=locus,
        raw_label=raw_label if raw_label is not None else label,
        notes=notes or [],
    )


def _description(label, text, raw_label=None):
    return Description(label=label, text=text, raw_label=raw_label if raw_label is not None else label)


def _read(reader, findings, latency_ms=10):
    return ReadResult(reader=reader, findings=findings, latency_ms=latency_ms)


def _read_descriptions(reader, descriptions, latency_ms=10):
    return ReadResult(reader=reader, descriptions=descriptions, latency_ms=latency_ms)


THRESHOLD = 0.5


def test_describer_merge_uses_cnn_positives_only():
    read_a = _read("a", [_finding("Cardiomegaly", 0.9), _finding("Pneumothorax", 0.1)])
    read_b_result = _read_descriptions(
        "b",
        [
            _description("Cardiomegaly", "心影明显增大，符合心脏扩大表现"),
            _description("Pneumothorax", "可见气胸征象"),
        ],
    )

    findings, disagreements, _ = merge_reads(read_a, read_b_result, THRESHOLD, mode="describer")

    assert disagreements == []
    positive = {f.label for f in findings if f.prob >= THRESHOLD}
    assert positive == {"Cardiomegaly"}


def test_describer_descriptions_attach_to_matching_finding():
    read_a = _read("a", [_finding("Cardiomegaly", 0.9)])
    read_b_result = _read_descriptions("b", [_description("Cardiomegaly", "心影增大")])

    findings, _, _ = merge_reads(read_a, read_b_result, THRESHOLD, mode="describer")

    cardiomegaly = next(f for f in findings if f.label == "Cardiomegaly")
    assert "心影增大" in cardiomegaly.notes


def test_describer_unmatched_description_is_not_discarded_and_not_a_finding():
    read_a = _read("a", [_finding("Cardiomegaly", 0.9)])
    read_b_result = _read_descriptions("b", [_description("纵隔气肿", "可见纵隔气肿", raw_label="纵隔气肿")])

    findings, disagreements, kappa = merge_reads(read_a, read_b_result, THRESHOLD, mode="describer")

    # Not present as its own Finding, and not silently attached to an
    # unrelated finding either -- describer descriptions never masquerade
    # as a judgement.
    assert all(f.label != "纵隔气肿" for f in findings)
    assert all("可见纵隔气肿" not in f.notes for f in findings)
    # Not discarded either: merge_reads never mutates its inputs, so the
    # description is still exactly where read_b put it -- a caller that
    # keeps state.read_b around (as StudyState does) can always recover it.
    assert "可见纵隔气肿" in read_b_result.descriptions[0].text


# ---------------------------------------------------------------------------
# Composition: a real read_b() output fed into a real merge_reads() call --
# no hand-built ReadResult standing in for reader_b. This is the test whose
# absence let Task 1.6's contract mismatch (merge_reads expected label-keyed
# Findings; a real describer-mode read_b() produced none) survive.
# ---------------------------------------------------------------------------


def test_describer_end_to_end_composition():
    text = json.dumps(
        [
            {"label": "Cardiomegaly", "description": "心影明显增大，符合心脏扩大表现"},
            {"label": "纵隔气肿", "description": "可见纵隔气肿"},  # unmatched by reader_a below
        ]
    )
    state = StudyState(study_id="CXR-composition", image_path="/tmp/fake.png")
    real_read_b_result = read_b(state, _StubClient(text), _describer_settings())

    # Sanity check on the premise: real read_b() output has no Findings for
    # merge to (wrongly) match on -- descriptions are its only channel.
    assert real_read_b_result.findings == []
    assert real_read_b_result.descriptions

    read_a_result = _read("a", [_finding("Cardiomegaly", 0.9), _finding("Pneumothorax", 0.1)])

    findings, disagreements, kappa = merge_reads(
        read_a_result, real_read_b_result, THRESHOLD, mode="describer"
    )

    assert disagreements == []
    positive = {f.label for f in findings if f.prob >= THRESHOLD}
    assert positive == {"Cardiomegaly"}  # reader_a's positive set exactly

    cardiomegaly = next(f for f in findings if f.label == "Cardiomegaly")
    assert "心影明显增大，符合心脏扩大表现" in cardiomegaly.notes

    # The unmatched pneumomediastinum description never becomes a Finding,
    # but it's still recoverable from real_read_b_result itself.
    assert all(f.label != "Pneumomediastinum" for f in findings)
    assert any(d.label == "Pneumomediastinum" for d in real_read_b_result.descriptions)
