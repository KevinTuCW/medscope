"""Tests for medscope.readers.vlm -- reader_b's client plumbing and the
JSON-to-Finding parser that turns a raw model reply into a ReadResult.

Covers: selecting the offline stand-in by default, failing loud (not
silently degrading) when a real VLM is requested but unconfigured, the
offline client's deterministic derivation of findings from ground-truth
report text (now surfaced through `chat_with_image`, exactly like the real
client -- see OfflineVLMClient's docstring for why that's plumbing
coverage, not model-quality coverage), the confidence-word mapping, and
`read_b`'s tolerant JSON parsing (fenced, bare-object, missing-field,
unparseable replies).
"""

import json

import pytest

from medscope.config import Settings
from medscope.llm import ModelResponse
from medscope.readers.vlm import (
    CONFIDENCE_CERTAIN,
    CONFIDENCE_POSSIBLE,
    CONFIDENCE_PROBABLE,
    OfflineVLMClient,
    _confidence_to_prob,
    build_vlm_client,
    read_b,
)
from medscope.state import StudyState


def test_build_vlm_client_returns_offline_when_use_real_vlm_false():
    settings = Settings(use_real_vlm=False)
    client = build_vlm_client(settings)
    assert isinstance(client, OfflineVLMClient)


def test_build_vlm_client_raises_when_real_vlm_requested_without_key():
    settings = Settings(use_real_vlm=True, vlm_api_key="", vlm_model="")
    with pytest.raises(RuntimeError):
        build_vlm_client(settings)


def test_build_vlm_client_raises_when_key_present_but_model_missing():
    settings = Settings(use_real_vlm=True, vlm_api_key="sk-real", vlm_model="")
    with pytest.raises(RuntimeError):
        build_vlm_client(settings)


def test_offline_client_is_deterministic_across_calls():
    client = OfflineVLMClient()
    text = "Cardiomegaly. Small right pleural effusion. Tortuous aorta."

    first = client.read(text)
    second = client.read(text)

    assert [f.model_dump() for f in first.findings] == [f.model_dump() for f in second.findings]


def test_offline_client_emits_vlm_source_and_no_locus():
    client = OfflineVLMClient()
    result = client.read("Cardiomegaly.")

    assert result.findings
    for finding in result.findings:
        assert finding.source == "vlm"
        assert finding.locus is None


def test_offline_client_canonicalizes_known_terms():
    client = OfflineVLMClient()
    result = client.read("Cardiomegaly. Pleural effusion.")

    labels = {f.label for f in result.findings}
    assert "Cardiomegaly" in labels
    assert "PleuralEffusion" in labels


def test_offline_client_preserves_unmapped_terms_via_raw_label():
    client = OfflineVLMClient()
    result = client.read("Tortuous aorta")

    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.label == "Tortuous aorta"
    assert finding.raw_label == "Tortuous aorta"


def test_offline_client_read_result_has_reader_b_and_zero_tokens():
    client = OfflineVLMClient()
    result = client.read("Cardiomegaly.")

    assert result.reader == "b"
    assert result.tokens == 0


# -- OfflineVLMClient.chat_with_image: the offline double now implements the
# same Protocol method the real client does, so reader_b has one code path
# and the offline suite genuinely exercises the JSON parser below rather
# than bypassing it through `.read()`. --


def test_offline_client_chat_with_image_returns_json_text():
    client = OfflineVLMClient(impression_text="Cardiomegaly. Small right pleural effusion.")
    response = client.chat_with_image("prompt text", "/tmp/fake.png")

    assert isinstance(response, ModelResponse)
    payload = json.loads(response.text)
    assert isinstance(payload, list)
    labels = {item["label"] for item in payload}
    assert "Cardiomegaly" in labels
    assert "Small right pleural effusion" in labels


def test_offline_client_chat_with_image_empty_when_no_impression_text():
    client = OfflineVLMClient()
    response = client.chat_with_image("prompt text", "/tmp/fake.png")

    assert json.loads(response.text) == []


def test_offline_client_drives_read_b_end_to_end():
    """The point of the refactor: build the offline client the way
    production code would (impression_text baked in), run it through the
    real `read_b`, and confirm the parser -- not a shortcut -- produced the
    Findings."""
    settings = Settings()
    client = OfflineVLMClient(impression_text="Cardiomegaly. Tortuous aorta.")
    state = StudyState(study_id="CXR9", image_path="/tmp/fake.png")

    result = read_b(state, client, settings)

    assert result.reader == "b"
    labels = {f.label for f in result.findings}
    assert "Cardiomegaly" in labels
    assert "Tortuous aorta" in labels  # unmapped -- preserved raw
    for finding in result.findings:
        assert finding.source == "vlm"
        assert finding.locus is None


# -- confidence-word mapping --


@pytest.mark.parametrize("word", ["certain", "definite", "confirmed", "clearly", "确定", "明确", "肯定"])
def test_confidence_certain_words_map_to_certain_level(word):
    assert _confidence_to_prob(word) == CONFIDENCE_CERTAIN


@pytest.mark.parametrize("word", ["probable", "likely", "suspicious", "可能性大", "怀疑", "考虑"])
def test_confidence_probable_words_map_to_probable_level(word):
    assert _confidence_to_prob(word) == CONFIDENCE_PROBABLE


@pytest.mark.parametrize(
    "word", ["possible", "cannot exclude", "cannot rule out", "equivocal", "不除外", "不排除", "待排"]
)
def test_confidence_possible_words_map_to_possible_level(word):
    assert _confidence_to_prob(word) == CONFIDENCE_POSSIBLE


def test_confidence_missing_defaults_to_possible():
    assert _confidence_to_prob(None) == CONFIDENCE_POSSIBLE
    assert _confidence_to_prob("") == CONFIDENCE_POSSIBLE


def test_confidence_unrecognized_word_defaults_to_possible():
    assert _confidence_to_prob("somewhat maybe kind of") == CONFIDENCE_POSSIBLE


# -- read_b: tolerant JSON parsing --


class _StubClient:
    name = "stub"

    def __init__(self, text: str, tokens: int = 7):
        self._text = text
        self._tokens = tokens

    def chat_with_image(self, prompt, image_path, *, system=None):
        return ModelResponse(text=self._text, tokens=self._tokens)


def _plain_state() -> StudyState:
    return StudyState(study_id="CXR-parse", image_path="/tmp/fake.png")


def test_read_b_parses_well_formed_json_list():
    text = json.dumps(
        [
            {"label": "Cardiomegaly", "confidence": "certain"},
            {"label": "Pneumothorax", "confidence": "possible"},
        ]
    )
    result = read_b(_plain_state(), _StubClient(text), Settings())

    labels = {f.label: f.prob for f in result.findings}
    assert labels["Cardiomegaly"] == CONFIDENCE_CERTAIN
    assert labels["Pneumothorax"] == CONFIDENCE_POSSIBLE
    assert result.tokens == 7


def test_read_b_parses_json_inside_code_fence():
    fenced = "Here are my findings:\n```json\n" + json.dumps(
        [{"label": "Cardiomegaly", "confidence": "probable"}]
    ) + "\n```\nLet me know if you need more."
    result = read_b(_plain_state(), _StubClient(fenced), Settings())

    assert len(result.findings) == 1
    assert result.findings[0].label == "Cardiomegaly"
    assert result.findings[0].prob == CONFIDENCE_PROBABLE


def test_read_b_parses_bare_object_as_single_finding():
    text = json.dumps({"label": "Cardiomegaly", "confidence": "certain"})
    result = read_b(_plain_state(), _StubClient(text), Settings())

    assert len(result.findings) == 1
    assert result.findings[0].label == "Cardiomegaly"


def test_read_b_missing_confidence_field_defaults_to_possible():
    text = json.dumps([{"label": "Cardiomegaly"}])
    result = read_b(_plain_state(), _StubClient(text), Settings())

    assert result.findings[0].prob == CONFIDENCE_POSSIBLE


def test_read_b_preserves_unmapped_label_via_raw_label():
    text = json.dumps([{"label": "Tortuous aorta", "confidence": "possible"}])
    result = read_b(_plain_state(), _StubClient(text), Settings())

    finding = result.findings[0]
    assert finding.label == "Tortuous aorta"
    assert finding.raw_label == "Tortuous aorta"


def test_read_b_locus_always_none():
    text = json.dumps([{"label": "Pneumothorax", "confidence": "certain", "location": "left"}])
    result = read_b(_plain_state(), _StubClient(text), Settings())

    assert result.findings[0].locus is None


def test_read_b_unrecoverable_output_returns_empty_result_with_note():
    text = "I'm sorry, I cannot analyze this image."
    result = read_b(_plain_state(), _StubClient(text), Settings())

    assert result.reader == "b"
    assert result.findings == []
    assert result.notes


def test_read_b_does_not_raise_on_malformed_json():
    text = "{not valid json at all"
    result = read_b(_plain_state(), _StubClient(text), Settings())

    assert result.findings == []
    assert result.notes


def test_read_b_populates_tokens_and_latency():
    text = json.dumps([{"label": "Cardiomegaly", "confidence": "certain"}])
    result = read_b(_plain_state(), _StubClient(text, tokens=42), Settings())

    assert result.tokens == 42
    assert result.latency_ms >= 0
