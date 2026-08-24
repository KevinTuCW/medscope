"""What must be true of the tracing layer, in two groups.

**Nothing exists when tracing is off.** Covered in `test_bootstrap.py` for
the decorator; here for the model-call and study-trace seams, which are the
ones on the pipeline's hot path.

**Nothing leaks when tracing is on.** This is the group that earns its
keep. medscope's premise includes a de-identification gate, and adding a
SaaS exporter is exactly the kind of change that routes around one: the
Langfuse SDK lifts base64 data URIs out of payloads and uploads them to
object storage, and the `intake` node runs *before* `deid`, so its span
carries whatever raw text a caller supplied. Both of those are leaks that
would look like working instrumentation. The tests below pin the two
defenses -- substituting a reference for the film before the SDK sees it,
and re-running `deid_text` at the export boundary.
"""

import pytest

from medscope.config import Settings
from medscope.llm import ModelResponse, _extract_token_split, _extract_tokens
from medscope.obs import (
    _mask_otel_spans,
    _scrub_attribute,
    _UNCHANGED,
    film_reference,
    _generation_input,
    mask_data_uris,
    observe_model_call,
    scrub,
    study_trace,
    tracing_enabled,
)
from medscope.state import StudyState


class _RecordingClient:
    """A stand-in that records what it was actually called with, so a test
    can prove the traced path and the untraced path call it identically."""

    name = "recording"
    model = "test-model"

    def __init__(self, text: str = "ok") -> None:
        self.text = text
        self.calls: list[tuple] = []

    def chat_with_image(self, prompt, image_path, *, system=None):
        self.calls.append((prompt, str(image_path), system))
        return ModelResponse(text=self.text, tokens=30, input_tokens=20, output_tokens=10)


# --- nothing exists when tracing is off ------------------------------------


def test_observe_model_call_is_a_plain_call_when_unconfigured():
    client = _RecordingClient()
    response = observe_model_call("read-film-vlm", client, "prompt", "/tmp/x.png", system="sys")

    assert response.text == "ok"
    assert client.calls == [("prompt", "/tmp/x.png", "sys")]


def test_study_trace_yields_no_callbacks_when_unconfigured():
    """LangGraph gets an empty callback list, not a no-op handler.

    A handler that does nothing still costs a callback dispatch on every
    node transition, which is precisely the timing the graph's
    critical-alert-ordering assertion depends on.
    """
    state = StudyState(study_id="s1", image_path="/tmp/x.png")
    with study_trace(state, Settings()) as trace:
        assert trace.callbacks == []
        trace.finish(state)  # must not raise


# --- nothing leaks when tracing is on --------------------------------------


def test_mask_data_uris_removes_the_pixels_but_keeps_an_identity():
    payload = "A" * 200
    text = f'{{"url": "data:image/png;base64,{payload}"}}'

    masked = mask_data_uris(text)

    assert payload not in masked
    assert "base64" not in masked
    assert "MASKED image/png" in masked
    # Same film masks to the same reference, so two calls on one study can
    # still be told apart from two calls on different studies.
    assert masked == mask_data_uris(text)
    assert masked != mask_data_uris(text.replace("A", "B"))


def test_generation_input_never_contains_the_film():
    """The substitution happens before the SDK sees the request.

    Masking at export cannot help here: Langfuse extracts and uploads media
    *before* the masking hook runs, so a film handed to the SDK is already
    gone by the time anything could redact it.
    """
    messages = _generation_input("read this film", "/tmp/x.png", "you are reader_b")

    rendered = repr(messages)
    assert "base64" not in rendered
    assert "MASKED" in rendered
    # Still shaped like the conversation the model had, per Langfuse's
    # guidance on renderable input.
    assert messages[0]["role"] == "system"
    assert messages[1]["content"][0]["text"] == "read this film"


def test_film_reference_survives_an_unreadable_path():
    """A tracing layer must not be able to fail a study."""
    reference = film_reference("/nonexistent/definitely-not-here.png")
    assert "unreadable" in reference


def test_scrub_applies_the_pipelines_own_deidentifier():
    """The export boundary runs the same gate the `deid` node does.

    This is what covers `intake`, whose span is created before any
    de-identification has happened. Note what this does and does not
    promise: export inherits `deid.py`'s coverage exactly -- the same
    context-anchored forms, no more. It closes the ordering hole, it does
    not add a second, stronger detector.
    """
    scrubbed = scrub("Patient: John Smith, MRN: 12345678")

    assert "John Smith" not in scrubbed
    assert "12345678" not in scrubbed
    assert "REDACTED" in scrubbed


def test_scrub_handles_a_film_and_phi_in_the_same_string():
    text = f'Patient: John Smith. data:image/png;base64,{"A" * 100}'
    scrubbed = scrub(text)

    assert "John Smith" not in scrubbed
    assert "A" * 100 not in scrubbed


def test_scrub_attribute_reports_clean_values_as_unchanged():
    """Patches must stay sparse -- the SDK asks for only what changed."""
    assert _scrub_attribute("cardiomegaly probability 0.82") is _UNCHANGED
    assert _scrub_attribute(42) is _UNCHANGED
    assert _scrub_attribute(["findings", "impression"]) is _UNCHANGED


def test_scrub_attribute_masks_string_sequences():
    payload = "B" * 100
    masked = _scrub_attribute(["clean", f"data:image/png;base64,{payload}"])

    assert masked is not _UNCHANGED
    assert masked[0] == "clean"
    assert payload not in masked[1]


# --- the export hook -------------------------------------------------------


class _FakeSpan:
    def __init__(self, attributes: dict) -> None:
        self.attributes = attributes


class _FakeParams:
    def __init__(self, spans: dict) -> None:
        self.spans = spans


def test_mask_otel_spans_patches_only_the_spans_that_need_it():
    dirty = _FakeSpan({"input": "Patient: John Smith", "count": 3})
    clean = _FakeSpan({"input": "no findings", "count": 1})
    params = _FakeParams({"dirty": dirty, "clean": clean})

    result = _mask_otel_spans(params=params)

    assert "clean" not in result.span_patches
    patched = result.span_patches["dirty"].set_attributes
    assert "John Smith" not in patched["input"]
    assert "count" not in patched


def test_mask_otel_spans_returns_none_when_the_batch_is_clean():
    params = _FakeParams({"a": _FakeSpan({"input": "left lower lobe opacity"})})
    assert _mask_otel_spans(params=params) is None


def test_mask_otel_spans_fails_closed_on_a_raising_attribute(monkeypatch):
    """One bad attribute must not export raw, and must not drop the batch.

    Letting the exception escape is the worse failure of the two: the SDK
    discards the *entire* export batch when the mask function raises, so a
    single malformed attribute would take every other span with it.
    """

    def _boom(value):
        raise ValueError("scrub exploded")

    monkeypatch.setattr("medscope.obs._scrub_attribute", _boom)

    params = _FakeParams({"a": _FakeSpan({"input": "Patient: John Smith"})})
    result = _mask_otel_spans(params=params)

    patched = result.span_patches["a"].set_attributes
    assert "John Smith" not in patched["input"]
    assert "withheld" in patched["input"]


# --- token accounting ------------------------------------------------------


class _Usage:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_token_split_is_reported_when_the_vendor_gives_one():
    usage = _Usage(total_tokens=30, prompt_tokens=20, completion_tokens=10)
    assert _extract_tokens(usage) == 30
    assert _extract_token_split(usage) == (20, 10)


def test_token_split_stays_zero_when_only_a_total_is_reported():
    """0 means "unknown", not "free" -- so a total-only response must not
    be reported to Langfuse as costing nothing on either side."""
    usage = _Usage(total_tokens=30)
    assert _extract_tokens(usage) == 30
    assert _extract_token_split(usage) == (0, 0)


def test_usage_details_omits_unknown_counts():
    from medscope.obs import _usage_details

    assert _usage_details(ModelResponse(text="x")) is None
    assert _usage_details(ModelResponse(text="x", tokens=30)) == {"total": 30}
    assert _usage_details(
        ModelResponse(text="x", tokens=30, input_tokens=20, output_tokens=10)
    ) == {"input": 20, "output": 10, "total": 30}


# --- configuration ---------------------------------------------------------


def test_tracing_needs_both_keys():
    assert tracing_enabled(Settings(langfuse_public_key="pk")) is False
    assert tracing_enabled(Settings(langfuse_secret_key="sk")) is False
    assert tracing_enabled(Settings(langfuse_public_key="pk", langfuse_secret_key="sk")) is True


def test_environment_defaults_to_development():
    """An unconfigured run is somebody experimenting, not production --
    same default posture as `use_real_vlm`."""
    assert Settings().langfuse_environment == "development"
