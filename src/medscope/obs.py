"""Langfuse tracing, off by default and harmless when unconfigured.

Three properties matter more than the tracing itself:

**The enabled check happens at call time, not import time.** Deciding once
when the module loads would bake in whatever configuration existed at
startup, so a test that constructs a differently-configured `Settings` would
still trace — or not — according to a decision made before it ran. The same
mistake in a sibling project let a real `.env` leak into a test run.

**Without keys nothing is constructed.** No client, no no-op spans, no
buffered events, no background flush thread. The pipeline must behave
identically whether or not anyone is watching, because a tracing layer that
subtly changes timing would corrupt the one assertion in this project that
depends on timing — the critical-alert-before-report ordering in the graph.

**Export goes through the same de-identification gate the pipeline does.**
`_mask_otel_spans` re-runs `deid.deid_text` over every string attribute
leaving the process, on top of stripping base64 film payloads. This is
belt-and-braces on purpose. The pipeline de-identifies at the `deid` node,
but `intake` runs *before* it — so the node span for `intake` carries the
raw `indication` and `history_text` a caller supplied. Whatever the graph
does internally, nothing reaches Langfuse without passing the gate.

Why the films never leave, mechanically: the Langfuse SDK extracts base64
data URIs from payloads and uploads them to its object storage, and that
extraction runs *before* `mask_otel_spans` gets a look. Masking at export
cannot un-upload a film. So the model calls are instrumented by hand here
(`observe_model_call`) with a `[MASKED …]` reference substituted for the
image *before* the SDK ever sees the request, rather than by importing
`langfuse.openai`, whose whole value is capturing the request verbatim.
That is a deliberate departure from "prefer the framework integration":
the integration is correct, it just captures the one thing we will not send.
"""

from __future__ import annotations

import functools
import hashlib
import mimetypes
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, TypeVar

from medscope.config import Settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from medscope.llm import ModelClient, ModelResponse
    from medscope.state import StudyState

F = TypeVar("F", bound=Callable[..., Any])

#: Matches the `data:<mime>;base64,<payload>` URIs `llm._encode_image_data_uri`
#: builds. Kept deliberately loose on the mime half: the point is to catch
#: anything shaped like an inlined binary, not to validate it.
_DATA_URI_RE = re.compile(r"data:(?P<mime>[\w.+-]+/[\w.+-]+);base64,(?P<payload>[A-Za-z0-9+/=]{32,})")


def tracing_enabled(settings: Settings | None = None) -> bool:
    return (settings or Settings()).tracing_enabled


# ---------------------------------------------------------------------------
# masking
# ---------------------------------------------------------------------------


def mask_data_uris(text: str) -> str:
    """Replace every inlined base64 payload with a bounded reference.

    The digest is over the base64 text rather than the decoded bytes purely
    so this stays a pure string function -- it is an identity check ("is
    this the same film as that one?"), not a content-addressable store.
    """

    def _replace(match: re.Match) -> str:
        payload = match.group("payload")
        digest = hashlib.sha256(payload.encode("ascii")).hexdigest()[:12]
        return f"[MASKED {match.group('mime')} · sha256:{digest} · {len(payload)} b64 chars]"

    return _DATA_URI_RE.sub(_replace, text)


def scrub(text: str) -> str:
    """Both gates, in order: drop pixel payloads, then de-identify.

    Order matters. `deid_text` normalizes and pattern-matches; running it
    over a multi-megabyte base64 blob first would be pointless work on data
    that is about to be discarded anyway.
    """
    from medscope.deid import deid_text

    scrubbed, _report = deid_text(mask_data_uris(text))
    return scrubbed


def _scrub_attribute(value: Any) -> Any:
    """Scrub one OpenTelemetry attribute value, or return it unchanged.

    Returns the sentinel `_UNCHANGED` when nothing needed rewriting, so the
    caller can keep patches sparse as the SDK asks.
    """
    if isinstance(value, str):
        masked = scrub(value)
        return masked if masked != value else _UNCHANGED

    if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
        masked_seq = [scrub(v) for v in value]
        return masked_seq if masked_seq != list(value) else _UNCHANGED

    return _UNCHANGED


_UNCHANGED = object()


def _mask_otel_spans(*, params: Any) -> Any:
    """Export-stage masking hook, wired into the client below.

    Fails closed per attribute: if scrubbing an attribute raises, that
    attribute is replaced with a marker rather than exported as-is. Letting
    the exception escape would be worse than useless -- the SDK drops the
    *whole batch* on a raising mask function, so one malformed attribute
    would silently take every other span in the batch with it.
    """
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    patches = {}
    for identifier, span in params.spans.items():
        replacements: dict[str, Any] = {}
        for key, value in span.attributes.items():
            try:
                masked = _scrub_attribute(value)
            except Exception:  # noqa: BLE001 - see docstring: fail closed
                replacements[key] = "[MASKING FAILED — withheld]"
                continue
            if masked is not _UNCHANGED:
                replacements[key] = masked
        if replacements:
            patches[identifier] = OtelSpanPatch(set_attributes=replacements)

    return MaskOtelSpansResult(span_patches=patches) if patches else None


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------

#: Keyed by credentials so a differently-configured `Settings` in a test
#: never gets handed the client built for another one.
_CLIENTS: dict[tuple[str, str, str, str], Any] = {}

_PROXY_VARS = ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy")

#: `graph.py`'s conditional-edge functions. LangGraph emits a span for each
#: one, and they are noise: a router is a branch, not a step, and its span
#: carries no input, no output and no time worth reading. They are leaves in
#: the tree, so dropping them orphans nothing.
_ROUTER_SPAN_PREFIX = "_route_after_"


def _should_export_span(span: Any) -> bool:
    return not (span.name or "").startswith(_ROUTER_SPAN_PREFIX)


@contextmanager
def _without_proxy_env() -> Iterator[None]:
    """Hide this machine's proxy variables while a client is constructed.

    `llm.OpenAICompatibleModelClient` solves the same problem with
    `httpx.Client(trust_env=False)`, and that is the first thing tried
    here -- but it only covers the *synchronous* transport. The Langfuse
    SDK also builds an internal `httpx.AsyncClient()` it accepts no
    override for, and that one reads the environment: with `ALL_PROXY` set
    to a SOCKS5 endpoint (as it is on this machine), constructing the
    client raises `ImportError: Using SOCKS proxy, but the 'socksio'
    package is not installed` before a single span is ever queued.

    httpx resolves proxies once, at client construction, so covering just
    the constructor is enough -- the variables are restored immediately and
    nothing else in the process sees them missing for longer than that.
    The narrow race this leaves (another thread building an HTTP client in
    the same window losing its proxy) is preferable to the alternatives:
    permanently mutating the environment, or requiring an extra dependency
    so that trace export can travel through a proxy it has no reason to use.
    """
    saved = {name: os.environ.pop(name) for name in _PROXY_VARS if name in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


def client(settings: Settings | None = None) -> Any | None:
    """The configured Langfuse client, or `None` when tracing is off.

    `None` rather than a no-op double: every caller here already has to
    handle "not tracing", and a double would mean constructing the SDK's
    background threads on a machine that never asked for them.

    `trust_env=False` on the transport is not incidental. This machine has
    `http_proxy` / `ALL_PROXY` set, and `ALL_PROXY` is SOCKS5 here -- a
    client that reads the environment routes span export through it, which
    additionally needs the optional `socksio` package to work at all. That
    is the same trap `llm.OpenAICompatibleModelClient` documents, and it
    fails the same silent way: exports queue up and never arrive.
    """
    settings = settings or Settings()
    if not settings.tracing_enabled:
        return None

    try:
        import httpx
        from langfuse import Langfuse
    except ImportError:
        # langfuse lives behind the `llm` extra. Its absence must never
        # break a run -- observability is not a dependency of correctness.
        return None

    key = (
        settings.langfuse_public_key,
        settings.langfuse_secret_key,
        settings.langfuse_host,
        settings.langfuse_environment,
    )
    if key not in _CLIENTS:
        with _without_proxy_env():
            _CLIENTS[key] = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                base_url=settings.langfuse_host or None,
                environment=settings.langfuse_environment or None,
                mask_otel_spans=_mask_otel_spans,
                should_export_span=_should_export_span,
                httpx_client=httpx.Client(trust_env=False, timeout=20.0),
            )
    return _CLIENTS[key]


def flush(settings: Settings | None = None) -> None:
    """Block until queued spans are sent. No-op when tracing is off."""
    active = client(settings)
    if active is not None:
        active.flush()


def traced(name: str) -> Callable[[F], F]:
    """Wrap a function in a Langfuse span when tracing is configured."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            active = client()
            if active is None:
                return func(*args, **kwargs)
            with active.start_as_current_observation(name=name, as_type="span"):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# model calls
# ---------------------------------------------------------------------------


def film_reference(image_path: str | Path) -> str:
    """A bounded stand-in for a film in a traced prompt.

    Carries what a reader of the trace actually needs -- which film, and
    whether two calls saw the same one -- without the pixels. An unreadable
    path yields a reference saying so rather than raising: a tracing layer
    must not be able to fail a study.
    """
    path = Path(image_path)
    mime, _ = mimetypes.guess_type(path.name)
    mime = mime or "image/png"
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    except OSError:
        digest = "unreadable"
    return f"[MASKED {mime} · sha256:{digest} · {path.name}]"


def _generation_input(prompt: str, image_path: str | Path, system: str | None) -> list[dict]:
    """The request as Langfuse renders it best: a role-labeled message list
    in OpenAI shape, with the film replaced by its reference.

    Deliberately mirrors `llm._build_messages` in structure rather than
    dumping the function's arguments, so what a reviewer reads in the UI is
    recognizably the conversation the model had.
    """
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": film_reference(image_path)}},
            ],
        }
    )
    return messages


def _usage_details(response: ModelResponse) -> dict[str, int] | None:
    """Token counts in Langfuse's shape, or `None` when the vendor gave us
    nothing. Reporting zeros would be a lie Langfuse then prices at $0.00 --
    `ModelResponse` is explicit that 0 means "unknown", not "free".
    """
    usage = {}
    if response.input_tokens:
        usage["input"] = response.input_tokens
    if response.output_tokens:
        usage["output"] = response.output_tokens
    if response.tokens:
        usage["total"] = response.tokens
    return usage or None


def observe_model_call(
    name: str,
    model_client: ModelClient,
    prompt: str,
    image_path: str | Path,
    *,
    system: str | None = None,
    metadata: dict | None = None,
) -> ModelResponse:
    """Call `model_client.chat_with_image`, recorded as a `generation`.

    The single seam every model call in this project goes through, which is
    why instrumenting here covers reader_b, the arbiter and the report
    writer without any of them knowing Langfuse exists. When tracing is off
    this is exactly the call it wraps, with no object constructed.

    `name` is the caller's role (`read-film-vlm`, `arbitrate-disagreement`,
    `write-report`), not the model's -- names are what dashboards and
    evaluators target, so they must survive swapping the model out.
    """
    active = client()
    if active is None:
        return model_client.chat_with_image(prompt, image_path, system=system)

    # Offline stand-ins have a `name` but no `model`; recording which
    # stand-in produced a reading matters more than the field being blank.
    model = getattr(model_client, "model", "") or getattr(model_client, "name", "unknown")

    with active.start_as_current_observation(
        name=name,
        as_type="generation",
        model=model,
        input=_generation_input(prompt, image_path, system),
        metadata={"image_path": str(image_path), **(metadata or {})},
    ) as generation:
        response = model_client.chat_with_image(prompt, image_path, system=system)
        generation.update(output=response.text, usage_details=_usage_details(response))
        return response


@contextmanager
def observe_retrieval(name: str, query: str, *, metadata: dict | None = None) -> Iterator[Any]:
    """A `retriever` observation around a guideline lookup.

    Typed `retriever` rather than a plain span so the retrieval step is
    filterable on its own -- "did the arbiter have the passage it needed?"
    is a different question from "what did the arbiter decide", and the
    answer to the first is the usual explanation for a bad second.
    """
    active = client()
    if active is None:
        yield None
        return
    with active.start_as_current_observation(
        name=name, as_type="retriever", input=query, metadata=metadata
    ) as span:
        yield span


# ---------------------------------------------------------------------------
# study trace
# ---------------------------------------------------------------------------


def _langgraph_callbacks() -> list:
    """LangGraph's Langfuse callback, or nothing if it can't be imported.

    Per-node spans come from this handler rather than from decorating the
    graph's nodes, which is what lets `graph.py` stay ignorant of Langfuse
    entirely. It needs the full `langchain` package (not just the
    `langchain-core` LangGraph pulls in), so it ships in the `llm` extra.

    Degrades instead of raising: without the handler you still get the root
    span, the generations, the retrievals and the scores -- you lose the
    node-by-node breakdown. Losing detail is an acceptable outcome for an
    observability layer. Taking down a study is not.
    """
    try:
        from langfuse.langchain import CallbackHandler
    except (ImportError, ModuleNotFoundError):
        return []
    return [CallbackHandler()]


class _NoTrace:
    """What every entry point gets when tracing is off: no callbacks to
    hand LangGraph, and a `finish` that does nothing."""

    callbacks: list = []

    def finish(self, result: StudyState) -> None:
        return None


class _StudyTrace:
    """The live counterpart of `_NoTrace`, holding the root observation."""

    def __init__(self, active: Any, root: Any, callbacks: list) -> None:
        self._client = active
        self._root = root
        self.callbacks = callbacks

    def finish(self, result: StudyState) -> None:
        """Close the trace out with the study's verdict and its gate scores.

        Scores rather than tags for all of it: tags are fixed at creation
        time, and every number here is only known once the study has run.
        """
        self._root.update(
            output={
                "status": result.status,
                "findings": len(result.findings),
                "needs_human": sum(1 for f in result.findings if f.needs_human),
                "critical_alerts": [a.label for a in result.alerts],
                "report_sections": sorted({s.section for s in result.report.sentences})
                if result.report
                else [],
                "notes": result.notes,
            }
        )

        self._root.score_trace(name="terminal-status", value=result.status, data_type="CATEGORICAL")
        numeric = {
            "critical-alerts": float(len(result.alerts)),
            "arbiter-calls": float(result.budget_spent),
            "findings-needing-human": float(sum(1 for f in result.findings if f.needs_human)),
            "disagreements": float(len(result.disagreements)),
        }
        # Both agreement numbers, or neither: a run that never reached
        # `merge` (QC failed, intake blocked) has no kappa, and scoring it
        # 0.0 would read as total disagreement rather than "not measured".
        if result.kappa is not None:
            numeric["kappa"] = float(result.kappa)
        if result.kappa_all_labels is not None:
            numeric["kappa-all-labels"] = float(result.kappa_all_labels)
        # Summed from the reads rather than read off `StudyState.tokens_used`.
        # That field is declared and displayed (workbench's cost panel) but no
        # node ever writes to it, so scoring it would report 0 forever on a
        # study that really did spend tokens -- `ReadResult.tokens` is where
        # the counts actually land.
        tokens = sum(read.tokens for read in (result.read_a, result.read_b) if read)
        if tokens:
            numeric["tokens-used"] = float(tokens)

        for score_name, value in numeric.items():
            self._root.score_trace(name=score_name, value=value, data_type="NUMERIC")


@contextmanager
def study_trace(state: StudyState, settings: Settings | None = None) -> Iterator[Any]:
    """One trace per study run — the unit of work this system reasons in.

    Yields something with `.callbacks` (hand it to LangGraph's `config`, so
    every node becomes a span without the graph importing Langfuse) and
    `.finish(result)` (call it with the finished `StudyState`).

    `session_id` is the study id: a study gets re-read after a NEEDS_REPEAT,
    and those runs belong together in the session view. There is no
    `user_id` -- this system has no authenticated users, and inventing one
    would only add a column that is constant.

    Flushes on exit. The workbench is a server and per-request flushing is
    normally wasteful, but a study run costs seconds of model latency, which
    makes the round trip free by comparison and means no entry point can
    lose a trace by forgetting to flush.
    """
    active = client(settings)
    if active is None:
        yield _NoTrace()
        return

    from langfuse import propagate_attributes

    settings = settings or Settings()
    with active.start_as_current_observation(
        name="read-study",
        as_type="chain",
        input={
            "study_id": state.study_id,
            "indication": state.indication,
            "history_text": state.history_text,
            "films": len(state.image_paths) or 1,
        },
    ) as root:
        with propagate_attributes(
            trace_name="read-study",
            session_id=state.study_id,
            tags=["chest-xray", f"reader-b-{settings.reader_b_mode}"],
            metadata={
                "reader_b_mode": settings.reader_b_mode,
                "cnn_prob_threshold": settings.cnn_prob_threshold,
                "critical_threshold": settings.critical_threshold,
                "magnitude_gap": settings.magnitude_gap,
                "max_llm_judgments": settings.max_llm_judgments,
                "use_real_vlm": settings.use_real_vlm,
                "vlm_model": settings.vlm_model,
            },
        ):
            try:
                yield _StudyTrace(active, root, _langgraph_callbacks())
            finally:
                active.flush()
