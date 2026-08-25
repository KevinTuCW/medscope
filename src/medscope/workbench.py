"""The radiologist-facing workbench -- turns a `StudyState` into the view
that makes the double-reading architecture legible, and streams a run as it
happens.

`build_dashboard` is the one function everything else in this module (and
`app.py`'s `/workbench/*` endpoints) is built around. It assembles five
blocks from a completed (or halted) `StudyState`:

- `image` -- the image path plus each reader's localization. reader_a
  (the CNN) contributes Grad-CAM loci, already normalized to 0-1 `cx`/`cy`/
  `r` by `readers.cnn`; reader_b (the VLM) never localizes, so its list is
  always empty -- that asymmetry is itself part of what the workbench is
  supposed to make visible, not a gap to paper over.
- `dual_read` -- the reader-versus-reader table, one row per label, built
  directly from `state.read_a`/`state.read_b` (the raw, pre-merge reads)
  rather than `state.findings` (the merged, post-arbiter set): the whole
  point of this view is to show what each reader said independently,
  agreement and disagreement side by side, which the merged view has
  already collapsed away.
- `report` -- the draft sentence by sentence, plus a map from every
  finding's `evidence_id` to the finding it names (label/probability/
  source/locus). Resolving that server-side, instead of shipping raw ids
  and making the browser re-join them against `findings`, is what lets a
  sentence-hover highlight the right image region with no duplicated
  join logic in JavaScript.
- `critical` -- alerts plus a timeline built from `state.trace_events`,
  sorted by timestamp. The timeline is the evidence that the critical
  bypass in `graph.py` actually ran before the (possibly slow) report,
  not an assertion about it.
- `audit` -- the de-identification report, QC result, token/latency
  figures, and the full per-node trace -- what makes a run inspectable
  after the fact.

Plus a top-level `status` and `blocked` (`status != "DRAFT_READY"`), so the
front end can grey out a draft that was held, escalated, or never
completed without re-deriving that logic itself.

`sse_events` streams one run of the graph: `start`, one `node` event per
pipeline stage as LangGraph's own `stream(stream_mode=["updates","values"])`
reports it completing, and a `complete` event carrying the full dashboard
built from the final state -- not a bare "done", so a consumer never has to
re-fetch to get the result it just watched being produced.

Runs are persisted through a `medscope.store.RunStore` (Task 3.3), replacing
what used to be `_RESULTS`, a process-local dict explicitly marked demo-only.
`_store()` builds it lazily on first use (not at import time) so
`Settings()` is only ever read after `conftest.py`'s environment-isolation
fixture has had a chance to run -- reading it at module-import time would
risk resolving against a real `.env` when this module gets pulled in
transitively at collection time (see `app.py` -> `workbench` -> here) before
any per-test fixture executes. `cached_result(study_id)` preserves the old
by-study-id lookup the endpoints depend on even though `RunStore.get` is
keyed by an opaque run id: it walks `RunStore.list` (newest-first) for the
most recent run of that study, matching the "last StudyState produced for
this study" semantics `_RESULTS` used to give for free.
"""

from __future__ import annotations

import base64
import io
import json
import mimetypes
from pathlib import Path

from medscope.bootstrap import build_runtime_deps, build_sample_deps
from medscope.config import Settings
from medscope.data.dicom import is_dicom_path, read_film
from medscope.data.openi import Study, load_studies
from medscope.graph import GraphDeps, build_graph
from medscope.ontology import canonical
from medscope.obs import study_trace
from medscope.runner import run_study
from medscope.state import StudyState
from medscope.store import RunStore, build_run_store
from medscope.views import primary_view

SAMPLES_DIR = Path("data/samples/studies")

# Generous "effectively all recent runs" bound for the by-study-id lookup in
# cached_result -- RunStore.list has no "no limit" sentinel, so this stands
# in for one; a demo/audit-trail-scale store won't come close to it.
_LOOKBACK_LIMIT = 10_000

_store_instance: RunStore | None = None


def _store() -> RunStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = build_run_store(Settings())
    return _store_instance


# ---------------------------------------------------------------------------
# Corpus wiring
# ---------------------------------------------------------------------------

#: The three studies committed to the repo. They are *not* a separate
#: corpus: all three also exist in the fetched OpenI dataset, there with two
#: views each instead of the one that fits in a git repo. So they are
#: floated to the top of an unfiltered listing rather than concatenated onto
#: it -- listing both copies would offer the same study twice, differing
#: only in how much of it reader_a gets to read.
PINNED_STUDY_IDS = ("38", "797", "1187")

#: Page size for the study listing. The fetched corpus is ~3.8k studies;
#: shipping all of them to a `<select>` on every page load is a lot of
#: payload for a list nobody reads past the top of.
DEFAULT_STUDY_LIMIT = 50

_corpus_cache: dict[str, list[Study]] = {}


def corpus_root(settings: Settings | None = None) -> Path:
    """Where studies are read from.

    The fetched OpenI dataset when it has been downloaded, the committed
    sample slice otherwise. Probing for the `ecgen-radiology/` subdirectory
    rather than calling `load_studies` keeps this cheap enough to call on
    every request: a full load of the real corpus takes seconds, and the
    answer to "which root" must not.

    The fallback is what preserves the project's offline-first property --
    a fresh clone with no `data/openi/` still gets a working workbench,
    with the three committed studies in it.
    """
    settings = settings or Settings()
    root = Path(settings.openi_root)
    if (root / "ecgen-radiology").is_dir():
        return root
    return SAMPLES_DIR


def load_corpus(settings: Settings | None = None) -> list[Study]:
    """Every loadable study under `corpus_root`, cached per root.

    Caching is not an optimization here so much as a precondition: loading
    the full OpenI corpus walks ~7.5k image files and parses ~3.9k report
    XMLs, which measured ~5.6s on the development machine. Paying that on
    every keystroke of the search box would make the feature unusable.

    Keyed by root so a differently-configured `Settings` (a test pointing
    at a fixture directory) can never be served the corpus loaded for
    another one.
    """
    root = corpus_root(settings)
    key = str(root)
    if key not in _corpus_cache:
        _corpus_cache[key] = [s for s in load_studies(root) if s.image_paths]
    return _corpus_cache[key]


def reset_corpus_cache() -> None:
    """Drop the cached corpora. For tests, and for a process that has just
    had a dataset downloaded underneath it."""
    _corpus_cache.clear()


def _matches(study: Study, needle: str) -> bool:
    """Match on study id, referral question, or MeSH terms.

    MeSH earns its place: the indication is what the *referrer* wrote
    ("chest pain"), while the MeSH terms are what the study turned out to
    show ("Pneumothorax"). Someone hunting for a case to demonstrate a
    critical finding is searching for the latter, and would find almost
    nothing searching only indications.
    """
    if needle in study.study_id.lower():
        return True
    if needle in (study.indication or "").lower():
        return True
    return any(needle in term.lower() for term in study.mesh)


def search_studies(
    q: str = "", limit: int = DEFAULT_STUDY_LIMIT, settings: Settings | None = None
) -> tuple[list[Study], int]:
    """One page of matching studies, plus how many matched in total.

    The total is returned separately rather than left for the caller to
    infer from the page length: "50 studies" and "50 of 812 studies" are
    different things to show someone, and a truncated page cannot tell
    them apart on its own.
    """
    studies = load_corpus(settings)
    needle = (q or "").strip().lower()

    if needle:
        matched = [s for s in studies if _matches(s, needle)]
    else:
        pinned = [s for s in studies if s.study_id in PINNED_STUDY_IDS]
        rest = [s for s in studies if s.study_id not in PINNED_STUDY_IDS]
        matched = pinned + rest

    total = len(matched)
    page = matched[:limit] if limit and limit > 0 else matched
    return page, total


def list_sample_studies(settings: Settings | None = None) -> list[Study]:
    """Every loadable study in the active corpus.

    Retained under its original name because it is what `eval` and the
    tests reach for when they mean "the studies this machine can run".
    """
    return load_corpus(settings)


def find_study(study_id: str, settings: Settings | None = None) -> Study:
    for study in load_corpus(settings):
        if study.study_id == study_id:
            return study
    raise KeyError(f"no study with id {study_id!r} under {corpus_root(settings)}")


#: Kept as the old name so nothing that imported it breaks.
find_sample_study = find_study


def build_initial_state(study: Study) -> StudyState:
    """Seed the pipeline with the whole study, not one file of it.

    `image_paths[0]` used to decide which film got read; it is filesystem
    order wearing a subscript. `views.primary_view` picks the film to
    display (and to hand the single-image consumers) from the pixels.
    """
    return StudyState(
        study_id=study.study_id,
        image_path=str(primary_view(study.image_paths)),
        image_paths=[str(p) for p in study.image_paths],
        indication=study.indication,
    )


def deps_for_study(study: Study, settings: Settings | None = None) -> GraphDeps:
    """Live clients when `use_real_vlm` is set, offline stand-ins otherwise.

    This used to be hardwired to `build_sample_deps`, which meant the
    workbench read every study with `OfflineVLMClient` even on a machine
    configured for a real VLM. That was defensible while the workbench
    could only run three committed studies; it stops being defensible once
    it can run the whole corpus, because the stand-in derives reader_b's
    findings from *the study's own ground-truth report*. Point that at
    3.8k studies and the dual-read panel shows agreement everywhere -- an
    artifact of reader_b having read the answer key, displayed in the exact
    panel built to show whether two readers independently agree.

    `build_runtime_deps` raises rather than quietly degrading when
    `use_real_vlm` is set without credentials; that refusal is the point,
    so it is left to propagate (see `bootstrap`'s module docstring).

    OfflineVLMClient is seeded per study, so the offline branch has to
    build deps per study rather than sharing one set.
    """
    settings = settings or Settings()
    if settings.use_real_vlm:
        return build_runtime_deps(settings)
    return build_sample_deps(settings, impression_text=study.impression_text)


def run_sample_study(study_id: str) -> StudyState:
    """Run one study from the active corpus end to end, persist it, and
    return the result."""
    study = find_study(study_id)
    state = build_initial_state(study)
    result = run_study(state, deps_for_study(study))
    _store().save(result)
    return result


def cached_result(study_id: str) -> StudyState | None:
    """The most recently produced `StudyState` for `study_id`, or `None` if
    it has never been run. See module docstring for why this walks
    `RunStore.list` rather than `RunStore.get` (keyed by run id, not
    study id).
    """
    for summary in _store().list(limit=_LOOKBACK_LIMIT):
        if summary["study_id"] == study_id:
            return _store().get(summary["run_id"])
    return None


def list_runs(limit: int = 20) -> list[dict]:
    """Newest-first run summaries for the `GET /runs` endpoint."""
    return _store().list(limit=limit)


# ---------------------------------------------------------------------------
# build_dashboard
# ---------------------------------------------------------------------------


def _image_data_url(image_path: str) -> str | None:
    """Inline the image as a base64 data URI so the workbench page can
    render it with nothing more than the documented endpoints -- no extra
    static-file route to serve `data/samples/studies/images/...` from, and
    it rides along for free inside the SSE `complete` event too. Returns
    None (rather than raising) when the path doesn't resolve, e.g. a study
    that never got past QC on a file the caller has since cleaned up.
    """
    path = Path(image_path)
    if not image_path or not path.is_file():
        return None

    # A DICOM film cannot be handed to an <img> tag: no browser decodes
    # it, and its raw bytes still carry the identifying tags. Render it
    # through the same de-identifying reader the pipeline used (so the
    # radiologist sees the pixels reader_a actually saw, MONOCHROME1
    # inversion included) and ship a PNG.
    if is_dicom_path(path):
        buffer = io.BytesIO()
        read_film(path).image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    mime, _ = mimetypes.guess_type(str(path))
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime or 'application/octet-stream'};base64,{encoded}"


def _image_block(state: StudyState) -> dict:
    """The displayed film, plus only the loci that belong on it.

    reader_a now reads every film of a study, so a Grad-CAM may have been
    computed on a film that is not the one on screen. Drawing it anyway
    would put a confident box over anatomy it was never computed from --
    the same audit-view failure the arbitration records exist to avoid: a
    view that misreports is worse than one that reports less. A locus is
    rendered only when its `image_ref` is the displayed film, or when the
    finding carries no attribution at all (single-image readers). The rest
    are counted, so "fewer boxes" never reads as "nothing found".
    """
    reader_a_loci = []
    loci_on_other_views = 0
    if state.read_a is not None:
        for finding in state.read_a.findings:
            if finding.locus is None:
                continue
            if finding.image_ref and finding.image_ref != state.image_path:
                loci_on_other_views += 1
                continue
            reader_a_loci.append(
                {
                    "label": finding.label,
                    "prob": finding.prob,
                    "cx": finding.locus.get("cx"),
                    "cy": finding.locus.get("cy"),
                    "r": finding.locus.get("r"),
                }
            )
    return {
        "path": state.image_path,
        "data_url": _image_data_url(state.image_path),
        "views": list(state.image_paths),
        "loci_on_other_views": loci_on_other_views,
        # reader_b (the VLM) never localizes -- see module docstring.
        "readers": {"a": reader_a_loci, "b": []},
    }


def _dual_read_rows(state: StudyState) -> list[dict]:
    a_mapped: dict[str, object] = {}
    if state.read_a is not None:
        for finding in state.read_a.findings:
            canon = canonical(finding.label)
            if canon is not None:
                a_mapped[canon] = finding

    b_mapped: dict[str, object] = {}
    if state.read_b is not None:
        for finding in state.read_b.findings:
            canon = canonical(finding.label)
            if canon is not None:
                b_mapped[canon] = finding

    disagreement_by_label = {d.label: d for d in state.disagreements}
    # Verdicts are read from `state.arbitration_records`, which the arbiter
    # produced and the graph stores verbatim. An earlier version inferred
    # them instead -- an arbiter-sourced Finding present with
    # needs_human=False meant CONFIRM, absent meant REJECT. That matched
    # `arbiter.py` exactly at the time, but coupled this audit view to that
    # module's internals: if REJECT ever stopped meaning "adds nothing to
    # the final set", the inference would go on reporting confidently and
    # wrongly. An audit view that misreports is worse than one that admits
    # it doesn't know.
    verdict_by_label = {r["label"]: r["verdict"] for r in state.arbitration_records}

    labels = sorted(set(a_mapped) | set(b_mapped) | set(disagreement_by_label))
    rows = []
    for label in labels:
        disagreement = disagreement_by_label.get(label)
        if disagreement is not None:
            verdict = verdict_by_label.get(label)
            rows.append(
                {
                    "label": label,
                    "a_prob": disagreement.a_prob,
                    "b_prob": disagreement.b_prob,
                    "agreed": False,
                    "kind": disagreement.kind,
                    "verdict": verdict,
                }
            )
        else:
            fa = a_mapped.get(label)
            fb = b_mapped.get(label)
            rows.append(
                {
                    "label": label,
                    "a_prob": fa.prob if fa is not None else None,
                    "b_prob": fb.prob if fb is not None else None,
                    "agreed": True,
                    "kind": None,
                    "verdict": None,
                }
            )
    return rows


def _dual_read_block(state: StudyState) -> dict:
    return {"kappa": state.kappa, "rows": _dual_read_rows(state)}


def _report_block(state: StudyState) -> dict:
    draft = state.report
    evidence_map = {
        f.evidence_id: {
            "label": f.label,
            "prob": f.prob,
            "source": f.source,
            "locus": f.locus,
        }
        for f in state.findings
    }
    if draft is None:
        return {"sentences": [], "evidence_map": {}, "disclaimer": "", "removed_count": 0}

    sentences = [
        {"text": s.text, "section": s.section, "evidence_ids": list(s.evidence_ids)}
        for s in draft.sentences
    ]
    return {
        "sentences": sentences,
        "evidence_map": evidence_map,
        "disclaimer": draft.disclaimer,
        "removed_count": draft.removed_count,
    }


def _critical_block(state: StudyState) -> dict:
    alerts = [
        {
            "label": a.label,
            "prob": a.prob,
            "source": a.source,
            "detected_at": a.detected_at.isoformat(),
            "image_ref": a.image_ref,
        }
        for a in state.alerts
    ]
    timeline = sorted(state.trace_events, key=lambda e: e["ts"])
    return {"alerts": alerts, "timeline": timeline}


def _audit_block(state: StudyState) -> dict:
    trace = sorted(state.trace_events, key=lambda e: e["ts"])
    latency_ms = None
    if len(trace) >= 2:
        latency_ms = (trace[-1]["ts"] - trace[0]["ts"]) * 1000
    return {
        "deid": state.deid_report,
        "qc": state.qc,
        "tokens": {
            "reader_a": state.read_a.tokens if state.read_a else 0,
            "reader_b": state.read_b.tokens if state.read_b else 0,
            "total": state.tokens_used,
        },
        "latency_ms": {
            "reader_a": state.read_a.latency_ms if state.read_a else None,
            "reader_b": state.read_b.latency_ms if state.read_b else None,
            "total": latency_ms,
        },
        "budget_spent": state.budget_spent,
        "trace": trace,
        "notes": list(state.notes),
    }


def build_dashboard(state: StudyState) -> dict:
    """Assemble the workbench's five blocks plus top-level status/blocked
    from a `StudyState`. See the module docstring for what each block
    contains and why.
    """
    return {
        "status": state.status,
        # A held or escalated study must never look like a finished
        # report -- see guardrails.output.enforce_output's docstring on
        # why this gate only ever downgrades, never promotes.
        "blocked": state.status != "DRAFT_READY",
        "image": _image_block(state),
        "dual_read": _dual_read_block(state),
        "report": _report_block(state),
        "critical": _critical_block(state),
        "audit": _audit_block(state),
    }


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_events(state: StudyState, deps: GraphDeps):
    """Run `state` through the graph, yielding SSE-formatted text chunks:
    `start`, one `node` event per pipeline stage as it completes, and
    `complete` carrying the full dashboard for the final state.

    Runs the graph directly (not through `runner.run_study`) via
    `stream(stream_mode=["updates", "values"])`: `"updates"` chunks name
    which node just ran (what drives the `node` events), and `"values"`
    chunks carry the full accumulated state after each superstep, so the
    last one observed is the final state -- without hand-reimplementing
    LangGraph's own reducer logic for `notes`/`trace_events`.

    Because it bypasses `runner.run_study`, it also has to open the study
    trace itself. Leaving that out would make the streamed path the one
    entry point that silently produces no trace -- and it is the path a
    person actually watches, so it is the last one that should be dark.
    """
    yield _sse("start", {"study_id": state.study_id})

    graph = build_graph(deps)
    last_values: dict | None = None
    with study_trace(state, deps.settings) as trace:
        for mode, chunk in graph.stream(
            state.model_dump(),
            stream_mode=["updates", "values"],
            config={"callbacks": trace.callbacks},
        ):
            if mode == "updates":
                for node_name in chunk:
                    yield _sse("node", {"node": node_name})
            else:
                last_values = chunk

        final_state = StudyState.model_validate(
            last_values if last_values is not None else state.model_dump()
        )
        trace.finish(final_state)

    _store().save(final_state)

    yield _sse("complete", build_dashboard(final_state))
