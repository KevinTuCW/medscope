"""FastAPI surface.

Phase 1 exposed only a liveness probe. Phase 3 adds the workbench: the page
itself, the run/dashboard/stream endpoints, a listing of the sample studies
available to run, and `/runs` (Task 3.3) listing recent persisted runs --
all built on `medscope.workbench`, which is where the actual
dashboard-assembly, streaming, and persistence logic lives. This module
stays thin on purpose: it wires routes to that module's functions and
translates their results into HTTP responses, nothing more.

Importing this module still must never load torch (see
`test_app.py::test_importing_the_app_does_not_load_torch`) -- `workbench`
imports `bootstrap`, which imports `readers.cnn.CNNReader`, but that class
only touches torch lazily inside `_load_model` (see that module's
docstring), so constructing the FastAPI app and its routes stays cheap even
though a `/workbench/run` call later on is not.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from medscope import workbench

app = FastAPI(
    title="medscope",
    description=(
        "Chest X-ray reading co-pilot — educational demo, NOT a medical device. "
        "Produces report drafts for physician review; it does not diagnose."
    ),
)

_STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "web" / "static"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "phase": "1"}


@app.get("/studies")
def list_studies(q: str = "", limit: int = workbench.DEFAULT_STUDY_LIMIT) -> dict:
    """One page of the active corpus, filtered by `q`.

    `total` travels with the page because the fetched OpenI corpus runs to
    ~3.8k studies: without it the front end cannot tell "these are all the
    matches" from "these are the first 50 of many".
    """
    studies, total = workbench.search_studies(q=q, limit=limit)
    return {
        "query": q,
        "total": total,
        "limit": limit,
        "corpus": str(workbench.corpus_root()),
        "studies": [
            {
                "study_id": s.study_id,
                "indication": s.indication,
                "impression_text": s.impression_text,
                "mesh": s.mesh,
                "views": len(s.image_paths),
            }
            for s in studies
        ],
    }


@app.get("/workbench", response_class=HTMLResponse)
def workbench_page() -> HTMLResponse:
    html = (_STATIC_DIR / "workbench.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


class RunRequest(BaseModel):
    study_id: str


@app.post("/workbench/run")
def workbench_run(payload: RunRequest) -> dict:
    try:
        result = workbench.run_sample_study(payload.study_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no study {payload.study_id!r}")
    except RuntimeError as exc:
        # `build_runtime_deps` refuses to run half-configured rather than
        # falling back to the offline stand-ins. Surfacing that as a 503
        # with its own message beats a bare traceback: the fix is always a
        # missing credential, and the message already says which.
        raise HTTPException(status_code=503, detail=str(exc))
    return workbench.build_dashboard(result)


@app.get("/workbench/dashboard")
def workbench_dashboard(study_id: str) -> dict:
    result = workbench.cached_result(study_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"study {study_id!r} has not been run yet")
    return workbench.build_dashboard(result)


@app.get("/runs")
def list_runs(limit: int = 20) -> dict:
    return {"runs": workbench.list_runs(limit=limit)}


@app.get("/workbench/stream")
def workbench_stream(study_id: str) -> StreamingResponse:
    try:
        study = workbench.find_study(study_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no study {study_id!r}")

    state = workbench.build_initial_state(study)
    try:
        deps = workbench.deps_for_study(study)
    except RuntimeError as exc:
        # Built before the stream opens, so a misconfigured VLM fails as an
        # ordinary HTTP error rather than as an exception thrown midway
        # through an already-200 event stream, where the browser would only
        # see the connection stop.
        raise HTTPException(status_code=503, detail=str(exc))
    return StreamingResponse(workbench.sse_events(state, deps), media_type="text/event-stream")
