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
def list_studies() -> dict:
    studies = workbench.list_sample_studies()
    return {
        "studies": [
            {
                "study_id": s.study_id,
                "indication": s.indication,
                "impression_text": s.impression_text,
            }
            for s in studies
        ]
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
        raise HTTPException(status_code=404, detail=f"no sample study {payload.study_id!r}")
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
        study = workbench.find_sample_study(study_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no sample study {study_id!r}")

    state = workbench.build_initial_state(study)
    deps = workbench.deps_for_study(study)
    return StreamingResponse(workbench.sse_events(state, deps), media_type="text/event-stream")
