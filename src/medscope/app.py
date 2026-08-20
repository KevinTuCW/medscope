"""FastAPI surface.

Phase 1 exposes only a liveness probe. The workbench page, the run endpoints
and the SSE stream land in Phase 3; this module stays deliberately free of
model imports so that starting the service — or running the test suite — never
pays for torch.
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(
    title="medscope",
    description=(
        "Chest X-ray reading co-pilot — educational demo, NOT a medical device. "
        "Produces report drafts for physician review; it does not diagnose."
    ),
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "phase": "1"}
