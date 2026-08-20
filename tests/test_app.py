"""Liveness endpoint.

Deliberately thin in Phase 1: the workbench, the run endpoints and the SSE
stream all arrive in Phase 3. What matters now is that the app imports without
dragging torch in — the CNN reader loads its weights lazily, and a health check
that took fifteen seconds and half a gigabyte of RAM would defeat its purpose.
"""

import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from medscope.app import app

client = TestClient(app)


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["phase"]


def test_importing_the_app_does_not_load_torch():
    """Guards the lazy-load contract from Task 1.5.

    `readers.cnn` imports torch inside `_load_model()` precisely so that
    importing anything else stays cheap. If someone later hoists that import to
    module scope, this test is what catches it.

    Runs in a subprocess on purpose: checking `sys.modules` in-process would
    only pass because this file happens to sort before `test_cnn_reader.py`,
    and would break the moment tests are renamed or run out of order.
    """
    probe = (
        "import medscope.app, sys; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parent.parent,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
    )
    assert result.returncode == 0, "importing medscope.app pulled in torch"
