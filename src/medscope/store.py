"""Run persistence -- Task 3.3.

`workbench._RESULTS` (Phase 3's workbench module) was a process-local dict
explicitly marked demo-only: fine for a single-user offline demo, gone the
moment the process restarts, and not something a real deployment should
extend. This module replaces it with a proper `RunStore`: an
`InMemoryRunStore` that behaves the same as the dict it replaces (the
default, so nothing changes for a fresh clone), and a `SqliteRunStore` for
when a run actually needs to survive a restart.

**What gets persisted is an audit record, not a mirror of `StudyState`.**
`_record_payload` below is an explicit allowlist of fields, not
`state.model_dump()` -- two fields are deliberately left out entirely:

- `indication` / `history_text`. These carry the clinician-written intake
  text, and by the time a *completed* run reaches `report_writer` they
  have already passed through `deid.deid_text` (see `graph._deid`, which
  overwrites them in place with the scrubbed versions). But not every
  terminal `StudyState` completed the graph: a study with
  `status == "GUARDRAIL_BLOCKED"` was stopped at `_route_after_intake`,
  *before* the `deid` node ever ran (see `graph.py`'s pipeline diagram) --
  its `indication`/`history_text` are still exactly the raw text that
  arrived. There is no field on `StudyState` that says "deid actually ran
  on this instance," so there is no safe way to persist these two fields
  only *sometimes*. Per the task brief: if a field can't be stored safely
  in every case that reaches this module, it doesn't get stored at all.
- `read_a` / `read_b`. The raw, pre-merge per-reader output -- including
  reader_b's free-text `notes`/`descriptions` in describer mode, which is
  generative-model output that never passes through `deid_text` the way
  `indication`/`history_text` do. The merged `findings` plus
  `arbitration_records` already carry what an audit trail needs from this
  stage (what each reader called, and how any disagreement was resolved),
  at materially lower risk and less bulk than the raw reads.

`deid_report` itself IS persisted -- it never carries raw text in the
first place (`deid.deid_text` returns pattern names and sha256 hashes,
never the matched strings; see that module's docstring), so it is exactly
the kind of "safe summary of what happened" this audit record exists to
keep.

`InMemoryRunStore` and `SqliteRunStore` share `_record_payload` /
`_summary`, specifically so the redaction logic can't drift between the
two backends -- `tests/test_store.py` parametrizes its whole contract
(including the privacy tests) over both for the same reason.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Protocol, runtime_checkable

from medscope.config import Settings
from medscope.state import StudyState

#: Explicit allowlist of what a run's audit record carries. See module
#: docstring for what is deliberately NOT here (`indication`/`history_text`,
#: `read_a`/`read_b`) and why.
_PERSISTED_FIELDS: tuple[str, ...] = (
    "study_id",
    "image_path",
    "deid_report",
    "qc",
    "findings",
    "disagreements",
    "kappa",
    "arbitration_records",
    "alerts",
    "report",
    "status",
    "notes",
    "trace_events",
    "budget_spent",
    "tokens_used",
)


def _record_payload(state: StudyState) -> dict:
    full = state.model_dump(mode="json")
    return {field: full[field] for field in _PERSISTED_FIELDS}


def _summary(run_id: str, study_id: str, status: str, created_at: float, alert_count: int) -> dict:
    return {
        "run_id": run_id,
        "study_id": study_id,
        "status": status,
        "created_at": created_at,
        "alert_count": alert_count,
    }


@runtime_checkable
class RunStore(Protocol):
    """Contract both backends satisfy -- see `tests/test_store.py`'s
    parametrized `store` fixture, which grades both against it directly."""

    def save(self, state: StudyState) -> str:
        """Persist `state`'s audit record; return a newly generated run id."""
        ...

    def get(self, run_id: str) -> StudyState | None:
        """Reconstruct the `StudyState` saved under `run_id`, or `None`."""
        ...

    def list(self, limit: int = 20) -> list[dict]:
        """Newest-first run summaries (run id, study id, status, timestamp,
        alert count), at most `limit` of them."""
        ...


class InMemoryRunStore:
    """The default -- process-local, lost on restart, same durability
    `workbench._RESULTS` had before this task. A single-user offline demo
    doesn't need more than this; `SqliteRunStore` is for when it does.
    """

    def __init__(self) -> None:
        self._records: dict[str, dict] = {}
        self._order: list[str] = []  # insertion order, oldest first

    def save(self, state: StudyState) -> str:
        run_id = uuid.uuid4().hex
        self._records[run_id] = {
            "study_id": state.study_id,
            "status": state.status,
            "created_at": time.time(),
            "alert_count": len(state.alerts),
            "payload": _record_payload(state),
        }
        self._order.append(run_id)
        return run_id

    def get(self, run_id: str) -> StudyState | None:
        record = self._records.get(run_id)
        if record is None:
            return None
        return StudyState.model_validate(record["payload"])

    def list(self, limit: int = 20) -> list[dict]:
        newest_first = list(reversed(self._order))[:limit]
        return [
            _summary(
                run_id,
                self._records[run_id]["study_id"],
                self._records[run_id]["status"],
                self._records[run_id]["created_at"],
                self._records[run_id]["alert_count"],
            )
            for run_id in newest_first
        ]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    study_id TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    alert_count INTEGER NOT NULL,
    payload TEXT NOT NULL
)
"""


class SqliteRunStore:
    """stdlib `sqlite3` only, no ORM. Every statement below is literal SQL
    with `?` parameter binding -- never string interpolation (see
    `tests/test_store.py::test_sqlite_store_uses_parameter_binding_not_
    string_interpolation`, which feeds a study_id crafted to look like SQL
    and checks it comes back verbatim rather than being executed).

    Holds one connection open for the store's lifetime instead of
    reconnecting per call, specifically so `path=":memory:"` works: a
    fresh connection to `:memory:` is a brand-new, empty database, so
    `save`/`get`/`list` need to share the *same* connection to see each
    other's writes -- this is what lets tests exercise the real SQL path
    without a file on disk.

    A single `threading.Lock` serializes access: `sqlite3` connections are
    not safe to share across threads without `check_same_thread=False`,
    and FastAPI's sync endpoints run in a threadpool, so more than one
    request can reach this store concurrently.
    """

    def __init__(self, path: str | Path = "data/runs.sqlite3") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    def save(self, state: StudyState) -> str:
        run_id = uuid.uuid4().hex
        payload = json.dumps(_record_payload(state))
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (run_id, study_id, status, created_at, alert_count, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, state.study_id, state.status, time.time(), len(state.alerts), payload),
            )
            self._conn.commit()
        return run_id

    def get(self, run_id: str) -> StudyState | None:
        with self._lock:
            row = self._conn.execute("SELECT payload FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return StudyState.model_validate(json.loads(row[0]))

    def list(self, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id, study_id, status, created_at, alert_count FROM runs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_summary(*row) for row in rows]


def build_run_store(settings: Settings | None = None) -> RunStore:
    """Select the configured backend: `settings.run_store` (`"memory"` |
    `"sqlite"`) picks it, `settings.sqlite_path` locates the sqlite file
    when selected.
    """
    settings = settings or Settings()
    if settings.run_store == "sqlite":
        return SqliteRunStore(settings.sqlite_path)
    return InMemoryRunStore()
