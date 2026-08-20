"""Loader for the arbiter's guideline corpus (`data/samples/guidelines.json`).

The corpus is synthesized teaching material, not a citation source -- see
the `disclaimer` field in the JSON file itself, which every loader here
preserves rather than strips, so nothing downstream can present a passage
as if it came from a real guideline document.
"""
import json
from pathlib import Path

from medscope.rag.embed import Embedder
from medscope.rag.store import Doc, InMemoryVectorStore, Retriever


def load_guidelines(data_dir: str = "data/samples") -> list[dict]:
    """Read `guidelines.json`'s raw `entries` list (id/text/meta dicts),
    without embedding or indexing them. Kept separate from
    `load_guideline_retriever` so callers (and tests) can inspect the
    corpus itself without needing an `Embedder`.
    """
    payload = json.loads((Path(data_dir) / "guidelines.json").read_text(encoding="utf-8"))
    return payload["entries"]


def load_guideline_retriever(data_dir: str, embedder: Embedder) -> Retriever:
    """Load `data/samples/guidelines.json` into an in-memory vector store.

    Each entry is a synthesized radiological interpretation teaching point
    (中文) -- see `load_guidelines`'s docstring and the file's own
    `disclaimer` field.
    """
    entries = load_guidelines(data_dir)
    store = InMemoryVectorStore(embedder)
    store.add([Doc(id=r["id"], text=r["text"], meta=r.get("meta", {})) for r in entries])
    return store
