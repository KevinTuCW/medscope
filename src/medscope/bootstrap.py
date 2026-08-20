"""Wiring the pipeline's dependencies, offline or live.

Two entry points, and the difference between them is the point of this
module:

`build_sample_deps()` is **fully hermetic** — offline stand-ins for every
model, a local hashing embedder, the committed sample corpus. No network, no
key, no weight download. It is what the test suite and a fresh clone run on.

`build_runtime_deps()` reads configuration and wires real clients. When the
configuration is incomplete it **raises rather than quietly returning the
offline stack**. That asymmetry is deliberate: an operator who believes a
study was read by a real VLM, when it was actually read by a stand-in that
derives its findings from the ground-truth report text, has been given a
false account of how the study was interpreted. A crash is recoverable; a
plausible-looking fabricated reading is not.
"""

from __future__ import annotations

from medscope.arbiter import OfflineArbiterClient
from medscope.config import Settings
from medscope.graph import GraphDeps
from medscope.rag.corpus import load_guideline_retriever
from medscope.rag.embed import LocalHashingEmbedder
from medscope.readers.cnn import CNNReader
from medscope.readers.vlm import OfflineVLMClient, build_vlm_client
from medscope.report import OfflineReportClient


def _retriever(data_dir: str = "data/samples"):
    return load_guideline_retriever(data_dir, LocalHashingEmbedder())


def build_sample_deps(
    settings: Settings | None = None,
    *,
    impression_text: str = "",
    data_dir: str = "data/samples",
) -> GraphDeps:
    """Fully offline dependency set — no network, no key, no weights.

    `impression_text` seeds `OfflineVLMClient`, which derives its findings
    from a study's own paired report. Anything built on this stack exercises
    plumbing, never model quality: a green run here says the pieces fit
    together, not that the system reads chest X-rays well.
    """
    settings = settings or Settings()
    return GraphDeps(
        settings=settings,
        cnn_reader=CNNReader(settings),
        vlm_client=OfflineVLMClient(impression_text=impression_text),
        arbiter_client=OfflineArbiterClient(),
        retriever=_retriever(data_dir),
        report_client=OfflineReportClient(),
    )


def build_runtime_deps(
    settings: Settings | None = None, *, data_dir: str = "data/samples"
) -> GraphDeps:
    """Live dependency set, gated on configuration.

    Raises when `use_real_vlm` is set without the credentials to honour it —
    see the module docstring for why silent degradation is the worse
    failure here.
    """
    settings = settings or Settings()

    # `build_vlm_client` already raises on the misconfigured combination;
    # calling it first means the failure surfaces before anything expensive
    # (weight loading, corpus indexing) has happened.
    vlm_client = build_vlm_client(settings)

    return GraphDeps(
        settings=settings,
        cnn_reader=CNNReader(settings),
        vlm_client=vlm_client,
        # The arbiter and report writer have no dedicated credentials of
        # their own yet; until they do, they stay on the offline stand-ins
        # rather than silently borrowing the VLM's configuration.
        arbiter_client=OfflineArbiterClient(),
        report_client=OfflineReportClient(),
        retriever=_retriever(data_dir),
    )
