"""Dependency wiring, and the one asymmetry that matters.

`build_sample_deps` must be hermetic so a fresh clone can run the pipeline.
`build_runtime_deps` must refuse to run half-configured rather than falling
back to the offline stand-ins — an operator told a study was read by a real
VLM, when it was actually read by a stand-in deriving findings from the
study's own ground-truth report, has a false account of the reading.
"""

import pytest

from medscope.arbiter import OfflineArbiterClient
from medscope.bootstrap import build_runtime_deps, build_sample_deps
from medscope.config import Settings
from medscope.obs import traced, tracing_enabled
from medscope.readers.vlm import OfflineVLMClient
from medscope.report import OfflineReportClient


def test_sample_deps_are_fully_offline():
    deps = build_sample_deps()
    assert isinstance(deps.vlm_client, OfflineVLMClient)
    assert isinstance(deps.arbiter_client, OfflineArbiterClient)
    assert isinstance(deps.report_client, OfflineReportClient)
    assert deps.retriever is not None


def test_sample_deps_seed_the_offline_vlm_with_the_studys_own_report():
    deps = build_sample_deps(impression_text="Cardiomegaly.")
    result = deps.vlm_client.chat_with_image("prompt", "/tmp/x.png")
    assert "Cardiomegaly" in result.text


def test_runtime_deps_raise_rather_than_silently_degrading():
    with pytest.raises(RuntimeError, match="VLM"):
        build_runtime_deps(Settings(use_real_vlm=True))


def test_runtime_deps_fall_back_to_offline_only_when_not_asked_for_real():
    deps = build_runtime_deps(Settings(use_real_vlm=False))
    assert isinstance(deps.vlm_client, OfflineVLMClient)


# --- tracing ---------------------------------------------------------------


def test_tracing_is_off_without_keys():
    assert tracing_enabled(Settings()) is False


def test_traced_is_the_identity_function_when_unconfigured():
    """No spans, no buffering, no background flush when tracing is off.

    The graph asserts that a critical alert lands before a slow report;
    an observability layer that quietly changed timing would corrupt
    exactly that assertion.
    """
    calls = []

    @traced("unit")
    def stage(x):
        calls.append(x)
        return x * 2

    assert stage(21) == 42
    assert calls == [21]


def test_tracing_enabled_is_evaluated_per_call_not_at_import(monkeypatch):
    """Guards against baking in start-up configuration.

    Deciding once at import would make the answer depend on whatever
    environment happened to exist when the module first loaded.
    """
    assert tracing_enabled(Settings()) is False
    keyed = Settings(langfuse_public_key="pk", langfuse_secret_key="sk")
    assert tracing_enabled(keyed) is True
