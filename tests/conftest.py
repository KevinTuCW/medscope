"""Shared pytest fixtures for medscope.

The most important fixture here is `_isolated_settings`. This developer
machine has a real `.env` with live GLM / SiliconFlow / Langfuse API keys.
Without isolation, `Settings()` could silently pick up those real
credentials during a test run — this is exactly what happened in the
sibling `aura` project, where a leaking `.env` poisoned a large number of
test cases. This fixture makes every test run against a clean, hermetic
environment: no real `.env` file is read, and no `MEDSCOPE_*` / `OPENAI_*`
/ `LANGFUSE_*` variable can leak in from the shell.
"""

import os

import pytest

from medscope.config import Settings

_ISOLATED_PREFIXES = ("MEDSCOPE_", "OPENAI_", "LANGFUSE_")


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    # Never let Settings read the real .env file during tests.
    monkeypatch.setitem(Settings.model_config, "env_file", None)

    # Strip any live credentials from the test process's environment so
    # Settings() can't pick them up even without a .env file.
    for key in list(os.environ):
        if key.startswith(_ISOLATED_PREFIXES):
            monkeypatch.delenv(key, raising=False)

    # Belt-and-suspenders: tracing must be off by default in tests.
    monkeypatch.delenv("MEDSCOPE_LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("MEDSCOPE_LANGFUSE_SECRET_KEY", raising=False)

    yield
