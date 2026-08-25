"""Shared pytest fixtures for medscope.

The most important fixture here is `_isolated_settings`. This developer
machine has a real `.env` with live GLM / SiliconFlow / Langfuse API keys.
Without isolation, `Settings()` could silently pick up those real
credentials during a test run — exactly what happened in the sibling `aura`
project, where a leaking `.env` poisoned a large number of test cases.

Note the fixture clears variables by **enumerating `Settings`' own fields**
rather than by matching a prefix. Settings deliberately has no `env_prefix`,
so its variables are bare names like `VLM_API_KEY` with nothing in common to
match on. Deriving the list from `model_fields` also means a field added
later is isolated automatically — a hand-maintained list would silently stop
covering new settings, and the failure mode of *that* is a test run quietly
picking up a real credential.
"""

import os

import pytest

from medscope.config import Settings

#: Third-party conventions that aren't medscope settings but can still steer
#: SDK behaviour if they leak in from the shell.
_ISOLATED_PREFIXES = ("OPENAI_", "LANGFUSE_", "MEDSCOPE_")


def _settings_env_names() -> set[str]:
    """Every environment variable `Settings` would read, upper and lower case.

    pydantic-settings matches case-insensitively; clearing both spellings
    avoids depending on that behaviour staying the same.
    """
    names = set(Settings.model_fields)
    return {n.upper() for n in names} | {n.lower() for n in names}


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch, tmp_path_factory):
    # Never let Settings read the real .env file during tests.
    monkeypatch.setitem(Settings.model_config, "env_file", None)

    managed = _settings_env_names()
    for key in list(os.environ):
        if key in managed or key.startswith(_ISOLATED_PREFIXES):
            monkeypatch.delenv(key, raising=False)

    # Belt-and-suspenders: tracing must be off by default in tests.
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    # Deleting OPENI_ROOT is not enough to make the fetched corpus invisible:
    # `Settings.openi_root` then falls back to its default, `data/openi`,
    # which exists on a machine that has run `scripts/fetch_openi.py` and
    # does not exist in CI or a fresh clone. A test asserting on the study
    # list would pass here and fail there -- or worse, quietly assert
    # against 3.8k studies on one machine and 3 on another. Pointing it at a
    # path that cannot exist makes every test see what CI sees; the tests
    # that genuinely want a corpus build one under `tmp_path` and pass it in
    # explicitly.
    monkeypatch.setenv("OPENI_ROOT", str(tmp_path_factory.mktemp("no-openi") / "absent"))

    from medscope import workbench

    workbench.reset_corpus_cache()
    yield
    workbench.reset_corpus_cache()
