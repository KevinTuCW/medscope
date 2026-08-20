"""Guards against the failure mode seen in the sibling `aura` project: a
real `.env` with live API keys silently leaking into `Settings()` during
tests. The autouse `_isolated_settings` fixture in conftest.py should make
this true regardless of what's actually sitting in this machine's `.env`.
"""

from medscope.config import Settings


def test_settings_are_isolated_from_real_env():
    assert Settings().tracing_enabled is False
