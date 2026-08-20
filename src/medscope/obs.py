"""Langfuse tracing, off by default and harmless when unconfigured.

Two properties matter more than the tracing itself:

**The enabled check happens at call time, not import time.** Deciding once
when the module loads would bake in whatever configuration existed at
startup, so a test that constructs a differently-configured `Settings` would
still trace — or not — according to a decision made before it ran. The same
mistake in a sibling project let a real `.env` leak into a test run.

**Without keys the decorator is the identity function.** No no-op spans, no
buffered events, no background flush thread. The pipeline must behave
identically whether or not anyone is watching, because a tracing layer that
subtly changes timing would corrupt the one assertion in this project that
depends on timing — the critical-alert-before-report ordering in the graph.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

from medscope.config import Settings

F = TypeVar("F", bound=Callable[..., Any])


def tracing_enabled(settings: Settings | None = None) -> bool:
    return (settings or Settings()).tracing_enabled


def traced(name: str) -> Callable[[F], F]:
    """Wrap a pipeline stage in a Langfuse span when tracing is configured."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not tracing_enabled():
                return func(*args, **kwargs)

            try:
                from langfuse import get_client
            except ImportError:
                # langfuse lives behind the `llm` extra. Its absence must
                # never break a run -- observability is not a dependency of
                # correctness.
                return func(*args, **kwargs)

            client = get_client()
            with client.start_as_current_span(name=name):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
