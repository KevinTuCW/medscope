"""Entry point for running one study through the compiled graph.

Deliberately three lines: build the graph fresh from `deps` (cheap -- it's
just wiring closures over already-constructed clients), invoke it with the
initial `StudyState` as a plain dict, and reconcile the result back into a
`StudyState`. See `medscope.graph`'s module docstring for why
`StudyState.model_validate` at the end is load-bearing, not decorative --
it's what drops `GraphState`'s graph-only routing keys.
"""

from __future__ import annotations

from medscope.graph import GraphDeps, build_graph
from medscope.state import StudyState


def run_study(state: StudyState, deps: GraphDeps) -> StudyState:
    graph = build_graph(deps)
    result = graph.invoke(state.model_dump())
    return StudyState.model_validate(result)
