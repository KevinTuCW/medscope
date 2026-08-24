"""Entry point for running one study through the compiled graph.

Build the graph fresh from `deps` (cheap -- it's just wiring closures over
already-constructed clients), invoke it with the initial `StudyState` as a
plain dict, and reconcile the result back into a `StudyState`. See
`medscope.graph`'s module docstring for why `StudyState.model_validate` at
the end is load-bearing, not decorative -- it's what drops `GraphState`'s
graph-only routing keys.

The tracing wrapper is the one addition to that. It is here, and not inside
`build_graph`, because a trace's scope is one study run -- the graph object
is reusable and knows nothing about how many studies pass through it. The
LangGraph callback goes in via `config` rather than by decorating nodes, so
`graph.py` keeps its property of importing nothing observability-related:
the pipeline does not know it is being watched.
"""

from __future__ import annotations

from medscope.graph import GraphDeps, build_graph
from medscope.obs import study_trace
from medscope.state import StudyState


def run_study(state: StudyState, deps: GraphDeps) -> StudyState:
    graph = build_graph(deps)
    with study_trace(state, deps.settings) as trace:
        result = graph.invoke(state.model_dump(), config={"callbacks": trace.callbacks})
        study = StudyState.model_validate(result)
        trace.finish(study)
    return study
