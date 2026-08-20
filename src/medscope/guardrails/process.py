"""Process guardrail — clean the finding set before it costs anything.

Sits between the readers and the arbiter, where each surviving disagreement
buys an LLM call, and before the report, where a malformed finding becomes a
malformed citation.

Two jobs:

* **Deduplicate by canonical label**, keeping the higher probability. Both
  readers naming the same finding is agreement, not two findings.
* **Drop malformed entries** — a blank label can't be cited or matched, so
  it can only travel through the pipeline causing confusion.

When truncating to `max_findings`, keep the **highest-probability** ones.
Dropping in arrival order could discard a confident critical finding while
retaining noise, which is the opposite of what a cap is for.
"""

from __future__ import annotations

from medscope.ontology import canonical
from medscope.state import Finding


def cap_findings(
    findings: list[Finding], max_findings: int
) -> tuple[list[Finding], int]:
    """Return `(kept, dropped_count)`.

    `dropped_count` covers both malformed entries and cap truncation, so the
    audit trail can report that something was discarded at all — a silent
    cap would misrepresent how much the readers actually said.
    """
    dropped = 0
    best: dict[str, Finding] = {}

    for f in findings:
        label = (f.label or "").strip()
        if not label:
            dropped += 1
            continue

        key = canonical(label) or label.casefold()
        current = best.get(key)
        if current is None or f.prob > current.prob:
            best[key] = f

    kept = sorted(best.values(), key=lambda f: f.prob, reverse=True)
    if len(kept) > max_findings:
        dropped += len(kept) - max_findings
        kept = kept[:max_findings]

    return kept, dropped
