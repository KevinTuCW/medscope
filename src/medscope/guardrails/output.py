"""Output guardrail — gates G2 and G3 enforced at run time.

The eval suite checks these gates on a fixed corpus. This function checks
them on the study actually in front of the radiologist, which is the one
that matters: a draft reaching a clinician with an uncited claim has already
failed, whatever a report says about it afterwards.

**This gate only ever downgrades.** It can turn a proposed `DRAFT_READY`
into `HELD`; it can never promote anything. A study halted for another
reason — QC failure, a critical escalation, blocked input — must not emerge
from here looking finished because its draft happened to be well-formed.
"""

from __future__ import annotations

from medscope.evidence import check_evidence
from medscope.language import detect_redlines, has_disclaimer
from medscope.state import Finding, ReportDraft

_REQUIRED_SECTIONS = frozenset({"technique", "findings", "impression", "recommendation"})


def enforce_output(
    draft: ReportDraft | None, findings: list[Finding], proposed_status: str
) -> tuple[str, list[str]]:
    """Return `(status, reasons)`.

    `reasons` is non-empty only when the status was downgraded, and each
    entry names which requirement failed — the workbench shows these to
    explain why a draft is being withheld.
    """
    if proposed_status != "DRAFT_READY":
        return proposed_status, []

    reasons: list[str] = []

    if draft is None:
        return "HELD", ["no draft was produced"]

    present = {s.section for s in draft.sentences}
    missing = _REQUIRED_SECTIONS - present
    if missing:
        reasons.append(f"missing section(s): {', '.join(sorted(missing))}")

    violations = check_evidence(draft, findings)
    if violations:
        reasons.append(f"evidence gate: {len(violations)} unsupported sentence(s)")

    redlined = [s.text for s in draft.sentences if detect_redlines(s.text)]
    if redlined:
        reasons.append(f"redline language in {len(redlined)} sentence(s)")

    if not has_disclaimer(draft):
        reasons.append("disclaimer missing or incomplete")

    return ("HELD", reasons) if reasons else ("DRAFT_READY", [])
