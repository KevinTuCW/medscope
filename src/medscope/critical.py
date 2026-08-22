"""Critical-finding triage -- the highest-stakes module in Phase 1.

Gate G1 requires critical-finding recall == 1.0: missing a pneumothorax
can kill, over-calling one costs a radiologist thirty seconds. That
asymmetry is deliberate and is not to be "balanced" into an F1 score --
recall is a hard gate, false-positive rate is only a soft advisory.

`triage` runs in parallel with report writing, not after it -- mirroring
real hospital critical-results communication policy, where a
life-threatening finding must reach a clinician immediately rather than
queue behind a report. Nothing here depends on the report existing, on
`merge_reads` having run, or on the arbiter having resolved anything: a
disagreement where only one reader calls a critical label positive must
still alert immediately rather than wait for arbitration, so callers pass
this function the readers' raw findings (e.g. `read_a.findings +
read_b.findings`), not `merge_reads`'s output -- `merge_reads` only
returns *agreement* findings, and a one-reader-positive critical call is
by definition not an agreement.

Pure function: no LLM, no network, no model.
"""

from __future__ import annotations

from datetime import datetime

from medscope.config import Settings
from medscope.ontology import CRITICAL_LABELS
from medscope.state import CriticalAlert, Finding


def triage(findings: list[Finding], settings: Settings, image_ref: str = "") -> list[CriticalAlert]:
    """Scan `findings` for `ontology.CRITICAL_LABELS` -- G1's single source
    of truth for which labels are life-threatening -- and raise one
    `CriticalAlert` per critical label present at or above
    `settings.critical_threshold`.

    Deliberately uses `critical_threshold` (0.3), not `cnn_prob_threshold`
    (0.5): a finding at 0.35 does not make the report, but does raise a
    critical alert here. See
    test_critical.py::test_alert_fires_below_report_threshold_but_above_critical_threshold,
    which pins this exact behaviour -- the load-bearing expression of the
    recall-over-precision asymmetry this module exists for.

    `image_ref` is the study-level fallback, not the answer. reader_a
    reads every film of a study, so the finding that raised the alert
    knows which film it was seen on -- and that is the film a radiologist
    needs pulled up. An alert pointing at the study's display film while
    the pneumothorax was seen on another projection sends someone to look
    at the wrong picture. A finding's own `image_ref` therefore wins; this
    parameter covers findings that carry no attribution (reader_b, and
    states written before the field existed).

    Deduplication: if more than one reader's Finding calls the same
    critical label positive, this emits exactly one alert for that label
    with `source="both"` and the higher of the two probabilities -- same
    over-calling-is-safe reasoning as the threshold choice: a human should
    know two independent readers agree, not receive it as two alerts.
    """
    by_label: dict[str, list[Finding]] = {}
    for finding in findings:
        if finding.label not in CRITICAL_LABELS:
            continue
        if finding.prob < settings.critical_threshold:
            continue
        by_label.setdefault(finding.label, []).append(finding)

    now = datetime.now()
    alerts: list[CriticalAlert] = []
    for label, group in by_label.items():
        sources = {f.source for f in group}
        best = max(group, key=lambda f: f.prob)
        source = "both" if len(sources) > 1 else best.source
        alerts.append(
            CriticalAlert(
                label=label,
                prob=best.prob,
                source=source,
                detected_at=now,
                image_ref=best.image_ref or image_ref,
            )
        )

    return alerts
