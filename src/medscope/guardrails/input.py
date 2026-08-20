"""Input guardrail — screen what arrives with the study.

`history_text` and `indication` are written by clinicians, travel with the
study, and are never validated at source. They reach reader_b's prompt and
the arbiter's prompt, so they are the injection surface of this system.

The screening policy is deliberately asymmetric:

* A **malformed study** (no image) is blocked — there is nothing to read.
* An **injection attempt is recorded, not blocked.** Discarding a real
  patient's study because someone typed something odd into a history field
  would be the wrong failure: the text is already defanged by
  `neutralize_untrusted()` at every prompt boundary, and the note preserves
  the fact for the audit trail. Refusing to read the film is a bigger harm
  than reading it with a flagged history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from medscope.security.sanitize import detect_injection
from medscope.state import StudyState


@dataclass
class IntakeVerdict:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def screen_intake(state: StudyState) -> IntakeVerdict:
    reasons: list[str] = []

    if not (state.image_path or "").strip():
        return IntakeVerdict(ok=False, reasons=["missing image path"])

    for name in ("history_text", "indication"):
        text = getattr(state, name, "") or ""
        if not text.strip():
            continue
        detected, category = detect_injection(text)
        if detected:
            reasons.append(f"injection detected in {name} ({category})")

    return IntakeVerdict(ok=True, reasons=reasons)
