"""Gate G3 — diagnostic-language redlines.

An AI-generated draft may *describe* what is visible. It may not diagnose,
exclude a diagnosis, or prescribe treatment: those are acts a licensed
physician performs and signs for. This module is the machine-checkable
expression of that boundary, and the run-time enforcement point sits in the
output guardrail, not only in the eval suite.

Two design points worth knowing before editing the word lists:

**Precision is as load-bearing as recall.** Radiology prose is built out of
hedges — "考虑", "提示", "建议结合临床", "建议进一步 CT 检查". A detector that
cannot distinguish those from "确诊为" would neutralise every real sentence in
the report while still scoring perfectly on the detection tests. Over-blocking
here degrades a deliverable just as surely as under-blocking breaches the
boundary; see the false-positive tests in `tests/test_language.py`.

**Normalisation before matching**, so zero-width or full-width characters can't
smuggle a redline past the rules. We reuse `deid._normalize`, which folds
full-width Latin letters and digits and strips zero-width characters, but
deliberately does *not* apply full NFKC — that would rewrite the CJK
punctuation of genuine Chinese report prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from medscope.deid import _normalize
from medscope.state import ReportDraft

REQUIRED_DISCLAIMER = "本报告由 AI 生成，仅为草稿，需由执业医师复核后方可使用。"

RedlineKind = Literal["diagnostic", "exclusion", "treatment"]


@dataclass(frozen=True)
class Redline:
    kind: RedlineKind
    matched: str
    span: tuple[int, int]


def _rule(kind: RedlineKind, pattern: str) -> tuple[RedlineKind, re.Pattern[str]]:
    return kind, re.compile(pattern, re.IGNORECASE)


# Ordered rules. Chinese patterns are written against the surface forms that
# actually appear in reports; English ones require the committing verb, not the
# disease name, so "pneumonia" alone is never a redline.
_RULES: tuple[tuple[RedlineKind, re.Pattern[str]], ...] = (
    # --- diagnostic: asserting a definite diagnosis -------------------------
    _rule("diagnostic", r"确诊(?:为|是)?"),
    _rule("diagnostic", r"明确诊断(?:为|是)?"),
    _rule("diagnostic", r"诊断(?:为|是)(?!不明)"),
    _rule("diagnostic", r"\bdiagnos(?:is|ed|e)\b\s*[::]?"),
    _rule("diagnostic", r"\bconfirms?\s+(?:the\s+)?diagnosis\b"),
    # --- exclusion: ruling a diagnosis out ----------------------------------
    # "可排除" / "除外" commit to absence; "未见明确…征象" merely describes what
    # the image shows and is left alone.
    _rule("exclusion", r"(?:可|能|已|完全)?排除"),
    _rule("exclusion", r"(?:完全|可)?除外"),
    _rule("exclusion", r"\brule[ds]?\s+out\b"),
    _rule("exclusion", r"\bis\s+ruled\s+out\b"),
    _rule("exclusion", r"\bexcludes?\s+(?:the\s+)?(?:diagnosis|possibility)\b"),
    # --- treatment: prescribing therapy -------------------------------------
    # Scoped to therapy. Recommending further *imaging* or clinical correlation
    # is normal radiological practice and must not trip.
    _rule("treatment", r"建议[^。；;]{0,6}?(?:服用|口服|静脉|输液|用药|抗感染治疗)"),
    _rule("treatment", r"建议[^。；;]{0,6}?(?:手术|治疗|化疗|放疗)"),
    _rule("treatment", r"\bprescrib(?:e|ed|ing)\b"),
    _rule("treatment", r"\b(?:start|initiate|administer)\s+\w+(?:cillin|mycin|azole)\b"),
    _rule("treatment", r"\btreat\s+with\b"),
)

# Applied left-to-right when downgrading. Each replacement must itself be free
# of redlines, or `neutralize` would not be idempotent.
_NEUTRALIZATIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"明确诊断(?:为|是)?"), "提示"),
    (re.compile(r"确诊(?:为|是)?"), "提示"),
    (re.compile(r"诊断(?:为|是)"), "考虑为"),
    (re.compile(r"(?:可|能|已|完全)?排除"), "未见明确"),
    (re.compile(r"(?:完全|可)?除外"), "未见明确"),
    (re.compile(r"建议([^。；;]{0,6}?)(?:服用|口服|静脉|输液|用药|抗感染治疗)"), r"建议结合临床"),
    (re.compile(r"建议([^。；;]{0,6}?)(?:手术|治疗|化疗|放疗)"), r"建议结合临床"),
    (re.compile(r"\bdiagnos(?:is|ed|e)\b\s*[::]?", re.IGNORECASE), "findings suggest"),
    (re.compile(r"\brule[ds]?\s+out\b", re.IGNORECASE), "without evidence of"),
    (re.compile(r"\bis\s+ruled\s+out\b", re.IGNORECASE), "is not evident"),
    (re.compile(r"\bprescrib(?:e|ed|ing)\b", re.IGNORECASE), "clinical correlation for"),
    (re.compile(r"\btreat\s+with\b", re.IGNORECASE), "clinical correlation regarding"),
)


def detect_redlines(text: str) -> list[Redline]:
    """Return every redline in `text`, normalised before matching."""
    normalized = _normalize(text)
    hits: list[Redline] = []
    for kind, pattern in _RULES:
        for m in pattern.finditer(normalized):
            hits.append(Redline(kind=kind, matched=m.group(0), span=m.span()))
    return hits


def neutralize(text: str) -> str:
    """Downgrade committing language to observational language.

    Idempotent: every replacement is itself redline-free, so re-running this on
    already-neutralised text is a no-op. Clean text is returned unchanged
    rather than rewritten.
    """
    out = text
    for pattern, replacement in _NEUTRALIZATIONS:
        out = pattern.sub(replacement, out)
    return out


def has_disclaimer(draft: ReportDraft) -> bool:
    """True only if the draft states both that AI wrote it and that a physician must review it.

    A generic "for reference only" is rejected deliberately. Those are the two
    facts a reader must not miss, and a vague disclaimer conveys neither.
    """
    text = _normalize(draft.disclaimer or "")
    if not text:
        return False
    says_ai = any(token in text.upper() for token in ("AI", "人工智能", "机器"))
    says_review = any(
        token in text for token in ("复核", "审核", "医师", "医生", "签发")
    )
    return says_ai and says_review
