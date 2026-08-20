"""Propose CANDIDATES for the critical-finding gold set (`data/evals/critical.json`).

*** This script's output is a candidate list, NOT a gold set. ***
Every candidate below still needs a human to read the actual report text
and decide. Do not copy this script's output directly into
`data/evals/critical.json` -- see the module docstring in
`medscope.critical` and the note printed at the end of this run for why.

Why this can't be fully automated: naive keyword matching on OpenI reports
is dominated by negation. Of the reports that mention "pneumothorax" or
"pneumomediastinum" at all, the large majority are negations ("no
pneumothorax") -- i.e. normal studies describing the *absence* of the
finding, which is exactly the sentence a keyword-only search can't tell
apart from an affirmed case. This script applies a NegEx-style heuristic
(negation/hedge trigger words in a window before the keyword, within the
same sentence) to narrow the field, and cross-checks MeSH terms (human
sub-heading assigned per report, independent of the free-text report
whose negation this script is trying to parse) as corroborating signal --
but the heuristic is still just a heuristic. It will pass through hedges
("cannot exclude a small pneumothorax") and miss unusual phrasings it
doesn't recognize. Reading each candidate's actual FINDINGS/IMPRESSION
text before it goes in the gold set is not optional.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/build_critical_goldset.py [openi_root]

Defaults to `Settings.openi_root`; pass an explicit path to point at a
local probe directory (e.g. /tmp/openi_probe) instead.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medscope.config import Settings  # noqa: E402
from medscope.data.openi import _find_images, _parse_report  # noqa: E402

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Keyword -> canonical critical label. "mediastinal emphysema" is MeSH's
# own term for pneumomediastinum (confirmed against real OpenI MeSH
# assignments -- see medscope.ontology's alias table, which now includes
# it as an English alias for exactly this reason).
_KEYWORDS: dict[str, str] = {
    "pneumothorax": "Pneumothorax",
    "pneumomediastinum": "Pneumomediastinum",
    "mediastinal emphysema": "Pneumomediastinum",
    "pleural effusion": "PleuralEffusion",
    "effusion": "PleuralEffusion",
}

# Negation triggers: if one appears in the `_WINDOW` words before the
# keyword within the same sentence, the sentence is denying the finding.
_NEGATION_TRIGGERS = (
    "no ",
    "no evidence of",
    "no significant",
    "without",
    "free of",
    "absence of",
    "not seen",
    "not appreciated",
    "not identified",
    "not present",
    "negative for",
    "ruled out",
    "rules out",
    "unremarkable for",
    "clear of",
)

# Hedge triggers: neither a clean affirmation nor a clean negation --
# flagged separately so a human looks closely rather than the heuristic
# guessing either way.
_HEDGE_TRIGGERS = (
    "cannot exclude",
    "can not exclude",
    "cannot rule out",
    "can not rule out",
    "can't exclude",
    "can't rule out",
    "possible",
    "questionable",
    "suspicious for",
    "concerning for",
    "may represent",
    "less apparent",
    "less perceptible",
    "history of",
)

# Severity words that matter specifically for pleural effusion: only a
# LARGE effusion is in CRITICAL_LABELS' clinical sense (see
# medscope.ontology's module docstring) -- a small/trace one is real but
# not life-threatening.
_LARGE_EFFUSION_WORDS = ("large", "massive", "sizable", "moderate to large")
_SMALL_EFFUSION_WORDS = ("small", "trace", "tiny", "minimal", "minute")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _classify_sentence(sentence: str, keyword: str) -> str:
    """Return 'negated', 'hedge', or 'affirmed' for one keyword occurrence
    in one sentence, based on trigger words appearing anywhere before it in
    the same sentence.

    Deliberately scans the whole sentence prefix rather than a fixed
    lookback window: radiology sentences routinely negate a whole list of
    findings up front ("Negative for consolidation, effusion, or
    pneumothorax"), so a short window clips the trigger off the front of
    exactly the sentences this needs to catch. A trailing space is added
    to the prefix so a single-word trigger like "no " still matches when
    the triggering word is the very last one before the keyword (e.g.
    "There is no pneumothorax" -- without the trailing space, "no " is not
    a substring of "...is no").
    """
    lower = sentence.lower()
    idx = lower.find(keyword)
    if idx == -1:
        return "affirmed"  # shouldn't happen; caller only calls on a match

    prefix = lower[:idx] + " "

    if any(trigger in prefix for trigger in _NEGATION_TRIGGERS):
        return "negated"
    if any(trigger in prefix for trigger in _HEDGE_TRIGGERS):
        return "hedge"
    return "affirmed"


def _effusion_severity(sentence: str) -> str:
    # Word-boundary matching, not substring: "large" is a substring of
    # "larger" ("...effusions, right larger than left"), which is a
    # laterality comparison, not a severity grade -- a plain `in` check
    # misclassifies it as "large" severity. Caught by manual review while
    # building the gold set; see data/evals/critical.json's notes for the
    # specific studies (313, 1957, 3249) this bit.
    lower = sentence.lower()
    if any(re.search(rf"\b{re.escape(w)}\b", lower) for w in _LARGE_EFFUSION_WORDS):
        return "large"
    if any(re.search(rf"\b{re.escape(w)}\b", lower) for w in _SMALL_EFFUSION_WORDS):
        return "small"
    return "unspecified"


def _scan_report(study_id: str, parsed: dict, image_paths: list) -> dict | None:
    text = f"{parsed['FINDINGS']} {parsed['IMPRESSION']}"
    if not text.strip():
        return None

    hits = []
    for sentence in _sentences(text):
        lower = sentence.lower()
        for keyword, label in _KEYWORDS.items():
            if keyword not in lower:
                continue
            classification = _classify_sentence(sentence, keyword)
            hit = {
                "label": label,
                "keyword": keyword,
                "classification": classification,
                "sentence": sentence,
            }
            if label == "PleuralEffusion":
                hit["effusion_severity"] = _effusion_severity(sentence)
            hits.append(hit)

    if not hits:
        return None

    return {
        "study_id": study_id,
        "hits": hits,
        "mesh": parsed["mesh"],
        "indication": parsed["INDICATION"],
        "image_available": bool(image_paths),
        "image_paths": [str(p) for p in image_paths],
    }


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Settings().openi_root
    reports_dir = root / "ecgen-radiology"
    print(f"Scanning OpenI reports under: {reports_dir}")

    # Deliberately does NOT reuse medscope.data.openi.load_studies here --
    # that function requires an image to be present and silently skips any
    # study without one, which is exactly wrong for this scan: we want
    # every report in the corpus (most of which have no locally-available
    # image, see Settings.openi_root vs the 108-image local probe slice),
    # and to record image availability as a separate fact per study rather
    # than using it as a filter.
    images_by_id = _find_images(root)

    report_paths = sorted(reports_dir.glob("*.xml")) if reports_dir.is_dir() else []
    print(f"Found {len(report_paths)} report XML files.")
    print(f"Found {sum(len(v) for v in images_by_id.values())} images locally, covering {len(images_by_id)} studies.")

    candidates = []
    total_mentions = 0
    negated = 0
    hedged = 0
    affirmed = 0
    parse_errors = 0
    for xml_path in report_paths:
        study_id = xml_path.stem
        try:
            parsed = _parse_report(xml_path)
        except Exception:  # noqa: BLE001 -- a malformed report shouldn't kill the scan
            parse_errors += 1
            continue

        scanned = _scan_report(study_id, parsed, images_by_id.get(study_id, []))
        if scanned is None:
            continue
        candidates.append(scanned)
        for hit in scanned["hits"]:
            total_mentions += 1
            if hit["classification"] == "negated":
                negated += 1
            elif hit["classification"] == "hedge":
                hedged += 1
            else:
                affirmed += 1

    if parse_errors:
        print(f"({parse_errors} report(s) failed to parse and were skipped.)")

    n_pneumo_reports = sum(
        1
        for c in candidates
        if any(h["label"] in ("Pneumothorax", "Pneumomediastinum") for h in c["hits"])
    )
    print(f"\n{n_pneumo_reports} reports mention pneumothorax or pneumomediastinum at all.")
    print(f"{len(candidates)} reports mention at least one target keyword (any of the 3 critical labels + 'effusion').")
    print(f"{total_mentions} total keyword-bearing sentences across those reports:")
    print(f"  negated:  {negated} ({negated / total_mentions:.1%})")
    print(f"  hedged:   {hedged} ({hedged / total_mentions:.1%})")
    print(f"  affirmed (heuristic only -- NOT verified): {affirmed} ({affirmed / total_mentions:.1%})")
    with_image = sum(1 for c in candidates if c["image_available"])
    print(f"\n{with_image} / {len(candidates)} candidate studies have a locally-available image.")

    out_path = Path("/tmp/critical_goldset_candidates.json")
    out_path.write_text(json.dumps(candidates, indent=2, ensure_ascii=False))

    print(f"\nWrote {len(candidates)} candidate studies to {out_path}")
    print(
        "\n*** THIS IS A CANDIDATE LIST, NOT A GOLD SET. ***\n"
        "Every 'affirmed' classification above is a heuristic guess, not a\n"
        "verified label. A human must read each candidate's FINDINGS/\n"
        "IMPRESSION text (and MeSH terms as corroboration) before anything\n"
        "from this file is fit to land in data/evals/critical.json."
    )


if __name__ == "__main__":
    main()
