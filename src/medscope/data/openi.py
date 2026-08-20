"""Loader for the OpenI / Indiana University chest X-ray dataset.

Pairs each radiology report XML (`ecgen-radiology/<N>.xml`) with the chest
X-ray image(s) whose filename starts with the matching `CXR<N>_` prefix.
This pairing is the foundation for everything downstream: the CNN reader
consumes `image_paths`, report-draft evaluation compares against
`impression_text`, and the critical-findings gold set is curated from these
reports.

Works against either the full fetched dataset (`Settings.openi_root`,
populated by `scripts/fetch_openi.py`) or the tiny committed sample slice
in `data/samples/studies/` — both use the same on-disk layout:

    <root>/ecgen-radiology/<N>.xml   -- report XML, matches the upstream
                                         archive's internal path
    <root>/**/CXR<N>_*.png           -- image(s) for that study (searched
                                         recursively so any extraction
                                         layout works)
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_IMAGE_PREFIX_RE = re.compile(r"^CXR(\d+)_")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:])")

_TEXT_LABELS = ("FINDINGS", "IMPRESSION", "INDICATION")

# Coordinating conjunctions that read as "orphaned" once the clause they
# joined to has been redacted away, e.g. "XXXX, XXXX and shortness of
# breath" -> naive substitution leaves "and shortness of breath". Only the
# leading position is stripped; mid-text occurrences are left alone.
_LEADING_CONJUNCTIONS = {"and", "or", "but"}


@dataclass
class Study:
    study_id: str
    image_paths: list[Path]
    findings_text: str
    impression_text: str
    mesh: list[str]
    indication: str
    # Per-study audit trail, e.g. records when XXXX de-identification
    # placeholders were stripped from the report text.
    notes: list[str] = field(default_factory=list)


def _normalize_xxxx(text: str) -> tuple[str, bool]:
    """Strip OpenI's literal `XXXX` de-identification placeholders.

    `XXXX` is not a uniform token in this corpus -- it stands in for a whole
    redacted word ("XXXX, XXXX and shortness of breath"), an age embedded in
    a compound word ("XXXX-year-old"), and a name-shaped fragment inside an
    ordinary word ("chest x-XXXX" for "chest x-ray", since their scrubber
    flags "ray" as a name). A blanket substring substitution leaves each of
    these as a different flavor of garbage -- most visibly a dangling
    leading hyphen ("-year-old female with chest pain").

    We deliberately do not try to reconstruct what was redacted (no
    "XXXX-year-old" -> "the patient is X years old" guessing). Instead: drop
    any *whole* whitespace-delimited token that contains "XXXX" -- covering
    both the standalone-word and the embedded-in-a-word cases uniformly --
    then reassemble with single spaces, so a run of dropped tokens can never
    leave a double space or a stray comma/period orphaned mid-sentence.
    After that: drop a leading coordinating conjunction that's now orphaned
    (see `_LEADING_CONJUNCTIONS`), drop any leading token that's pure
    punctuation, and collapse space-before-punctuation. If nothing
    alphanumeric survives, the field becomes "" rather than leftover
    punctuation.

    Returns the cleaned text and whether anything changed, so the caller can
    record a per-study note rather than silently rewriting the report.
    """
    if not text or "XXXX" not in text:
        return text, False

    tokens = [tok for tok in text.split() if "XXXX" not in tok]

    while tokens and tokens[0].strip(".,;:").lower() in _LEADING_CONJUNCTIONS:
        tokens = tokens[1:]
    while tokens and not any(ch.isalnum() for ch in tokens[0]):
        tokens = tokens[1:]

    cleaned = " ".join(tokens)
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned).strip()

    if not any(ch.isalnum() for ch in cleaned):
        cleaned = ""

    return cleaned, True


def _parse_report(xml_path: Path) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    texts = {label: "" for label in _TEXT_LABELS}
    for node in root.findall(".//AbstractText"):
        label = node.get("Label", "")
        if label in texts:
            texts[label] = (node.text or "").strip()

    mesh = [m.text.strip() for m in root.findall(".//MeSH/major") if m.text and m.text.strip()]

    return {**texts, "mesh": mesh}


def _find_images(root: Path) -> dict[str, list[Path]]:
    images_by_id: dict[str, list[Path]] = {}
    for png_path in sorted(root.rglob("CXR*.png")):
        match = _IMAGE_PREFIX_RE.match(png_path.name)
        if not match:
            continue
        images_by_id.setdefault(match.group(1), []).append(png_path)
    return images_by_id


def load_studies(root: Path) -> list[Study]:
    """Load and pair every (report, image set) under `root`.

    A study is skipped — with a logged reason — rather than raising, if its
    report XML is missing or unparseable, or if it has no images. This keeps
    one bad record from crashing an entire dataset load.
    """
    root = Path(root)
    reports_dir = root / "ecgen-radiology"

    report_ids = {p.stem for p in reports_dir.glob("*.xml")} if reports_dir.is_dir() else set()
    images_by_id = _find_images(root)

    all_ids = report_ids | set(images_by_id)

    def _sort_key(study_id: str):
        return (0, int(study_id)) if study_id.isdigit() else (1, study_id)

    studies: list[Study] = []
    for study_id in sorted(all_ids, key=_sort_key):
        xml_path = reports_dir / f"{study_id}.xml"
        images = images_by_id.get(study_id, [])

        if not xml_path.exists():
            logger.warning(
                "skipping study %s: report XML missing at %s", study_id, xml_path
            )
            continue
        if not images:
            logger.warning("skipping study %s: no images found under %s", study_id, root)
            continue

        try:
            parsed = _parse_report(xml_path)
        except ET.ParseError as exc:
            logger.warning("skipping study %s: corrupt report XML (%s)", study_id, exc)
            continue

        notes: list[str] = []
        findings, findings_changed = _normalize_xxxx(parsed["FINDINGS"])
        impression, impression_changed = _normalize_xxxx(parsed["IMPRESSION"])
        indication, indication_changed = _normalize_xxxx(parsed["INDICATION"])
        if findings_changed or impression_changed or indication_changed:
            notes.append("normalized XXXX de-identification placeholder(s) (dropped, not reconstructed)")

        studies.append(
            Study(
                study_id=study_id,
                image_paths=images,
                findings_text=findings,
                impression_text=impression,
                mesh=parsed["mesh"],
                indication=indication,
                notes=notes,
            )
        )

    return studies
