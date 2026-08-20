"""Tests for the OpenI data loader.

These run entirely offline against the committed sample slice in
data/samples/studies/ — no network access, no dependency on the real
(gitignored) data/openi/ dataset. See data/samples/studies/README.md for
which 3 studies were picked and why.
"""

import re
from pathlib import Path

import pytest

from medscope.data.openi import Study, load_studies

SAMPLES_ROOT = Path("data/samples/studies")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s[.,;:]")
_LEADING_CONJUNCTIONS = {"and", "or", "but"}


def test_load_studies_returns_exactly_three():
    studies = load_studies(SAMPLES_ROOT)
    assert len(studies) == 3


def test_first_study_has_required_fields():
    studies = load_studies(SAMPLES_ROOT)
    first = studies[0]

    assert isinstance(first, Study)
    assert first.findings_text or first.impression_text
    assert len(first.image_paths) >= 1
    assert all(isinstance(p, Path) for p in first.image_paths)
    assert isinstance(first.mesh, list)
    assert len(first.mesh) >= 1


def test_xxxx_normalization_invariants_hold_for_every_field():
    """Property check across every sample study and every extracted text
    field, not a hardcoded example.

    Asserting only "XXXX" not in text (the original version of this test)
    is exactly what let a corrupted result ship: '-year-old female with
    chest pain' has no literal 'XXXX' in it but is not usable text. This
    checks the invariants the loader actually promises:
      1. no XXXX placeholder survives
      2. non-empty fields start with an alphanumeric char -- no leading
         hyphen/comma, no orphaned leading conjunction
      3. no doubled spaces, no space-before-punctuation
      4. a field never ends up as leftover punctuation-only content -- a
         fully-redacted field must become ""
    It loops over studies/fields so it keeps working when a 4th sample is
    added, and fails on the class of bug rather than one instance of it.
    """
    studies = load_studies(SAMPLES_ROOT)
    assert studies, "expected at least one sample study to check invariants against"

    checked_fields = 0
    saw_normalization_note = False

    for study in studies:
        if study.notes:
            saw_normalization_note = True

        for field_name in ("findings_text", "impression_text", "indication"):
            text = getattr(study, field_name)
            checked_fields += 1
            label = f"study {study.study_id} .{field_name}"

            # 1. no XXXX placeholder survives
            assert "XXXX" not in text, f"{label}: {text!r}"

            if not text:
                continue

            # 2. starts with an alphanumeric char; no orphaned conjunction
            assert text[0].isalnum(), f"{label} has a non-alnum leading char: {text!r}"
            first_word = text.split(" ", 1)[0].strip(".,;:").lower()
            assert first_word not in _LEADING_CONJUNCTIONS, (
                f"{label} starts with an orphaned conjunction: {text!r}"
            )

            # 3. no doubled spaces, no space-before-punctuation
            assert "  " not in text, f"{label} has a doubled space: {text!r}"
            assert not _SPACE_BEFORE_PUNCT_RE.search(text), (
                f"{label} has space-before-punctuation: {text!r}"
            )

            # 4. never pure leftover punctuation -- a fully-redacted field
            # must have been emptied out instead.
            assert any(ch.isalnum() for ch in text), (
                f"{label} is leftover punctuation with no content: {text!r}"
            )

    assert checked_fields > 0
    # At least one of our 3 sample reports actually contained XXXX in the
    # raw XML (OpenI's de-identification placeholder) — confirm the loader
    # noticed and recorded that it normalized something, not just that the
    # invariants happen to hold vacuously because nothing needed cleanup.
    assert saw_normalization_note


def test_missing_report_xml_is_skipped_not_raised(tmp_path, caplog):
    root = tmp_path / "studies"
    (root / "ecgen-radiology").mkdir(parents=True)
    images_dir = root / "images"
    images_dir.mkdir()
    # Image exists for study 99, but no matching ecgen-radiology/99.xml.
    (images_dir / "CXR99_IM-0001-1001.png").write_bytes(b"stand-in, not a real png")

    with caplog.at_level("WARNING"):
        studies = load_studies(root)

    assert studies == []
    assert "99" in caplog.text


def test_corrupt_report_xml_is_skipped_not_raised(tmp_path, caplog):
    root = tmp_path / "studies"
    reports_dir = root / "ecgen-radiology"
    reports_dir.mkdir(parents=True)
    images_dir = root / "images"
    images_dir.mkdir()

    (reports_dir / "7.xml").write_text("<eCitation><unclosed>", encoding="utf-8")
    (images_dir / "CXR7_IM-0001-1001.png").write_bytes(b"stand-in, not a real png")

    with caplog.at_level("WARNING"):
        studies = load_studies(root)

    assert studies == []
    assert "7" in caplog.text


def test_study_with_no_images_is_skipped_not_raised(tmp_path, caplog):
    root = tmp_path / "studies"
    reports_dir = root / "ecgen-radiology"
    reports_dir.mkdir(parents=True)
    (root / "images").mkdir()

    (reports_dir / "5.xml").write_text(
        "<eCitation><MedlineCitation><Article><Abstract>"
        '<AbstractText Label="FINDINGS">Clear lungs.</AbstractText>'
        '<AbstractText Label="IMPRESSION">Normal.</AbstractText>'
        "</Abstract></Article></MedlineCitation></eCitation>",
        encoding="utf-8",
    )

    with caplog.at_level("WARNING"):
        studies = load_studies(root)

    assert studies == []
    assert "5" in caplog.text


def test_multi_view_study_keeps_every_image_in_order(tmp_path):
    # Multi-view is spec'd behavior (a study commonly has a frontal + a
    # lateral) but none of the 3 committed sample studies happen to have
    # more than one image, so it was previously unexercised. Build a
    # synthetic two-image study to cover it directly.
    root = tmp_path / "studies"
    reports_dir = root / "ecgen-radiology"
    reports_dir.mkdir(parents=True)
    images_dir = root / "images"
    images_dir.mkdir()

    (reports_dir / "42.xml").write_text(
        "<eCitation><MedlineCitation><Article><Abstract>"
        '<AbstractText Label="FINDINGS">Clear lungs.</AbstractText>'
        '<AbstractText Label="IMPRESSION">Normal.</AbstractText>'
        "</Abstract></Article></MedlineCitation></eCitation>",
        encoding="utf-8",
    )
    frontal = images_dir / "CXR42_IM-0001-1001.png"
    lateral = images_dir / "CXR42_IM-0001-2001.png"
    frontal.write_bytes(b"frontal-stand-in")
    lateral.write_bytes(b"lateral-stand-in")

    studies = load_studies(root)

    assert len(studies) == 1
    study = studies[0]
    assert len(study.image_paths) == 2
    assert {p.name for p in study.image_paths} == {"CXR42_IM-0001-1001.png", "CXR42_IM-0001-2001.png"}
    # Deterministic ordering (sorted by path), not incidental dict order.
    assert study.image_paths == sorted(study.image_paths)
