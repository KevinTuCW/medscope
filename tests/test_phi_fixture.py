"""Tests for the committed PHI eval fixture (data/evals/phi.json).

Phase 3's G4 suite consumes this file directly, so it's checked here for
shape and internal consistency: the ground truth each sample carries must
actually be scrubbable by the current deid.py, and the polluted text/DICOM
attributes must actually contain what the ground truth claims. Regenerate
with `python scripts/gen_phi_fixture.py` (see that file's docstring).
"""

from __future__ import annotations

import json
from pathlib import Path

from medscope.deid import deid_text, scan_payload

FIXTURE_PATH = Path("data/evals/phi.json")


def _load_fixture() -> list[dict]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_fixture_file_exists_with_twenty_samples():
    assert FIXTURE_PATH.exists(), f"missing {FIXTURE_PATH} -- run scripts/gen_phi_fixture.py"
    samples = _load_fixture()
    assert len(samples) == 20


def test_fixture_samples_have_required_shape():
    for sample in _load_fixture():
        assert "seed" in sample
        assert "report_text_before" in sample
        assert "report_text_after" in sample
        assert "dicom_attributes" in sample
        assert "ground_truth" in sample
        assert sample["ground_truth"], "expected at least one ground-truth PHI item"
        for item in sample["ground_truth"]:
            assert set(item) == {"kind", "value", "location"}


def test_fixture_ground_truth_actually_appears_in_polluted_output():
    for sample in _load_fixture():
        haystack = sample["report_text_after"] + " " + " ".join(sample["dicom_attributes"].values())
        for item in sample["ground_truth"]:
            assert item["value"] in haystack, (
                f"seed {sample['seed']}: {item['kind']}={item['value']!r} not found in polluted output"
            )


def test_fixture_is_deid_solvable():
    # This is the point of the fixture: current deid.py must be able to
    # clear every sample's ground truth from both the report text and the
    # DICOM attribute values.
    for sample in _load_fixture():
        scrubbed_text, _ = deid_text(sample["report_text_after"])
        payload = {
            "report_text": scrubbed_text,
            "dicom_attributes": {k: v for k, v in sample["dicom_attributes"].items()},
        }
        # DICOM attributes here are raw (un-deid_dicom'd) values -- phi.json
        # stores what was injected, not a scrubbed dataset. Scrub the text
        # representation the same way deid_text would for a cloud payload.
        scrubbed_attrs = {k: deid_text(v)[0] for k, v in sample["dicom_attributes"].items()}
        payload["dicom_attributes"] = scrubbed_attrs

        leaked = scan_payload(payload, _as_phi_items(sample["ground_truth"]))
        assert leaked == [], f"seed {sample['seed']}: {[i.value for i in leaked]} survived scrubbing"


def _as_phi_items(raw_items: list[dict]):
    from medscope.data.synth_phi import PhiItem

    return [PhiItem(**item) for item in raw_items]
