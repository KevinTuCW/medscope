"""Generate the committed PHI eval fixture at data/evals/phi.json.

Plants synthetic PHI (medscope.data.synth_phi) into 20 deterministic
seeds x a small pool of clinical report templates, and records:
  - the DICOM attributes injected (PatientName, PatientID, ...)
  - the report text before/after injection
  - the ground-truth PhiItem list for both

Phase 3's G4 suite loads this file directly and grades deid.py /
scan_payload against the ground truth it contains -- see the design note
at the top of medscope/deid.py for why the ground truth must come from the
injector rather than be recovered by pattern-matching afterwards.

Usage:
    PYTHONPATH=src python scripts/gen_phi_fixture.py

Regenerate whenever synth_phi's value pools or phrasing change -- the
output is deterministic for fixed seeds, so an unrelated code change should
reproduce byte-identical output.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydicom.dataset import Dataset

from medscope.data.synth_phi import inject_phi, inject_phi_text

_OUTPUT_PATH = Path("data/evals/phi.json")
_N_SAMPLES = 20

# Keep in sync with the DICOM tags medscope.data.synth_phi.inject_phi
# writes to (PatientName, PatientID, PatientBirthDate, AccessionNumber,
# InstitutionName, ReferringPhysicianName).
_DICOM_PHI_KEYWORDS = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "AccessionNumber",
    "InstitutionName",
    "ReferringPhysicianName",
)

# A small pool of clinical report templates covering English and Chinese
# prose, plain findings and numbered impressions, and a measurement -- the
# same shapes test_deid.test_clinical_content_preserved checks survive
# scrubbing untouched.
_TEMPLATES = [
    "Lungs are clear without focal consolidation. No pleural effusion or pneumothorax. Heart size is normal.",
    "There is a 2.5 cm nodule in the right upper lobe. No cardiomegaly. Osseous structures are intact.",
    "Cardiomegaly without lung infiltrates. Mild tortuosity of the aorta noted.",
    "右侧胸腔积液，左肺纹理清晰。心影大小正常。未见气胸征象。",
    "双肺纹理增粗，未见明确实变影。心影不大。膈肌光滑。",
    "Mild interstitial edema versus atypical infectious process. No acute cardiopulmonary process.",
    "The mediastinal contour is within normal limits. No visible pneumothorax or pleural fluid.",
    "1. Cardiomegaly. 2. Tortuous aorta. Otherwise unremarkable chest radiograph.",
]


def _fresh_dataset() -> Dataset:
    ds = Dataset()
    ds.Modality = "CR"
    ds.Rows = 512
    ds.Columns = 512
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.ViewPosition = "PA"
    ds.PatientSex = "F"
    ds.PatientAge = "045Y"
    return ds


def generate() -> list[dict]:
    samples = []
    for seed in range(_N_SAMPLES):
        template = _TEMPLATES[seed % len(_TEMPLATES)]

        ds, dicom_items = inject_phi(_fresh_dataset(), seed=seed)
        text_after, text_items = inject_phi_text(template, seed=seed)

        dicom_attributes = {keyword: str(getattr(ds, keyword)) for keyword in _DICOM_PHI_KEYWORDS}
        ground_truth = [item.model_dump() for item in (*dicom_items, *text_items)]

        samples.append(
            {
                "seed": seed,
                "report_text_before": template,
                "report_text_after": text_after,
                "dicom_attributes": dicom_attributes,
                "ground_truth": ground_truth,
            }
        )
    return samples


def main() -> None:
    samples = generate()
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(samples)} samples to {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
