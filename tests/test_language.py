"""Gate G3: diagnostic-language redlines.

An AI-generated draft may describe what it sees; it may not diagnose, exclude,
or prescribe. Those are acts a licensed physician performs and signs for. The
rules here are the machine-checkable expression of that boundary.

The false-positive tests matter as much as the detection tests: a redline
detector that flags ordinary radiological hedging would neutralise every real
sentence in the report, turning a useful draft into mush while scoring 100% on
every detection test.
"""

import pytest

from medscope.language import (
    REQUIRED_DISCLAIMER,
    detect_redlines,
    has_disclaimer,
    neutralize,
)
from medscope.state import ReportDraft, ReportSentence


# --- detection -------------------------------------------------------------


@pytest.mark.parametrize(
    "text,kind",
    [
        ("确诊为肺炎。", "diagnostic"),
        ("明确诊断为左下肺炎症。", "diagnostic"),
        ("Diagnosis: pneumonia.", "diagnostic"),
        ("可排除气胸。", "exclusion"),
        ("完全除外肺栓塞。", "exclusion"),
        ("Pneumothorax is ruled out.", "exclusion"),
        ("建议服用阿莫西林。", "treatment"),
        ("建议立即手术治疗。", "treatment"),
        ("Prescribe amoxicillin 500mg.", "treatment"),
    ],
)
def test_detect_redline_kinds(text, kind):
    hits = detect_redlines(text)
    assert hits, f"expected a redline in {text!r}"
    assert kind in {h.kind for h in hits}


def test_zero_width_evasion_still_detected():
    """Zero-width characters must not smuggle a redline past the detector.

    Normalisation runs before matching, same as in deid.py.
    """
    assert detect_redlines("确​诊为肺炎。")


def test_fullwidth_evasion_still_detected():
    assert detect_redlines("Ｄｉａｇｎｏｓｉｓ: pneumonia.")


# --- false positives: real radiology prose must survive ---------------------


@pytest.mark.parametrize(
    "text",
    [
        "右下肺斑片影，考虑感染性病变可能，建议结合临床。",
        "双肺纹理增粗，未见明显实变。",
        "心影增大，主动脉迂曲。",
        "提示左侧胸腔积液，建议随访复查。",
        "未见明确气胸征象。",
        "Patchy opacity in the right lower lobe, likely infectious.",
        "No acute cardiopulmonary process.",
        "Findings are nonspecific; clinical correlation is recommended.",
        "建议进一步 CT 检查以明确。",
    ],
)
def test_clinical_hedging_is_not_a_redline(text):
    """These are exactly how a radiologist is supposed to write.

    'Suggest further CT' is a recommendation about imaging, not a prescription;
    'no definite pneumothorax' is a description of absence, not a clinical
    exclusion. A detector that cannot tell these apart is unusable.
    """
    assert detect_redlines(text) == []


# --- neutralisation --------------------------------------------------------


def test_neutralize_downgrades_certainty():
    out = neutralize("确诊为肺炎。")
    assert any(w in out for w in ("提示", "考虑"))
    assert detect_redlines(out) == []


def test_neutralize_is_idempotent():
    once = neutralize("可排除气胸。")
    assert neutralize(once) == once


def test_neutralize_leaves_clean_text_untouched():
    clean = "右下肺斑片影，考虑感染性病变可能。"
    assert neutralize(clean) == clean


# --- disclaimer ------------------------------------------------------------


def _draft(disclaimer: str) -> ReportDraft:
    return ReportDraft(
        sentences=[
            ReportSentence(text="胸部正位片。", section="technique", evidence_ids=[])
        ],
        disclaimer=disclaimer,
    )


def test_missing_disclaimer_detected():
    assert has_disclaimer(_draft("")) is False


def test_required_disclaimer_accepted():
    assert has_disclaimer(_draft(REQUIRED_DISCLAIMER)) is True


def test_disclaimer_must_name_both_ai_origin_and_physician_review():
    """A vague 'for reference only' is not the disclaimer we require.

    The two facts a reader must not miss are that a machine wrote it and that a
    physician has to sign it. Anything omitting either is rejected.
    """
    assert has_disclaimer(_draft("仅供参考。")) is False
