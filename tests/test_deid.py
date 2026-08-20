"""Tests for the PHI scrubber.

Graded against ground truth from `medscope.data.synth_phi`, not against
OpenI's already-de-identified sample data -- that corpus has nothing left
in it to find, so a scrubber tested only against it would pass no matter
what it did. See the design note at the top of `deid.py`.

`test_clinical_content_preserved` is the reverse check: a `deid_text` that
returns the empty string would score a perfect 100% on every other test in
this file. OpenI's own de-identification pass corrupted "chest x-ray" into
"chest x-XXXX" (see medscope.data.openi._normalize_xxxx) -- over-redaction
is a real, common failure mode and it's invisible unless you check for it
directly.
"""

from __future__ import annotations

import base64

from pydicom.dataset import Dataset

from medscope.data.openi import load_studies
from medscope.data.synth_phi import inject_phi, inject_phi_text
from medscope.deid import DICOM_ALLOWLIST, deid_dicom, deid_text, scan_payload

SAMPLES_ROOT = "data/samples/studies"


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
    ds.PixelData = b"\x00\x01\x02\x03"
    return ds


def test_dicom_whitelist():
    ds, ground_truth = inject_phi(_fresh_dataset(), seed=13)
    scrubbed, _report = deid_dicom(ds)

    serialized = str(scrubbed)
    for item in ground_truth:
        assert item.value not in serialized, f"{item.kind}={item.value!r} leaked past deid_dicom"

    kept_keywords = {elem.keyword for elem in scrubbed}
    assert kept_keywords <= DICOM_ALLOWLIST
    # Sanity: allowlisted imaging tags actually survived scrubbing.
    assert "Rows" in kept_keywords
    assert "Modality" in kept_keywords


def test_dicom_whitelist_is_allowlist_not_blocklist():
    # A private/vendor tag medscope has never heard of must still be
    # dropped -- proving DICOM_ALLOWLIST is opt-in, not an enumerated
    # blocklist of "known bad" tags.
    ds = _fresh_dataset()
    ds.add_new(0x00410010, "LO", "ACME VENDOR SECRET PATIENT LINKAGE ID")
    scrubbed, report = deid_dicom(ds)

    assert "ACME VENDOR SECRET" not in str(scrubbed)
    assert len(report["dropped_tags"]) >= 1


def test_text_phi():
    base_report = (
        "Lungs are clear without focal consolidation. "
        "No pleural effusion or pneumothorax. "
        "Heart size is normal."
    )
    polluted, ground_truth = inject_phi_text(base_report, seed=21)
    scrubbed, _report = deid_text(polluted)

    for item in ground_truth:
        assert item.value not in scrubbed, f"{item.kind}={item.value!r} leaked past deid_text"


def test_text_phi_across_many_seeds_including_cjk_glued_placement():
    # A single fixed seed can't guarantee it exercises the CJK-glued
    # (zero-separator) placement inject_phi_text now sometimes chooses --
    # sweep enough seeds to cover both glued and separated placements for
    # every kind, and confirm deid_text clears every one, not just the
    # always-separated shapes a narrower sweep might land on.
    base_report = "The lungs are clear. No cardiomegaly. No pleural effusion."
    for seed in range(40):
        polluted, ground_truth = inject_phi_text(base_report, seed=seed)
        scrubbed, _report = deid_text(polluted)
        for item in ground_truth:
            assert item.value not in scrubbed, (
                f"seed {seed}: {item.kind}={item.value!r} leaked past deid_text in {scrubbed!r}"
            )


def test_chinese_names_from_injector_never_leave_a_residual_character():
    # Stronger than "the whole name string is gone": checks contiguous
    # 2-character fragments of every Chinese-shaped injected name (not
    # just the full string) -- a check on the full string alone would not
    # have caught "姓名：[REDACTED:name]明" (the exact partial-redaction bug
    # this was written against), since "王小明" the whole string is indeed
    # absent from that output even though "明" survived.
    #
    # 2-character fragments, not single characters: a lone character can
    # coincidentally reuse a common Han glyph from unrelated surviving
    # text with no connection to the name at all -- caught during
    # development, "伊丽莎白・史密斯" was fully and correctly redacted, but
    # a single-character check flagged "史" as "surviving" purely because
    # the continuation phrase "...无特殊病史。" ("no notable medical
    # history") happens to end in that character too. A coincidental
    # 2-character run is implausible enough that this stays a meaningful
    # fragment check without that false-positive mode.
    for seed in range(200):
        text, items = inject_phi_text("Lungs are clear.", seed=seed)
        scrubbed, _report = deid_text(text)
        for item in items:
            if item.kind not in ("patient_name", "physician_name"):
                continue
            if not any("一" <= ch <= "鿿" for ch in item.value):
                continue  # Western-shaped name, not the case under test
            for i in range(len(item.value) - 1):
                fragment = item.value[i : i + 2]
                assert fragment not in scrubbed, (
                    f"seed {seed}: fragment {fragment!r} of Chinese name {item.value!r} "
                    f"survived in {scrubbed!r}"
                )


def test_cjk_glued_identifiers_are_still_caught():
    # `\b` treats Han characters as \w, so it does NOT assert a boundary
    # between a Chinese label and an adjacent digit/letter -- and Chinese
    # is written with no space there, so "label glued directly to value"
    # is the normal case, not an edge case. Every \b-anchored rule leaked
    # silently under this condition; this pins the fix across every rule
    # family, not just the ones first found by probing. Includes both the
    # "glued to the left" (label immediately before value) and "glued to
    # the right" (value immediately before trailing prose) directions --
    # \b fails symmetrically on both sides.
    cases = [
        ("身份证110101199003074512", "110101199003074512"),  # national_id
        ("检查号ACC1234567", "ACC1234567"),  # accession
        ("日期2019-11-04", "2019-11-04"),  # date, iso
        ("住院号MRN0483921号", "MRN0483921"),  # patient_id, glued both sides
        ("姓名ZHANG^SAN情况良好", "ZHANG^SAN"),  # DICOM caret name, glued both sides
        ("联系电话138-1234-5678", "138-1234-5678"),  # phone, cn_dashed
        ("费用555-123-4567请核对", "555-123-4567"),  # phone, us_dashed
        ("就诊于Riverside Memorial Hospital报告", "Riverside Memorial Hospital"),  # institution
    ]
    for text, value in cases:
        scrubbed, _report = deid_text(text)
        assert value not in scrubbed, f"{value!r} survived in {scrubbed!r} (CJK-adjacent \\b failure)"


def test_fullwidth_parens_phone_caught():
    # A phone number wrapped in full-width parentheses (（）, U+FF08/09)
    # rather than ASCII ones -- a natural choice when the surrounding text
    # is otherwise full-width-punctuated Chinese prose. _FULLWIDTH_TRANS
    # only folds Latin letters/digits (see its own comment for why it
    # can't be blanket NFKC), so this needs the phone pattern itself to
    # accept both paren shapes.
    scrubbed, _report = deid_text("（555）123-4567联系")
    assert "555" not in scrubbed
    assert "123-4567" not in scrubbed


def test_chinese_name_is_fully_redacted_never_a_fragment():
    # A prior fix made the Chinese-name value pattern lazy ({2,4}?) to stop
    # "患者：张三入院复查。" over-matching into "入院" ("admitted to hospital").
    # That traded over-redaction for something strictly worse: partial
    # redaction. "姓名：王小明" (3-char name) came out as
    # "姓名：[REDACTED:name]明" -- 明 leaked AND the output reads as
    # corrupted garbage, failing G4 while also degrading the clinical text
    # fed to the VLM. The invariant: within a context-anchored match,
    # either redact the COMPLETE name or don't match at all -- never emit
    # a fragment. Met via two mechanisms, not a length guess: a delimiter
    # terminator for structured fields ("姓名：张三，男") and a common-surname
    # anchor (including compound surnames) for prose that runs directly
    # into more text with no delimiter ("患者：欧阳明月入院复查。"). The
    # surname anchor is safe specifically because it only ever applies
    # behind a context anchor -- clinical prose never writes "患者：陈旧性".
    exact_cases = [
        ("姓名：王小明，男，45岁。", "姓名：[REDACTED:name]，男，45岁。"),  # 3-char, delimiter-terminated
        ("姓名：司马相如，主治医师签发。", "姓名：[REDACTED:name]，主治医师签发。"),  # compound surname, delimited
        ("患者：欧阳明月入院复查。", "患者：[REDACTED:name]入院复查。"),  # compound surname, prose (no delimiter)
    ]
    for text, expected in exact_cases:
        scrubbed, _report = deid_text(text)
        assert scrubbed == expected, f"{text!r} -> {scrubbed!r}, expected {expected!r}"


def test_chinese_single_char_given_name_glued_to_prose_is_a_disclosed_edge_case():
    # "张三" -- surname + a SINGLE-character given name -- immediately
    # followed by more clinical prose with no delimiter anywhere reachable
    # is genuinely ambiguous: nothing in the text distinguishes "the given
    # name is 1 character, the rest is unrelated prose" from "the given
    # name is 2 characters". _GIVEN_NAME_VAL resolves that ambiguity by
    # preferring 2 characters when no delimiter settles it -- correct for
    # every one of the reported examples (all of which have 2-character
    # given names) and for the statistically more common case. The PHI
    # itself is never left exposed either way (the full name is always
    # consumed), but this specific shape -- short given name, zero
    # delimiter, more CJK text immediately after -- can consume one
    # adjacent clinical character along with it. Confirmed structural, not
    # specific to this string: the same one-character spillover reproduces
    # with any 1-char given name in this shape (see deid.py's comment
    # above _GIVEN_NAME_VAL). Disclosed here rather than silently shipped.
    scrubbed, _report = deid_text("患者：张三入院复查。")
    assert "张三" not in scrubbed, "the name itself must never survive"
    assert scrubbed == "患者：[REDACTED:name]院复查。"


def test_reported_by_context_caught():
    cases = [
        "Reported by Donnelly, Margaret",
        "Read by Chen, Wei",
        "Dictated by Smith, John",
        "Signed by Garcia, Maria",
    ]
    for text in cases:
        scrubbed, _report = deid_text(text)
        assert "," in text and text.split("by ", 1)[1] not in scrubbed, (
            f"reporting physician name survived in {scrubbed!r}"
        )


def test_phone_format_variants_all_caught():
    # Pins team-lead's independent adversarial probe: the injector only
    # ever produced dash/paren-separated numbers, so the bare 11-digit CN
    # mobile form (the most common real-world way to write one) went
    # untested and unmatched. These are the exact miss cases from that
    # probe, plus the two that were already caught.
    cases = [
        ("联系电话：13812345678", "13812345678"),
        ("Phone: 13812345678", "13812345678"),
        ("电话 138-1234-5678", "138-1234-5678"),
        ("Call 555-123-4567", "555-123-4567"),
        ("Contact: (555) 123-4567", "(555) 123-4567"),
    ]
    for text, value in cases:
        scrubbed, _report = deid_text(text)
        assert value not in scrubbed, f"{value!r} survived in {scrubbed!r}"


def test_axis_generalized_shapes_are_caught():
    # A second independent adversarial review found four more leaks --
    # not new PHI kinds, but formatting axes the hand-enumerated variant
    # lists never named: a country-code phone prefix, a space-separated
    # phone, a dot-separated date, and lowercase alphabetic ID prefixes.
    # deid.py's rules are now composed from named axes (separator/prefix/
    # case/script -- see the module docstring) instead of one literal
    # regex per concrete shape, specifically so a whole axis is fixed at
    # once rather than shape-by-shape.
    cases = [
        ("Phone: +8613812345678", "13812345678"),  # prefix axis: country code
        ("Phone: 138 1234 5678", "138 1234 5678"),  # separator axis: space
        ("DOB: 2019.11.04", "2019.11.04"),  # separator axis: dot
        ("MRN: mrn0483921", "0483921"),  # case axis: lowercase prefix
        ("住院号mrn0483921", "0483921"),  # case axis, glued to CJK label
    ]
    for text, value in cases:
        scrubbed, _report = deid_text(text)
        assert value not in scrubbed, f"{value!r} survived in {scrubbed!r}"


def test_axis_generalized_name_scripts_are_caught():
    # Same review: an interpunct-joined name ("Mary·Curie", the middle dot
    # U+00B7 sometimes used in place of a space) and a Chinese
    # transliteration of a foreign name, also interpunct-joined
    # ("玛丽·居里" -- Marie Curie). The transliterated-name shape doesn't
    # fit the surname-anchored Chinese-name rule at all (玛丽/居里 aren't
    # Chinese surnames); it's a structurally distinct script variant, not
    # a longer name pool.
    cases = [
        ("Patient: Mary·Curie presented", "Mary·Curie"),
        ("患者：玛丽·居里，女", "玛丽·居里"),
    ]
    for text, value in cases:
        scrubbed, _report = deid_text(text)
        assert value not in scrubbed, f"{value!r} survived in {scrubbed!r}"


def test_bare_phone_rule_requires_context_and_does_not_eat_other_numbers():
    # The bare-digit phone rule is context-anchored specifically so it
    # can't start eating measurements or accession-shaped numbers that
    # happen to run 11 digits long with no phone label nearby.
    text = "Accession reference 12345678901 confirmed. There is a 2.5 cm nodule."
    scrubbed, _report = deid_text(text)
    assert "12345678901" in scrubbed
    assert "2.5 cm nodule" in scrubbed


def test_mrn_bare_digit_variant_caught():
    cases = [
        "MRN: 0483921",
        "Medical Record Number: 0483921",
        "病案号：0483921",
        "住院号：0483921",
    ]
    for text in cases:
        scrubbed, _report = deid_text(text)
        assert "0483921" not in scrubbed, f"MRN survived in {scrubbed!r}"


def test_name_last_comma_first_variant_caught():
    scrubbed, _report = deid_text("Patient: Smith, John presented for follow-up.")
    assert "Smith, John" not in scrubbed
    assert "presented for follow-up." in scrubbed


def test_date_slash_mdy_variant_caught():
    scrubbed, _report = deid_text("DOB: 03/12/1965.")
    assert "03/12/1965" not in scrubbed


def test_national_id_caught():
    base_report = "The lungs are clear."
    polluted, ground_truth = inject_phi_text(base_report, seed=9)
    national_id_items = [i for i in ground_truth if i.kind == "national_id"]
    assert national_id_items, "expected inject_phi_text to plant a national_id item"

    scrubbed, _report = deid_text(polluted)
    for item in national_id_items:
        assert item.value not in scrubbed, f"national_id {item.value!r} survived scrubbing"


def test_general_date_rule_uses_neutral_date_kind_not_birth_date():
    # A date matched by the generic shape rules could be a comparison
    # study date, an admission date, anything -- deid_text has no way to
    # know it's a birth date just from the shape. Only the DICOM
    # PatientBirthDate tag (dropped wholesale by deid_dicom) carries that
    # semantic; the audit label here must not claim more than it knows.
    scrubbed, report = deid_text("Comparison study 2019-11-04 reviewed.")
    assert "2019-11-04" not in scrubbed
    assert "[REDACTED:date]" in scrubbed
    assert "birth_date" not in report["patterns_fired"]


def test_unlabelled_chinese_name_is_a_documented_gap_not_a_silent_one():
    # Deliberate trade-off, not an oversight: requiring an explicit label
    # (colon/whitespace after 患者/姓名/医生/医师/Patient/Dr.) before treating
    # a Chinese-shaped run as a name is what keeps test_clinical_content_
    # preserved passing -- a surname-list heuristic without that anchor
    # would false-positive on ordinary vocabulary like 陈旧性 ("chronic",
    # starts with the common surname 陈). The cost: a name glued directly
    # onto a role noun with no delimiter is missed. Pinned here as known,
    # not silently absent from the coverage matrix.
    text = "会诊医师欧阳明月认为需复查。"
    scrubbed, _report = deid_text(text)
    assert "欧阳明月" in scrubbed


def test_payload_scan():
    base_report = "The lungs are clear. No cardiomegaly. No pleural effusion."
    polluted, text_ground_truth = inject_phi_text(base_report, seed=5)
    ds, dicom_ground_truth = inject_phi(_fresh_dataset(), seed=5)
    ground_truth = text_ground_truth + dicom_ground_truth

    scrubbed_ds, _ = deid_dicom(ds)
    scrubbed_text, _ = deid_text(polluted)

    fake_image_b64 = base64.b64encode(b"not-a-real-image-but-realistic-shaped-blob" * 4).decode()
    payload = {
        "study": {
            "report_text": scrubbed_text,
            "dicom_summary": str(scrubbed_ds),
        },
        "images": [{"content_b64": fake_image_b64, "mime": "image/png"}],
    }

    leaked = scan_payload(payload, ground_truth)
    assert leaked == []


def test_payload_scan_detects_unscrubbed_leak():
    # Sanity check on scan_payload itself: it must not be a function that
    # always returns [] regardless of input.
    base_report = "The lungs are clear."
    polluted, ground_truth = inject_phi_text(base_report, seed=5)

    payload = {"study": {"report_text": polluted}}  # deliberately NOT scrubbed
    leaked = scan_payload(payload, ground_truth)

    assert leaked, "scan_payload should catch PHI in an unscrubbed payload"
    assert {item.kind for item in leaked} == {item.kind for item in ground_truth}


def test_unicode_evasion():
    zero_width_phone = "​".join("(555) 201-3344")
    fullwidth_name = "Ｐａｔｉｅｎｔ：　Ｊｏｈｎ　Ｓｍｉｔｈ"

    scrubbed_phone, _ = deid_text(f"Contact number on file: {zero_width_phone}.")
    scrubbed_name, _ = deid_text(fullwidth_name)

    assert "555" not in scrubbed_phone
    assert "3344" not in scrubbed_phone
    assert "John" not in scrubbed_name and "Smith" not in scrubbed_name


def test_deid_report():
    ds, _ = inject_phi(_fresh_dataset(), seed=2)
    _scrubbed_ds, dicom_report = deid_dicom(ds)

    assert "dropped_tags" in dicom_report
    assert "PatientName" in dicom_report["dropped_tags"]
    assert "before_hash" in dicom_report and "after_hash" in dicom_report
    assert dicom_report["before_hash"] != dicom_report["after_hash"]

    base_report = "Lungs are clear."
    polluted, _ = inject_phi_text(base_report, seed=2)
    _scrubbed_text, text_report = deid_text(polluted)

    assert "patterns_fired" in text_report
    assert text_report["patterns_fired"]
    assert "before_hash" in text_report and "after_hash" in text_report
    assert text_report["before_hash"] != text_report["after_hash"]


def test_clinical_content_preserved():
    samples = [
        "右侧胸腔积液，左肺纹理清晰，心影大小正常。",
        "Cardiomegaly without lung infiltrates.",
        "There is a 2.5 cm nodule in the right upper lobe.",
        # 陈 is a common surname, so this checks the name rule doesn't fire
        # on ordinary clinical vocabulary that happens to share a character
        # with it -- confirmed by team-lead's independent adversarial probe.
        "陈旧性肺结核，较 2 年前无明显变化。",
        "PA and lateral chest x-ray dated earlier.",
        # Re-pinned after replacing \b with CJK-safe lookarounds
        # ((?<![0-9A-Za-z]) / (?![0-9A-Za-z])) -- the lookaround change
        # widens what can match, so these confirm measurements and
        # accession-shaped numbers are still untouched. Also from
        # team-lead's probe.
        "Accession 13812345678 archived.",
        "ETT 12345678901 catalog ref.",
        "Compared with study 12/2018.",
        "SpO2 98% on 2 L nasal cannula.",
    ]
    for text in samples:
        scrubbed, _report = deid_text(text)
        assert scrubbed == text, f"clinical content mangled: {text!r} -> {scrubbed!r}"

    for study in load_studies(SAMPLES_ROOT):
        for field_name in ("findings_text", "impression_text", "indication"):
            text = getattr(study, field_name)
            if not text:
                continue
            scrubbed, _report = deid_text(text)
            assert scrubbed == text, (
                f"study {study.study_id}.{field_name} clinical content mangled: "
                f"{text!r} -> {scrubbed!r}"
            )
