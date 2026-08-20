"""Tests for the synthetic PHI injector.

This module intentionally exists *before* deid.py, and its tests must pass
first. The scrubber (Task 1.3 Step B) is only meaningful if it is measured
against PHI we know we planted -- the already-de-identified OpenI sample
data has nothing left in it to find, so a scrubber tested only against that
data would pass even if it returned its input unchanged. The injector's own
returned `PhiItem` list is the ground truth `deid.py` gets graded against;
see the design note at the top of `medscope/deid.py`.
"""

from __future__ import annotations

from pydicom.dataset import Dataset

from medscope.data.synth_phi import PhiItem, inject_phi, inject_phi_text


def _fresh_dataset() -> Dataset:
    ds = Dataset()
    ds.Modality = "CR"
    ds.Rows = 512
    ds.Columns = 512
    return ds


def test_phi_item_model_shape():
    item = PhiItem(kind="patient_name", value="John Smith", location="PatientName")
    assert item.kind == "patient_name"
    assert item.value == "John Smith"
    assert item.location == "PatientName"


def test_inject_phi_is_deterministic_for_fixed_seed():
    ds_a, items_a = inject_phi(_fresh_dataset(), seed=42)
    ds_b, items_b = inject_phi(_fresh_dataset(), seed=42)

    assert str(ds_a) == str(ds_b)
    assert [i.model_dump() for i in items_a] == [i.model_dump() for i in items_b]


def test_inject_phi_differs_across_seeds():
    _ds_a, items_a = inject_phi(_fresh_dataset(), seed=1)
    _ds_b, items_b = inject_phi(_fresh_dataset(), seed=2)

    assert [i.value for i in items_a] != [i.value for i in items_b]


def test_inject_phi_every_item_appears_in_polluted_dataset():
    ds, items = inject_phi(_fresh_dataset(), seed=7)
    assert items, "expected at least one injected PHI item"

    serialized = str(ds)
    for item in items:
        assert item.value in serialized, f"{item.kind}={item.value!r} not found in polluted dataset"


def test_inject_phi_covers_expected_dicom_tags():
    ds, items = inject_phi(_fresh_dataset(), seed=3)
    kinds = {item.kind for item in items}
    assert kinds == {
        "patient_name",
        "patient_id",
        "birth_date",
        "accession",
        "institution",
        "physician_name",
    }
    # Every item's value actually landed on the DICOM tag it claims to.
    for item in items:
        assert str(getattr(ds, item.location)) == item.value


def test_inject_phi_uses_dicom_caret_convention_for_names():
    ds, items = inject_phi(_fresh_dataset(), seed=3)
    name_items = [i for i in items if i.kind in ("patient_name", "physician_name")]
    assert name_items
    assert all("^" in i.value for i in name_items), "DICOM PN values must use caret delimiter"


def test_inject_phi_ground_truth_is_complete_no_extra_leakage():
    # Nothing should appear in the serialized dataset's PHI-bearing tags that
    # isn't accounted for in the returned ground truth.
    ds, items = inject_phi(_fresh_dataset(), seed=17)
    reported_values = {item.value for item in items}
    for keyword in (
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "AccessionNumber",
        "InstitutionName",
        "ReferringPhysicianName",
    ):
        assert str(getattr(ds, keyword)) in reported_values


def test_inject_phi_text_is_deterministic_for_fixed_seed():
    base = "Lungs are clear. No pleural effusion."
    text_a, items_a = inject_phi_text(base, seed=42)
    text_b, items_b = inject_phi_text(base, seed=42)

    assert text_a == text_b
    assert [i.model_dump() for i in items_a] == [i.model_dump() for i in items_b]


def test_inject_phi_text_differs_across_seeds():
    base = "Lungs are clear. No pleural effusion."
    _text_a, items_a = inject_phi_text(base, seed=1)
    _text_b, items_b = inject_phi_text(base, seed=2)
    assert [i.value for i in items_a] != [i.value for i in items_b]


def test_inject_phi_text_every_item_appears_in_polluted_text():
    base = (
        "Lungs are clear without focal consolidation. "
        "No pleural effusion or pneumothorax. "
        "Heart size is normal."
    )
    text, items = inject_phi_text(base, seed=11)
    assert items, "expected at least one injected PHI item"
    for item in items:
        assert item.value in text, f"{item.kind}={item.value!r} not found in polluted text"


def test_inject_phi_text_preserves_original_clinical_sentences():
    base = (
        "Lungs are clear without focal consolidation. "
        "No pleural effusion or pneumothorax. "
        "Heart size is normal."
    )
    text, _items = inject_phi_text(base, seed=11)
    for sentence in (
        "Lungs are clear without focal consolidation.",
        "No pleural effusion or pneumothorax.",
        "Heart size is normal.",
    ):
        assert sentence in text


def test_inject_phi_text_interleaves_rather_than_appends_a_block():
    # A naive "prepend/append everything as one block" implementation would
    # put every injected value entirely before or entirely after all
    # original sentences. Require at least one injected value to land before
    # the last original sentence, proving real interleaving.
    base = (
        "Lungs are clear without focal consolidation. "
        "No pleural effusion or pneumothorax. "
        "Heart size is normal."
    )
    text, items = inject_phi_text(base, seed=11)
    last_original_pos = text.index("Heart size is normal.")
    assert any(text.index(item.value) < last_original_pos for item in items)


def test_inject_phi_text_includes_both_western_and_chinese_names():
    text, items = inject_phi_text("Lungs are clear.", seed=5)
    name_values = [i.value for i in items if i.kind in ("patient_name", "physician_name")]
    has_western = any(all(ord(ch) < 128 for ch in v) for v in name_values)
    has_chinese = any(any("一" <= ch <= "鿿" for ch in v) for v in name_values)
    assert has_western, f"expected a Western-shaped name among {name_values!r}"
    assert has_chinese, f"expected a Chinese-shaped name among {name_values!r}"


def test_inject_phi_text_chinese_names_span_multiple_lengths_and_compound_surnames():
    # A pool of exclusively 2-character names is exactly why a 3-4
    # character-name bug in deid.py stayed invisible for four rounds of
    # this task: the injector's vocabulary silently defined the tested
    # surface. Across a spread of seeds, Chinese-shaped names must include
    # 2-character, 3-character, and 4-character (compound-surname) forms.
    lengths = set()
    for seed in range(100):
        _text, items = inject_phi_text("Lungs are clear.", seed=seed)
        for item in items:
            if item.kind in ("patient_name", "physician_name") and any(
                "一" <= ch <= "鿿" for ch in item.value
            ):
                lengths.add(len(item.value))
    assert lengths >= {2, 3, 4}, f"expected 2/3/4-char Chinese names, saw lengths {lengths}"


def test_inject_phi_text_chinese_names_appear_both_delimited_and_running_into_prose():
    # A Chinese name is only genuinely stress-tested if it's planted both
    # ways real reports write it: terminated by punctuation in a
    # structured field ("姓名：张三，男") and running directly into more
    # clinical prose with no delimiter at all ("患者：欧阳明月入院复查。").
    # Only the second shape exercises deid.py's surname-anchor fallback.
    delimited_seen = prose_seen = False
    delimiters = "，。、；：,;:"
    for seed in range(100):
        text, items = inject_phi_text("Lungs are clear.", seed=seed)
        for item in items:
            if item.kind not in ("patient_name", "physician_name"):
                continue
            if not any("一" <= ch <= "鿿" for ch in item.value):
                continue
            idx = text.index(item.value)
            after = idx + len(item.value)
            following = text[after] if after < len(text) else ""
            if following in delimiters:
                delimited_seen = True
            elif "一" <= following <= "鿿":
                prose_seen = True
    assert delimited_seen, "never saw a Chinese name terminated by a delimiter"
    assert prose_seen, "never saw a Chinese name running directly into more CJK prose"


def test_inject_phi_text_ground_truth_kinds_are_complete():
    text, items = inject_phi_text("Lungs are clear.", seed=9)
    reported_kinds = {i.kind for i in items}
    assert reported_kinds == {
        "patient_name",
        "patient_id",
        "birth_date",
        "national_id",
        "institution",
        "physician_name",
        "phone",
    }
    reported_values = {item.value for item in items}
    for item in items:
        assert item.value in reported_values
    # No orphaned duplicate content sneaked in beyond what's reported.
    assert len(items) == len(reported_kinds)


def test_inject_phi_text_exercises_multiple_format_variants_across_seeds():
    # A monoculture injector (one canonical shape per kind) only ever
    # proves the scrubber against that one shape -- see the design note in
    # deid.py. Across a spread of seeds, phone/date/MRN/name formatting
    # must actually vary, not just the values within one fixed template.
    phones, dates, mrns, name_shapes = set(), set(), set(), set()
    for seed in range(60):
        _text, items = inject_phi_text("Lungs are clear.", seed=seed)
        by_kind = {item.kind: item.value for item in items}
        phones.add(_phone_shape(by_kind["phone"]))
        dates.add(_date_shape(by_kind["birth_date"]))
        mrns.add(_mrn_shape(by_kind["patient_id"]))
        for name_kind in ("patient_name", "physician_name"):
            name_shapes.add(_name_shape(by_kind[name_kind]))

    assert len(phones) >= 3, f"expected multiple phone formats, saw {phones!r}"
    assert len(dates) >= 3, f"expected multiple date formats, saw {dates!r}"
    assert len(mrns) >= 2, f"expected both MRN forms (prefixed/bare), saw {mrns!r}"
    assert "western_plain" in name_shapes and "western_comma" in name_shapes, name_shapes


def test_inject_phi_text_exercises_named_axes_across_seeds():
    # Restructured after a second adversarial review found leaks that
    # weren't new PHI kinds, just formatting axes no hand-enumerated
    # variant list had named: a phone country-code prefix, a space
    # separator, a dot-separated date, and lowercase ID prefixes. The
    # injector now samples separator/prefix/case/script axes
    # independently rather than choosing among a fixed list of pre-baked
    # concrete shapes, so each axis's values must actually appear
    # independently across a seed sweep -- not just co-occur inside one
    # hardcoded variant string.
    phone_separators, phone_prefixed = set(), False
    date_separators = set()
    mrn_cases, mrn_prefixes = set(), set()
    name_scripts = set()

    for seed in range(300):
        text, items = inject_phi_text("Lungs are clear.", seed=seed)
        by_kind: dict[str, list[str]] = {}
        for item in items:
            by_kind.setdefault(item.kind, []).append(item.value)

        for phone in by_kind.get("phone", []):
            if phone.startswith(("+86", "0086", "86")):
                phone_prefixed = True
            for sep in ("-", ".", " ", "－", "．"):
                if sep in phone:
                    phone_separators.add(sep)
            if not any(c in phone for c in "-.－． ()（）"):
                phone_separators.add("")  # bare, no separator at all

        for dob in by_kind.get("birth_date", []):
            for sep in ("-", "/", ".", "－", "／", "．"):
                if sep in dob:
                    date_separators.add(sep)

        for mrn in by_kind.get("patient_id", []):
            if mrn.startswith("MRN"):
                mrn_cases.add("upper")
                mrn_prefixes.add("MRN")
            elif mrn[:3].lower() == "mrn":
                mrn_cases.add("lower_or_mixed")
                mrn_prefixes.add("mrn")
            elif mrn.startswith("No."):
                mrn_prefixes.add("No.")
            else:
                mrn_prefixes.add("")  # bare digits, labeled externally

        for name_kind in ("patient_name", "physician_name"):
            for name in by_kind.get(name_kind, []):
                if any(ip in name for ip in "·・"):
                    if any("一" <= ch <= "鿿" for ch in name):
                        name_scripts.add("chinese_interpunct")
                    else:
                        name_scripts.add("western_interpunct")

    assert phone_separators == {"", "-", ".", " ", "－", "．"}, phone_separators
    assert phone_prefixed, "expected at least one country-code-prefixed phone"
    assert date_separators == {"-", "/", ".", "－", "／", "．"}, date_separators
    assert mrn_cases == {"upper", "lower_or_mixed"}, mrn_cases
    assert mrn_prefixes == {"MRN", "mrn", "No.", ""}, mrn_prefixes
    assert name_scripts == {"chinese_interpunct", "western_interpunct"}, name_scripts


def test_inject_phi_text_exercises_glued_and_separated_cjk_placement_across_seeds():
    # Chinese is written with no space between a label and the value that
    # follows it, so a label glued directly onto an identifier
    # ("身份证110101...") is the *normal* case in real Chinese report text,
    # not an edge case. An injector that only ever inserts a colon or a
    # space before a value can never exercise deid.py's CJK-adjacency
    # handling (see _bounded() there) -- the fixtures would look complete
    # while silently never covering the condition that actually leaks.
    # Across a spread of seeds, each of these kinds must appear BOTH glued
    # directly onto a Han-character label with zero separator AND with an
    # explicit separator (colon/space/English label).
    kinds = ("national_id", "birth_date", "institution", "phone")
    glued_seen = {k: False for k in kinds}
    separated_seen = {k: False for k in kinds}

    for seed in range(80):
        text, items = inject_phi_text("Lungs are clear.", seed=seed)
        by_kind: dict[str, list[str]] = {}
        for item in items:
            by_kind.setdefault(item.kind, []).append(item.value)
        for kind in kinds:
            for value in by_kind.get(kind, []):
                idx = text.index(value)
                preceding = text[idx - 1] if idx > 0 else ""
                if "一" <= preceding <= "鿿":
                    glued_seen[kind] = True
                else:
                    separated_seen[kind] = True

    for kind in kinds:
        assert glued_seen[kind], f"{kind}: never saw a CJK-glued (zero-separator) placement across 80 seeds"
        assert separated_seen[kind], f"{kind}: never saw a separated placement across 80 seeds"


def test_inject_phi_text_mrn_prefixed_glue_placement_varies():
    # The self-labeling "MRN0483921" form doesn't need a separator to be
    # identifiable, so it should be exercised glued to a Chinese label on
    # both sides (leading "住院号MRN0483921" and trailing "MRN0483921号"),
    # not just left bare with punctuation around it.
    leading_glue_seen = trailing_glue_seen = plain_seen = False
    for seed in range(80):
        text, items = inject_phi_text("Lungs are clear.", seed=seed)
        mrn_value = next(i.value for i in items if i.kind == "patient_id")
        if not mrn_value.startswith("MRN"):
            continue  # "bare" MRN variant, not the self-labeling one under test
        idx = text.index(mrn_value)
        before = text[idx - 1] if idx > 0 else ""
        after = text[idx + len(mrn_value)] if idx + len(mrn_value) < len(text) else ""
        if "一" <= before <= "鿿":
            leading_glue_seen = True
        elif "一" <= after <= "鿿":
            trailing_glue_seen = True
        else:
            plain_seen = True

    assert leading_glue_seen, "never saw a leading-CJK-glued prefixed MRN"
    assert trailing_glue_seen, "never saw a trailing-CJK-glued prefixed MRN"
    assert plain_seen, "never saw a plain (non-glued) prefixed MRN"


def _phone_shape(value: str) -> str:
    if value.startswith("("):
        return "us_paren"
    if "-" in value and value.split("-")[0].startswith("1") and len(value.split("-")[0]) == 3:
        return "cn_dashed"
    if "-" in value:
        return "us_dashed"
    return "cn_bare"


def _date_shape(value: str) -> str:
    if "年" in value:
        return "chinese"
    if "-" in value:
        return "iso"
    if "/" in value:
        return "slash_mdy" if value.index("/") <= 2 and len(value.split("/")[-1]) == 4 else "slash_ymd"
    return "compact8"


def _mrn_shape(value: str) -> str:
    return "prefixed" if value.startswith("MRN") else "bare"


def _name_shape(value: str) -> str:
    if any("一" <= ch <= "鿿" for ch in value):
        return "chinese"
    return "western_comma" if "," in value else "western_plain"
