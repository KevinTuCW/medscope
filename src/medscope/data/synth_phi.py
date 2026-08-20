"""Synthetic PHI injector -- manufactures the pollution that gate G4 exists
to catch.

The OpenI dataset (`medscope.data.openi`) is already de-identified. If
`medscope.deid` were built and tested only against that data, it would pass
no matter what it did -- including a scrubber that returns its input
unchanged, because there is no PHI left in that corpus to find. That would
make G4 a tautology: permanently green, detecting nothing.

This module exists to fix that. `inject_phi` / `inject_phi_text` plant fake
but realistically-*shaped* patient identifiers into an in-memory DICOM
`Dataset` and into report prose, and return the ground-truth list of
exactly what they planted. `medscope.deid` is then graded against that
list, not against its own idea of what it removed -- see the design note
at the top of `deid.py` for why the ground truth must come from here and
not be recovered after the fact by pattern-matching the polluted output.

Shape matters more than realism: a scrubber that only catches
`"John Smith"` but misses `"ZHANG^SAN"` (DICOM's caret-delimited
`Family^Given` convention) or a Chinese name in report prose isn't doing
its job. Every call to either function plants at least one Western-shaped
and one Chinese-shaped name for that reason.

`inject_phi_text` also varies the *format* it uses for a given kind
(phone, date, MRN, name), chosen deterministically from the seed, rather
than always emitting one canonical shape. A monoculture injector is a
narrower version of the same tautology this module exists to avoid: if
the injector only ever plants dash-separated phone numbers, then a
scrubber whose phone rule requires a dash will pass every test while
missing the single most common way a Chinese mobile number is actually
written (11 bare digits). The injector and scrubber quietly agreeing on
one formatting convention is exactly as misleading as them quietly
agreeing on one literal value. See `deid.py` for the rules this format
diversity is meant to keep honest.

It also varies *placement*, not just format: national_id, birth_date,
institution, and the self-labeling MRN form are each sometimes embedded
with an explicit English label and separator ("ID: 1101011990...") and
sometimes glued directly onto a Chinese label with zero separator
("身份证1101011990..."). Chinese is written with no space between a label
and the value that follows it, so the glued form is the *normal* case for
Chinese prose, not an edge case -- an injector that only ever inserts a
colon or space before a value can never exercise (or protect against a
regression in) deid.py's CJK-adjacency boundary handling, since that bug
only bites when there is no separator at all.

Restructured as a COMBINATORIAL generator over named axes, rather than a
hand-written list of variants per PHI kind. A second adversarial review
found four more leaks (a phone country-code prefix, a space-separated
phone, a dot-separated date, a lowercase MRN prefix) plus two more via a
follow-up probe (an interpunct-joined name, a transliterated Chinese
name) -- none of them a new PHI kind, all of them a formatting dimension
the prior variant lists never named. A hand-enumerated list of shapes is
permanently a subset of reality, no matter how many rounds it grows
through; naming the axis instead ("what separators exist" is a question
someone can answer completely; "list every way a phone number is
written" is not) means one addition applies to every value of that kind
at once. The axes sampled independently here, per call:

  - separator: none / `-` / `.` / ` ` / full-width `－` / full-width `．`
    (phone, `_PHONE_SEPARATORS`), plus `/` and full-width `／` for dates
    (`_DATE_SEPARATORS`).
  - prefix: a phone country code (none/`+86`/`0086`/`86`,
    `_PHONE_CN_PREFIXES`), or an ID prefix (`MRN`/`mrn`/`Mrn`/`No.`,
    `_MRN_SELF_LABEL_PREFIXES`).
  - case: folded into the prefix axis above -- MRN's case varies
    independently of every other axis.
  - script/join style: for names, space / `Last, First` / interpunct
    (`·`/`・`) for Western names, and surname+given-name / interpunct-
    joined transliteration for Chinese-script names (`_random_prose_name`).
  - placement: separated-by-label or glued directly onto a Chinese label
    with zero separator (`_glued_or_labeled`, `_name_clause`).

`medscope.deid` composes its rules from the SAME named axes (see that
module's docstring) rather than one literal regex per shape, so the two
modules' coverage is driven by the same small, reviewable axis list
instead of two independently hand-maintained variant enumerations that
can silently drift apart.

Known gaps this generator deliberately does NOT plant, matching what
`deid.py` deliberately does not attempt to catch (see that module's
docstring for the reasoning behind each): bare unlabeled digit runs with
no separator/prefix/context; Chinese numeral dates (二〇一九年); hyphenated
national ID numbers; a bare unseparated 10-digit US-style phone number
(the injector always forces at least one separator for that style,
specifically so it never emits an identifier deid.py has no rule for).
"""

from __future__ import annotations

import random
import re
from typing import Literal

from pydantic import BaseModel
from pydicom.dataset import Dataset

PhiKind = Literal[
    "patient_name",
    "patient_id",
    "birth_date",
    "accession",
    "institution",
    "physician_name",
    "phone",
    "national_id",
]


class PhiItem(BaseModel):
    kind: PhiKind
    value: str  # the literal string injected -- what deid.py must remove
    location: str  # DICOM tag keyword, or report field name it went into


# -- fake-but-shaped value pools -------------------------------------------
# DICOM PN (Person Name) values use the caret-delimited Family^Given
# convention regardless of whether the underlying name is Western or a
# pinyin transliteration -- that's the DICOM shape a scrubber must catch,
# independent of which script/culture the name comes from.
_WESTERN_SURNAMES = ["SMITH", "JOHNSON", "GARCIA", "MILLER", "DAVIS", "WALKER", "BROWN", "WILSON"]
_WESTERN_GIVEN = ["JOHN", "MARY", "JAMES", "PATRICIA", "ROBERT", "LINDA", "MICHAEL", "BARBARA"]
_CHINESE_SURNAMES_PINYIN = ["ZHANG", "WANG", "LI", "LIU", "CHEN", "YANG", "ZHAO", "HUANG"]
_CHINESE_GIVEN_PINYIN = ["SAN", "WEI", "FANG", "MING", "LEI", "JUAN", "QIANG", "YAN"]

# Report prose gets actual Western full names and actual Chinese characters
# (not pinyin) -- prose PHI doesn't carry DICOM's caret convention, but it
# does need to exercise both scripts.
_WESTERN_FIRST = ["John", "Mary", "James", "Patricia", "Robert", "Linda", "Michael", "Barbara"]
_WESTERN_LAST = ["Smith", "Johnson", "Garcia", "Miller", "Davis", "Walker", "Brown", "Wilson"]
# Deliberately spans 2/3/4-character names, including compound surnames
# (司马/欧阳/上官/诸葛) -- a pool of exclusively 2-character names is exactly
# what let deid.py's Chinese-name matching leak on 3-4 character names for
# several rounds: the injector's own vocabulary silently defined the
# tested surface (see the exchange pinned in test_deid.py's
# test_chinese_name_is_fully_redacted_never_a_fragment). Every surname
# used here must exist in deid.py's own surname list -- see
# test_inject_phi_text_chinese_names_span_multiple_lengths_and_compound_surnames.
_CHINESE_NAMES_HANZI = [
    "张三", "李明", "王芳", "陈伟", "刘娟", "杨强", "赵磊", "黄燕", "周涛", "吴敏",  # 2-char
    "王小明", "李文博", "陈家豪", "黄雅婷",  # 3-char, single surname + 2-char given name
    "欧阳明月", "司马相如", "上官婉儿", "诸葛亮华",  # 4-char, compound surname + 2-char given name
]
# A Chinese transliteration of a foreign name ("玛丽·居里" = Marie Curie) is
# a structurally distinct script, not a longer surname+given-name entry --
# 玛丽/居里 aren't Chinese surnames, so deid.py's surname-anchored rule
# can't and shouldn't catch these; they need their own interpunct-joined
# pattern (see _TRANSLITERATED_NAME_VAL in deid.py).
_TRANSLITERATED_FIRST = ["玛丽", "安娜", "伊丽莎白", "苏珊"]
_TRANSLITERATED_LAST = ["居里", "史密斯", "约翰逊", "威尔逊"]
# U+00B7 MIDDLE DOT and U+30FB KATAKANA MIDDLE DOT -- both used in
# practice in place of a space when joining name components.
_INTERPUNCT_CHARS = ("·", "・")

_INSTITUTIONS = [
    "Riverside Memorial Hospital",
    "Lakeside General Hospital",
    "St. Augustine Medical Center",
    "Cedar Valley Clinic",
    "北京仁和医院",
    "上海惠民医院",
    "广州协和诊所",
    "天河区中心医院",
]

_DICOM_PHI_TAGS: tuple[tuple[PhiKind, str], ...] = (
    ("patient_name", "PatientName"),
    ("patient_id", "PatientID"),
    ("birth_date", "PatientBirthDate"),
    ("accession", "AccessionNumber"),
    ("institution", "InstitutionName"),
    ("physician_name", "ReferringPhysicianName"),
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _random_dicom_name(rng: random.Random, *, chinese: bool) -> str:
    if chinese:
        surname, given = rng.choice(_CHINESE_SURNAMES_PINYIN), rng.choice(_CHINESE_GIVEN_PINYIN)
    else:
        surname, given = rng.choice(_WESTERN_SURNAMES), rng.choice(_WESTERN_GIVEN)
    return f"{surname}^{given}"


def _random_prose_name(rng: random.Random, *, chinese: bool) -> str:
    """A name as it would appear in report prose.

    Script/join axis, sampled independently of which script this call was
    asked for: Western names are "First Last", "Last, First", or
    interpunct-joined ("First·Last"); Chinese-script names are either a
    real surname+given-name (deid.py's surname-anchored rule) or a
    Chinese transliteration of a foreign name, interpunct-joined
    ("玛丽·居里") -- a structurally distinct shape, not a longer name pool
    (see _TRANSLITERATED_FIRST/_LAST above).
    """
    if chinese:
        if rng.choice((True, True, True, False)):
            return rng.choice(_CHINESE_NAMES_HANZI)
        interpunct = rng.choice(_INTERPUNCT_CHARS)
        return f"{rng.choice(_TRANSLITERATED_FIRST)}{interpunct}{rng.choice(_TRANSLITERATED_LAST)}"

    first, last = rng.choice(_WESTERN_FIRST), rng.choice(_WESTERN_LAST)
    style = rng.choice(("space", "comma", "interpunct"))
    if style == "comma":
        return f"{last}, {first}"
    if style == "interpunct":
        return f"{first}{rng.choice(_INTERPUNCT_CHARS)}{last}"
    return f"{first} {last}"


def _random_dicom_date(rng: random.Random) -> str:
    year, month, day = rng.randint(1940, 2005), rng.randint(1, 12), rng.randint(1, 28)
    return f"{year:04d}{month:02d}{day:02d}"


# Separator axis (independent of field order): none (compact), hyphen,
# slash, dot, and their full-width forms. "compact"/"chinese" are their
# own styles below since a separator-less \d{1,2} month/day would be
# ambiguous (deid.py handles compact8 and 年月日 as their own patterns for
# the same reason).
_DATE_SEPARATORS = ("-", "/", ".", "－", "／", "．")
_DATE_ORDERS = ("ymd", "mdy")


def _random_prose_date(rng: random.Random) -> str:
    """A date as it would appear in report prose. Separator and field-order
    axes are sampled independently of each other and of the "compact
    (no separator)"/"Chinese 年月日" alternative styles, so e.g. a dot
    separator crosses with both orderings rather than being hand-listed
    as its own variant per combination.
    """
    year, month, day = rng.randint(1940, 2005), rng.randint(1, 12), rng.randint(1, 28)
    style = rng.choice(("separated", "separated", "separated", "compact", "chinese"))
    if style == "compact":
        return f"{year:04d}{month:02d}{day:02d}"
    if style == "chinese":
        return f"{year:04d}年{month}月{day}日"
    sep = rng.choice(_DATE_SEPARATORS)
    order = rng.choice(_DATE_ORDERS)
    if order == "ymd":
        return f"{year:04d}{sep}{month:02d}{sep}{day:02d}"
    return f"{month:02d}{sep}{day:02d}{sep}{year:04d}"


_MRN_BARE_LABELS = ("MRN: ", "Medical Record Number: ", "病案号：", "住院号：")
_MRN_CJK_LEADING_LABELS = ("住院号", "病案号")
_MRN_CJK_TRAILING_SUFFIXES = ("号", "档")


_MRN_SELF_LABEL_PREFIXES = ("MRN", "mrn", "Mrn", "No.")


def _random_mrn(rng: random.Random) -> tuple[str, str]:
    """Returns (ground-truth value, how it's embedded in prose).

    Half the time the value is self-labeling -- a prefix (case axis: MRN/
    mrn/Mrn, plus the generic "No." prefix) baked directly onto the
    digits -- for that half, it's further glued directly onto a Chinese
    label with zero separator about a third of the time (leading or
    trailing), since the self-labeling form doesn't need one. The other
    half of the time it's bare digits with an explicit label word in the
    surrounding prose instead (always separated -- an unlabeled,
    unseparated run of bare digits is indistinguishable from an arbitrary
    clinical number, so deid.py deliberately requires a separator there;
    see the "unlabelled Chinese name" trade-off note in deid.py for the
    same reasoning applied to names).
    """
    digits = f"{rng.randint(1000000, 9999999)}"
    if rng.choice([True, False]):
        value = f"{rng.choice(_MRN_SELF_LABEL_PREFIXES)}{digits}"
        placement = rng.choice(("plain", "plain", "cjk_leading", "cjk_trailing"))
        if placement == "cjk_leading":
            return value, f"{rng.choice(_MRN_CJK_LEADING_LABELS)}{value}"
        if placement == "cjk_trailing":
            return value, f"{value}{rng.choice(_MRN_CJK_TRAILING_SUFFIXES)}"
        return value, value
    label = rng.choice(_MRN_BARE_LABELS)
    return digits, f"{label}{digits}"


_NATIONAL_ID_CJK_LABELS = ("身份证", "身份证号", "证件号")
_DOB_CJK_LABELS = ("出生日期", "生日", "日期")
_PHONE_CJK_LABELS = ("电话", "联系电话", "手机")
_INSTITUTION_CJK_LABELS = ("就诊于", "来自")


def _glued_or_labeled(rng: random.Random, value: str, *, en_label: str, cjk_labels: tuple[str, ...]) -> str:
    """Either an English label + separator before `value`, or a Chinese
    label glued directly onto it with zero separator. See the module
    docstring for why both need exercising, not just the separated form.
    """
    if rng.choice([True, False]):
        return f"{en_label}{value}"
    return f"{rng.choice(cjk_labels)}{value}"


_NAME_PROSE_CONTINUATIONS = (
    "入院复查。",
    "情况良好，建议随访。",
    "既往体健，无特殊病史。",
)
_CHINESE_PATIENT_LABELS = ("患者", "姓名")
_CHINESE_PHYSICIAN_LABELS = ("医生", "医师")


def _name_clause(rng: random.Random, name: str, *, en_prefix: str, cn_labels: tuple[str, ...]) -> str:
    """A name-bearing clause. Varies whether the name sits behind an
    English or a Chinese context label, and -- for the Chinese label --
    whether the name is delimiter-terminated (a structured field shape,
    "姓名：张三，详见下列记录。") or runs directly into more clinical prose
    with no separator at all ("患者：欧阳明月入院复查。"). Both are real
    report shapes; deid.py's Chinese-name rule must redact the whole name
    either way and never leave a fragment behind -- see
    test_chinese_name_is_fully_redacted_never_a_fragment in test_deid.py.
    """
    placement = rng.choice(("english", "chinese_delimited", "chinese_prose"))
    if placement == "chinese_prose":
        return f"{rng.choice(cn_labels)}：{name}{rng.choice(_NAME_PROSE_CONTINUATIONS)}"
    if placement == "chinese_delimited":
        return f"{rng.choice(cn_labels)}：{name}，详见下列记录。"
    return f"{en_prefix}{name}."


_PHONE_SEPARATORS = ("", "-", ".", " ", "－", "．")
_PHONE_CN_PREFIXES = ("", "+86", "0086", "86")


def _random_phone(rng: random.Random) -> str:
    """A phone number. Separator axis (none/hyphen/dot/space and
    full-width forms) and, for the CN style, a country-code prefix axis
    (none/+86/0086/86) are each sampled independently -- their
    cross-product covers shapes like "+8613812345678" (prefix, no
    separator) and "138 1234 5678" (no prefix, space separator) without
    either being its own hand-listed variant. A bare 11-digit CN mobile
    with no prefix and no separator is included deliberately (rng
    happening to draw sep="" and prefix="" together) -- that's the most
    common real-world way to write one, and an injector that only ever
    produces separated numbers would never exercise deid.py's
    context-anchored rule for the unseparated, unprefixed form.
    """
    style = rng.choice(("cn", "us_dashed", "us_paren"))
    sep = rng.choice(_PHONE_SEPARATORS)
    if style == "cn":
        prefix = rng.choice(_PHONE_CN_PREFIXES)
        second = rng.choice("35678")
        g1 = f"1{second}{rng.randint(0, 9)}"
        g2 = f"{rng.randint(0, 9999):04d}"
        g3 = f"{rng.randint(0, 9999):04d}"
        prefix_join = sep if prefix else ""
        return f"{prefix}{prefix_join}{g1}{sep}{g2}{sep}{g3}"
    if style == "us_dashed":
        return f"{rng.randint(200, 999)}{sep or '-'}{rng.randint(200, 999)}{sep or '-'}{rng.randint(1000, 9999)}"
    return f"({rng.randint(200, 999)}){sep or ' '}{rng.randint(200, 999)}-{rng.randint(1000, 9999)}"


def _random_national_id(rng: random.Random) -> str:
    """A Chinese resident-ID-shaped string: 6-digit region code + 8-digit
    birth date + 3-digit sequence + 1 check character (digit or X).
    """
    year, month, day = rng.randint(1940, 2005), rng.randint(1, 12), rng.randint(1, 28)
    region = "110101"
    birth = f"{year:04d}{month:02d}{day:02d}"
    seq = f"{rng.randint(0, 999):03d}"
    check = rng.choice("0123456789X")
    return f"{region}{birth}{seq}{check}"


def inject_phi(dataset: Dataset, seed: int) -> tuple[Dataset, list[PhiItem]]:
    """Write fake PHI onto `dataset`'s standard identifying tags.

    Deterministic given `seed`. Always plants one Western-shaped and one
    Chinese-shaped `Family^Given` name (assigned to patient/physician in a
    random order) so both forms are exercised on every call.
    """
    rng = random.Random(seed)

    patient_is_chinese = rng.choice([True, False])
    values: dict[str, str] = {
        "PatientName": _random_dicom_name(rng, chinese=patient_is_chinese),
        "PatientID": f"MRN{rng.randint(1000000, 9999999)}",
        "PatientBirthDate": _random_dicom_date(rng),
        "AccessionNumber": f"ACC{rng.randint(10000000, 99999999)}",
        "InstitutionName": rng.choice(_INSTITUTIONS),
        "ReferringPhysicianName": _random_dicom_name(rng, chinese=not patient_is_chinese),
    }

    for keyword, value in values.items():
        setattr(dataset, keyword, value)

    items = [PhiItem(kind=kind, value=values[keyword], location=keyword) for kind, keyword in _DICOM_PHI_TAGS]
    return dataset, items


def inject_phi_text(report_text: str, seed: int) -> tuple[str, list[PhiItem]]:
    """Insert fake PHI into report prose, interleaved with the original
    sentences rather than appended as a single block.

    Deterministic given `seed`. Plants a demographics clause up front (a
    patient name, MRN, and DOB -- the way real reports open), a comparison
    sentence referencing an institution partway through, and a
    physician/contact sentence at the end. Patient and physician names are
    always one Western-shaped and one Chinese-shaped, in random order.
    """
    rng = random.Random(seed)

    patient_is_chinese = rng.choice([True, False])
    patient_name = _random_prose_name(rng, chinese=patient_is_chinese)
    physician_name = _random_prose_name(rng, chinese=not patient_is_chinese)
    mrn_value, mrn_fragment = _random_mrn(rng)
    dob = _random_prose_date(rng)
    national_id = _random_national_id(rng)
    institution = rng.choice(_INSTITUTIONS)
    phone = _random_phone(rng)

    items = [
        PhiItem(kind="patient_name", value=patient_name, location="report:demographics_line"),
        PhiItem(kind="patient_id", value=mrn_value, location="report:demographics_line"),
        PhiItem(kind="birth_date", value=dob, location="report:demographics_line"),
        PhiItem(kind="national_id", value=national_id, location="report:demographics_line"),
        PhiItem(kind="institution", value=institution, location="report:comparison_sentence"),
        PhiItem(kind="physician_name", value=physician_name, location="report:physician_sentence"),
        PhiItem(kind="phone", value=phone, location="report:physician_sentence"),
    ]

    dob_fragment = _glued_or_labeled(rng, dob, en_label="DOB: ", cjk_labels=_DOB_CJK_LABELS)
    national_id_fragment = _glued_or_labeled(rng, national_id, en_label="ID: ", cjk_labels=_NATIONAL_ID_CJK_LABELS)
    phone_fragment = _glued_or_labeled(rng, phone, en_label="Contact: ", cjk_labels=_PHONE_CJK_LABELS)

    patient_clause = _name_clause(rng, patient_name, en_prefix="Patient: ", cn_labels=_CHINESE_PATIENT_LABELS)
    physician_clause = _name_clause(rng, physician_name, en_prefix="Dr. ", cn_labels=_CHINESE_PHYSICIAN_LABELS)

    demographics_line = f"{patient_clause} {mrn_fragment}. {dob_fragment}. {national_id_fragment}."
    physician_sentence = f"{physician_clause} {phone_fragment}."

    if rng.choice([True, False]):
        comparison_sentence = f"Comparison exam performed at {institution} was reviewed."
    else:
        comparison_sentence = f"{rng.choice(_INSTITUTION_CJK_LABELS)}{institution}的既往检查已复核。"

    sentences = [s for s in _SENTENCE_SPLIT_RE.split(report_text.strip()) if s] if report_text.strip() else []
    if not sentences:
        polluted = " ".join([demographics_line, comparison_sentence, physician_sentence])
        return polluted, items

    mid = max(1, len(sentences) // 2)
    parts = [demographics_line, *sentences[:mid], comparison_sentence, *sentences[mid:], physician_sentence]
    polluted = " ".join(parts)
    return polluted, items
