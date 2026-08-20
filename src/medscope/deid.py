"""PHI scrubber -- the implementation behind gate G4 (zero PHI leakage in
any cloud-bound payload).

Design note, read before touching the rules below:

The OpenI dataset (`medscope.data.openi`) is already de-identified, so it
was deliberately NOT used to develop or test this module. Testing a
scrubber only against already-clean data lets a no-op scrubber pass --
there's nothing left in that corpus to catch. Instead, `tests/test_deid.py`
plants known PHI with `medscope.data.synth_phi.inject_phi` /
`inject_phi_text` first, and grades this module against the ground-truth
`PhiItem` list the injector returns. The ground truth is produced by the
injector's own record of what it wrote, not recovered afterwards by
pattern-matching the polluted text -- if the same regex both created the
answer key and graded the exam, whatever it missed would be invisible on
both sides at once.

`DICOM_ALLOWLIST` is an allowlist, not a blocklist, on purpose. DICOM has
thousands of tags, private vendor tags routinely carry identifiers, and no
blocklist can enumerate what it doesn't know about -- a blocklist is
guaranteed to leak eventually. Keep only what imaging/reading actually
needs; everything else is dropped by default.

`deid_text` normalizes (full-width Latin letters/digits to ASCII, see
`_FULLWIDTH_TRANS`) and strips zero-width characters BEFORE pattern
matching, specifically so a full-width or zero-width-joined identifier
can't be smuggled past the rules in a different Unicode form. This is
narrower than a blanket `unicodedata.normalize("NFKC", ...)` on purpose --
full NFKC also compatibility-maps full-width CJK punctuation onto ASCII
look-alikes, which rewrites ordinary Chinese clinical prose (caught by
`test_clinical_content_preserved`; see `_FULLWIDTH_TRANS`'s comment).
`scan_payload` applies the same normalization when checking whether a
ground-truth value survived, so it can't be fooled by the same trick
either -- it is G4's scoring function; Phase 3's eval consumes it directly.

Every rule here matches on *shape* (context keyword + name pattern, a
caret-delimited DICOM person-name convention, a date format, an
institution-name suffix, ...), not on the literal values `synth_phi`
happens to generate -- otherwise this module would only ever catch its own
test fixtures. The one test that keeps this module honest in the other
direction is `test_clinical_content_preserved`: OpenI's own
de-identification pass once corrupted "chest x-ray" into "chest x-XXXX"
(see `medscope.data.openi._normalize_xxxx`) by being too aggressive.
Over-redaction is just as real a defect as under-redaction, and a
`deid_text` that returns the empty string would otherwise score 100% on
every other test in this file.

Rules are composed from named AXES, not written one literal regex per
concrete shape. A second adversarial review found leaks (a phone country
code, a space-separated phone, a dot-separated date, lowercase ID
prefixes, an interpunct-joined name) that weren't new PHI kinds -- they
were formatting dimensions no hand-enumerated variant list had named. A
list of shapes is always a strict subset of reality; an axis, once named,
applies to every value of that kind at once. The axes actually used here:

  - separator (`_SEP`): none, `-`, `/`, `.`, and their full-width forms,
    or a single whitespace char -- shared by phone and date.
  - prefix: a phone country code (`+86`/`0086`/`86`, `_PHONE_CN_PREFIXED`)
    or an ID prefix (`MRN`/`No.`, `_MRN_PREFIX`/`_ACC_PREFIX`).
  - case (`(?i:...)` scoped to just the prefix, e.g. `_MRN_PREFIX`):
    MRN/ACC/No. prefixes are matched case-insensitively -- digits aren't
    affected by case, so this doesn't widen the numeric part at all.
  - script/join style: a name joined by an interpunct (`·`/`・`,
    `_WESTERN_NAME_VAL_INTERPUNCT`) instead of a space, or a Chinese
    transliteration of a foreign name, itself interpunct-joined
    (`_TRANSLITERATED_NAME_VAL`) -- structurally distinct from the
    surname-anchored Chinese-name rule, not a longer name pool.
  - placement: a value separated from its context label, or glued
    directly onto a Chinese one with zero separator (`_bounded()`,
    `_MRN_CTX`/`_PHONE_CTX_GROUP` allowing an empty separator).

`medscope.data.synth_phi` samples these same axes independently per call
rather than choosing among a fixed list of pre-baked shapes, so their
cross-product is what gets tested, not a hand-picked subset of it --
see that module's docstring.

Ordering within `_TEXT_RULES` matters where two patterns could both match
a prefix of the same text: the more specific/longer pattern must run
first, or the shorter one wins and leaves a fragment. Two concrete
instances: compound surnames listed before single-character ones in
`_SURNAME_ALT` (so "欧阳" wins over "欧" at the same position), and the
transliterated-interpunct name rule placed before the surname-anchored
rule (some surnames, e.g. 苏, are also the first character of a
transliterated given name -- "苏珊·居里" would otherwise match only "苏珊"
and leave "·居里" behind).

Known unsupported PHI shapes
-----------------------------
The lesson this whole module is built on, stated plainly: a hand-
enumerated list of PHI shapes is permanently a subset of real-world
shapes, and a test suite built from that list can never be more
adversarial than the list itself. Every round of review against this
module found leaks that weren't new PHI *kinds* -- they were formatting
axes (a separator, a prefix, a case, a script/join style) nobody had
named yet. Naming an axis, not enumerating another shape, is what
closes a whole class of gap at once; see the AXES section above for how
that reshaped this module's rules. Four such gaps found and closed
during development, kept here as concrete illustration of the pattern
(the code paths handling them are `_PHONE_CN_PREFIXED`/`_SEP` for the
first two, `_MRN_PREFIX`'s `(?i:...)` for the third, and
`_WESTERN_NAME_VAL_INTERPUNCT`/`_TRANSLITERATED_NAME_VAL` for the
fourth -- pinned by `test_axis_generalized_shapes_are_caught` and
`test_axis_generalized_name_scripts_are_caught`):
  - `+86`-prefixed and space-separated CN mobiles ("+8613812345678",
    "138 1234 5678") -- no separator/prefix axis had been named.
  - a lowercase "mrn" prefix ("mrn0483921") -- no case axis had been
    named for ID prefixes.
  - dot-separated dates ("2019.11.04") -- the separator axis had only
    hyphen and slash named, not dot.
  - interpunct-joined transliterated names, both scripts ("Mary·Curie",
    "玛丽·居里") -- no script/join-style axis had been named for names.

What's still genuinely unsupported, and why -- named here deliberately
rather than left as an unenumerated gap (see
test_unlabelled_chinese_name_is_a_documented_gap_not_a_silent_one and
test_chinese_single_char_given_name_glued_to_prose_is_a_disclosed_edge_
case for the two pinned by tests):
  - A bare, unlabeled digit run with no context anchor, no prefix, and no
    separator is indistinguishable from an arbitrary clinical number
    (e.g. a bare 10-digit US-style phone). Deliberately not matched --
    the alternative is redacting measurements and accession numbers.
  - A Chinese name with no context label at all (no colon/whitespace
    after 患者/姓名/医生/医师), e.g. "会诊医师欧阳明月认为需复查。" -- requiring
    an explicit separator after the context keyword is what keeps
    ordinary vocabulary like "陈旧性" (starts with the common surname 陈)
    from false-positiving; the cost is a name glued directly onto a role
    noun with no delimiter is missed.
  - A single-character Chinese given name with no delimiter and more CJK
    text immediately after ("患者：张三入院复查。") is genuinely ambiguous --
    nothing distinguishes "1-char given name, rest is prose" from
    "2-char given name" without a delimiter to settle it; the fallback
    prefers 2 characters (correct for the statistically more common
    case), which can over-redact one adjacent clinical character.
  - Chinese numeral dates (e.g. "二〇一九年十一月四日") use a different
    digit script entirely; would need a CJK-numeral-to-Arabic mapping
    table, not implemented.
  - Hyphenated/segmented national ID numbers (e.g. "110101-19900307-4512")
    -- real Chinese resident IDs are essentially always written as one
    contiguous 18-character string in practice; adding hyphen tolerance
    would risk colliding with other hyphenated digit runs.
  - Case-insensitivity for Western person names or the institution
    "Hospital"/"Clinic" suffix rule is deliberately NOT applied, unlike
    the MRN/ACC prefixes: the leading-capital-letter requirement is the
    primary precision signal for those two rules (see
    test_clinical_content_preserved), and relaxing it would match
    ordinary lowercase English clinical prose. Case tolerance is safe for
    a 3-letter ID prefix; it is not safe for "the next word is
    capitalized."

More shapes than this list almost certainly exist and haven't been
found yet -- that is the point being made above, not a gap in this list.
Treat this module as "good against everything found so far," not "PHI-
proof."
"""

from __future__ import annotations

import hashlib
import re

from pydicom.dataset import Dataset

from medscope.data.synth_phi import PhiItem

# Only tags an imaging pipeline (CNN/VLM readers, the report writer) needs.
# Everything on a DICOM dataset that isn't in this set is dropped, no
# matter what it's called -- see the module docstring for why this must be
# an allowlist rather than an enumerated blocklist.
DICOM_ALLOWLIST: set[str] = {
    "Modality",
    "ImageType",
    "ViewPosition",
    "PatientOrientation",
    "PatientSex",
    "PatientAge",
    "BodyPartExamined",
    "Rows",
    "Columns",
    "PhotometricInterpretation",
    "SamplesPerPixel",
    "PlanarConfiguration",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "PixelSpacing",
    "WindowCenter",
    "WindowWidth",
    "RescaleIntercept",
    "RescaleSlope",
    "PixelData",
}

_ZERO_WIDTH_RE = re.compile("[​‌‍⁠﻿]")

# Deliberately NOT full unicodedata.normalize("NFKC", ...): NFKC's
# compatibility decomposition also maps full-width CJK punctuation (e.g.
# "，" U+FF0C, "。" U+FF01) onto their ASCII look-alikes, which silently
# rewrites ordinary Chinese clinical prose -- caught by
# test_clinical_content_preserved during development (NFKC turned
# "右侧胸腔积液，左肺..." into "...，"-comma-free "...,"-comma text). The
# actual evasion risk is narrower than "any full-width character": an
# attacker smuggling a Latin identifier past ASCII-anchored rules writes it
# in full-width *letters/digits* (e.g. "Ｊｏｈｎ Ｓｍｉｔｈ"). Map only
# that range, plus the full-width space used to pad it, and leave CJK
# punctuation untouched.
_FULLWIDTH_TRANS = str.maketrans(
    {chr(c): chr(c - 0xFEE0) for c in (*range(0xFF21, 0xFF3B), *range(0xFF41, 0xFF5B), *range(0xFF10, 0xFF1A))}
    | {"　": " "}
)

def _bounded(pattern: str) -> str:
    """Wrap a digit/ASCII-letter-run pattern so it doesn't match when glued
    directly onto another digit/letter on either side -- WITHOUT using
    `\\b`. `\\b` is defined purely on the \\w/\\W transition, and Han
    characters ARE \\w, so `\\b` asserts nothing at all between a Chinese
    label and an adjacent digit or Latin letter. Since Chinese is written
    with no space there, "label glued to value" is the *normal* case for
    Chinese prose, not an edge case -- every rule that used `\\b` leaked
    silently under exactly that condition (see
    test_cjk_glued_identifiers_are_still_caught). `(?<![0-9A-Za-z])` /
    `(?![0-9A-Za-z])` assert "not glued to an ASCII digit/letter" without
    that false assumption about CJK.
    """
    return f"(?<![0-9A-Za-z]){pattern}(?![0-9A-Za-z])"


# Context keywords that must precede a bare name/MRN/phone for it to be
# treated as PHI. Requiring context (rather than matching "two capitalized
# words" or "2-4 CJK characters" or "11 digits" anywhere) is what keeps
# this from redacting ordinary clinical prose or accession-shaped numbers
# -- see test_clinical_content_preserved and
# test_bare_phone_rule_requires_context_and_does_not_eat_other_numbers.
# Includes the reporting-physician phrasing radiology reports actually use
# ("Reported by"/"Read by"/"Dictated by"/"Signed by"/报告医师/审核医师), not
# just the patient-facing "Patient"/"Dr." forms.
# Known trade-off: a name/number with NO such label (e.g. glued directly
# onto a role noun with no delimiter, "医师欧阳明月") is missed by design --
# see test_unlabelled_chinese_name_is_a_documented_gap_not_a_silent_one.
_NAME_CTX = (
    r"(?:Patient|Dr\.?|Referring physician|Ordering physician|Attending physician"
    r"|Reported by|Read by|Dictated by|Signed by"
    r"|患者|姓名|医生|医师|报告医师|审核医师)[:：\s]+"
)
_WESTERN_NAME_VAL = r"[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)+"
_WESTERN_NAME_VAL_COMMA = r"[A-Z][a-zA-Z'\-]+,\s*[A-Z][a-zA-Z'\-]+"  # "Last, First"

# Chinese has no space between a name and the words that follow it, so
# neither greedy nor lazy {2,4} can know where a name ends: greedy
# over-matches into clinical prose ("张三入院" -> eating "入院", "admitted to
# hospital"); lazy under-matches a real 3-4 character name, LEAKING the
# rest as a fragment ("王小明" -> "[REDACTED:name]明" -- 明 survives, and
# the sentence reads as corrupted garbage). Partial redaction is strictly
# worse than either failure it "solves": it fails G4 (PHI leaked) AND
# corrupts the clinical text fed downstream. The fix is not a length
# guess -- it's two mechanisms that actually carry the information a bare
# length pattern doesn't have:
#   1. A delimiter terminator, for structured fields ("姓名：张三，男，45岁"):
#      match up to 2 given-name characters, but only accept a length that
#      is immediately followed by punctuation/whitespace/end-of-string.
#   2. A surname anchor, for prose with no delimiter ("患者：欧阳明月入院复查。"):
#      require the match to start with a real Chinese surname (including
#      compound ones like 欧阳/司马), then take up to 2 more characters as
#      the given name. This is safe specifically because it only ever
#      applies behind a context anchor -- clinical prose never writes
#      "患者：陈旧性", so the false-positive risk that rules out a bare
#      surname-anywhere heuristic doesn't exist here.
# Compound (2-character) surnames are listed before single-character ones
# in the alternation so e.g. "欧阳" is tried -- and wins -- before the
# single-character surname "欧" at the same position.
_COMPOUND_SURNAMES = (
    "欧阳", "司马", "上官", "诸葛", "东方", "皇甫", "尉迟", "公孙",
    "长孙", "宇文", "慕容", "令狐", "独孤", "南宫", "万俟", "拓跋",
)
_SINGLE_SURNAMES = (
    "李", "王", "张", "刘", "陈", "杨", "黄", "赵", "周", "吴", "徐", "孙", "胡", "朱",
    "高", "林", "何", "郭", "马", "罗", "梁", "宋", "郑", "谢", "韩", "唐", "冯", "于",
    "董", "萧", "程", "曹", "袁", "邓", "许", "傅", "沈", "曾", "彭", "吕", "苏", "卢",
    "蒋", "蔡", "贾", "丁", "魏", "薛", "叶", "阎", "余", "潘", "杜", "戴", "夏", "钟",
    "汪", "田", "任", "姜", "范", "方", "石", "姚", "谭", "廖", "邹", "熊", "金", "陆",
    "郝", "孔", "白", "崔", "康", "毛", "邱", "秦", "江", "史", "顾", "侯", "邵", "孟",
    "龙", "万", "段", "雷", "钱", "汤", "尹", "黎", "易", "常", "武", "乔", "贺", "赖",
    "龚", "文", "欧",
)
_SURNAME_ALT = "|".join(_COMPOUND_SURNAMES + _SINGLE_SURNAMES)
_NAME_DELIMITERS = r"，。、；：,;:\s"
# Given-name portion: prefer a length that lands exactly on a delimiter
# (or end of string) -- unambiguous when the field is structured. If no
# such length exists within reach (free-running prose), fall back to
# always taking 2 characters if available, matching real-world Chinese
# given-name length (1-2 characters, with 2 at least as common).
#
# Residual, disclosed gap: a SINGLE-character given name with no
# delimiter and more CJK text immediately after ("患者：张三入院复查。") is
# genuinely ambiguous -- nothing distinguishes "1-char given name, rest
# is prose" from "2-char given name" without a delimiter to settle it.
# The greedy-2 fallback gets this wrong by one character (it also
# consumes the first character of "入院"). The name itself is still
# always fully removed either way -- see
# test_chinese_single_char_given_name_glued_to_prose_is_a_disclosed_edge_case.
_GIVEN_NAME_VAL = (
    rf"(?:[一-鿿]{{1,2}}(?=[{_NAME_DELIMITERS}]|$))"
    rf"|(?:[一-鿿]{{1,2}})"
)
_CHINESE_NAME_VAL = rf"(?:{_SURNAME_ALT})(?:{_GIVEN_NAME_VAL})"

# Script axis: an interpunct (·/・) in place of a space is a real way both
# a Western name ("Mary·Curie") and a Chinese transliteration of a foreign
# name ("玛丽·居里", Marie Curie) get written. The transliterated-Chinese
# case is NOT a longer entry for the surname-anchored rule above -- 玛丽
# and 居里 aren't Chinese surnames, so it needs its own pattern, gated by
# the same context anchor for the same reason (a bare CJK-interpunct-CJK
# run could otherwise collide with enumerations elsewhere in prose).
_INTERPUNCT = "·・"
_WESTERN_NAME_VAL_INTERPUNCT = rf"[A-Z][a-zA-Z'\-]+[{_INTERPUNCT}][A-Z][a-zA-Z'\-]+"
_TRANSLITERATED_NAME_VAL = rf"[一-鿿]{{2,4}}[{_INTERPUNCT}][一-鿿]{{2,4}}"

# ID prefixes (MRN/accession self-labeling values): case axis, since a
# report field is not guaranteed to preserve the canonical uppercase form
# ("MRN"/"ACC"), and a "No." prefix, which is a generic but real way to
# introduce a record number. `(?i:...)` scopes case-insensitivity to just
# the prefix -- \d is unaffected by case regardless.
_MRN_PREFIX = r"(?i:MRN|No\.)"
_ACC_PREFIX = r"(?i:ACC|No\.)"

_MRN_CTX = r"(?i:(?:MRN|Medical Record(?: Number)?)|病案号|住院号)[:：\s]+"
_BARE_DIGITS_VAL = r"\d{6,10}"

# Separator axis, shared by phone and date: no separator, ASCII/full-width
# hyphen, ASCII/full-width slash, ASCII/full-width dot, or a single
# whitespace char. One class covers "138-1234-5678", "2019/11/04",
# "138.1234.5678", "138 1234 5678", and their full-width-punctuated
# equivalents, rather than a literal regex per separator character.
_SEP = r"[-－/／.．\s]"

_PHONE_CTX = r"(?i:Phone|Tel|Contact(?: number)?)|电话|联系电话|手机"
_PHONE_CTX_GROUP = rf"(?:{_PHONE_CTX})[:：\s]*"
# Prefix axis: a country code (+86 / 0086 / 86) in front of a CN mobile
# number is, on its own, distinctive enough to be context-free -- nothing
# else in a radiology report is preceded by a country code. Separators
# after the prefix and between digit groups are each independently
# optional (covers "+8613812345678" glued solid as well as
# "+86-138-1234-5678" fully separated).
_PHONE_CN_PREFIXED = rf"(?:\+86|0086|86){_SEP}?1[3-9]\d{_SEP}?\d{{4}}{_SEP}?\d{{4}}"
# Separator axis, no country code: BOTH gaps must carry some separator --
# that's what makes this context-free-safe. A fully bare, unseparated,
# unprefixed 11-digit run is indistinguishable from an arbitrary clinical
# number, so it still requires the context anchor below (_CN_MOBILE_VAL).
_PHONE_CN_SEPARATED = rf"1[3-9]\d{_SEP}\d{{4}}{_SEP}\d{{4}}"
_PHONE_US_SEPARATED = rf"\d{{3}}{_SEP}\d{{3}}{_SEP}\d{{4}}"
# Full-width parentheses (（）, U+FF08/09) alongside ASCII ones -- a
# natural choice in otherwise full-width-punctuated Chinese prose, and
# not something _FULLWIDTH_TRANS folds (see its comment for why that
# normalization is deliberately narrow).
_PHONE_US_PAREN = rf"[（(]\d{{3}}[）)]{_SEP}?\d{{3}}{_SEP}?\d{{4}}"
# Trailing-only guard: _PHONE_CTX_GROUP allows a zero-width separator
# (e.g. "Contact13812345678", context ending in the ASCII letter "t"), so
# this can't use the full `_bounded()` -- its leading lookbehind would
# wrongly reject that case. Only the tail needs protection against
# over-matching into more digits/letters.
_CN_MOBILE_VAL = r"1[3-9]\d{9}(?![0-9A-Za-z])"  # 11-digit Chinese mobile, unseparated, context-anchored


def _rule(kind: str, value_pattern: str, ctx_pattern: str | None = None) -> tuple[str, re.Pattern]:
    if ctx_pattern:
        pattern = re.compile(f"(?P<ctx>{ctx_pattern})(?P<value>{value_pattern})")
    else:
        pattern = re.compile(f"(?P<value>{value_pattern})")
    return kind, pattern


# Order matters a little (e.g. the labeled MRN/accession prefixes are
# self-contained in the value itself, so they can't collide with the bare
# 8-digit date fallback -- neither can match a substring of the other's
# digit run once `_bounded` is applied consistently). Each pattern targets
# a *shape*, not a literal fixture value.
#
# The generic date rules use the neutral "date" kind, not "birth_date":
# shape alone can't tell a date of birth apart from a comparison-study
# date or any other date in the report, so the audit label mustn't claim
# more than it knows. "birth_date" is reserved for the DICOM
# PatientBirthDate tag in deid_dicom, where the semantics are actually
# known from the tag itself.
_TEXT_RULES: list[tuple[str, re.Pattern]] = [
    _rule("name", _WESTERN_NAME_VAL, _NAME_CTX),
    _rule("name", _WESTERN_NAME_VAL_COMMA, _NAME_CTX),
    _rule("name", _WESTERN_NAME_VAL_INTERPUNCT, _NAME_CTX),
    # Transliterated-interpunct MUST run before the surname-anchored rule:
    # some surnames (e.g. 苏) are also the first character of a
    # transliterated given name ("苏珊·居里"), so the surname rule alone
    # would match just "苏珊" and leave "·居里" behind as a fragment --
    # exactly the failure this module exists to avoid (see
    # test_chinese_names_from_injector_never_leave_a_residual_character).
    # Same principle as ordering compound surnames before single-character
    # ones within _SURNAME_ALT: the more specific/longer pattern goes
    # first so it gets the chance to consume the whole identifier before a
    # shorter pattern can partially match a prefix of it.
    _rule("name", _TRANSLITERATED_NAME_VAL, _NAME_CTX),
    _rule("name", _CHINESE_NAME_VAL, _NAME_CTX),
    # DICOM's caret-delimited Family^Given convention, wherever it shows up
    # (e.g. a DICOM tag value serialized into a JSON payload as plain text).
    _rule("name", _bounded(r"[A-Z]{2,}\^[A-Z]{2,}(?:\^[A-Z]*)*")),
    _rule("patient_id", _bounded(rf"{_MRN_PREFIX}\d{{6,9}}")),
    _rule("patient_id", _bounded(_BARE_DIGITS_VAL), _MRN_CTX),
    _rule("accession", _bounded(rf"{_ACC_PREFIX}\d{{7,9}}")),
    _rule("national_id", _bounded(r"\d{17}[\dXx]")),
    _rule("date", _bounded(rf"\d{{4}}{_SEP}\d{{1,2}}{_SEP}\d{{1,2}}")),
    _rule("date", _bounded(rf"\d{{1,2}}{_SEP}\d{{1,2}}{_SEP}\d{{4}}")),
    _rule("date", r"\d{4}年\d{1,2}月\d{1,2}日"),
    _rule("date", _bounded(r"\d{8}")),
    _rule(
        "institution",
        _bounded(r"[A-Z][A-Za-z.\s]{2,40}?(?:Hospital|Medical Center|Clinic|Health System|Medical Group)"),
    ),
    _rule("institution", r"[一-鿿]{2,10}(?:医院|诊所|医疗中心)"),
    _rule("phone", _PHONE_US_PAREN),
    _rule("phone", _bounded(_PHONE_CN_PREFIXED)),
    _rule("phone", _bounded(_PHONE_CN_SEPARATED)),
    _rule("phone", _bounded(_PHONE_US_SEPARATED)),
    _rule("phone", _CN_MOBILE_VAL, _PHONE_CTX_GROUP),
]


def _normalize(text: str) -> str:
    """Fold full-width Latin letters/digits to ASCII and drop zero-width
    characters (see `_FULLWIDTH_TRANS` for why this isn't full NFKC).

    Must run before any pattern matching (in deid_text) or literal
    comparison (in scan_payload) -- otherwise a full-width identifier
    (e.g. "Ｊｏｈｎ") or a zero-width-joined one can smuggle PHI past both.
    """
    return _ZERO_WIDTH_RE.sub("", text).translate(_FULLWIDTH_TRANS)


def deid_dicom(dataset: Dataset) -> tuple[Dataset, dict]:
    """Return a copy of `dataset` containing only allowlisted tags, plus an
    audit report of what was dropped.
    """
    before_hash = hashlib.sha256(str(dataset).encode("utf-8")).hexdigest()

    scrubbed = Dataset()
    dropped_tags: list[str] = []
    for elem in dataset:
        keyword = elem.keyword or f"tag{elem.tag}"
        if keyword in DICOM_ALLOWLIST:
            scrubbed.add(elem)
        else:
            dropped_tags.append(keyword)

    after_hash = hashlib.sha256(str(scrubbed).encode("utf-8")).hexdigest()
    report = {
        "dropped_tags": dropped_tags,
        "kept_tags": [elem.keyword for elem in scrubbed],
        "before_hash": before_hash,
        "after_hash": after_hash,
    }
    return scrubbed, report


def deid_text(text: str) -> tuple[str, dict]:
    """Normalize (see `_normalize`), then redact PHI-shaped substrings.
    Returns the scrubbed text plus an audit report.
    """
    before_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    result = _normalize(text)
    patterns_fired: list[str] = []

    for kind, pattern in _TEXT_RULES:
        def _replacement(match: re.Match, kind: str = kind) -> str:
            ctx = match.groupdict().get("ctx") or ""
            return f"{ctx}[REDACTED:{kind}]"

        result, count = pattern.subn(_replacement, result)
        if count:
            patterns_fired.append(kind)

    after_hash = hashlib.sha256(result.encode("utf-8")).hexdigest()
    report = {
        "patterns_fired": patterns_fired,
        "before_hash": before_hash,
        "after_hash": after_hash,
    }
    return result, report


def _iter_strings(payload) -> list[str]:
    """Recursively collect every string leaf out of a dict/list/str payload
    (a cloud-bound request body: text fields plus base64 image data).
    """
    strings: list[str] = []

    def _walk(node) -> None:
        if isinstance(node, dict):
            for value in node.values():
                _walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                _walk(value)
        elif isinstance(node, str):
            strings.append(node)

    _walk(payload)
    return strings


def scan_payload(payload: dict, ground_truth: list[PhiItem]) -> list[PhiItem]:
    """Return which `ground_truth` items are still present anywhere in
    `payload`. This is G4's scoring function -- Phase 3's eval consumes it
    directly, so it must not be fooled by the same Unicode tricks deid_text
    guards against.
    """
    normalized_blob = "\n".join(_normalize(s) for s in _iter_strings(payload))

    leaked = []
    for item in ground_truth:
        if _normalize(item.value) in normalized_blob:
            leaked.append(item)
    return leaked
