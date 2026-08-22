"""Shared label vocabulary for both readers (reader_a/CNN, reader_b/VLM).

Both readers emit `Finding.label` values that must land on the same
vocabulary, or `merge` (Task 1.6) can't tell agreement from disagreement --
two findings with different spellings of the same pathology would look like
a "unique" (one-reader-only) disagreement instead of the presence/magnitude
agreement they actually are.

`CANONICAL` is a synonym table, not a taxonomy: TorchXRayVision's own
pathology names, common English aliases, and Chinese clinical terms all map
onto one canonical name per pathology. The Chinese aliases are
load-bearing, not decoration -- reader_b (the VLM) emits free-form Chinese,
so an unmapped Chinese phrase silently becomes an untracked disagreement.

`CRITICAL_LABELS` is the single source of truth for gate G1
(critical-finding recall must be 1.0). It deliberately does not live in
`medscope.config.Settings` (see that module's docstring) -- two copies of
this list would inevitably drift in spelling and silently disable the
gate. It lists label *identity* only; "large" effusion vs. any effusion is
a magnitude judgment made downstream against `Settings.critical_threshold`,
not a separate ontology entry -- the ontology has no way to represent
severity that isn't already carried by `Finding.prob`.
"""

from __future__ import annotations

import re
import unicodedata

# TorchXRayVision's own pathology names (from `densenet121-res224-all`,
# verified at runtime via `model.pathologies` -- see readers/cnn.py; never
# hardcoded there, only mirrored here as alias source text), common English
# aliases, and Chinese clinical terms, all mapped to one canonical name.
CANONICAL: dict[str, str] = {
    # -- Atelectasis --
    "Atelectasis": "Atelectasis",
    "肺不张": "Atelectasis",
    # Open-i's own MeSH major term for this finding (330 studies) -- the
    # corpus indexes it with the anatomical qualifier, and an exact-match
    # lookup does not see through that.
    "Pulmonary Atelectasis": "Atelectasis",
    # -- Consolidation --
    "Consolidation": "Consolidation",
    "实变": "Consolidation",
    "肺实变": "Consolidation",
    # -- Infiltration --
    "Infiltration": "Infiltration",
    "浸润影": "Infiltration",
    "肺浸润": "Infiltration",
    # -- Pneumothorax --
    "Pneumothorax": "Pneumothorax",
    "气胸": "Pneumothorax",
    "左侧气胸": "Pneumothorax",
    "右侧气胸": "Pneumothorax",
    # -- Edema --
    "Edema": "Edema",
    "Pulmonary Edema": "Edema",
    "肺水肿": "Edema",
    # -- Emphysema --
    "Emphysema": "Emphysema",
    "肺气肿": "Emphysema",
    # -- Fibrosis --
    "Fibrosis": "Fibrosis",
    "Pulmonary Fibrosis": "Fibrosis",
    "肺纤维化": "Fibrosis",
    # -- PleuralEffusion (TorchXRayVision calls this "Effusion") --
    "Effusion": "PleuralEffusion",
    "Pleural Effusion": "PleuralEffusion",
    "胸腔积液": "PleuralEffusion",
    "右侧胸腔积液": "PleuralEffusion",
    "左侧胸腔积液": "PleuralEffusion",
    "胸水": "PleuralEffusion",
    # -- Pneumonia --
    "Pneumonia": "Pneumonia",
    "肺炎": "Pneumonia",
    # -- PleuralThickening --
    "Pleural_Thickening": "PleuralThickening",
    "Pleural Thickening": "PleuralThickening",
    "胸膜增厚": "PleuralThickening",
    # -- Cardiomegaly --
    "Cardiomegaly": "Cardiomegaly",
    "心影增大": "Cardiomegaly",
    "心脏扩大": "Cardiomegaly",
    # -- Nodule --
    "Nodule": "Nodule",
    "结节": "Nodule",
    "肺结节": "Nodule",
    # -- Mass --
    "Mass": "Mass",
    "肿块": "Mass",
    # -- Hernia --
    "Hernia": "Hernia",
    "疝": "Hernia",
    "膈疝": "Hernia",
    # -- LungLesion --
    "Lung Lesion": "LungLesion",
    "肺部病变": "LungLesion",
    # -- Fracture --
    # Site-qualified aliases are enumerated one by one rather than matched
    # by substring, and that restraint is the point: "subcutaneous
    # emphysema" contains "emphysema" but names soft-tissue air, not the
    # pulmonary disease `Emphysema` stands for -- a substring rule would
    # map it and hand the merge layer a confident false agreement.
    # Measured over 40 real reader_b studies, "rib fracture" (5 mentions)
    # and "肋骨骨折" (1) were the most common unmappable outputs for which
    # the ontology already had a home.
    "Fracture": "Fracture",
    "骨折": "Fracture",
    "Rib Fracture": "Fracture",
    "肋骨骨折": "Fracture",
    "Clavicle Fracture": "Fracture",
    "锁骨骨折": "Fracture",
    "Vertebral Fracture": "Fracture",
    "椎体骨折": "Fracture",
    # MeSH's inverted heading, as it appears in the corpus (89 studies).
    "Fractures, Bone": "Fracture",
    # -- LungOpacity --
    "Lung Opacity": "LungOpacity",
    "肺部阴影": "LungOpacity",
    "肺部密度增高影": "LungOpacity",
    # -- EnlargedCardiomediastinum --
    "Enlarged Cardiomediastinum": "EnlargedCardiomediastinum",
    "纵隔增宽": "EnlargedCardiomediastinum",
    # -- Pneumomediastinum (critical; NOT a densenet121-res224-all output --
    # see readers/cnn.py report / CRITICAL_LABELS note above) --
    "Pneumomediastinum": "Pneumomediastinum",
    # MeSH's own term for this finding on real OpenI reports (confirmed
    # against the corpus while building the critical-finding gold set --
    # see scripts/build_critical_goldset.py / data/evals/critical.json).
    "Mediastinal Emphysema": "Pneumomediastinum",
    "纵隔气肿": "Pneumomediastinum",
    "气纵隔": "Pneumomediastinum",
}

# Every canonical value must also be a valid lookup key, so `canonical()`
# round-trips on its own output (e.g. re-canonicalizing merge() output, or
# looking up a CRITICAL_LABELS entry directly). Built automatically rather
# than duplicated by hand above, so it can't drift out of sync.
for _value in set(CANONICAL.values()):
    CANONICAL.setdefault(_value, _value)
del _value

CRITICAL_LABELS: tuple[str, ...] = (
    "Pneumothorax",
    "PleuralEffusion",
    "Pneumomediastinum",
)

#: Labels on which the two readers can actually be compared.
#:
#: This is a **measurement**, not a decree, and it exists because the
#: alternative was measuring the wrong thing. Over 40 real studies
#: (scripts/calibrate_vlm.py --limit 40, Qwen3-VL-32B), reader_a called
#: 10.22 of its 18 labels positive per study -- LungOpacity on 40/40 --
#: while reader_b named 2.575 findings, of which only 5 distinct labels
#: ever landed on this ontology at all: Cardiomegaly (19 mentions),
#: LungOpacity (6), PleuralEffusion (2), Pneumonia (1), Pneumothorax (1),
#: plus Fracture once the site-qualified aliases above were added. Scoring
#: agreement across all 18 labels therefore graded reader_b on a
#: vocabulary it demonstrably does not use, and the resulting kappa
#: (-0.041) was mostly a statement about reader_a's calibration.
#:
#: The critical labels are unioned in unconditionally. A safety gate's
#: vocabulary must never be decided by what a model happened to say in a
#: 40-study sample -- narrowing that could quietly drop Pneumothorax from
#: comparison is the same move as deleting an inconvenient gold-set label.
#:
#: What this does NOT do: hide anything. Labels outside it still produce
#: findings, still reach `critical.triage` (which reads the raw
#: per-reader findings, never the merged set), and still get recorded as
#: out-of-vocabulary disagreements -- they just stop being counted as
#: evidence that two readers disagreed about something both of them read.
#: `merge.agreement` reports the wide kappa alongside the narrow one for
#: exactly this reason.
COMPARISON_LABELS: frozenset[str] = frozenset(
    {
        "Cardiomegaly",
        "LungOpacity",
        "PleuralEffusion",
        "Pneumonia",
        "Pneumothorax",
        "Fracture",
    }
    | set(CRITICAL_LABELS)
)

# Chinese full/half-width punctuation joins the ASCII set here because NFKC
# does not compatibility-map it onto ASCII look-alikes (unlike full-width
# Latin letters/digits, which NFKC does normalize).
_STRIP_RE = re.compile(r"[\s_\-.,;:!?'\"()/、，。！？；：（）]+")

def _normalize_key(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw)
    text = text.casefold()
    return _STRIP_RE.sub("", text)


_LOOKUP: dict[str, str] = {
    _normalize_key(alias): canonical_name for alias, canonical_name in CANONICAL.items()
}


def canonical(raw: str) -> str | None:
    """Map a raw label (English, Chinese, or already-canonical) to its
    canonical ontology name. Returns None for unknown labels.

    Callers must record an unknown label as a `unique`-kind Disagreement
    (see medscope.state.Disagreement) rather than dropping it silently --
    an unmapped label is exactly the case where one reader saw something
    the other didn't.
    """
    if not raw:
        return None
    return _LOOKUP.get(_normalize_key(raw))
