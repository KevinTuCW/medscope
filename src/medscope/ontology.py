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
    "Fracture": "Fracture",
    "骨折": "Fracture",
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
