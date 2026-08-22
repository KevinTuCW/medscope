"""Tests for medscope.ontology -- the shared label vocabulary both readers
must speak so `merge` (Task 1.6) can tell agreement from disagreement.
"""

from medscope.ontology import CANONICAL, CRITICAL_LABELS, canonical


def test_english_synonyms_converge():
    assert canonical("Effusion") == canonical("Pleural Effusion")
    assert canonical("Effusion") == "PleuralEffusion"


def test_txv_underscore_name_converges_with_spaced_alias():
    assert canonical("Pleural_Thickening") == canonical("Pleural Thickening")


def test_chinese_free_form_maps_to_canonical():
    assert canonical("右侧胸腔积液") == "PleuralEffusion"
    assert canonical("胸腔积液") == "PleuralEffusion"
    assert canonical("气胸") == "Pneumothorax"
    assert canonical("心影增大") == "Cardiomegaly"


def test_normalization_handles_whitespace_case_and_width():
    # full-width letters + surrounding whitespace + mixed case
    assert canonical("  effusion  ") == "PleuralEffusion"
    assert canonical("ｅｆｆｕｓｉｏｎ") == "PleuralEffusion"


def test_unknown_label_returns_none():
    assert canonical("some totally unmapped finding") is None
    assert canonical("") is None


def test_canonical_names_round_trip():
    # Re-canonicalizing an already-canonical label (e.g. merge() output)
    # must be a no-op, not a lookup miss.
    for value in set(CANONICAL.values()):
        assert canonical(value) == value


def test_critical_labels_are_all_valid_canonical_names():
    # If this fails, gate G1 is scoring against labels nothing can ever
    # produce -- i.e. silently green and useless.
    for label in CRITICAL_LABELS:
        assert canonical(label) == label


def test_site_qualified_fracture_maps_to_fracture():
    """"rib fracture" was reader_b's single most common unmappable output
    over 40 real studies (5 mentions, plus 肋骨骨折), while `Fracture` was
    sitting in the ontology the whole time -- `canonical()` is an exact
    lookup, not a substring match."""
    for alias in ("rib fracture", "Rib Fracture", "肋骨骨折", "clavicle fracture", "锁骨骨折"):
        assert canonical(alias) == "Fracture", alias


def test_subcutaneous_emphysema_does_not_map_to_emphysema():
    """The reason the fracture aliases above are enumerated rather than
    substring-matched. Subcutaneous emphysema is soft-tissue air;
    `Emphysema` is the pulmonary disease. A substring rule would map this
    and hand the merge layer a confident false agreement between two
    readers talking about different organs."""
    assert canonical("subcutaneous emphysema") != "Emphysema"
    assert canonical("皮下气肿") != "Emphysema"
