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
