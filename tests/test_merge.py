"""Tests for medscope.merge -- where reader_a (CNN) and reader_b (VLM)
independent reads are compared. Agreement collapses to one Finding;
disagreement is the signal the arbiter (a later task) spends LLM budget
on, so getting agreement-vs-disagreement classification right here is
load-bearing for the whole double-reading architecture.
"""

import math

import pytest

from medscope.config import Settings
from medscope.critical import triage
from medscope.merge import agreement, cohens_kappa, merge_reads
from medscope.ontology import COMPARISON_LABELS, CRITICAL_LABELS
from medscope.state import Description, Finding, ReadResult


def _finding(label, prob, source="cnn", locus=None, raw_label=None, notes=None):
    return Finding(
        label=label,
        prob=prob,
        source=source,
        locus=locus,
        raw_label=raw_label if raw_label is not None else label,
        notes=notes or [],
    )


def _description(label, text, raw_label=None):
    return Description(label=label, text=text, raw_label=raw_label if raw_label is not None else label)


def _read(reader, findings, latency_ms=10):
    return ReadResult(reader=reader, findings=findings, latency_ms=latency_ms)


def _read_descriptions(reader, descriptions, latency_ms=10):
    return ReadResult(reader=reader, descriptions=descriptions, latency_ms=latency_ms)


THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# merge_reads, mode="reader"
# ---------------------------------------------------------------------------


def test_agreement_merges_to_one_finding_with_higher_prob_and_locus():
    locus_a = {"cx": 0.5, "cy": 0.5, "r": 0.1}
    read_a = _read("a", [_finding("Cardiomegaly", 0.9, source="cnn", locus=locus_a)])
    read_b = _read("b", [_finding("Cardiomegaly", 0.7, source="vlm", locus=None)])

    findings, disagreements, kappa = merge_reads(read_a, read_b, THRESHOLD)

    assert disagreements == []
    assert len(findings) == 1
    merged = findings[0]
    assert merged.label == "Cardiomegaly"
    assert merged.prob == 0.9  # the higher of the two
    assert merged.source == "cnn"  # source of the higher-prob finding
    assert merged.locus == locus_a  # taken from whichever reader has one


def test_agreement_takes_locus_from_reader_b_when_a_has_none():
    locus_b = {"cx": 0.3, "cy": 0.4, "r": 0.2}
    read_a = _read("a", [_finding("Nodule", 0.6, source="cnn", locus=None)])
    read_b = _read("b", [_finding("Nodule", 0.8, source="vlm", locus=locus_b)])

    findings, _, _ = merge_reads(read_a, read_b, THRESHOLD)

    assert findings[0].locus == locus_b


def test_both_negative_is_also_an_agreement_not_a_disagreement():
    # This is exactly the "both readers say no" case Task 1.5 called out --
    # both below threshold on the same label should merge, not vanish or
    # get flagged, so it stays distinguishable from "reader never
    # considered it" (a `unique` disagreement).
    read_a = _read("a", [_finding("Pneumothorax", 0.05)])
    read_b = _read("b", [_finding("Pneumothorax", 0.1, source="vlm")])

    findings, disagreements, _ = merge_reads(read_a, read_b, THRESHOLD)

    assert disagreements == []
    assert len(findings) == 1
    assert findings[0].prob == 0.1


def test_presence_disagreement_one_positive_one_negative():
    read_a = _read("a", [_finding("Effusion", 0.8)])
    read_b = _read("b", [_finding("Pleural Effusion", 0.2, source="vlm")])

    findings, disagreements, _ = merge_reads(read_a, read_b, THRESHOLD)

    assert findings == []
    assert len(disagreements) == 1
    d = disagreements[0]
    assert d.kind == "presence"
    assert d.label == "PleuralEffusion"
    assert d.a_prob == 0.8
    assert d.b_prob == 0.2


def test_magnitude_disagreement_both_positive_but_far_apart():
    # gap 0.35 >= Settings().magnitude_gap (0.3 default)
    read_a = _read("a", [_finding("Cardiomegaly", 0.9)])
    read_b = _read("b", [_finding("Cardiomegaly", 0.55, source="vlm")])

    findings, disagreements, _ = merge_reads(read_a, read_b, THRESHOLD)

    assert findings == []
    assert len(disagreements) == 1
    d = disagreements[0]
    assert d.kind == "magnitude"
    assert d.a_prob == 0.9
    assert d.b_prob == 0.55


def test_both_positive_close_together_is_agreement_not_magnitude():
    read_a = _read("a", [_finding("Cardiomegaly", 0.85)])
    read_b = _read("b", [_finding("Cardiomegaly", 0.95, source="vlm")])

    findings, disagreements, _ = merge_reads(read_a, read_b, THRESHOLD)

    assert disagreements == []
    assert len(findings) == 1


def test_unique_disagreement_label_only_in_reader_a():
    read_a = _read("a", [_finding("Hernia", 0.6)])
    read_b = _read("b", [])

    findings, disagreements, _ = merge_reads(read_a, read_b, THRESHOLD)

    assert findings == []
    assert len(disagreements) == 1
    d = disagreements[0]
    assert d.kind == "unique"
    assert d.label == "Hernia"
    assert d.a_prob == 0.6
    assert d.b_prob is None


def test_unique_disagreement_label_only_in_reader_b():
    read_a = _read("a", [])
    read_b = _read("b", [_finding("气胸", 0.7, source="vlm")])

    findings, disagreements, _ = merge_reads(read_a, read_b, THRESHOLD)

    assert len(disagreements) == 1
    d = disagreements[0]
    assert d.kind == "unique"
    assert d.label == "Pneumothorax"  # canonicalized before matching
    assert d.a_prob is None
    assert d.b_prob == 0.7


def test_unmapped_label_is_unique_disagreement_never_dropped():
    read_a = _read("a", [_finding("some totally unmapped finding", 0.4)])
    read_b = _read("b", [])

    findings, disagreements, _ = merge_reads(read_a, read_b, THRESHOLD)

    assert findings == []
    assert len(disagreements) == 1
    d = disagreements[0]
    assert d.kind == "unique"
    assert d.label == "some totally unmapped finding"
    assert d.a_prob == 0.4


def test_unmapped_labels_from_both_readers_never_cross_matched():
    # Two different unmapped phrases must not accidentally merge just
    # because canonical() returns None for both -- there's no shared
    # identity to match unmapped labels on.
    read_a = _read("a", [_finding("odd english phrase", 0.6)])
    read_b = _read("b", [_finding("奇怪的中文描述", 0.6, source="vlm")])

    findings, disagreements, _ = merge_reads(read_a, read_b, THRESHOLD)

    assert findings == []
    assert len(disagreements) == 2
    assert all(d.kind == "unique" for d in disagreements)


def test_one_sided_negative_call_is_agreement_not_a_unique_disagreement():
    # The readers are structurally asymmetric: CNNReader.read emits a
    # Finding for every one of the model's labels, including the ones it
    # confidently calls absent, while reader_b only names what it saw.
    # Scoring "reader_a says Hernia 0.02, reader_b never mentioned Hernia"
    # as a disagreement makes ~17 of them per study for any VLM, which
    # (a) floods the arbiter's max_llm_judgments budget with non-events and
    # (b) pins disagreement_rate at 1.0, making the reader/describer
    # demotion rule in scripts/calibrate_vlm.py impossible to pass. Two
    # readers who both decline to call a label are agreeing it is absent.
    read_a = _read("a", [_finding("Hernia", 0.02)])
    read_b = _read("b", [])

    findings, disagreements, _ = merge_reads(read_a, read_b, THRESHOLD)

    assert disagreements == []
    assert len(findings) == 1
    assert findings[0].label == "Hernia"
    assert findings[0].prob == 0.02


def test_one_sided_positive_call_is_still_a_unique_disagreement():
    # The other half of the rule above: a label one reader called POSITIVE
    # and the other never mentioned is a real conflict, and must stay one.
    read_a = _read("a", [_finding("Hernia", 0.6)])
    read_b = _read("b", [])

    _findings, disagreements, _ = merge_reads(read_a, read_b, THRESHOLD)

    assert [d.kind for d in disagreements] == ["unique"]


def test_kappa_is_not_perfect_when_reader_b_named_nothing_mappable():
    # Observed against a real VLM: reader_b returned nine free-text labels
    # ("right rib fracture", "no pleural effusion", ...), none of which
    # canonical() could map. b_mapped is then empty, so the universe comes
    # entirely from reader_a, both positive sets are empty, and
    # cohens_kappa's degenerate branch reports 1.0 -- perfect agreement
    # from a reader that never once agreed with anything. kappa_floor is
    # what decides whether reader_b stays a peer, so this must not read as
    # agreement.
    read_a = _read("a", [_finding("Pneumothorax", 0.05), _finding("Effusion", 0.1)])
    read_b = _read(
        "b",
        [
            _finding("right rib fracture", 0.9, source="vlm"),
            _finding("no pleural effusion", 0.9, source="vlm"),
        ],
    )

    _findings, _disagreements, kappa = merge_reads(read_a, read_b, THRESHOLD)

    assert kappa != 1.0


# ---------------------------------------------------------------------------
# merge_reads, mode="describer"
# ---------------------------------------------------------------------------


def test_describer_mode_positive_set_equals_reader_a_exactly():
    read_a = _read(
        "a",
        [
            _finding("Cardiomegaly", 0.9),
            _finding("Pneumothorax", 0.1),
        ],
    )
    # reader_b contributes descriptions, not a positive/negative judgement
    # -- Description (unlike Finding) has no prob at all to ignore.
    read_b = _read_descriptions(
        "b",
        [
            _description("Cardiomegaly", "心影明显增大，符合心脏扩大表现"),
            _description("Pneumothorax", "可见气胸征象"),
        ],
    )

    findings, disagreements, _ = merge_reads(read_a, read_b, THRESHOLD, mode="describer")

    assert disagreements == []
    positive = {f.label for f in findings if f.prob >= THRESHOLD}
    assert positive == {"Cardiomegaly"}  # reader_a's positive set, unaffected by reader_b


def test_describer_mode_attaches_matching_descriptions_to_finding_notes():
    read_a = _read("a", [_finding("Cardiomegaly", 0.9)])
    read_b = _read_descriptions("b", [_description("Cardiomegaly", "心影增大")])

    findings, _, _ = merge_reads(read_a, read_b, THRESHOLD, mode="describer")

    cardiomegaly = next(f for f in findings if f.label == "Cardiomegaly")
    assert "心影增大" in cardiomegaly.notes


def test_describer_mode_unmatched_description_is_not_a_finding():
    # Task 2.3 revisited this twice: an unmatched description first
    # synthesized a standalone prob=0.0 Finding (Task 1.6), which risked
    # being picked up as a citation target downstream (a description is
    # not a judgement). merge_reads never mutates its inputs, so the fix
    # isn't to relocate the description -- it just never enters the merged
    # findings list; it stays exactly where read_b put it (see
    # tests/test_describer_mode.py::
    # test_describer_unmatched_description_is_not_discarded_and_not_a_finding
    # for the fuller "not discarded" check against a real read_b() output).
    read_a = _read("a", [_finding("Cardiomegaly", 0.9)])
    # reader_b mentions something reader_a's finding set has no entry for.
    read_b = _read_descriptions("b", [_description("纵隔气肿", "可见纵隔气肿")])

    findings, _, _ = merge_reads(read_a, read_b, THRESHOLD, mode="describer")

    assert all(f.label != "纵隔气肿" for f in findings)
    all_notes = [n for f in findings for n in f.notes]
    assert "可见纵隔气肿" not in all_notes


# ---------------------------------------------------------------------------
# cohens_kappa
# ---------------------------------------------------------------------------


def test_kappa_is_one_for_identical_label_sets():
    universe = {"A", "B", "C"}
    kappa, note = cohens_kappa({"A", "B"}, {"A", "B"}, universe)

    assert kappa == pytest.approx(1.0)
    assert note is None


def test_kappa_is_zero_at_chance_level():
    # both_pos={A}, only_a={B}, only_b={C}, both_neg={D}: po=pe=0.5
    universe = {"A", "B", "C", "D"}
    kappa, note = cohens_kappa({"A", "B"}, {"A", "C"}, universe)

    assert kappa == pytest.approx(0.0, abs=1e-6)
    assert note is None


def test_kappa_degenerate_when_both_readers_all_negative():
    # The common case: normal chest X-ray, both readers call every label
    # negative. pe == 1 -- the formula's denominator is 0.
    universe = {"Cardiomegaly", "Pneumothorax", "Effusion"}
    kappa, note = cohens_kappa(set(), set(), universe)

    assert kappa == 1.0
    assert not math.isnan(kappa)
    assert note == "kappa_degenerate"


def test_kappa_empty_universe_is_degenerate_not_a_crash():
    kappa, note = cohens_kappa(set(), set(), set())

    assert kappa == 1.0
    assert note == "kappa_degenerate"


def test_merge_reads_kappa_matches_all_negative_degenerate_case():
    read_a = _read("a", [_finding("Pneumothorax", 0.05), _finding("Effusion", 0.1)])
    read_b = _read("b", [_finding("Pneumothorax", 0.02, source="vlm"), _finding("Effusion", 0.05, source="vlm")])

    _, _, kappa = merge_reads(read_a, read_b, THRESHOLD)

    assert kappa == 1.0


# ---------------------------------------------------------------------------
# Comparison vocabulary: what the two readers can actually be compared on
# ---------------------------------------------------------------------------


def _labelled_read(reader, pairs):
    return ReadResult(
        reader=reader,
        latency_ms=1,
        findings=[
            Finding(label=label, prob=prob, source="cnn" if reader == "a" else "vlm")
            for label, prob in pairs
        ],
    )


def test_kappa_ignores_labels_reader_b_never_speaks_about():
    """The defect this exists for: reader_a calls ~10 of its 18 labels
    positive per study while reader_b names ~2.5 findings, so scoring
    agreement across all 18 grades reader_b on a vocabulary it does not
    use. Measured that way over 40 real studies, kappa was -0.041.

    Both readers agree here on the one label they can both speak about;
    the disagreement is entirely on labels outside the comparison
    vocabulary. The narrowed number must see the agreement, and the wide
    number must still show the gap.
    """
    read_a = _labelled_read(
        "a",
        [("Cardiomegaly", 0.9), ("Emphysema", 0.9), ("Fibrosis", 0.9), ("Infiltration", 0.9)],
    )
    read_b = _labelled_read("b", [("Cardiomegaly", 0.9)])

    stats = agreement(read_a, read_b, 0.5)

    assert stats.narrow > stats.wide
    assert stats.n_narrow_labels < stats.n_wide_labels


def test_wide_kappa_is_still_reported_alongside_the_narrow_one():
    """Narrowing a comparison must never be a way to improve the number by
    redefining it -- the figure that made narrowing necessary stays
    visible."""
    read_a = _labelled_read("a", [("Cardiomegaly", 0.9), ("Emphysema", 0.9)])
    read_b = _labelled_read("b", [("Cardiomegaly", 0.9)])

    stats = agreement(read_a, read_b, 0.5)

    assert stats.wide is not None
    assert stats.n_wide_labels == 2


def test_out_of_vocabulary_disagreements_are_flagged_not_dropped():
    """reader_a naming a label reader_b never uses is information, but it
    is not two readers disagreeing about the same thing. It stays in the
    list, marked."""
    read_a = _labelled_read("a", [("Emphysema", 0.9), ("Cardiomegaly", 0.9)])
    read_b = _labelled_read("b", [])

    _findings, disagreements, _kappa = merge_reads(read_a, read_b, 0.5)
    by_label = {d.label: d for d in disagreements}

    assert by_label["Emphysema"].in_vocabulary is False
    assert by_label["Cardiomegaly"].in_vocabulary is True
    assert len(disagreements) == 2  # neither was dropped


def test_every_critical_label_stays_in_the_comparison_vocabulary():
    """A safety gate's vocabulary must not be decided by what a model
    happened to say in a 40-study sample."""
    assert set(CRITICAL_LABELS) <= COMPARISON_LABELS


def test_narrowing_does_not_touch_the_critical_channel():
    """G1 reads the raw per-reader findings, never the merged set, so an
    out-of-vocabulary narrowing cannot silence an alert. Pinned here
    because the two modules are edited independently."""
    read_a = _labelled_read("a", [("Pneumothorax", 0.62), ("Emphysema", 0.9)])
    read_b = _labelled_read("b", [])
    settings = Settings()

    _findings, _disagreements, _kappa = merge_reads(read_a, read_b, 0.5)
    alerts = triage(list(read_a.findings) + list(read_b.findings), settings)

    assert "Pneumothorax" in {a.label for a in alerts}
