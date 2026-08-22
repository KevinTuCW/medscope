"""Where reader_a (CNN) and reader_b (VLM) meet.

Both readers read independently, and their failure modes are orthogonal:
the CNN is quantitative and localized but blind to clinical context, the
VLM is contextual and descriptive but gives no numbers or boxes. A
disagreement between them isn't noise to average away -- it's a precise
pointer at a finding a human (or, later, an LLM arbiter) needs to look at.
Everything downstream depends on this module classifying agreement vs
disagreement correctly: the arbiter only spends LLM budget on
disagreements, and the workbench's main explanatory view is this
reader-vs-reader table.

`merge_reads` fixes its signature (including `mode`) now even though only
`mode="reader"` is exercised in Phase 1 -- Task 2.3 switches reader_b to
`mode="describer"` if the VLM proves too weak to be a peer, and call sites
shouldn't need rewriting when that happens.
"""

from __future__ import annotations

from dataclasses import dataclass

from medscope.config import Settings
from medscope.ontology import COMPARISON_LABELS, canonical
from medscope.state import Description, Disagreement, Finding, ReadResult
from medscope.thresholds import report_threshold


def _split_mapped(findings: list[Finding]) -> tuple[dict[str, Finding], list[Finding]]:
    """Split a reader's findings into {canonical label -> Finding} and a
    list of findings whose label `canonical()` can't map.

    Always re-canonicalizes `finding.label` here rather than trusting it's
    already canonical -- `canonical()` is idempotent on an already-canonical
    name, so this is a no-op for a well-behaved reader, and a safety net
    for one that isn't. Unmapped findings never enter the cross-reader
    label matching below (there's no shared identity to match on) and
    instead become `unique` disagreements directly -- see `_merge_reader`.
    """
    mapped: dict[str, Finding] = {}
    unmapped: list[Finding] = []
    for finding in findings:
        canon = canonical(finding.label)
        if canon is None:
            unmapped.append(finding)
        else:
            mapped[canon] = finding
    return mapped, unmapped


def _group_description_text(descriptions: list[Description]) -> dict[str, list[str]]:
    """Group describer-mode `Description.text` by label, re-canonicalizing
    each `Description.label` for the same defensive reason `_split_mapped`
    re-canonicalizes `Finding.label` -- `_description_from_item` already
    canonicalizes, so this is normally a no-op, but never trust a caller-
    supplied label as already canonical.

    A label with no canonical mapping keys on its own (raw) text here too:
    it simply won't match anything in `a_mapped` (reader_a's findings are
    already canonical-only), so it naturally stays unmatched rather than
    needing separate unmapped-tracking the way `_split_mapped` does for
    Findings feeding `_merge_reader`'s two-sided matching.
    """
    grouped: dict[str, list[str]] = {}
    for description in descriptions:
        canon = canonical(description.label)
        key = canon if canon is not None else description.label
        grouped.setdefault(key, []).append(description.text)
    return grouped


def _merge_finding(fa: Finding, fb: Finding, label: str) -> Finding:
    """Agreement (both readers on the same side of `threshold` -- both
    positive, or both negative): take the higher prob, take the locus from
    whichever reader has one (in practice the CNN, since the VLM gives no
    localization).
    """
    higher, lower = (fa, fb) if fa.prob >= fb.prob else (fb, fa)
    locus = higher.locus if higher.locus is not None else lower.locus
    return Finding(
        label=label,
        prob=higher.prob,
        source=higher.source,
        locus=locus,
        raw_label=higher.raw_label,
        notes=list(higher.notes),
    )


def _merge_reader(
    read_a: ReadResult, read_b: ReadResult, threshold: float
) -> tuple[list[Finding], list[Disagreement], float]:
    """`threshold` is the caller's global default; the per-label measured
    operating point wins where one exists (see `medscope.thresholds`).
    Both readers are judged by the same number for a given label -- a
    per-label threshold applied on one side only would turn a calibration
    into a systematic disagreement."""
    a_mapped, a_unmapped = _split_mapped(read_a.findings)
    b_mapped, b_unmapped = _split_mapped(read_b.findings)

    findings: list[Finding] = []
    disagreements: list[Disagreement] = []

    for label in sorted(set(a_mapped) | set(b_mapped)):
        fa = a_mapped.get(label)
        fb = b_mapped.get(label)
        in_vocab = label in COMPARISON_LABELS

        # A label only one reader mentioned is a `unique` disagreement only
        # if that reader actually CALLED it. The two readers are
        # structurally asymmetric: reader_a emits a Finding for every label
        # its model knows (most of them confident negatives), while reader_b
        # names only what it saw. Treating each unmentioned negative as a
        # conflict produced ~17 per study against a real VLM -- see
        # test_one_sided_negative_call_is_agreement_not_a_unique_disagreement.
        label_threshold = report_threshold(label, threshold)
        if fa is None:
            if fb.prob >= label_threshold:
                disagreements.append(
                    Disagreement(
                        label=label, a_prob=None, b_prob=fb.prob, kind="unique",
                        in_vocabulary=in_vocab,
                    )
                )
            else:
                findings.append(fb)
            continue
        if fb is None:
            if fa.prob >= label_threshold:
                disagreements.append(
                    Disagreement(
                        label=label, a_prob=fa.prob, b_prob=None, kind="unique",
                        in_vocabulary=in_vocab,
                    )
                )
            else:
                findings.append(fa)
            continue

        a_pos = fa.prob >= label_threshold
        b_pos = fb.prob >= label_threshold
        if a_pos != b_pos:
            disagreements.append(
                Disagreement(
                    label=label, a_prob=fa.prob, b_prob=fb.prob, kind="presence",
                    in_vocabulary=in_vocab,
                )
            )
        elif a_pos and abs(fa.prob - fb.prob) >= _magnitude_gap():
            # Both positive, but far enough apart to matter clinically --
            # see Settings.magnitude_gap for the chosen threshold and why.
            disagreements.append(
                Disagreement(
                    label=label, a_prob=fa.prob, b_prob=fb.prob, kind="magnitude",
                    in_vocabulary=in_vocab,
                )
            )
        else:
            findings.append(_merge_finding(fa, fb, label))

    # Labels canonical() couldn't map are never dropped. An unmapped label is by definition outside the comparison vocabulary:
    # there is no shared identity to compare on. It is still recorded --
    # "reader_b saw a central venous catheter" is information, it is just
    # not evidence that two readers disagreed about the same thing.
    for finding in a_unmapped:
        disagreements.append(
            Disagreement(
                label=finding.label, a_prob=finding.prob, b_prob=None, kind="unique",
                in_vocabulary=False,
            )
        )
    for finding in b_unmapped:
        disagreements.append(
            Disagreement(
                label=finding.label, a_prob=None, b_prob=finding.prob, kind="unique",
                in_vocabulary=False,
            )
        )

    positive_a = {l for l, f in a_mapped.items() if f.prob >= report_threshold(l, threshold)}
    positive_b = {l for l, f in b_mapped.items() if f.prob >= report_threshold(l, threshold)}
    # Narrowed to the labels the two readers can actually be compared on --
    # see `ontology.COMPARISON_LABELS` for the measurement behind it. The
    # wide universe is still computed, by `agreement()` below, and reported
    # next to this one: narrowing the comparison must never mean losing the
    # number that made narrowing necessary.
    universe = (set(a_mapped) | set(b_mapped)) & COMPARISON_LABELS

    if b_unmapped and not b_mapped:
        # reader_b judged, but every label it produced fell outside the
        # shared vocabulary, so the universe is reader_a's alone and both
        # positive sets are trivially comparable. cohens_kappa would take
        # its degenerate branch and report 1.0 -- perfect agreement from a
        # reader that never once agreed with anything (observed against a
        # real VLM returning "right rib fracture", "no pleural effusion",
        # ...). `kappa_floor` is what decides whether reader_b stays a peer,
        # so this reports no measurable agreement rather than total
        # agreement. It is not a computed kappa and does not pretend to be;
        # the reason is recorded per-study by scripts/calibrate_vlm.py.
        return findings, disagreements, 0.0

    kappa, _note = cohens_kappa(positive_a, positive_b, universe)

    return findings, disagreements, kappa


def _merge_describer(
    read_a: ReadResult, read_b: ReadResult, threshold: float
) -> tuple[list[Finding], list[Disagreement], float]:
    """reader_b demoted to a description generator: it contributes no
    positive/negative judgement, so there is nothing for it to disagree
    with reader_a about. The positive set equals reader_a's exactly.

    reader_b's `read_b.descriptions` (a list of `Description`, not
    `Finding` -- see that type's docstring in state.py for why) attach to
    the matching output Finding's `.notes` by canonical label. A
    description with no matching reader_a finding is not discarded, but it
    does not become a Finding either (Task 2.3 revisited Task 1.6's
    original choice here, which matched on `read_b.findings` and
    synthesized a standalone `prob=0.0` Finding for an unmatched one --
    that shape only ever worked against hand-built test fixtures, since a
    real describer-mode `read_b()` always returns `findings == []`; see
    tests/test_describer_mode.py::test_describer_end_to_end_composition).
    This function never mutates or drops its `read_a`/`read_b` inputs, so
    an unmatched description stays fully recoverable from the caller's own
    `read_b` (e.g. `state.read_b.descriptions`) -- it just never enters the
    *merged* view, which is reserved for findings only.
    """
    a_mapped, a_unmapped = _split_mapped(read_a.findings)
    b_text_by_label = _group_description_text(read_b.descriptions)

    findings: list[Finding] = []
    for label, fa in a_mapped.items():
        extra_text = b_text_by_label.get(label, [])
        notes = list(fa.notes) + extra_text if extra_text else fa.notes
        findings.append(fa if not extra_text else fa.model_copy(update={"notes": notes}))
    findings.extend(a_unmapped)

    # No second judgement to compare reader_a against, so there is nothing
    # to classify as a disagreement in describer mode.
    disagreements: list[Disagreement] = []

    # Comparing reader_a's positive set against itself is not a gamed
    # shortcut: kappa is 1.0 by definition whenever the two label sets are
    # identical, which is exactly true here since reader_b contributes no
    # judgement of its own.
    positive_a = {l for l, f in a_mapped.items() if f.prob >= report_threshold(l, threshold)}
    universe = set(a_mapped)
    kappa, _note = cohens_kappa(positive_a, positive_a, universe)

    return findings, disagreements, kappa


def merge_reads(
    read_a: ReadResult, read_b: ReadResult, threshold: float, mode: str = "reader"
) -> tuple[list[Finding], list[Disagreement], float]:
    """Merge reader_a and reader_b's independent reads.

    `mode` mirrors `Settings.reader_b_mode`:
      - "reader" (default): both readers are peers -- see `_merge_reader`.
      - "describer": reader_b only describes, doesn't judge -- see
        `_merge_describer`.
    """
    if mode == "describer":
        return _merge_describer(read_a, read_b, threshold)
    if mode == "reader":
        return _merge_reader(read_a, read_b, threshold)
    raise ValueError(f"unknown mode {mode!r}; expected 'reader' or 'describer'")


def _magnitude_gap() -> float:
    return Settings().magnitude_gap


def cohens_kappa(labels_a: set[str], labels_b: set[str], universe: set[str]) -> tuple[float, str | None]:
    """Cohen's kappa for inter-reader agreement: each label in `universe`
    is a binary present/absent call by each reader (membership in
    `labels_a` / `labels_b`), scored via the standard 2x2 confusion table
    and `(po - pe) / (1 - pe)`. Hand-rolled on purpose -- no sklearn.

    Returns `(kappa, note)`. `note` is `None` for a normally-computed
    kappa, or `"kappa_degenerate"` when `pe == 1` -- both readers unanimous
    in the same direction across all of `universe`, most commonly
    both-negative on a clean study (also covers an empty `universe`, where
    there is nothing to compare at all). In that case `po == pe == 1` too,
    so the formula is 0/0. Returning `nan` would be doubly wrong: this
    isn't "no agreement data", it's "the readers agree on everything, but
    the denominator can't say how much of that was luck" -- and `nan`
    silently breaks any downstream `kappa < threshold` comparison (Task
    2.3 compares against `Settings.kappa_floor` to decide whether the VLM
    stays a peer reader; `nan < kappa_floor` is `False`, which would look
    exactly like "kappa is fine"). Returning `1.0` keeps the number usable
    while the note flags that it wasn't a meaningful signal.

    Note: this function's return type carries `note` alongside the float
    because callers otherwise cannot distinguish this degenerate case from
    a genuine, fully-informative kappa of 1.0 -- both would just be `1.0`.
    `merge_reads` calls this internally and reports only the float, per its
    own fixed return signature.
    """
    n = len(universe)
    if n == 0:
        return 1.0, "kappa_degenerate"

    both_pos = both_neg = only_a = only_b = 0
    for label in universe:
        a = label in labels_a
        b = label in labels_b
        if a and b:
            both_pos += 1
        elif not a and not b:
            both_neg += 1
        elif a:
            only_a += 1
        else:
            only_b += 1

    po = (both_pos + both_neg) / n
    p_a_pos = (both_pos + only_a) / n
    p_b_pos = (both_pos + only_b) / n
    pe = p_a_pos * p_b_pos + (1 - p_a_pos) * (1 - p_b_pos)

    if pe >= 1.0 - 1e-9:
        return 1.0, "kappa_degenerate"
    return (po - pe) / (1 - pe), None


@dataclass(frozen=True)
class Agreement:
    """Both kappas, side by side, plus what each was computed over.

    Reporting only the narrowed number would be the move this project
    keeps refusing to make: redefining the measurement until it looks
    better. `narrow` is the honest agreement signal (labels both readers
    can speak about); `wide` is the number that made narrowing necessary
    (-0.041 over 40 studies) and stays visible next to it.
    """

    narrow: float
    wide: float
    n_narrow_labels: int
    n_wide_labels: int
    narrow_note: str | None = None
    wide_note: str | None = None


def agreement(read_a: ReadResult, read_b: ReadResult, threshold: float) -> Agreement:
    """Inter-reader agreement, measured twice: narrowed and wide."""
    a_mapped, _a_unmapped = _split_mapped(read_a.findings)
    b_mapped, _b_unmapped = _split_mapped(read_b.findings)

    positive_a = {l for l, f in a_mapped.items() if f.prob >= report_threshold(l, threshold)}
    positive_b = {l for l, f in b_mapped.items() if f.prob >= report_threshold(l, threshold)}
    wide_universe = set(a_mapped) | set(b_mapped)
    narrow_universe = wide_universe & COMPARISON_LABELS

    narrow, narrow_note = cohens_kappa(positive_a, positive_b, narrow_universe)
    wide, wide_note = cohens_kappa(positive_a, positive_b, wide_universe)
    return Agreement(
        narrow=narrow,
        wide=wide,
        n_narrow_labels=len(narrow_universe),
        n_wide_labels=len(wide_universe),
        narrow_note=narrow_note,
        wide_note=wide_note,
    )
