"""Deterministic view ordering for a multi-film study.

A radiologist reads a *study*, not a file. Open-i studies carry 1-4 films
(2.4 on average across the G1 gold set; 38 of 53 cases have more than
one), and until this module existed the pipeline picked one of them with
`next(root.rglob(pattern))` or `image_paths[0]` -- filesystem enumeration
order. That made the measurement itself unstable: same code, same data,
a different G1 score depending on what the OS handed back first.

Two rules, and the second one is the load-bearing one:

1. **Order is a pure function of pixels and filenames.** Frontal-like
   films first (they are what `densenet121-res224-all` was trained on),
   then by descending symmetry, then by filename. Nothing in that key can
   change between two runs over the same bytes.

2. **Ordering never becomes filtering.** The frontal/lateral call is the
   same cheap mirror-symmetry heuristic `qc.py` uses as a *soft* advisory,
   and it is much mushier on real films than its documented band suggests:
   gold-set study 1704 -- a confirmed large pleural effusion -- has two
   films scoring 0.467 and 0.494, both landing on the lateral side of the
   0.5 threshold. A "read only frontals" policy would leave that study
   with nothing to read at all. So every view stays in the list; the
   heuristic decides what gets read *first*, not what gets read.

Feeding a lateral film to a frontal-trained model does produce garbage
(measured on this dataset: 40 of 47 laterals raised a false pneumothorax
alarm), which is exactly why the ordering exists -- but the response to a
mushy classifier is to rank and attribute, not to discard films on its
say-so.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from medscope.config import Settings

FRONTAL = "frontal"
LATERAL = "lateral"

#: Symmetry scores are rounded before they enter the sort key. Two films
#: differing in the 12th decimal must not be able to swap places between
#: runs -- that would reintroduce, in float noise, exactly the
#: nondeterminism this module removes.
_SYMMETRY_PRECISION = 6


@dataclass(frozen=True)
class ViewRef:
    """One film of a study, with the view call that ranked it.

    `symmetry` is kept rather than discarded so the workbench and the eval
    report can show *why* a film was read first, and so a future validated
    view classifier can be compared against this heuristic on the record.
    """

    path: Path
    view: str
    symmetry: float


def score_view(image_path: str | Path) -> float:
    """Mirror-symmetry score for one film. Higher = more frontal-like.

    Delegates to `qc._lateral_symmetry_score` on purpose: one heuristic,
    one implementation. If the QC advisory and the read order ever
    disagreed about what counts as a lateral, the workbench would show a
    view call that contradicts the film it drew the Grad-CAM on.
    """
    from contextlib import closing

    from medscope.film import open_film
    from medscope.qc import _lateral_symmetry_score

    # Closed explicitly: `order_views` runs over every film of every study
    # in an eval sweep, and leaving each one to the garbage collector is
    # how a long run meets the open-file limit.
    with closing(open_film(image_path)) as image:
        return _lateral_symmetry_score(image)


def order_views(
    image_paths: list[str | Path] | list[Path] | list[str],
    settings: Settings | None = None,
) -> list[ViewRef]:
    """Rank a study's films deterministically. Never drops one."""
    settings = settings or Settings()
    threshold = settings.lateral_symmetry_threshold

    refs = []
    for raw in image_paths:
        path = Path(raw)
        symmetry = score_view(path)
        view = FRONTAL if symmetry >= threshold else LATERAL
        refs.append(ViewRef(path=path, view=view, symmetry=symmetry))

    return sorted(
        refs,
        key=lambda r: (
            0 if r.view == FRONTAL else 1,
            -round(r.symmetry, _SYMMETRY_PRECISION),
            r.path.name,
        ),
    )


def primary_view(
    image_paths: list[str | Path] | list[Path] | list[str],
    settings: Settings | None = None,
) -> Path | None:
    """The one film to show, and to hand to the single-image consumers.

    reader_b (the VLM), the arbiter and the report writer each get one
    image, not the whole study -- a cost decision, and one worth naming:
    reader_a sees more of the study than reader_b does. Returns None only
    for a study with no films at all.
    """
    ordered = order_views(image_paths, settings)
    return ordered[0].path if ordered else None
