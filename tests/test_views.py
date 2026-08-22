"""Tests for medscope.views -- deterministic view ordering for a study.

The defect this module exists to kill: reader_a used to be handed
`next(root.rglob(pattern))` (eval) or `image_paths[0]` (workbench), i.e.
whichever film the filesystem happened to return first. Same code, same
data, different score depending on OS enumeration order. Every test here
is really one assertion in different clothes -- *the reading order of a
study must be a function of its pixels and filenames, nothing else*.

The second thing under test is that ordering never becomes filtering.
Study 1704 in the G1 gold set has two films scoring 0.467 and 0.494 on
the frontal/lateral symmetry heuristic -- both below the 0.5 threshold --
while its report confirms a large pleural effusion. A "frontal only"
policy would leave that study with zero readable images. So the
classification is allowed to *order* views and never to drop one.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from medscope.views import FRONTAL, LATERAL, order_views, primary_view, score_view


def _write(path: Path, arr: np.ndarray) -> Path:
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)
    return path


def _symmetric(size: int = 256, seed: int = 0) -> np.ndarray:
    """A left-right mirror-symmetric image -- frontal-like."""
    rng = np.random.default_rng(seed)
    half = rng.integers(0, 255, size=(size, size // 2))
    return np.concatenate([half, half[:, ::-1]], axis=1)


def _asymmetric(size: int = 256, seed: int = 1) -> np.ndarray:
    """A left-right gradient -- as un-mirror-symmetric as it gets."""
    ramp = np.linspace(0, 255, size)
    rng = np.random.default_rng(seed)
    return np.tile(ramp, (size, 1)) + rng.normal(0, 3, size=(size, size))


@pytest.fixture
def frontal_and_lateral(tmp_path: Path) -> tuple[Path, Path]:
    return (
        _write(tmp_path / "CXR1_IM-0001-1001.png", _symmetric()),
        _write(tmp_path / "CXR1_IM-0001-2001.png", _asymmetric()),
    )


def test_score_view_separates_symmetric_from_asymmetric(frontal_and_lateral):
    frontal, lateral = frontal_and_lateral
    assert score_view(frontal) > score_view(lateral)


def test_frontal_like_view_is_ordered_first(frontal_and_lateral):
    frontal, lateral = frontal_and_lateral
    ordered = order_views([lateral, frontal])

    assert [v.path for v in ordered] == [frontal, lateral]
    assert ordered[0].view == FRONTAL
    assert ordered[1].view == LATERAL


def test_order_is_independent_of_input_order(frontal_and_lateral):
    """The whole point: shuffling the caller's list changes nothing."""
    frontal, lateral = frontal_and_lateral
    forwards = order_views([frontal, lateral])
    backwards = order_views([lateral, frontal])

    assert [v.path for v in forwards] == [v.path for v in backwards]


def test_same_class_views_tie_break_by_filename(tmp_path: Path):
    """Two films of the same class must not depend on enumeration order.

    Both are byte-identical here, so symmetry cannot separate them and the
    filename is the only remaining deterministic key.
    """
    arr = _symmetric()
    b = _write(tmp_path / "CXR9_IM-0002-2001.png", arr)
    a = _write(tmp_path / "CXR9_IM-0002-1001.png", arr)

    assert [v.path for v in order_views([b, a])] == [a, b]


def test_no_view_is_ever_dropped(frontal_and_lateral, tmp_path: Path):
    """Ordering is not filtering -- see module docstring (study 1704)."""
    frontal, lateral = frontal_and_lateral
    extra = _write(tmp_path / "CXR1_IM-0001-3001.png", _asymmetric(seed=7))

    ordered = order_views([lateral, extra, frontal])

    assert len(ordered) == 3
    assert {v.path for v in ordered} == {frontal, lateral, extra}


def test_lateral_only_study_still_yields_a_primary_view(tmp_path: Path):
    """A study whose every film reads lateral-like is still readable.

    This is the study-1704 case. The most frontal-like film available wins;
    refusing to pick one would turn a soft heuristic into a hard reject.
    """
    mild = _write(tmp_path / "CXR2_IM-0001-2001.png", _asymmetric(seed=3))
    strong = _write(tmp_path / "CXR2_IM-0001-1001.png", _asymmetric(seed=4))
    ordered = order_views([mild, strong])

    assert all(v.view == LATERAL for v in ordered)
    assert primary_view([mild, strong]) == ordered[0].path
    assert ordered[0].symmetry == max(v.symmetry for v in ordered)


def test_primary_view_of_empty_study_is_none():
    assert primary_view([]) is None
    assert order_views([]) == []


def test_order_views_accepts_str_paths(frontal_and_lateral):
    frontal, lateral = frontal_and_lateral
    ordered = order_views([str(lateral), str(frontal)])

    assert [v.path for v in ordered] == [frontal, lateral]
