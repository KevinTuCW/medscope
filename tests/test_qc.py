"""Tests for image quality control (medscope.qc) -- the gate that stops an
image from ever reaching the CNN reader when it can't support a reading.

Degenerate cases (low-res, blank, over-exposed) are synthesized in-memory
with Pillow/numpy. The lateral-view check is exercised against the real
committed fixture in data/samples/qc/ and the 3 real frontal images in
data/samples/studies/images/ -- see both READMEs for provenance. Everything
here runs fully offline.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from medscope.config import Settings
from medscope.qc import QCResult, check_quality

QC_FIXTURE = Path("data/samples/qc/CXR3399_IM-1643-2001.png")
FRONTAL_IMAGES = [
    Path("data/samples/studies/images/CXR38_IM-1911-1001.png"),
    Path("data/samples/studies/images/CXR1187_IM-0126-1001.png"),
    Path("data/samples/studies/images/CXR797_IM-2332-1001.png"),
]


def _solid_image(size, value):
    return Image.fromarray(np.full((size, size), value, dtype=np.uint8), mode="L")


def test_low_resolution_is_hard_failure():
    img = _solid_image(64, 128)
    result = check_quality(img)

    assert isinstance(result, QCResult)
    assert result.ok is False
    assert "resolution_too_low" in result.issues


def test_all_black_image_has_no_signal():
    img = _solid_image(256, 0)
    result = check_quality(img)

    assert result.ok is False
    assert "no_signal" in result.issues


def test_all_white_image_has_no_signal():
    img = _solid_image(256, 255)
    result = check_quality(img)

    assert result.ok is False
    assert "no_signal" in result.issues


def test_over_exposed_image_is_soft_advisory():
    # Bright but with real variance, so it isn't also caught by no_signal --
    # isolates the exposure check.
    rng = np.random.default_rng(42)
    arr = rng.integers(230, 256, size=(256, 256), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")

    result = check_quality(img)

    assert "exposure_out_of_range" in result.issues
    assert result.ok is True


def test_normal_frontal_image_passes_clean():
    img = Image.open(FRONTAL_IMAGES[0])
    result = check_quality(img)

    assert result.ok is True
    assert result.issues == []


def test_real_lateral_fixture_is_flagged():
    img = Image.open(QC_FIXTURE)
    result = check_quality(img)

    assert "lateral_view" in result.issues
    assert result.ok is True  # soft advisory, not a hard reject
    assert result.metrics["lateral_symmetry_score"] < Settings().lateral_symmetry_threshold


@pytest.mark.parametrize("image_path", FRONTAL_IMAGES)
def test_real_frontal_images_are_not_flagged_lateral(image_path):
    img = Image.open(image_path)
    result = check_quality(img)

    assert "lateral_view" not in result.issues
    assert result.metrics["lateral_symmetry_score"] >= Settings().lateral_symmetry_threshold


def test_metrics_carry_the_measured_symmetry_score():
    img = Image.open(QC_FIXTURE)
    result = check_quality(img)

    assert isinstance(result.metrics["lateral_symmetry_score"], float)
    # Matches the value measured directly against the real fixture (see
    # data/samples/qc/README.md); a wide tolerance guards against float
    # drift across numpy/Pillow versions without masking a real regression.
    assert result.metrics["lateral_symmetry_score"] == pytest.approx(0.151, abs=0.01)
