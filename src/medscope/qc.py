"""Image quality control -- the gate that stops an image from ever
reaching the CNN reader when the image can't support a reading.

Pure Pillow + numpy statistics on purpose. No CNN, no torch: this gate
must be cheap enough to run before any model loads, so a bad image never
burns model-load time or LLM budget before being turned away.

Two checks are HARD failures (`ok=False`): resolution too low, and no
signal (a flat/blank image). These let the pipeline terminate immediately
with status NEEDS_REPEAT -- a rejection here costs nothing, unlike a bad
image that proceeds and produces a confident reading of a picture the
model can't actually interpret.

Two checks are SOFT advisories (issues recorded, `ok` stays True):
exposure out of a sane band, and the lateral-view screen. Both are cheap
heuristics that can misfire on a real, usable film; a hard reject on
either would throw away readable studies more often than it would save
budget.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from medscope.config import Settings

_MIN_DIMENSION = 128  # below this, there isn't enough detail to read
_LOW_STD_THRESHOLD = 5.0  # near-zero variance -> flat/blank image, no signal
_EXPOSURE_LOW = 15.0
_EXPOSURE_HIGH = 235.0


class QCResult(BaseModel):
    ok: bool
    issues: list[str] = Field(default_factory=list)
    # Measured values behind each issue, so the workbench audit can show
    # *why* something was rejected or flagged -- a bare boolean is useless
    # to a radiologist reviewing a NEEDS_REPEAT.
    metrics: dict = Field(default_factory=dict)


def check_quality(image: Image.Image, settings: Settings | None = None) -> QCResult:
    """Run all QC checks against a single image and return a QCResult.

    `settings` defaults to a fresh `Settings()` if not given (picks up
    `lateral_symmetry_threshold`, tunable via LATERAL_SYMMETRY_THRESHOLD).
    """
    settings = settings or Settings()
    issues: list[str] = []
    metrics: dict = {}

    width, height = image.size
    metrics["width"] = width
    metrics["height"] = height
    if width < _MIN_DIMENSION or height < _MIN_DIMENSION:
        issues.append("resolution_too_low")

    gray = np.asarray(image.convert("L"), dtype=float)
    mean = float(gray.mean())
    std = float(gray.std())
    metrics["mean"] = mean
    metrics["std"] = std

    if std < _LOW_STD_THRESHOLD:
        issues.append("no_signal")

    if not (_EXPOSURE_LOW <= mean <= _EXPOSURE_HIGH):
        issues.append("exposure_out_of_range")

    symmetry_score = _lateral_symmetry_score(image)
    metrics["lateral_symmetry_score"] = symmetry_score
    if symmetry_score < settings.lateral_symmetry_threshold:
        issues.append("lateral_view")

    hard_fail = "resolution_too_low" in issues or "no_signal" in issues
    return QCResult(ok=not hard_fail, issues=issues, metrics=metrics)


def _lateral_symmetry_score(image: Image.Image) -> float:
    """Score how left-right symmetric an image is about its vertical axis.

    This is a cheap screening heuristic, NOT a validated view classifier.
    A frontal chest radiograph (PA/AP) is approximately symmetric about
    the spine; a lateral is not. Score the image by the correlation
    between it and its own horizontal mirror -- high score, frontal-like;
    low or negative score, lateral-like. Measured on real OpenI images,
    frontals score ~0.83-0.85 and laterals score -0.05-0.15, with
    `Settings.lateral_symmetry_threshold` (default 0.5) sitting in the
    empty band between them (see data/samples/qc/README.md).

    Treated purely as a soft advisory: wrongly discarding a genuinely
    frontal study is worse than flagging one for a second look.
    """
    a = np.asarray(image.convert("L").resize((128, 128)), dtype=float)
    a = (a - a.mean()) / (a.std() + 1e-6)
    return float((a * a[:, ::-1]).mean())
