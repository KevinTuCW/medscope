"""Tests for medscope.readers.cnn -- the discriminative CNN reader
(reader_a). It is the only component that produces quantitative
probabilities and spatial localization; gate G2 requires every report
sentence to cite a Finding carrying a source, a probability, and a
location, so correctness here is load-bearing for that gate.

The fast tests below stub both `_infer` and `_gradcam` -- the two seams
that touch `_load_model()` -- so they run fully offline with no weight
download and no network access. Preprocessing runs for real against a
committed sample image; it only needs Pillow/numpy/torchxrayvision's pure
array utilities, no model weights.

One real-weights test is marked `slow` and deselected by default (see
pyproject.toml `addopts`); run it explicitly with `-m slow`.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from medscope.config import Settings
from medscope.readers import cnn as cnn_mod
from medscope.readers.cnn import CNNReader
from medscope.state import ReadResult

SAMPLE_IMAGE = Path("data/samples/studies/images/CXR38_IM-1911-1001.png")
CARDIOMEGALY_IMAGE = Path("data/samples/studies/images/CXR797_IM-2332-1001.png")

STUB_PROBS = {
    "Cardiomegaly": 0.82,
    "Effusion": 0.61,
    "Pneumothorax": 0.05,
    "Atelectasis": 0.12,
    "some unmapped finding": 0.4,
}


def _stub_gradcam(tensor, label):
    return {"cx": 0.5, "cy": 0.5, "r": 0.1}


@pytest.fixture
def reader(monkeypatch):
    monkeypatch.setattr(cnn_mod, "_infer", lambda tensor: dict(STUB_PROBS))
    monkeypatch.setattr(cnn_mod, "_gradcam", _stub_gradcam)
    return CNNReader(settings=Settings())


def test_read_returns_read_result_for_reader_a(reader):
    result = reader.read(SAMPLE_IMAGE)

    assert isinstance(result, ReadResult)
    assert result.reader == "a"
    assert result.latency_ms >= 0
    assert len(result.findings) == len(STUB_PROBS)


def test_all_findings_source_cnn_with_probs_in_range(reader):
    result = reader.read(SAMPLE_IMAGE)

    for finding in result.findings:
        assert finding.source == "cnn"
        assert 0.0 <= finding.prob <= 1.0


def test_known_labels_are_canonicalized(reader):
    result = reader.read(SAMPLE_IMAGE)
    by_raw = {f.raw_label: f for f in result.findings}

    assert by_raw["Cardiomegaly"].label == "Cardiomegaly"
    assert by_raw["Effusion"].label == "PleuralEffusion"


def test_unmappable_label_keeps_raw_label_instead_of_being_dropped(reader):
    result = reader.read(SAMPLE_IMAGE)
    labels = {f.raw_label for f in result.findings}

    assert "some unmapped finding" in labels
    unmapped = next(f for f in result.findings if f.raw_label == "some unmapped finding")
    # canonical() can't map it -- kept, not dropped, and not silently
    # coerced into looking like a real canonical ontology name.
    assert unmapped.label == "some unmapped finding"


def test_sub_threshold_findings_have_no_locus(reader):
    result = reader.read(SAMPLE_IMAGE)
    settings = Settings()

    below = [f for f in result.findings if f.prob < settings.cnn_prob_threshold]
    assert below  # Pneumothorax (0.05) and Atelectasis (0.12) qualify
    for finding in below:
        assert finding.locus is None


def test_above_threshold_findings_have_normalized_locus(reader):
    result = reader.read(SAMPLE_IMAGE)
    settings = Settings()

    above = [f for f in result.findings if f.prob >= settings.cnn_prob_threshold]
    assert above  # Cardiomegaly (0.82), Effusion (0.61), unmapped (0.4)
    for finding in above:
        assert finding.locus is not None
        for key in ("cx", "cy", "r"):
            assert 0.0 <= finding.locus[key] <= 1.0


def test_gradcam_only_called_for_above_threshold_labels(monkeypatch, reader):
    calls = []

    def _tracking_gradcam(tensor, label):
        calls.append(label)
        return {"cx": 0.5, "cy": 0.5, "r": 0.1}

    monkeypatch.setattr(cnn_mod, "_gradcam", _tracking_gradcam)
    reader.read(SAMPLE_IMAGE)

    settings = Settings()
    expected = {label for label, prob in STUB_PROBS.items() if prob >= settings.cnn_prob_threshold}
    assert set(calls) == expected


# ---------------------------------------------------------------------------
# Multi-film reading: a radiologist reads a study, not a file
# ---------------------------------------------------------------------------


def _film(tmp_path: Path, name: str, fill: int) -> Path:
    """A flat film of a given brightness -- brightness is the handle the
    stubbed `_infer` below uses to tell one film from another."""
    path = tmp_path / name
    Image.new("L", (256, 320), color=fill).save(path)
    return path


@pytest.fixture
def two_films(tmp_path: Path) -> tuple[Path, Path]:
    """Two films of one study: `dark` and `bright`."""
    return _film(tmp_path, "CXR5_IM-0001-1001.png", 40), _film(tmp_path, "CXR5_IM-0001-2001.png", 200)


@pytest.fixture
def per_film_reader(monkeypatch):
    """reader_a whose stubbed probabilities differ per film.

    The bright film calls a confident pneumothorax and a negative
    cardiomegaly; the dark film calls the reverse. Any implementation that
    reads one film of the study gets one of those two answers, so the
    tests below can tell "read the study" from "read a file".
    """

    def _infer_by_brightness(tensor):
        bright = float(tensor.mean()) > 0
        return {
            "Pneumothorax": 0.91 if bright else 0.02,
            "Cardiomegaly": 0.10 if bright else 0.77,
        }

    monkeypatch.setattr(cnn_mod, "_infer", _infer_by_brightness)
    monkeypatch.setattr(cnn_mod, "_gradcam", _stub_gradcam)
    return CNNReader(settings=Settings())


def test_read_study_takes_the_max_across_films(per_film_reader, two_films):
    """A finding visible on one projection and not another is still a
    finding. Reading a single film of a multi-film study is what cost G1
    five confirmed-positive cases."""
    dark, bright = two_films
    by_label = {f.label: f for f in per_film_reader.read_study([dark, bright]).findings}

    assert by_label["Pneumothorax"].prob == pytest.approx(0.91)
    assert by_label["Cardiomegaly"].prob == pytest.approx(0.77)


def test_read_study_attributes_each_finding_to_the_film_it_was_seen_on(per_film_reader, two_films):
    dark, bright = two_films
    by_label = {f.label: f for f in per_film_reader.read_study([dark, bright]).findings}

    assert by_label["Pneumothorax"].image_ref == str(bright)
    assert by_label["Cardiomegaly"].image_ref == str(dark)


def test_read_study_is_independent_of_the_order_films_arrive_in(per_film_reader, two_films):
    """Same study, same bytes, same answer -- whatever order the caller (or
    the filesystem) hands the films over in."""
    dark, bright = two_films

    def _shape(result):
        return sorted((f.label, round(f.prob, 6), f.image_ref) for f in result.findings)

    assert _shape(per_film_reader.read_study([dark, bright])) == _shape(
        per_film_reader.read_study([bright, dark])
    )


def test_gradcam_runs_on_the_film_that_won_the_label(monkeypatch, per_film_reader, two_films):
    """A locus computed on one film and attributed to another is a
    confidently wrong overlay -- worse than no overlay at all."""
    dark, bright = two_films
    seen: dict[str, float] = {}

    def _tracking_gradcam(tensor, label):
        seen[label] = float(tensor.mean())
        return {"cx": 0.5, "cy": 0.5, "r": 0.1}

    monkeypatch.setattr(cnn_mod, "_gradcam", _tracking_gradcam)
    per_film_reader.read_study([dark, bright])

    # Only the two above-threshold labels, each localized on its own film:
    # xrv normalizes around mid-grey, so the bright film's tensor has a
    # positive mean and the dark film's a negative one.
    assert set(seen) == {"Pneumothorax", "Cardiomegaly"}
    assert seen["Pneumothorax"] > 0
    assert seen["Cardiomegaly"] < 0


def test_read_study_records_which_films_it_read(per_film_reader, two_films):
    """Without the films named in the audit trail, "reader_a read the
    study" is an unverifiable claim."""
    dark, bright = two_films
    note = " ".join(per_film_reader.read_study([dark, bright]).notes)

    assert dark.name in note and bright.name in note
    assert "2 film" in note


def test_read_delegates_to_read_study(per_film_reader, two_films):
    _dark, bright = two_films

    def _shape(result):
        return [(f.label, f.prob, f.image_ref) for f in result.findings]

    assert _shape(per_film_reader.read(bright)) == _shape(per_film_reader.read_study([bright]))


def test_read_study_with_no_films_returns_an_empty_result(per_film_reader):
    result = per_film_reader.read_study([])

    assert result.findings == []
    assert result.notes == ["reader_a received no films"]


def test_preprocess_keeps_the_apices_and_costophrenic_angles(tmp_path: Path):
    """No center crop: the top and bottom of a portrait film survive.

    `XRayCenterCrop` squares the image by trimming the long axis, which on
    Open-i's portrait films cuts the lung apices (where a pneumothorax
    collects) and the costophrenic angles (where an effusion collects) --
    the two findings G1 exists for. This test fails if the crop comes back.
    """
    arr = np.zeros((320, 128), dtype=np.uint8)
    arr[:8, :] = 255  # a bright band along the very top edge only
    path = tmp_path / "portrait.png"
    Image.fromarray(arr, mode="L").save(path)

    kept = cnn_mod._preprocess(path)
    cropped = cnn_mod._preprocess(path, center_crop=True)

    assert float(kept[0, 0, 0, :].max()) > float(cropped[0, 0, 0, :].max())


@pytest.mark.slow
def test_real_model_on_cardiomegaly_case_returns_well_formed_result():
    reader = CNNReader(settings=Settings())
    result = reader.read(CARDIOMEGALY_IMAGE)

    assert result.reader == "a"
    assert len(result.findings) > 0
    for finding in result.findings:
        assert finding.source == "cnn"
        assert 0.0 <= finding.prob <= 1.0
        if finding.locus is not None:
            for key in ("cx", "cy", "r"):
                assert 0.0 <= finding.locus[key] <= 1.0
