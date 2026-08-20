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

import pytest

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
