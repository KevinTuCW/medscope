"""Tests for medscope.data.dicom -- reading a real DICOM *file*.

Everything DICOM-shaped in this repo used to be a `Dataset` built in
memory by a test. That is a comfortable fiction: an in-memory dataset has
no file meta group, no transfer syntax, no MONOCHROME1, and no 15-bit
pixels. Each fixture here therefore writes an actual `.dcm` to disk with
`pydicom` and reads it back through the real path, and each one is shaped
after what a genuine Open-i CR film actually carries (verified against
`1_IM-0001-3001.dcm` from the NLMCXR_dcm archive: MONOCHROME1,
BitsStored 15, PatientBirthDate present, `SourceApplicationEntityTitle`
= 'REALVIEWSERVER' in group 0002).

The one test that would fail today if the code were wrong in the way this
module was written to fix: `test_file_meta_identifiers_are_reported` --
`deid_dicom`'s `for elem in dataset` cannot see group 0002, so those
identifiers were invisible to the allowlist, not scrubbed by it.
"""

from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ComputedRadiographyImageStorage, ExplicitVRLittleEndian, generate_uid

from medscope.data.dicom import is_dicom_path, read_film
from medscope.film import open_film

CR_ROWS, CR_COLS = 256, 192  # above QC's 128px floor, so a film can pass QC


def _write_cr(
    path: Path,
    *,
    photometric: str = "MONOCHROME1",
    pixels: np.ndarray | None = None,
    slope: float | None = None,
    intercept: float | None = None,
) -> Path:
    """Write a single-frame CR file carrying the tags a real one carries,
    identifying ones included -- the allowlist has nothing to prove
    against a file that was clean to begin with."""
    ds = Dataset()
    ds.PatientName = "Zhang^Wei"
    ds.PatientID = "P-4471"
    ds.PatientBirthDate = "19880317"
    ds.PatientSex = "F"
    ds.StudyDate = "20120229"
    ds.AccessionNumber = "1290436665346037"
    ds.InstitutionName = "Indiana University Hospital"
    ds.ReferringPhysicianName = "Li^Ming"
    ds.StudyInstanceUID = generate_uid()

    ds.Modality = "CR"
    ds.BodyPartExamined = "CHEST"
    ds.ViewPosition = "PA"
    ds.PhotometricInterpretation = photometric
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 15
    ds.HighBit = 14
    ds.PixelRepresentation = 0
    if slope is not None:
        ds.RescaleSlope = slope
    if intercept is not None:
        ds.RescaleIntercept = intercept

    if pixels is None:
        # A left-right gradient: bright on one side under MONOCHROME2,
        # which makes the inversion check directional rather than
        # statistical.
        pixels = np.tile(
            np.linspace(0, 30000, CR_COLS, dtype=np.uint16), (CR_ROWS, 1)
        ).astype(np.uint16)
    ds.Rows, ds.Columns = pixels.shape
    ds.PixelData = pixels.tobytes()

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = ComputedRadiographyImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationVersionName = "OSIRIX"
    meta.SourceApplicationEntityTitle = "REALVIEWSERVER"
    ds.file_meta = meta
    ds.preamble = b"\0" * 128

    ds.save_as(str(path), enforce_file_format=True)
    return path


@pytest.fixture
def cr_file(tmp_path: Path) -> Path:
    return _write_cr(tmp_path / "study.dcm")


def test_is_dicom_path_matches_by_suffix():
    assert is_dicom_path("a/b/film.dcm")
    assert is_dicom_path(Path("FILM.DICOM"))
    assert not is_dicom_path("data/samples/studies/images/CXR38_IM-1911-1001.png")


def test_read_film_returns_8bit_grayscale(cr_file):
    film = read_film(cr_file)

    assert film.image.mode == "L"
    assert film.image.size == (CR_COLS, CR_ROWS)


def test_monochrome1_is_inverted_to_monochrome2(tmp_path: Path):
    """A MONOCHROME1 film handed over as-is is a photographic negative.

    The same gradient is written twice, once under each convention; after
    reading, the two images must be mirror images in intensity, not
    copies of each other.
    """
    inverted = read_film(_write_cr(tmp_path / "m1.dcm", photometric="MONOCHROME1")).image
    upright = read_film(_write_cr(tmp_path / "m2.dcm", photometric="MONOCHROME2")).image

    m1 = np.asarray(inverted, dtype=float)
    m2 = np.asarray(upright, dtype=float)

    # Left edge is dark under MONOCHROME2 and bright under MONOCHROME1.
    assert m2[:, 0].mean() < m2[:, -1].mean()
    assert m1[:, 0].mean() > m1[:, -1].mean()


def test_identifying_tags_do_not_survive_the_read(cr_file):
    """The scrub happens on the way in, not as a step a caller can skip."""
    film = read_film(cr_file)
    kept = set(film.deid_report["kept_tags"])
    dropped = set(film.deid_report["dropped_tags"])

    for identifier in (
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "AccessionNumber",
        "InstitutionName",
        "ReferringPhysicianName",
        "StudyDate",
        "StudyInstanceUID",
    ):
        assert identifier in dropped
        assert identifier not in kept

    # ...while the tags a reader actually needs stay.
    assert {"Modality", "Rows", "Columns", "PhotometricInterpretation"} <= kept


def test_file_meta_identifiers_are_reported(cr_file):
    """Group 0002 is not part of the dataset, so the allowlist never saw it.

    `deid_dicom` iterates `for elem in dataset`, and pydicom keeps the file
    meta group on `ds.file_meta` -- a separate object. A real Open-i film
    carries `SourceApplicationEntityTitle = 'REALVIEWSERVER'` and a
    `PrivateInformationCreatorUID` there: site and device identity that an
    allowlist cannot drop if it cannot see it. They are named in the report
    instead of riding along unmentioned.
    """
    film = read_film(cr_file)

    assert "SourceApplicationEntityTitle" in film.dropped_file_meta
    assert "MediaStorageSOPInstanceUID" in film.dropped_file_meta
    assert "dropped_file_meta" in film.deid_report
    assert any("file meta" in note for note in film.notes)


def test_rescale_slope_and_intercept_are_applied(tmp_path: Path):
    flat = np.full((CR_ROWS, CR_COLS), 100, dtype=np.uint16)
    flat[:, CR_COLS // 2 :] = 200

    plain = read_film(_write_cr(tmp_path / "plain.dcm", pixels=flat))
    scaled = read_film(
        _write_cr(tmp_path / "scaled.dcm", pixels=flat, slope=2.0, intercept=-50.0)
    )

    assert any("rescale" in note for note in scaled.notes)
    assert not any("rescale" in note for note in plain.notes)
    # The transform is affine and the window is percentile-based, so the
    # rendered image is unchanged -- what matters is that it was applied and
    # said so, not that it moved the pixels.
    assert np.asarray(plain.image).shape == np.asarray(scaled.image).shape


def test_multiframe_file_is_refused_rather_than_guessed(tmp_path: Path):
    """Picking a frame out of a multi-frame file is a clinical decision, and
    this reader is not entitled to make it silently."""
    ds = pydicom.dcmread(str(_write_cr(tmp_path / "single.dcm")))
    frames = np.stack([np.asarray(ds.pixel_array)] * 3)
    ds.NumberOfFrames = 3
    ds.Rows, ds.Columns = frames.shape[1], frames.shape[2]
    ds.PixelData = frames.astype(np.uint16).tobytes()
    multi = tmp_path / "multi.dcm"
    ds.save_as(str(multi), enforce_file_format=True)

    with pytest.raises(RuntimeError, match="single-frame"):
        read_film(multi)


def test_open_film_dispatches_dicom_and_png(cr_file):
    """QC, view ordering and reader_a all go through `open_film`; if it
    only understood PNG, two of the three would raise on a `.dcm` and the
    third would be wrong in some interesting way."""
    png = Path("data/samples/studies/images/CXR38_IM-1911-1001.png")

    assert open_film(cr_file).size == (CR_COLS, CR_ROWS)
    assert open_film(png).size[0] > 0


def test_dicom_film_runs_through_qc_and_view_scoring(cr_file):
    """End to end on a file, not on an in-memory Dataset: the two cheap
    stages that touch pixels before the model does."""
    from medscope.qc import check_quality
    from medscope.views import order_views

    result = check_quality(open_film(cr_file))
    ordered = order_views([cr_file])

    assert result.ok
    assert "resolution_too_low" not in result.issues
    assert "no_signal" not in result.issues
    assert ordered[0].path == cr_file
