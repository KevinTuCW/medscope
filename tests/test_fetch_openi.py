"""Tests for the two guards in scripts/fetch_openi.py that decide whether
a download becomes data.

Both exist because of something that actually happened while wiring the
DICOM path:

- openi.nlm.nih.gov answered `200` with an HTML maintenance page instead
  of the archive. Written to disk unchecked, that page becomes a
  `.tgz` that fails three steps later with a confusing error.
- A 136 KB prefix of the archive produced a *13 MB* CR film on disk that
  parsed cleanly and reported sane Rows/Columns -- and was 93% zeros. tar
  creates the member at its declared size regardless of how much of it
  arrived, so "the file opened" is not evidence that the pixels are there.

No network here: the archive is reconstructed locally, then truncated the
way a Range prefix truncates it.
"""

from __future__ import annotations

import gzip
import importlib.util
import io
import tarfile
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ComputedRadiographyImageStorage, ExplicitVRLittleEndian, generate_uid

_SPEC = importlib.util.spec_from_file_location(
    "fetch_openi", Path(__file__).resolve().parents[1] / "scripts" / "fetch_openi.py"
)
fetch_openi = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fetch_openi)

ROWS, COLS = 200, 160
MAINTENANCE_PAGE = b"<!DOCTYPE html>\n<html lang=\"en\">\n<title>There seems to be an error</title>"


def _cr_bytes(seed: int = 0) -> bytes:
    ds = Dataset()
    ds.PatientName = f"Case^{seed}"
    ds.Modality = "CR"
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 15
    ds.HighBit = 14
    ds.PixelRepresentation = 0
    ds.Rows, ds.Columns = ROWS, COLS
    rng = np.random.default_rng(seed)
    ds.PixelData = rng.integers(0, 30000, size=(ROWS, COLS), dtype=np.uint16).tobytes()

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = ComputedRadiographyImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta = meta
    ds.preamble = b"\0" * 128

    buffer = io.BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def _archive(n_films: int = 3) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for i in range(n_films):
            payload = _cr_bytes(i)
            info = tarfile.TarInfo(name=f"./{i}/{i}_IM-0001-1001.dcm")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return gzip.compress(raw.getvalue())


def test_maintenance_page_is_refused_not_saved():
    with pytest.raises(SystemExit, match="not a gzip stream"):
        fetch_openi._require_gzip(MAINTENANCE_PAGE, fetch_openi.DICOM_URL)


def test_real_gzip_passes_the_guard():
    fetch_openi._require_gzip(_archive(1), fetch_openi.DICOM_URL)


def test_complete_film_is_recognized_as_complete(tmp_path: Path):
    path = tmp_path / "whole.dcm"
    path.write_bytes(_cr_bytes())

    assert fetch_openi._dicom_pixels_complete(path)


def test_film_with_truncated_pixels_is_rejected(tmp_path: Path):
    """The failure this check exists for: a file that parses, reports the
    right Rows/Columns, and is missing most of its pixels."""
    ds = pydicom.dcmread(io.BytesIO(_cr_bytes()))
    ds.PixelData = ds.PixelData[: len(ds.PixelData) // 8]
    path = tmp_path / "truncated.dcm"
    ds.save_as(str(path), enforce_file_format=True)

    assert pydicom.dcmread(str(path)).Rows == ROWS  # still looks fine...
    assert not fetch_openi._dicom_pixels_complete(path)  # ...but it isn't


def test_truncated_prefix_keeps_only_whole_films(tmp_path: Path):
    """A Range prefix ends mid-film by construction. What arrived whole is
    data; what did not is discarded rather than written."""
    archive = _archive(3)
    prefix = archive[: int(len(archive) * 0.55)]

    dest = tmp_path / "dicom"
    dest.mkdir()
    kept = discarded = 0
    try:
        with tarfile.open(fileobj=gzip.GzipFile(fileobj=io.BytesIO(prefix)), mode="r|") as tar:
            kept, discarded = fetch_openi._extract_dicom_stream(tar, dest)
    except (tarfile.TarError, EOFError, gzip.BadGzipFile, OSError):
        pass

    on_disk = sorted(dest.glob("*.dcm"))
    assert 0 < kept < 3
    assert len(on_disk) == kept
    for path in on_disk:
        assert fetch_openi._dicom_pixels_complete(path)


def test_whole_archive_keeps_every_film(tmp_path: Path):
    """The other direction, so the test above can't pass by discarding
    everything."""
    dest = tmp_path / "dicom"
    dest.mkdir()

    with tarfile.open(fileobj=gzip.GzipFile(fileobj=io.BytesIO(_archive(3))), mode="r|") as tar:
        kept, discarded = fetch_openi._extract_dicom_stream(tar, dest)

    assert (kept, discarded) == (3, 0)
    assert len(list(dest.glob("*.dcm"))) == 3


def test_proxy_is_off_by_default():
    """The environment's SOCKS proxy measured ~1 KB/s against ~34 KB/s
    direct, and httpx refuses SOCKS entirely without `httpx[socks]`. Opting
    in is a decision; inheriting it is an ambush."""
    assert fetch_openi.USE_ENV_PROXY is False
    assert fetch_openi._client(1.0).trust_env is False
