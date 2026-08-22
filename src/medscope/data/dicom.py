"""Read a real DICOM film into the shape the rest of the pipeline expects.

Everything upstream of this module was written against PNGs, and the
DICOM allowlist in `deid.py` was only ever exercised by `Dataset` objects
built in memory by tests. Both of those are comfortable fictions. Opening
one genuine Open-i CR file turns up four things a synthesized `Dataset`
cannot:

1. **`PhotometricInterpretation = MONOCHROME1`** -- the film is stored
   inverted (high value = dark). Handing those pixels to a model trained
   on MONOCHROME2 is not a subtle degradation; it is showing the model a
   photographic negative. Nothing in the PNG path ever had to know this.
2. **`BitsStored = 15` with `BitsAllocated = 16`** -- the useful range is
   not the container's range, and it is not 12 bits either. Windowing has
   to come from the data, not from an assumption.
3. **The file meta group (0002) is not part of the dataset.** pydicom
   keeps it on `ds.file_meta`, so `deid_dicom`'s `for elem in dataset`
   never sees it -- and in the real file that group carries
   `SourceApplicationEntityTitle` ('REALVIEWSERVER'),
   `PrivateInformationCreatorUID` and a `MediaStorageSOPInstanceUID`.
   Device and site identifiers, sitting outside the allowlist's reach.
   `read_film` reports them as dropped rather than letting them ride
   along invisibly.
4. **"De-identified" does not mean "no identifiers".** That same file
   carries `PatientBirthDate = 19880317`, an `AccessionNumber` and a
   `StudyDate`. The allowlist drops all three -- which is the point: on
   this archive G4's DICOM half is doing real work, not a tautology.

Scope, stated plainly: this handles the single-frame, uncompressed
(Explicit VR Little Endian) CR/DX films Open-i ships. Compressed transfer
syntaxes (JPEG2000, RLE) need pylibjpeg/gdcm and are not installed; they
raise rather than silently returning a wrong image. **Burned-in pixel
annotation is not addressed at all** -- PHI rendered into the pixels
themselves survives every tag-level rule in this repo, and detecting it
would need OCR that isn't here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from medscope.deid import deid_dicom

#: Group 0002 keywords that carry site/device identity. Reported, never
#: propagated. Not an allowlist: the whole file meta group is dropped --
#: this names the ones worth telling an auditor about by name.
FILE_META_IDENTIFIERS = (
    "SourceApplicationEntityTitle",
    "PrivateInformationCreatorUID",
    "MediaStorageSOPInstanceUID",
    "ImplementationVersionName",
    "ImplementationClassUID",
)

DICOM_SUFFIXES = (".dcm", ".dicom")


@dataclass
class DicomFilm:
    """One film read from a DICOM file, with the audit trail of what the
    read dropped. The image is 8-bit grayscale, MONOCHROME2 convention
    (high value = bright), ready for QC / reader_a / display."""

    image: object  # PIL.Image.Image -- typed loosely to keep PIL off the import path
    deid_report: dict
    dropped_file_meta: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def is_dicom_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in DICOM_SUFFIXES


def read_film(path: str | Path) -> DicomFilm:
    """Read `path` as a DICOM film: scrub tags, render pixels to 8-bit.

    The scrub runs on the way in, not as a later step a caller might
    forget: by the time anything else in the pipeline holds this film,
    the identifying tags are already gone and the report says which.
    """
    import pydicom
    from PIL import Image

    dataset = pydicom.dcmread(str(path))
    notes: list[str] = []

    dropped_meta = [
        keyword
        for keyword in FILE_META_IDENTIFIERS
        if getattr(dataset, "file_meta", None) is not None
        and getattr(dataset.file_meta, keyword, None) not in (None, "")
    ]

    transfer_syntax = getattr(getattr(dataset, "file_meta", None), "TransferSyntaxUID", None)
    photometric = str(dataset.get("PhotometricInterpretation", "MONOCHROME2"))

    try:
        pixels = dataset.pixel_array
    except Exception as exc:  # pragma: no cover - depends on optional codecs
        raise RuntimeError(
            f"cannot decode pixel data of {path} (transfer syntax {transfer_syntax}). "
            "Compressed DICOM needs pylibjpeg or gdcm, which this project does not "
            "install -- failing loudly rather than returning a wrong image."
        ) from exc

    if pixels.ndim != 2:
        raise RuntimeError(
            f"{path} holds a {pixels.ndim}-dimensional pixel array; this reader handles "
            "single-frame films only, and guessing which frame to read would be a silent "
            "clinical decision."
        )

    array = pixels.astype(np.float32)
    slope = float(dataset.get("RescaleSlope", 1) or 1)
    intercept = float(dataset.get("RescaleIntercept", 0) or 0)
    if slope != 1.0 or intercept != 0.0:
        array = array * slope + intercept
        notes.append(f"applied rescale slope {slope} / intercept {intercept}")

    # Window from the data rather than from BitsStored: a 15-bit container
    # says what the pixels *could* span, not what they do. Percentiles clip
    # the detector's hot/cold outliers that would otherwise flatten the
    # whole lung field into a couple of grey levels.
    low, high = np.percentile(array, (0.5, 99.5))
    if high <= low:  # flat film -- let QC's no_signal check be the one to reject it
        low, high = float(array.min()), float(array.max()) or 1.0
        notes.append("degenerate window; fell back to full range")
    array = np.clip((array - low) / (high - low), 0.0, 1.0)

    if photometric.upper() == "MONOCHROME1":
        # Stored inverted. Un-inverting here rather than anywhere downstream
        # because this is the only layer that knows the convention.
        array = 1.0 - array
        notes.append("MONOCHROME1: inverted to MONOCHROME2 convention")

    image = Image.fromarray((array * 255).astype(np.uint8), mode="L")

    _scrubbed, report = deid_dicom(dataset)
    if dropped_meta:
        report = {**report, "dropped_file_meta": dropped_meta}
        notes.append(
            "dropped file meta group (0002) identifiers: " + ", ".join(dropped_meta)
        )

    return DicomFilm(
        image=image, deid_report=report, dropped_file_meta=dropped_meta, notes=notes
    )
