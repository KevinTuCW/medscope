"""One way to open a film, whatever container it arrived in.

Three places need pixels -- QC, `views` (the frontal/lateral score) and
reader_a's preprocessing -- and each of them used to call
`Image.open(path)` directly. That is fine right up until a `.dcm` shows
up, at which point two of the three would raise and the third would be
wrong in some interesting way. Routing all of them through here means a
new container format is one dispatch, not a scavenger hunt.

DICOM films are de-identified as they are read (see
`medscope.data.dicom.read_film`), so nothing downstream can hold an
un-scrubbed dataset by accident.
"""

from __future__ import annotations

from pathlib import Path


def open_film(path: str | Path):
    """Return a PIL grayscale-capable image for `path`.

    PNG/JPEG go straight to Pillow. DICOM goes through the reader that
    knows about MONOCHROME1, rescale slope and the file meta group.
    """
    from PIL import Image

    from medscope.data.dicom import is_dicom_path, read_film

    if is_dicom_path(path):
        return read_film(path).image
    return Image.open(path)
