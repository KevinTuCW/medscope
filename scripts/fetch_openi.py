"""Fetch the OpenI / Indiana University chest X-ray dataset.

Populates `Settings.openi_root` (default `data/openi/`, gitignored):

- Always downloads and extracts the reports archive (`NLMCXR_reports.tgz`,
  ~1.1 MB) -- it's small enough to always fetch in full.
- Optionally fetches images:
    --image-bytes N   pull only the first N bytes of the 1.36 GB image
                       archive via an HTTP Range request, then gunzip/untar
                       what that partial download contains. Cheap way to get
                       a handful of sample PNGs.
    --full            download the entire 1.36 GB image archive. Slow -- see
                       the printed warning before it starts.

- Optionally fetches DICOM:
    --dicom-bytes N   pull the first N bytes of the 80.7 GB DICOM archive
                       (`NLMCXR_dcm.tgz`) the same way, keeping only the
                       films whose pixel data arrived intact.

An earlier version of this docstring said NLMCXR_dcm.tgz was "confirmed
unreachable from this network (connection failure)". That is no longer
true and the correction matters more than the capability: the archive
serves `206 Partial Content` with a real gzip stream. What it also does,
intermittently and with a `200`, is serve an HTML maintenance page --
which is why `_require_gzip` exists. Writing that page to disk as
`prefix.tgz` would have produced a confusing failure three steps later
instead of one clear one here. Measured throughput has ranged from
379 KB/s down to ~1 KB/s on the same day, so size the prefix small.

The Open-i JSON search API (/api/search?...) is still not attempted --
that one remains unreachable. Don't rely on it.

Usage:
    python scripts/fetch_openi.py                    # reports only
    python scripts/fetch_openi.py --image-bytes 20000000   # + ~20MB of images
    python scripts/fetch_openi.py --dicom-bytes 40000000   # + a few real DICOM films
    python scripts/fetch_openi.py --full              # + the full image archive
"""

from __future__ import annotations

import argparse
import gzip
import io
import logging
import tarfile
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("fetch_openi")

REPORTS_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
IMAGES_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"
DICOM_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_dcm.tgz"

# Measured from this project's dev environment; used only to size the
# --full warning message, not to make network decisions.
MEASURED_DOWNLOAD_KBPS = 379
FULL_IMAGE_ARCHIVE_BYTES = 1_360_814_128


def _fail(url: str, reason: str) -> None:
    # Fail loudly and stop -- never silently fall back to an empty dataset,
    # which would make every downstream test vacuously pass.
    raise SystemExit(
        f"fetch_openi: failed to fetch {url}: {reason}\n"
        f"Check: is openi.nlm.nih.gov reachable from this network? "
        f"Try `curl -I {url}` to confirm, and check for a proxy/VPN "
        f"requirement. Not falling back to an empty dataset."
    )


def _download_full(url: str, timeout: float = 60.0) -> bytes:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/151.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Referer": "https://openi.nlm.nih.gov/",
    }

    try:
        with _client(timeout) as client:
            resp = client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        _fail(url, f"connection error ({exc})")
        raise

    if resp.status_code != 200:
        _fail(url, f"HTTP {resp.status_code}")

    # gzip 文件的 magic number 是 1f 8b
    if not resp.content.startswith(b"\x1f\x8b"):
        preview = resp.content[:200].decode("utf-8", errors="replace")
        _fail(
            url,
            f"response is not gzip data "
            f"(Content-Type={resp.headers.get('content-type')}, "
            f"size={len(resp.content)}, preview={preview!r})",
        )

    return resp.content


#: Whether to honour HTTP(S)_PROXY / ALL_PROXY from the environment.
#: Off by default -- see `_client`. Flipped by `--use-proxy`.
USE_ENV_PROXY = False


def _client(timeout: float) -> httpx.Client:
    """An httpx client that ignores the ambient proxy by default.

    These archives are public static files on openi.nlm.nih.gov and need
    no proxy to reach. Measured on this dev machine, which has
    `ALL_PROXY=socks5://...` exported: **~1 KB/s through the proxy versus
    ~34 KB/s direct**, and httpx refuses a SOCKS proxy outright unless
    `httpx[socks]` is installed -- so inheriting the environment turns a
    15-minute download into a multi-hour one, or into an ImportError.
    `medscope.llm` sets `trust_env=False` for the same reason.

    `--use-proxy` exists for the network where the proxy is the only way
    out; it is a choice someone makes, not a default they inherit.
    """
    return httpx.Client(timeout=timeout, follow_redirects=True, trust_env=USE_ENV_PROXY)


def _download_range(url: str, num_bytes: int, timeout: float = 60.0) -> bytes:
    headers = {"Range": f"bytes=0-{num_bytes - 1}"}
    try:
        with _client(timeout) as client:
            resp = client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        _fail(url, f"connection error ({exc})")
        raise
    if resp.status_code not in (200, 206):
        _fail(url, f"HTTP {resp.status_code}")
    return resp.content


def _require_gzip(data: bytes, url: str) -> None:
    """Refuse anything that isn't a gzip stream.

    openi.nlm.nih.gov intermittently answers `200` with an HTML page
    titled "There seems to be an error" (maintenance) in place of the
    archive. Without this check that page lands on disk as an archive and
    the failure surfaces two steps later as an unreadable tar, which is a
    far worse error message than this one.
    """
    if data[:2] != b"\x1f\x8b":
        preview = data[:80].decode("utf-8", errors="replace").replace("\n", " ")
        _fail(url, f"response is not a gzip stream (starts with: {preview!r})")


def _dicom_pixels_complete(path: Path) -> bool:
    """True only if the film's pixel data actually all arrived.

    A truncated gzip prefix still yields a *plausible* file: tar creates
    the member at its declared size, so a 13 MB CR film can materialize
    out of 136 KB of downloaded bytes, parse cleanly, report sane
    Rows/Columns -- and be 93% zeros. Length is the only honest check;
    "it opened" is not evidence.
    """
    try:
        import pydicom
    except ImportError:  # pragma: no cover - pydicom ships with the cv extra
        logger.warning("pydicom missing; cannot verify %s, keeping it unverified", path.name)
        return True
    try:
        ds = pydicom.dcmread(str(path))
        expected = (
            int(ds.Rows)
            * int(ds.Columns)
            * int(ds.get("SamplesPerPixel", 1))
            * (int(ds.BitsAllocated) // 8)
        )
        return len(ds.PixelData) >= expected
    except Exception as exc:
        logger.info("discarding %s: not a readable DICOM (%s)", path.name, exc)
        return False


def _extract_dicom_stream(tar: tarfile.TarFile, dest: Path) -> tuple[int, int]:
    """Extract whole members only; return (kept, discarded).

    `member.size` is the size the tar header *claims*. Reading fewer bytes
    than that means the prefix ended mid-film, so the member is dropped
    rather than written: an incomplete film that parses is worse than no
    film at all, because it looks like data.
    """
    kept = discarded = 0
    for member in tar:
        if not member.isfile() or not member.name.lower().endswith(".dcm"):
            continue
        fobj = tar.extractfile(member)
        if fobj is None:
            continue
        try:
            payload = fobj.read()
        except (tarfile.TarError, EOFError, OSError) as exc:
            logger.info("stream ended inside %s (%s)", member.name, exc)
            discarded += 1
            break
        if len(payload) != member.size:
            logger.info(
                "discarding %s: %d of %d bytes arrived", member.name, len(payload), member.size
            )
            discarded += 1
            break
        out_path = dest / Path(member.name).name
        out_path.write_bytes(payload)
        if _dicom_pixels_complete(out_path):
            kept += 1
        else:
            out_path.unlink()
            discarded += 1
    return kept, discarded


def fetch_dicom_partial(root: Path, dicom_bytes: int) -> None:
    """Pull a prefix of the DICOM archive and keep the intact films in it.

    The archive is 80.7 GB and one uncompressed CR film is ~13 MB, so a
    prefix buys a handful of films rather than a dataset -- which is all
    the real-DICOM path needs: the tag allowlist and the MONOCHROME1 /
    rescale handling either hold up on a genuine file or they don't.
    """
    logger.info("downloading %d bytes (Range prefix) of DICOM archive: %s", dicom_bytes, DICOM_URL)
    data = _download_range(DICOM_URL, dicom_bytes, timeout=600.0)
    _require_gzip(data, DICOM_URL)

    dest = root / "dicom"
    dest.mkdir(parents=True, exist_ok=True)
    logger.info("downloaded %d bytes, extracting complete films to %s", len(data), dest)

    kept = discarded = 0
    try:
        with tarfile.open(fileobj=gzip.GzipFile(fileobj=io.BytesIO(data)), mode="r|") as tar:
            kept, discarded = _extract_dicom_stream(tar, dest)
    except (tarfile.TarError, EOFError, gzip.BadGzipFile, OSError) as exc:
        # Expected, not exceptional: a Range prefix ends mid-stream by
        # construction. Whatever completed before that point stands.
        logger.info("stream ended (%s) -- keeping whatever completed", exc)

    logger.info("DICOM films kept: %d (incomplete, discarded: %d)", kept, discarded)
    if kept == 0:
        logger.warning(
            "no complete film in this prefix. One CR film is ~13 MB uncompressed -- "
            "raise --dicom-bytes rather than treating the partial files as data."
        )


def fetch_reports(root: Path) -> None:
    logger.info("downloading reports archive: %s", REPORTS_URL)
    data = _download_full(REPORTS_URL)
    logger.info("downloaded %d bytes, extracting to %s", len(data), root)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(root, filter="data")
    logger.info("reports extracted")


def _extract_image_stream(tar: tarfile.TarFile, dest: Path) -> int:
    extracted = 0
    for member in tar:
        if not member.isfile():
            continue
        fobj = tar.extractfile(member)
        if fobj is None:
            continue
        out_path = dest / Path(member.name).name
        out_path.write_bytes(fobj.read())
        extracted += 1
    return extracted


def fetch_images_partial(root: Path, image_bytes: int) -> None:
    logger.info(
        "downloading %d bytes (Range prefix) of image archive: %s", image_bytes, IMAGES_URL
    )
    data = _download_range(IMAGES_URL, image_bytes)
    dest = root / "images"
    dest.mkdir(parents=True, exist_ok=True)

    # `extracted` must live in this scope, not be returned from a helper --
    # the expected truncation can interrupt mid-loop, after several members
    # were already written to disk but before a `return` is ever reached.
    # Counting in a helper would silently report 0 despite real progress.
    extracted = 0
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz, tarfile.open(
            fileobj=gz, mode="r|"
        ) as tar:
            for member in tar:
                if not member.isfile():
                    continue
                fobj = tar.extractfile(member)
                if fobj is None:
                    continue
                out_path = dest / Path(member.name).name
                out_path.write_bytes(fobj.read())
                extracted += 1
    except (tarfile.TarError, EOFError, OSError) as exc:
        # Expected: a byte-range prefix of a .tgz truncates mid-stream. This
        # is normal for --image-bytes and must not crash the script or be
        # silently swallowed -- log it plainly.
        logger.info(
            "partial archive, extracted %d images as requested (truncation: %s)",
            extracted,
            exc,
        )
        return

    logger.info("extracted %d images (archive prefix happened to end cleanly)", extracted)


def fetch_images_full(root: Path) -> None:
    minutes = FULL_IMAGE_ARCHIVE_BYTES / 1024 / MEASURED_DOWNLOAD_KBPS / 60
    logger.warning(
        "WARNING: --full downloads the entire %.2f GB image archive. At this "
        "environment's measured ~%d KB/s download speed, that takes OVER AN "
        "HOUR (~%.0f min). Consider --image-bytes N for a small sample instead.",
        FULL_IMAGE_ARCHIVE_BYTES / 1e9,
        MEASURED_DOWNLOAD_KBPS,
        minutes,
    )
    logger.info("downloading full image archive: %s", IMAGES_URL)
    data = _download_full(IMAGES_URL, timeout=7200.0)
    dest = root / "images"
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        extracted = _extract_image_stream(tar, dest)
    logger.info("extracted %d images", extracted)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--image-bytes",
        type=int,
        default=None,
        help="Download only this many leading bytes of NLMCXR_png.tgz via an HTTP "
        "Range request (gunzipped/untarred, tolerating end-of-stream truncation).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download the entire 1.36 GB image archive. Slow -- see the printed warning.",
    )
    parser.add_argument(
        "--dicom-bytes",
        type=int,
        default=None,
        help="Download only this many leading bytes of NLMCXR_dcm.tgz (80.7 GB) via an "
        "HTTP Range request, keeping only the films whose pixel data arrived intact. "
        "One CR film is ~13 MB uncompressed.",
    )
    parser.add_argument(
        "--use-proxy",
        action="store_true",
        help="Honour HTTP(S)_PROXY / ALL_PROXY from the environment. Off by default -- "
        "a proxy can slow these large downloads down by orders of magnitude, and httpx "
        "needs httpx[socks] to use a SOCKS one at all.",
    )
    parser.add_argument(
        "--skip-reports",
        action="store_true",
        help="Skip the reports archive (rarely useful; it's only 1.1 MB).",
    )
    args = parser.parse_args()

    global USE_ENV_PROXY
    USE_ENV_PROXY = args.use_proxy

    if args.full and args.image_bytes:
        raise SystemExit("fetch_openi: pass either --full or --image-bytes, not both.")

    from medscope.config import Settings

    root = Settings().openi_root
    root.mkdir(parents=True, exist_ok=True)

    if not args.skip_reports:
        fetch_reports(root)

    if args.dicom_bytes:
        fetch_dicom_partial(root, args.dicom_bytes)

    if args.full:
        fetch_images_full(root)
    elif args.image_bytes:
        fetch_images_partial(root, args.image_bytes)
    elif not args.dicom_bytes:
        logger.info(
            "no image option given (--image-bytes N, --dicom-bytes N or --full) -- "
            "reports only fetched."
        )

    logger.info(
        "NOTE: the Open-i JSON search API (/api/search) is not attempted by this "
        "script -- it was confirmed unreachable from this network. NLMCXR_dcm.tgz "
        "IS reachable (--dicom-bytes), but intermittently serves a maintenance page "
        "instead of the archive; _require_gzip turns that into a clear failure."
    )


if __name__ == "__main__":
    main()
