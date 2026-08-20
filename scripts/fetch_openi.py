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

Deliberately does NOT attempt:
  - NLMCXR_dcm.tgz (the DICOM archive) -- confirmed unreachable from this
    network (connection failure).
  - the Open-i JSON search API (/api/search?...) -- also confirmed
    unreachable. Don't rely on it either.

Usage:
    python scripts/fetch_openi.py                    # reports only
    python scripts/fetch_openi.py --image-bytes 20000000   # + ~20MB of images
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
        resp = httpx.get(
            url,
            headers=headers,
            timeout=timeout,
            follow_redirects=True,
        )
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


def _download_range(url: str, num_bytes: int, timeout: float = 60.0) -> bytes:
    headers = {"Range": f"bytes=0-{num_bytes - 1}"}
    try:
        resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
    except httpx.HTTPError as exc:
        _fail(url, f"connection error ({exc})")
        raise
    if resp.status_code not in (200, 206):
        _fail(url, f"HTTP {resp.status_code}")
    return resp.content


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
        "--skip-reports",
        action="store_true",
        help="Skip the reports archive (rarely useful; it's only 1.1 MB).",
    )
    args = parser.parse_args()

    if args.full and args.image_bytes:
        raise SystemExit("fetch_openi: pass either --full or --image-bytes, not both.")

    from medscope.config import Settings

    root = Settings().openi_root
    root.mkdir(parents=True, exist_ok=True)

    if not args.skip_reports:
        fetch_reports(root)

    if args.full:
        fetch_images_full(root)
    elif args.image_bytes:
        fetch_images_partial(root, args.image_bytes)
    else:
        logger.info(
            "no image option given (--image-bytes N or --full) -- reports only fetched."
        )

    logger.info(
        "NOTE: NLMCXR_dcm.tgz (DICOM) and the Open-i JSON search API are not "
        "attempted by this script -- both were confirmed unreachable from this network."
    )


if __name__ == "__main__":
    main()
