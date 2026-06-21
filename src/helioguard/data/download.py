"""
Download raw OMNI2 and NOAA NCEI satellite-anomaly data.

Design notes
------------
* Downloads are **idempotent**: if a file is already present and non-empty,
  we skip it. This makes ``python -m helioguard.data.download`` cheap to
  re-run, which matters for DVC and for development iteration.
* OMNI files are organised one per calendar year (``omni2_YYYY.dat``).
  We mirror that on disk.
* All network calls have a timeout and a clear error message - the worst
  thing a downloader can do is hang silently.

Run as a module::

    python -m helioguard.data.download
    python -m helioguard.data.download --years 1986 1987 1988
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
from pathlib import Path
from typing import Iterable

from helioguard.config import (
    NCEI_ANOMALY_URL,
    OMNI2_BASE_URL,
    RAW_DIR,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


DEFAULT_YEARS: tuple[int, ...] = tuple(range(1985, 1998))


def _download(url: str, dest: Path, timeout: float = 60.0) -> Path:
    """Download ``url`` to ``dest`` unless ``dest`` already exists non-empty.

    Returns the destination path. Raises ``RuntimeError`` on network failure
    so DVC / CI surface a clear error.
    """
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("cache hit: %s (%d bytes)", dest.name, dest.stat().st_size)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("downloading %s -> %s", url, dest)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            tmp.write_bytes(resp.read())
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"failed to download {url}: {exc}") from exc
    tmp.rename(dest)
    logger.info("ok: %s (%d bytes)", dest.name, dest.stat().st_size)
    return dest


def download_omni_year(year: int, raw_dir: Path = RAW_DIR) -> Path:
    """Download a single year of OMNI2 hourly data."""
    fname = f"omni2_{year}.dat"
    url = f"{OMNI2_BASE_URL}/{fname}"
    return _download(url, raw_dir / "omni" / fname)


def download_omni(
    years: Iterable[int] = DEFAULT_YEARS,
    raw_dir: Path = RAW_DIR,
) -> list[Path]:
    """Download a range of OMNI2 years. Errors on individual years do not
    stop the rest - we log and continue, returning the successful paths."""
    out: list[Path] = []
    for y in years:
        try:
            out.append(download_omni_year(y, raw_dir))
        except RuntimeError as exc:
            logger.error("skipping year %d: %s", y, exc)
    return out


def download_ncei_anomalies(raw_dir: Path = RAW_DIR) -> Path:
    """Download the NOAA NCEI Spacecraft Anomalies bulk Excel file."""
    return _download(NCEI_ANOMALY_URL, raw_dir / "ncei" / "anom5j.xls")


def download_all(years: Iterable[int] = DEFAULT_YEARS) -> dict[str, list[Path]]:
    """Run all downloads and return a manifest dict."""
    ensure_dirs()
    return {
        "omni": download_omni(years),
        "ncei": [download_ncei_anomalies()],
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download HelioGuard raw data.")
    p.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(DEFAULT_YEARS),
        help="OMNI years to fetch (default: 1985-1997 inclusive).",
    )
    p.add_argument(
        "--skip-omni", action="store_true", help="Skip OMNI download."
    )
    p.add_argument(
        "--skip-ncei", action="store_true", help="Skip NCEI download."
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging."
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ensure_dirs()
    if not args.skip_omni:
        download_omni(args.years)
    if not args.skip_ncei:
        download_ncei_anomalies()
    return 0


if __name__ == "__main__":
    sys.exit(main())
