"""
Parse NASA OMNI2 hourly fixed-width records into a tidy ``DataFrame``.

The OMNI2 format is documented at
https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text — 55 numeric
fields per record, whitespace-delimited. Different columns use different
"missing data" fill values (e.g. ``999.9`` for field magnitudes, ``999999.``
for proton temperature). We translate all of them to ``NaN`` so downstream
code never has to remember which sentinel applies where.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from helioguard.config import (
    OMNI2_COLUMNS,
    OMNI_FILL_THRESHOLDS,
    RAW_DIR,
)

logger = logging.getLogger(__name__)


def _parse_one_year(path: Path) -> pd.DataFrame:
    """Read one ``omni2_YYYY.dat`` file into a DataFrame.

    The file is whitespace-delimited with no header; ``OMNI2_COLUMNS`` gives
    the field names in order. We coerce a proper UTC timestamp from
    ``year + doy + hour`` and drop the three components after the join.
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=OMNI2_COLUMNS,
        engine="python",
    )
    df["timestamp"] = (
        pd.to_datetime(df["year"].astype(int).astype(str), format="%Y", utc=True)
        + pd.to_timedelta(df["doy"].astype(int) - 1, unit="D")
        + pd.to_timedelta(df["hour"].astype(int), unit="h")
    )
    return df.drop(columns=["year", "doy", "hour"])


def _apply_fill_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace OMNI sentinel "missing" values with NaN, in place-ish.

    We use a per-column threshold: any value ``>= threshold`` is treated as
    missing. This is robust to the different fill conventions across columns
    (999.9, 9999.99, 999999., etc.).
    """
    out = df.copy()
    for col, thresh in OMNI_FILL_THRESHOLDS.items():
        if col in out.columns:
            mask = out[col].abs() >= thresh
            out.loc[mask, col] = np.nan
    return out


def _decode_kp(df: pd.DataFrame) -> pd.DataFrame:
    """OMNI2 stores Kp as the integer ``Kp × 10`` (so Kp 2.3 → 23). Restore
    the real-valued planetary K-index in a new ``Kp_real`` column and keep
    the raw integer for traceability."""
    if "Kp" in df.columns:
        df = df.copy()
        df["Kp_real"] = df["Kp"] / 10.0
    return df


def load_omni(
    years: Iterable[int] | None = None,
    raw_dir: Path = RAW_DIR,
) -> pd.DataFrame:
    """Load and concatenate OMNI2 hourly data for ``years``.

    If ``years`` is None, every ``omni2_*.dat`` file present on disk is read.
    The result is sorted by timestamp, indexed by it, and has fill values
    converted to NaN.
    """
    omni_dir = raw_dir / "omni"
    if years is None:
        files = sorted(omni_dir.glob("omni2_*.dat"))
    else:
        files = [omni_dir / f"omni2_{y}.dat" for y in years]
        files = [f for f in files if f.exists()]
    if not files:
        raise FileNotFoundError(
            f"No OMNI files found in {omni_dir}. "
            "Run `python -m helioguard.data.download` first."
        )
    logger.info("loading %d OMNI year-files from %s", len(files), omni_dir)
    df = pd.concat([_parse_one_year(f) for f in files], ignore_index=True)
    df = _apply_fill_values(df)
    df = _decode_kp(df)
    df = df.set_index("timestamp").sort_index()
    df = df.dropna(how="all")
    logger.info(
        "OMNI loaded: %d rows, range %s -> %s",
        len(df), df.index.min(), df.index.max(),
    )
    return df


def omni_coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column missingness report.

    Useful in the audit notebook to show *which* OMNI fields are actually
    usable in our chosen window — the early OMNI years are notoriously
    sparse for plasma fields, for example.
    """
    n = len(df)
    miss = df.isna().sum()
    return (
        pd.DataFrame({"missing": miss, "present": n - miss})
        .assign(pct_missing=lambda x: 100.0 * x["missing"] / n)
        .sort_values("pct_missing")
    )
