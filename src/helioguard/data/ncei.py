"""
Parse the NOAA NCEI Spacecraft Anomalies bulk Excel file into a tidy DataFrame.

The raw file (``anom5j.xls``) is a legacy CDFV2 ``.xls`` workbook with
5,033 events covering **actual anomaly dates from 1963-02-14 to 1994-09-11**.

.. important::
   **Data-audit finding (logged Day 1).** The intuitive-looking ``EDATE``
   column is in fact the *catalog batch-entry date* - there are only 49
   distinct values, the largest single one (1994-01-19) carrying 883 events
   that span years of real history. The genuine anomaly date is ``ADATE``,
   which has 2,508 distinct values across 32 years. We therefore use
   ``ADATE`` as the primary time index and rename it ``anomaly_date`` in
   the tidied output. This is the kind of leakage trap a careless join
   would walk straight into.

Column glossary (from the NCEI documentation, with our corrections):

* ``VER``      - record format version (5.0 for this file)
* ``EDATE``    - **catalog entry date** (not the anomaly date - see above)
* ``ADATE``    - **actual anomaly date** (UTC) ← used as the event timestamp
* ``BIRD``     - spacecraft identifier (some have a leading ``@``)
* ``STIMEU``   - universal time of the event (HHMM; ``9999`` = missing)
* ``DUR``      - duration
* ``STIMEL``   - local time of the event
* ``ORBIT``    - orbit-type code (G, C, I, …)
* ``NS``       - N/S hemisphere flag for latitude
* ``LAT/LON``  - sub-satellite position
* ``ATYPE``    - anomaly type
* ``ADIAG``    - anomaly diagnosis (ESD / ECEMP / SEU / RFI / SDC / UNK)
* ``ACOMMENT`` - free-text comment
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pandas as pd

from helioguard.config import RAW_DIR

logger = logging.getLogger(__name__)


def load_ncei_anomalies(
    raw_dir: Path = RAW_DIR,
    *,
    keep_unknown: bool = True,
) -> pd.DataFrame:
    """Load and tidy the NCEI anomaly catalogue.

    Parameters
    ----------
    raw_dir
        Project ``data/raw`` directory.
    keep_unknown
        If False, rows with ``ADIAG == "UNK"`` are dropped. Many UNK rows are
        clearly environment-related but not formally diagnosed, so keeping
        them and labelling them as a separate class is usually better than
        dropping them.

    Returns
    -------
    DataFrame indexed by event date, with parsed types, cleaned strings,
    and an ``is_environmental`` boolean flag (True for ESD/ECEMP/SEU,
    which are the three diagnoses with a clean space-weather link).
    """
    path = raw_dir / "ncei" / "anom5j.xls"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m helioguard.data.download` first."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_excel(path, engine="xlrd")
    logger.info("NCEI loaded: %d rows, %d cols", *df.shape)

    df["anomaly_date"] = pd.to_datetime(df["ADATE"], errors="coerce")
    df["catalog_entry_date"] = pd.to_datetime(df["EDATE"], errors="coerce")
    df = df.dropna(subset=["anomaly_date"])

    df["stimeu"] = df["STIMEU"].where(df["STIMEU"] != 9999)

    for col in ("BIRD", "ORBIT", "NS", "EW", "ATYPE", "ADIAG"):
        if col in df.columns:
            df[col] = (
                df[col].astype("string").str.strip().str.lstrip("@").str.upper()
            )

    environmental = {"ESD", "ECEMP", "SEU"}
    df["is_environmental"] = df["ADIAG"].isin(environmental)

    if not keep_unknown:
        df = df[df["ADIAG"] != "UNK"]

    df = df.set_index("anomaly_date").sort_index()
    logger.info(
        "NCEI cleaned: %d rows, range %s -> %s, %d environmental",
        len(df), df.index.min(), df.index.max(), int(df["is_environmental"].sum()),
    )
    return df


def daily_anomaly_counts(
    df: pd.DataFrame,
    *,
    only_environmental: bool = True,
    fill_calendar: bool = True,
) -> pd.DataFrame:
    """Aggregate the event catalogue to a daily panel.

    Parameters
    ----------
    df
        Output of :func:`load_ncei_anomalies` (indexed by ``anomaly_date``).
    only_environmental
        If True (default), restrict the ``n_anomalies`` count to
        ESD/ECEMP/SEU events. Per-diagnosis columns are always returned
        in full so the caller can re-aggregate if needed.
    fill_calendar
        If True, the returned index is a continuous daily ``DatetimeIndex``
        from the first to the last anomaly date, with zero-counts on
        quiet days. This is what you want for joining to OMNI.

    Returns
    -------
    DataFrame with columns ``n_anomalies``, ``n_<diag>`` per diagnosis,
    ``any_anomaly``, and ``any_environmental``.
    """
    src = df[df["is_environmental"]] if only_environmental else df
    daily = (
        src.assign(date=lambda x: x.index.normalize())
        .groupby("date")
        .size()
        .rename("n_anomalies")
        .to_frame()
    )

    diag = df.assign(date=df.index.normalize())
    counts = diag.pivot_table(
        index="date",
        columns="ADIAG",
        values="BIRD",
        aggfunc="count",
        fill_value=0,
    )
    counts.columns = [f"n_{c.lower()}" for c in counts.columns]
    out = daily.join(counts, how="outer").fillna(0).astype(int)

    if fill_calendar and not out.empty:
        full = pd.date_range(out.index.min(), out.index.max(), freq="D")
        out = out.reindex(full, fill_value=0)
        out.index.name = "date"

    out["any_anomaly"] = out["n_anomalies"] > 0
    env_cols = [c for c in ("n_esd", "n_ecemp", "n_seu") if c in out.columns]
    out["any_environmental"] = out[env_cols].sum(axis=1) > 0
    return out
