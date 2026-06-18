"""Smoke tests for the data parsers.

These do **not** exercise the network — they assume the raw files are
already cached under ``data/raw/``. The full pipeline test that downloads
data lives in ``tests/test_pipeline_integration.py`` (added later).
"""

from __future__ import annotations

import pandas as pd
import pytest

from helioguard.config import RAW_DIR
from helioguard.data.ncei import daily_anomaly_counts, load_ncei_anomalies
from helioguard.data.omni import load_omni, omni_coverage_report


# Skip these if the raw data has not been downloaded yet — keeps CI useful
# even on a clean checkout.
pytestmark = pytest.mark.skipif(
    not (RAW_DIR / "ncei" / "anom5j.xls").exists(),
    reason="Raw data not downloaded; run `python -m helioguard.data.download`.",
)


class TestOmniParser:
    def test_load_one_year(self):
        df = load_omni(years=[2015])
        # OMNI2 hourly has 8760 rows in a non-leap year.
        assert len(df) == 8760
        assert df.index.is_monotonic_increasing
        assert df.index.tz is not None  # UTC-aware

    def test_kp_decoded(self):
        df = load_omni(years=[2015])
        # Kp_real should be in [0, 9].
        kp = df["Kp_real"].dropna()
        assert kp.min() >= 0
        assert kp.max() <= 9

    def test_fill_values_become_nan(self):
        df = load_omni(years=[2015])
        # No remaining 999.* sentinels in B_mag_avg after fill conversion.
        assert (df["B_mag_avg"].abs() < 900).all() | df["B_mag_avg"].isna().all()

    def test_coverage_report_shape(self):
        df = load_omni(years=[2015])
        rep = omni_coverage_report(df[["B_mag_avg", "Bz_gsm"]])
        assert set(rep.columns) == {"missing", "present", "pct_missing"}
        assert (rep["pct_missing"] >= 0).all()
        assert (rep["pct_missing"] <= 100).all()


class TestNceiParser:
    def test_uses_adate_not_edate(self):
        """The non-negotiable Day-1 finding: ADATE is the anomaly date."""
        ncei = load_ncei_anomalies()
        # ADATE-based index should span ~1963 to ~1994, not just 1986+.
        assert ncei.index.min().year <= 1970
        assert ncei.index.max().year >= 1993
        assert ncei.index.name == "anomaly_date"

    def test_diagnosis_categories(self):
        ncei = load_ncei_anomalies()
        diags = set(ncei["ADIAG"].dropna().unique())
        # The three environmental classes must all be present.
        assert {"ESD", "ECEMP", "SEU"}.issubset(diags)

    def test_environmental_flag(self):
        ncei = load_ncei_anomalies()
        env_subset = ncei[ncei["is_environmental"]]
        # The flag must agree with the explicit diagnosis set.
        assert env_subset["ADIAG"].isin({"ESD", "ECEMP", "SEU"}).all()

    def test_daily_panel_is_continuous(self):
        ncei = load_ncei_anomalies()
        daily = daily_anomaly_counts(ncei, fill_calendar=True)
        # Continuous daily index — no gaps after fill_calendar=True.
        expected = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        assert len(daily) == len(expected)
        assert (daily.index == expected).all()

    def test_base_rate_in_plausible_range(self):
        """A sanity check on the headline target: the environmental-day
        base rate must lie in a realistic range. If it ever drifts outside
        5-25 % we want the test to fail loudly so we re-audit."""
        ncei = load_ncei_anomalies()
        daily = daily_anomaly_counts(ncei, fill_calendar=True)
        rate = daily["any_environmental"].mean()
        assert 0.05 < rate < 0.25, f"unexpected base rate {rate:.3f}"
