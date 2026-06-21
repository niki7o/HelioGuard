"""
Project-wide constants and configuration.

Everything that is "a magic number or magic string" lives here so that the
notebooks and modules never duplicate it. Single source of truth.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
MLRUNS_DIR: Path = PROJECT_ROOT / "mlruns"

OMNI2_BASE_URL: str = (
    "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni"
)
"""NASA SPDF mirror of the OMNI2 hourly archive. One file per calendar year,
named ``omni2_YYYY.dat`` (fixed-width ASCII, ~2.8 MB)."""

NCEI_ANOMALY_URL: str = (
    "https://www.ngdc.noaa.gov/stp/space-weather/satellite-data/"
    "spacecraft-anomalies/data/anom5j.xls"
)
"""NOAA NCEI Spacecraft Anomalies database - bulk Excel file (~2.6 MB)
covering 5,033 cataloged on-orbit anomalies."""

OMNI2_COLUMNS: list[str] = [
    "year", "doy", "hour",
    "bartels_rotation", "imf_sc_id", "plasma_sc_id",
    "n_pts_imf", "n_pts_plasma",
    "B_mag_avg",
    "B_vec_mag",
    "B_lat_angle", "B_lon_angle",
    "Bx_gse",
    "By_gse", "Bz_gse",
    "By_gsm", "Bz_gsm",
    "sigma_B_mag", "sigma_B_vec",
    "sigma_Bx", "sigma_By", "sigma_Bz",
    "proton_temp",
    "proton_density",
    "flow_speed",
    "flow_lon_angle", "flow_lat_angle",
    "alpha_proton_ratio",
    "flow_pressure",
    "sigma_T", "sigma_N", "sigma_V",
    "sigma_phi_V", "sigma_theta_V", "sigma_NaNp",
    "E_field",
    "plasma_beta",
    "alfven_mach",
    "Kp",
    "R_sunspot",
    "Dst",
    "AE",
    "p_flux_gt_1MeV",
    "p_flux_gt_2MeV",
    "p_flux_gt_4MeV",
    "p_flux_gt_10MeV",
    "p_flux_gt_30MeV",
    "p_flux_gt_60MeV",
    "flag",
    "ap",
    "F107",
    "PCN",
    "AL", "AU",
    "magnetosonic_mach",
]

OMNI_FILL_THRESHOLDS: dict[str, float] = {
    "B_mag_avg": 999.0, "B_vec_mag": 999.0,
    "Bx_gse": 999.0, "By_gse": 999.0, "Bz_gse": 999.0,
    "By_gsm": 999.0, "Bz_gsm": 999.0,
    "sigma_B_mag": 999.0, "sigma_B_vec": 999.0,
    "sigma_Bx": 999.0, "sigma_By": 999.0, "sigma_Bz": 999.0,
    "proton_temp": 1e7,
    "proton_density": 999.0,
    "flow_speed": 9999.0,
    "flow_lon_angle": 999.0, "flow_lat_angle": 999.0,
    "alpha_proton_ratio": 9.0,
    "flow_pressure": 99.0,
    "E_field": 999.0,
    "plasma_beta": 999.0,
    "alfven_mach": 999.0,
    "magnetosonic_mach": 99.0,
    "Kp": 99.0,
    "R_sunspot": 999.0,
    "Dst": 99999.0,
    "AE": 9999.0, "AL": 9999.0, "AU": 9999.0,
    "ap": 999.0,
    "F107": 999.0,
    "PCN": 999.0,
}

ADIAG_DESCRIPTIONS: dict[str, str] = {
    "ESD":   "Electrostatic discharge (surface charging by keV electrons)",
    "ECEMP": "NCEI diagnosis code for internal / deep-dielectric charging (MeV electrons)",
    "SEU":   "Single-event upset (energetic particles, esp. cosmic rays)",
    "RFI":   "Radio-frequency interference",
    "SDC":   "Spacecraft dynamic / configuration",
    "UNK":   "Unknown / unattributed",
}

ORBIT_DESCRIPTIONS: dict[str, str] = {
    "G": "Geostationary",
    "C": "Circular (non-GEO)",
    "I": "Inclined",
    "E": "Elliptical",
    "P": "Polar",
    "V": "Vertical (suborbital / sounding rocket)",
    "S": "Sun-synchronous",
    "M": "Molniya",
    "D": "Deep space",
}

RANDOM_STATE: int = 42
"""Single global random seed used everywhere a seed is needed."""


def ensure_dirs() -> None:
    """Create the standard data / report directories if they do not exist."""
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, MLRUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)
