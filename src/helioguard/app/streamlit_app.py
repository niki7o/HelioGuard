"""
HelioGuard interactive demo (Streamlit).

Run from the project root with::

    streamlit run src/helioguard/app/streamlit_app.py

The app loads the serialised pipeline produced in section 11 of the
notebook (``mlartifacts/helioguard_pipeline.joblib``) and lets a user
obtain a calibrated next-day anomaly probability in one of two modes:

* **Historical date** — pick a day inside the OMNI coverage window; the
  app builds the exact same daily feature vector the model was trained
  on (via :class:`helioguard.features.FeaturePipeline`) and scores it.
  This is the rigorous path.
* **What-if scenario** — set the daily-mean solar-wind drivers with
  sliders; the app fills each driver's lag / rolling columns with the
  chosen value and lets the fitted imputer fill the rest with training
  medians. This is an approximation, clearly labelled as such, useful
  for building intuition.

In both modes the output is the calibrated probability, the confidence
margin |p - 0.5|, and the abstain / alert decision using the abstention
margin selected on the locked test set (stored inside the bundle).
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- make `helioguard` importable when run via `streamlit run ...` -------
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from helioguard.config import DATA_DIR  # noqa: E402
from helioguard.features import DEFAULT_DRIVERS, FeaturePipeline  # noqa: E402

BUNDLE_PATH = ROOT / "mlartifacts" / "helioguard_pipeline.joblib"

# Sliders we expose in scenario mode, with (label, min, max, default, unit).
# These are the physically dominant drivers; everything else falls back to
# the training-set median through the fitted imputer.
SCENARIO_DRIVERS = {
    "Bz_gsm": ("IMF Bz (GSM)", -30.0, 30.0, -5.0, "nT"),
    "flow_speed": ("Solar-wind speed", 250.0, 1000.0, 450.0, "km/s"),
    "proton_density": ("Proton density", 0.5, 60.0, 7.0, "n/cc"),
    "Dst": ("Dst index", -400.0, 50.0, -30.0, "nT"),
    "Kp_real": ("Kp index", 0.0, 9.0, 3.0, ""),
}


# =========================================================================
# Loading
# =========================================================================
@st.cache_resource
def load_bundle() -> dict:
    """Load and cache the serialised pipeline bundle."""
    if not BUNDLE_PATH.exists():
        st.error(
            f"Pipeline bundle not found at {BUNDLE_PATH}.\n\n"
            "Run the notebook through section 11 first to create it "
            "(`python scripts/build_notebook.py` then execute the notebook)."
        )
        st.stop()
    return joblib.load(BUNDLE_PATH)


@st.cache_data
def score_probability(_bundle: dict, feature_row: pd.DataFrame) -> float:
    """Run a single feature row through model + calibrator → probability.

    The leading underscore on ``_bundle`` tells Streamlit not to try to
    hash the (unhashable) fitted estimators.
    """
    X = _bundle["imputer"].transform(feature_row[_bundle["feature_names"]])
    X = _bundle["scaler"].transform(X)
    p_raw = _bundle["model"].predict_proba(X)[:, 1]
    return float(_bundle["calibrator"].transform(p_raw)[0])


# =========================================================================
# Feature-row construction
# =========================================================================
def scenario_feature_row(bundle: dict, scenario: dict, doy: int) -> pd.DataFrame:
    """Approximate feature row from daily-mean driver values.

    Every lag / rolling-mean column of a controlled driver is filled with
    that driver's scenario value; rolling-std columns and uncontrolled
    drivers are left NaN so the fitted median imputer fills them. Storm
    phase is derived from the Dst slider; day-of-year is encoded cyclically.
    """
    names = bundle["feature_names"]
    row = pd.Series(np.nan, index=names, dtype=float)

    for driver, value in scenario.items():
        for col in names:
            if col.startswith(f"{driver}_lag") or col.startswith(f"{driver}_rmean"):
                row[col] = value

    # Storm-phase dummies from the Dst slider (mirrors features._storm_phase).
    dst = scenario.get("Dst", 0.0)
    for col, on in (
        ("phase_quiet", dst > -30),
        ("phase_main", dst <= -30),       # assume deepening for a what-if
        ("phase_recovery", False),
    ):
        if col in row.index:
            row[col] = 1.0 if on else 0.0

    if "doy_sin" in row.index:
        row["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
    if "doy_cos" in row.index:
        row["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)

    return row.to_frame().T


@st.cache_data
def historical_feature_row(_bundle: dict, date: pd.Timestamp) -> pd.DataFrame | None:
    """Build the exact training-time feature vector for a historical day.

    Loads the cached OMNI year containing ``date``, runs the real feature
    pipeline (with the bundle's fitted transformers swapped in), and
    returns the single daily row. Returns None if OMNI for that day is
    not on disk.
    """
    from helioguard.data.omni import load_omni

    year = date.year
    try:
        omni = load_omni(years=[year - 1, year])  # need prior year for lags
    except FileNotFoundError:
        return None

    pipe = FeaturePipeline()
    pipe.imputer_ = _bundle["imputer"]
    pipe.scaler_ = _bundle["scaler"]
    pipe.feature_names_ = _bundle["feature_names"]

    # Daily features are built internally; we ask for just our target day.
    daily = pipe._build_daily(omni)
    if date.normalize() not in daily.index:
        return None
    # Return the raw (un-scaled) engineered row; score_probability scales it.
    return daily.loc[[date.normalize()], _bundle["feature_names"]]


# =========================================================================
# Decision + gauge
# =========================================================================
def decide(p: float, margin: float) -> tuple[str, str]:
    """Map a probability to a decision string + colour, given the margin."""
    if abs(p - 0.5) < margin:
        return "ABSTAIN — low confidence", "#9e9e9e"
    if p >= 0.5:
        return "ALERT — elevated anomaly risk", "#d32f2f"
    return "NO ALERT — low anomaly risk", "#388e3c"


def probability_gauge(p: float, margin: float) -> go.Figure:
    """A 0–1 gauge with the abstention band shaded around 0.5."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(100 * p, 1),
            number={"suffix": " %"},
            title={"text": "P(next-day environmental anomaly)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1565c0"},
                "steps": [
                    {"range": [100 * (0.5 - margin), 100 * (0.5 + margin)],
                     "color": "#eceff1"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(t=60, b=10, l=30, r=30))
    return fig


# =========================================================================
# Layout
# =========================================================================
def main() -> None:
    st.set_page_config(page_title="HelioGuard", page_icon="🛰️", layout="centered")
    st.title("🛰️ HelioGuard")
    st.caption(
        "Calibrated next-day satellite-anomaly risk from solar-wind state. "
        "Model: gradient boosting + isotonic calibration, evaluated on a "
        "locked 1992–1994 test fold."
    )

    bundle = load_bundle()
    margin = float(bundle.get("abstention_margin", 0.05))

    mode = st.radio(
        "Input mode",
        ["What-if scenario", "Historical date"],
        horizontal=True,
        help="Scenario mode is an approximation; historical mode uses the "
             "exact training-time feature pipeline.",
    )

    if mode == "What-if scenario":
        st.subheader("Set the daily-mean solar-wind state")
        scenario: dict[str, float] = {}
        cols = st.columns(2)
        for i, (driver, (label, lo, hi, default, unit)) in enumerate(
            SCENARIO_DRIVERS.items()
        ):
            with cols[i % 2]:
                scenario[driver] = st.slider(
                    f"{label}" + (f" [{unit}]" if unit else ""),
                    lo, hi, default,
                )
        doy = st.slider("Day of year", 1, 366, 80,
                        help="Captures the equinoctial activity peak.")
        row = scenario_feature_row(bundle, scenario, doy)
        st.info(
            "Scenario mode fills each driver's lag/rolling columns with the "
            "slider value and imputes the rest with training medians. Treat "
            "the number as indicative, not a rigorous forecast."
        )
    else:
        st.subheader("Pick a day inside the OMNI coverage window")
        date = st.date_input(
            "Date (UTC)",
            value=pd.Timestamp("1989-03-13"),  # the Quebec storm
            min_value=pd.Timestamp("1974-01-02"),
            max_value=pd.Timestamp("1994-09-11"),
        )
        date = pd.Timestamp(date)
        row = historical_feature_row(bundle, date)
        if row is None:
            st.warning(
                "OMNI data for that day is not cached locally. Run "
                "`python -m helioguard.data.download` for the relevant year, "
                "or pick a date inside the downloaded window."
            )
            st.stop()
        st.caption(f"Scoring the real solar-wind state of {date.date()}.")

    # --- score + display ------------------------------------------------
    p = score_probability(bundle, row)
    decision, colour = decide(p, margin)

    st.plotly_chart(probability_gauge(p, margin), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Probability", f"{p:.1%}")
    c2.metric("Confidence |p−0.5|", f"{abs(p - 0.5):.3f}")
    c3.metric("Abstention margin", f"{margin:.3f}")

    st.markdown(
        f"<div style='padding:0.75rem;border-radius:0.5rem;"
        f"background:{colour};color:white;font-weight:600;text-align:center'>"
        f"{decision}</div>",
        unsafe_allow_html=True,
    )

    with st.expander("How to read this"):
        st.markdown(
            "- The model outputs a **calibrated** probability: a value of "
            "0.3 means roughly a 30 % empirical chance, not just a high score.\n"
            "- Because environmental-anomaly days are rare (~8 %), a "
            "well-calibrated model **rarely crosses 0.5**. The shaded band on "
            "the gauge is the **abstention zone**: inside it the model is too "
            "unsure and declines to call.\n"
            "- The abstention margin was selected on the locked test fold "
            "(section 10 of the notebook) and is stored inside the model "
            "bundle, so the app and the notebook always agree."
        )


if __name__ == "__main__":
    main()
