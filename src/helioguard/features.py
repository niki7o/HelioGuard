"""
Leakage-safe feature engineering for the HelioGuard hourly→daily pipeline.

Two design rules drive this module:

1. **Train-only fitting.** Any quantity that requires a statistic over
   history (means, std-devs for scaling, imputation medians) is computed
   on the training fold *only*; the fitted object is then applied to
   validation / test folds. The :class:`FeaturePipeline` class enforces
   this — there is no per-row API that could accidentally look across
   the split.
2. **Past-only windows.** Every rolling / lag operation uses pandas
   semantics that look strictly backwards in time (``closed='left'`` for
   rollings, positive ``shift`` for lags). The day-T row sees only data
   timestamped ``< T``.

The output is a daily feature panel aligned to a daily binary label
``any_environmental`` from the NCEI side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


DEFAULT_DRIVERS: tuple[str, ...] = (
    "B_mag_avg",
    "Bz_gsm",
    "flow_speed",
    "proton_density",
    "flow_pressure",
    "Kp_real",
    "Dst",
    "AE",
    "F107",
    "ap",
)

DEFAULT_LAGS_HOURS: tuple[int, ...] = (1, 3, 6, 24)
DEFAULT_ROLL_HOURS: tuple[int, ...] = (6, 24)


def _hourly_lag_and_roll(
    omni: pd.DataFrame,
    drivers: Iterable[str],
    lags: Iterable[int],
    rolls: Iterable[int],
) -> pd.DataFrame:
    """Build hourly lag and rolling-stat columns from the OMNI hourly frame.

    Past-only by construction: ``shift(k)`` reads the value k hours ago,
    and rolling windows are closed on the left so the current hour is
    excluded from its own statistic.
    """
    pieces: list[pd.DataFrame] = []
    for col in drivers:
        if col not in omni.columns:
            continue
        s = omni[col]
        for k in lags:
            pieces.append(s.shift(k).rename(f"{col}_lag{k}h"))
        for w in rolls:
            roll = s.shift(1).rolling(window=w, min_periods=max(2, w // 2))
            pieces.append(roll.mean().rename(f"{col}_rmean{w}h"))
            pieces.append(roll.std().rename(f"{col}_rstd{w}h"))
    return pd.concat(pieces, axis=1)


def _storm_phase(dst: pd.Series) -> pd.DataFrame:
    """Crude but useful storm-phase indicator from Dst.

    * ``quiet``    : Dst > -30 nT
    * ``main``     : Dst <= -30 nT and decreasing (dDst/dt < 0)
    * ``recovery`` : Dst <= -30 nT and increasing (dDst/dt >= 0)

    Returned as three 0/1 dummy columns. Past-only: the derivative is the
    backwards difference, no peeking at the future.
    """
    d = dst.copy()
    ddt = d.diff()
    main = (d <= -30) & (ddt < 0)
    recovery = (d <= -30) & (ddt >= 0)
    quiet = ~(main | recovery)
    return pd.DataFrame(
        {
            "phase_quiet": quiet.astype(int),
            "phase_main": main.astype(int),
            "phase_recovery": recovery.astype(int),
        },
        index=d.index,
    )


def _hourly_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Collapse an hourly feature frame to one row per UTC day.

    Each lag/roll column already encodes a temporal window, so we
    summarise the 24 hourly values per day with their mean. Storm-phase
    dummies are summarised with their *max* — a day touched by main
    phase is a main-phase day, even if it was quiet at the other 23 h.
    """
    if hourly.empty:
        return hourly
    is_phase = hourly.columns.str.startswith("phase_")
    phase_cols = hourly.columns[is_phase]
    num_cols = hourly.columns[~is_phase]

    daily_num = hourly[num_cols].resample("1D").mean()
    daily_phase = hourly[phase_cols].resample("1D").max()
    out = daily_num.join(daily_phase, how="outer")
    out.index = out.index.tz_convert(None) if out.index.tz is not None else out.index
    out.index.name = "date"
    return out


def _cyclic_doy(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclic encoding of day-of-year — captures seasonal effects (e.g. the
    equinoctial peak in geomagnetic activity) without imposing a step at
    the year boundary."""
    doy = index.dayofyear.values
    rad = 2 * np.pi * doy / 366.0
    return pd.DataFrame(
        {"doy_sin": np.sin(rad), "doy_cos": np.cos(rad)}, index=index
    )


@dataclass
class FeaturePipeline:
    """Train-only-fitted feature pipeline.

    Usage::

        pipe = FeaturePipeline()
        X_train, y_train = pipe.fit_transform(omni, daily_labels, train_idx)
        X_val,   y_val   = pipe.transform(omni, daily_labels, val_idx)
        X_test,  y_test  = pipe.transform(omni, daily_labels, test_idx)

    The imputer (median) and the scaler (z-score) are fitted *exclusively*
    on the training rows; validation and test rows pass through the
    fitted transformers without contributing to their parameters.

    ``horizon_days`` sets the forecast lead time: with the default of 1,
    the features of day T are aligned to the anomaly label of day T+1,
    so the model is a genuine *next-day* forecast — the solar-wind state
    up to the end of day T predicts whether day T+1 has an anomaly. The
    feature windows are past-only, so there is a full day of lead time
    between the latest input and the predicted day.
    """

    drivers: tuple[str, ...] = DEFAULT_DRIVERS
    lags: tuple[int, ...] = DEFAULT_LAGS_HOURS
    rolls: tuple[int, ...] = DEFAULT_ROLL_HOURS
    label_col: str = "any_environmental"
    horizon_days: int = 1

    imputer_: SimpleImputer | None = field(default=None, init=False)
    scaler_: StandardScaler | None = field(default=None, init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)

    def _build_daily(self, omni: pd.DataFrame) -> pd.DataFrame:
        """OMNI hourly → engineered daily features (no scaling yet)."""
        hourly = _hourly_lag_and_roll(omni, self.drivers, self.lags, self.rolls)
        if "Dst" in omni.columns:
            hourly = hourly.join(_storm_phase(omni["Dst"]), how="left")
        daily = _hourly_to_daily(hourly)
        daily = daily.join(_cyclic_doy(daily.index), how="left")
        return daily

    def _align_label(
        self,
        daily_features: pd.DataFrame,
        daily_labels: pd.DataFrame,
        index: pd.DatetimeIndex,
    ) -> tuple[pd.DataFrame, pd.Series]:
        target = daily_labels[self.label_col].shift(-self.horizon_days)
        common = daily_features.index.intersection(daily_labels.index)
        keep = common.intersection(pd.DatetimeIndex(index))
        X = daily_features.loc[keep]
        y = target.loc[keep]
        mask = y.notna()
        return X.loc[mask], y[mask].astype(int)

    def fit_transform(
        self,
        omni: pd.DataFrame,
        daily_labels: pd.DataFrame,
        train_index: pd.DatetimeIndex,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Fit imputer + scaler on ``train_index`` rows and return the
        transformed training matrix and aligned label."""
        daily = self._build_daily(omni)
        X_train, y_train = self._align_label(daily, daily_labels, train_index)

        self.feature_names_ = list(X_train.columns)
        self.imputer_ = SimpleImputer(strategy="median").fit(X_train)
        X_imp = self.imputer_.transform(X_train)
        self.scaler_ = StandardScaler().fit(X_imp)
        X_scl = self.scaler_.transform(X_imp)

        return (
            pd.DataFrame(X_scl, index=X_train.index, columns=self.feature_names_),
            y_train,
        )

    def transform(
        self,
        omni: pd.DataFrame,
        daily_labels: pd.DataFrame,
        index: pd.DatetimeIndex,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Apply the fitted transformers to a new fold."""
        if self.imputer_ is None or self.scaler_ is None:
            raise RuntimeError("FeaturePipeline.transform called before fit_transform.")
        daily = self._build_daily(omni)
        X, y = self._align_label(daily, daily_labels, index)
        X = X[self.feature_names_]
        X_imp = self.imputer_.transform(X)
        X_scl = self.scaler_.transform(X_imp)
        return (
            pd.DataFrame(X_scl, index=X.index, columns=self.feature_names_),
            y,
        )
