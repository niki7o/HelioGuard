# Serving the HelioGuard pipeline

The final calibrated pipeline is logged as an MLflow artifact during
§11 of [`notebooks/helioguard.ipynb`](../notebooks/helioguard.ipynb).
This document is the minimal recipe for loading it from another process
and computing a calibrated next-day anomaly probability for a single
day of OMNI driver inputs.

## Artifact layout

`mlartifacts/helioguard_pipeline.joblib` is a single `dict` produced by
`joblib.dump`:

| key | type | role |
|---|---|---|
| `model` | fitted sklearn classifier | the headline supervised model (selected by validation TSS in §7) |
| `calibrator` | fitted `sklearn.isotonic.IsotonicRegression` | post-hoc recalibrator from §10 |
| `feature_names` | `list[str]` | the exact column order the model expects |
| `imputer` | fitted `sklearn.impute.SimpleImputer` | median fill, fitted on the train fold |
| `scaler` | fitted `sklearn.preprocessing.StandardScaler` | z-score, fitted on the train fold |

The imputer + scaler are the same ones from `helioguard.features.FeaturePipeline`
that were fitted at training time, so re-fitting is not needed (and
not allowed - it would re-introduce the leakage the project was built
to avoid).

## Minimal serving snippet

```python
import joblib
import numpy as np
import pandas as pd

from helioguard.data.omni import load_omni
from helioguard.features import FeaturePipeline

bundle = joblib.load("mlartifacts/helioguard_pipeline.joblib")

# Build the same feature representation the model was trained on.
# In production you would point load_omni at a streaming OMNI mirror.
omni_recent = load_omni(years=[2024])  # whatever window you have

# We need the FeaturePipeline only for its feature-building logic;
# transformers are already fitted in the bundle.
pipe = FeaturePipeline()
pipe.imputer_ = bundle["imputer"]
pipe.scaler_ = bundle["scaler"]
pipe.feature_names_ = bundle["feature_names"]

# Synthesise a tiny daily_labels frame for the dates you want to score -
# the labels here are only used for index alignment; their values are
# ignored by the scoring code.
dates = pd.date_range("2024-06-01", "2024-06-30", freq="D")
dummy_labels = pd.DataFrame(
    {"any_environmental": np.zeros(len(dates), dtype=int)},
    index=dates,
)
X_today, _ = pipe.transform(omni_recent, dummy_labels, dates)

p_raw = bundle["model"].predict_proba(X_today)[:, 1]
p_cal = bundle["calibrator"].transform(p_raw)

for d, p in zip(X_today.index, p_cal):
    print(f"{d.date()}  P(environmental anomaly) = {p:.3f}")
```

## Using the abstention threshold from §10

The §10 risk-coverage analysis selects the abstention margin `m*` that
maximises TSS on the covered subset subject to a minimum-coverage
constraint. That value is computed in the notebook and **stored inside
the bundle** as `abstention_margin`, so serving code never has to
hardcode it:

```python
M_STAR = bundle["abstention_margin"]  # selected in §10, not hardcoded

confidence = np.abs(p_cal - 0.5)
abstain = confidence < M_STAR
decision = np.where(abstain, "ABSTAIN",
                    np.where(p_cal >= 0.5, "ALERT", "NO ALERT"))
```

On the 1992-1994 locked test fold the selected margin is small
(`m* ~ 0.05`) and coverage stays near 100 %, because the risk-coverage
curve is nearly flat - confident predictions are only marginally more
skillful than the full set. That is the honest operational reading:
selective prediction does not buy much here. The mechanism is in place
and the threshold is data-selected; the curve simply does not reward
abstention strongly on this data.

## MLflow registry

§11 of the notebook **registers the model automatically** - it calls
`mlflow.sklearn.log_model(..., registered_model_name="helioguard-anomaly")`
and then sets the `@staging` alias on the new version. Browse it in the
UI with:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
```

For programmatic loading of the registered estimator:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")

# MLflow 3.x alias form (set in §11). The classic stage form
# "models:/helioguard-anomaly/Staging" also resolves on older MLflow.
model_uri = "models:/helioguard-anomaly@staging"
estimator = mlflow.sklearn.load_model(model_uri)
```

Note that the registered MLflow model is the **raw sklearn estimator
only**. The calibrator, scaler, imputer, feature-name list, and the
selected abstention margin live in the joblib bundle - load both and
combine them as shown in the snippet above. The bundle is the
single-file source of truth for serving; the registry entry exists for
versioning and stage management.

## What this serving recipe does not do

* It does not stream OMNI in real time. NASA SPDF's
  `omniweb.gsfc.nasa.gov` updates with a multi-day latency; a true
  operational system would consume the near-real-time NOAA SWPC feed
  and re-derive the OMNI-equivalent fields locally.
* It does not provide ensemble uncertainty. The calibrated probability
  *is* an uncertainty quantification, but it does not separate epistemic
  from aleatoric uncertainty. See §12 of the notebook for the longer
  discussion.
