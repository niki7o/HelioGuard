# HelioGuard — project walkthrough

A plain-language tour of every file, so you can explain any part of the
project in an oral defense. Read top to bottom once; after that you can
answer "what does this file do and why is it here?" for anything a
grader points at.

---

## The one-sentence pitch

> We join NASA solar-wind data to a NOAA satellite-anomaly catalogue and
> ask whether the solar-wind state of the past hours can give a
> *calibrated* probability of a satellite anomaly the next day — and
> whether letting the model *abstain* on unsure days improves it. The
> honest answer is: it produces well-calibrated probabilities but only
> weak threshold-0.5 skill, which is itself a meaningful result.

If you can say that paragraph in your own words, you've defended the
"problem formulation and significance" rubric line.

---

## The shape of the project

There are three layers, and the separation between them is deliberate:

1. **`src/helioguard/`** — reusable Python modules. Anything that runs
   more than once, or that is "plumbing" rather than "narrative", lives
   here. This is what earns the *code quality / separation of concerns*
   rubric points, and it is exactly what the instructor asked for in
   earlier feedback ("move the boilerplate, like plotting, out of the
   notebook").
2. **`notebooks/helioguard.ipynb`** — the actual report. It tells the
   story top-to-bottom and *orchestrates* the modules; it does not
   contain plumbing. Every modelling decision and result is visible
   here. This is the primary deliverable.
3. **`scripts/`, `tests/`, `docs/`, `data/`** — supporting cast:
   the notebook generator, the test suite, the documentation, and the
   DVC-tracked data.

A grader who only opens the notebook gets the whole story. A grader who
opens `src/` sees clean, tested, documented code. Both are satisfied.

---

## File-by-file

### `src/helioguard/config.py`
The single source of truth for every constant: file paths, the 55-column
OMNI record layout, the per-column "missing value" sentinels, the
anomaly-diagnosis codes, and the global random seed. **Why it exists:**
so no magic number is ever duplicated between the notebook and the
modules. If you change the data window, you change it in one place.

### `src/helioguard/data/download.py`
Downloads the raw OMNI year-files and the NCEI Excel catalogue. It is
**idempotent** — if a file is already on disk it is skipped — so you can
re-run it safely. **Why it matters:** reproducibility. Anyone can clone
the repo and run one command to fetch the exact data.

### `src/helioguard/data/omni.py`
Parses the NASA OMNI2 fixed-width hourly files. Three non-obvious jobs:
(1) it builds a proper UTC timestamp from the year/day-of-year/hour
columns; (2) it converts each column's "missing" sentinel (999.9,
9999.0, 999999.0 …) into a real `NaN`; (3) it decodes Kp, which OMNI
stores as an integer ten times too large (Kp 2.3 is stored as 23).
**Defense point:** if you skipped any of these the model would train on
garbage — this is the "garbage in, garbage out" discipline.

### `src/helioguard/data/ncei.py`
Parses the NOAA spacecraft-anomaly catalogue and aggregates it to a
daily panel. **The key finding lives here:** the obvious-looking
`EDATE` column is *not* the anomaly date — it is the catalogue
data-entry date (only 49 distinct values, one batch holding 883 events
spanning years). The real anomaly date is `ADATE`. Joining on `EDATE`
would silently corrupt every space-weather link. This is the single
most important data-audit catch in the project.

### `src/helioguard/features.py`
Turns the hourly OMNI frame into a daily feature matrix: lagged values
(1, 3, 6, 24 h), rolling mean and standard deviation (6, 24 h), storm-
phase flags from Dst, and a cyclic day-of-year encoding. **The crucial
design choice:** the `FeaturePipeline` class fits its imputer and scaler
on the *training rows only* and then applies them to validation and test
rows. There is no method that could accidentally compute a statistic
across the train/test boundary. Every rolling window looks strictly
backwards in time. **Defense point:** this is how we guarantee no
leakage — the thing the instructor most cares about.

### `src/helioguard/metrics.py`
The scoring functions: TSS and HSS (the standard space-weather skill
scores, insensitive to class imbalance), Brier score and Expected
Calibration Error (for probability quality), and the risk–coverage
curve (for selective prediction). **Why custom:** scikit-learn does not
ship TSS/HSS or a risk–coverage helper, and writing them yourself shows
you understand the maths rather than calling a black box.

### `src/helioguard/plots.py`
The reliability diagram and risk–coverage plot. Pure formatting, kept
out of the notebook on purpose so the narrative reads cleanly — exactly
the "move boilerplate plotting out of the notebook" advice.

### `src/helioguard/tracking.py`
A three-line helper that points MLflow at a project-local SQLite store.
**Why SQLite:** MLflow 3.x deprecated the plain-file backend, so the
modern, supported choice is a small `mlruns/mlflow.db`. Used for
experiment tracking and the model registry.

### `src/helioguard/app/streamlit_app.py`
The interactive demo. Loads the saved model bundle and gives a
calibrated probability for either a historical date (rigorous: it
rebuilds the real feature vector) or a what-if slider scenario
(approximate, clearly labelled). Shows the probability, the confidence
margin, and the abstain/alert decision. **Defense point:** the app and
the notebook share the *same* abstention margin because it is stored
inside the model bundle — they cannot disagree.

### `notebooks/helioguard.ipynb`
The report, in 13 sections (abstract → data → EDA → three hypotheses →
features → regression & logistic baselines → SVM/RF/GBM → PCA/Isomap +
clustering → calibration → selective prediction on the locked test set →
MLflow registry → limitations → references). It is regenerated from
`scripts/build_notebook.py`, which means the notebook can be reviewed as
plain Python and rebuilt deterministically.

### `scripts/build_notebook.py`
Builds the notebook from Python. **Why not just edit the `.ipynb`:**
a notebook is JSON and painful to review in a pull request; a build
script is readable, diff-able, and guarantees the notebook can always be
regenerated.

### `tests/test_parsers.py`
Nine sanity tests: the OMNI parser produces 8760 rows for a non-leap
year, Kp decodes into [0, 9], sentinels become NaN; the NCEI parser uses
ADATE not EDATE, the daily panel is gap-free, the base rate is in a
plausible range. **Why it matters:** the rubric explicitly rewards
tested code, and these catch exactly the kind of silent data corruption
that would invalidate the whole analysis.

### `data/` + DVC
The raw OMNI and NCEI files are tracked by **DVC**, not git: the small
`.dvc` pointer files are committed, the ~63 MB of payload lives in DVC's
content store. **Why:** keeping large/raw data out of git is best
practice, and the instructor specifically praised using DVC for data
versioning.

### `docs/serving.md`
How to load the saved pipeline in another process and score new days,
including the abstention threshold and the MLflow registry alias.

### `Makefile`
Shortcuts: `make data` (download), `make test` (pytest), `make
mlflow-ui` (open the tracking UI), `make clean`.

---

## The three questions a grader is most likely to ask

**"Where is the leakage protection?"**
`features.py` — the `FeaturePipeline.fit_transform` / `transform` split
fits all statistics on the train fold only, and the test fold (1992–94)
is locked at the top of §2 of the notebook and only touched in §10.

**"Why is your TSS so low if ROC-AUC is 0.67?"**
Because the model is *calibrated*. On a rare event (~8 % positive), a
calibrated model almost never outputs a probability above 0.5, so its
skill *at the 0.5 threshold* is low even though its *ranking* is good.
That is why the headline deliverable is the risk–coverage curve, not a
single confusion matrix. This is the "honest null" — see §12.

**"What is genuinely novel here?"**
Two public datasets that are usually studied separately, fused at daily
resolution, with calibration and selective prediction treated as
first-class deliverables and a strictly leakage-safe chronological
protocol — rather than chasing a single accuracy number.
