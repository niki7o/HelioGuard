"""
Scoring metrics for HelioGuard.

We add space-weather conventions on top of the sklearn defaults:

* :func:`tss`  - True Skill Statistic (a.k.a. Peirce Skill Score) =
  ``TPR - FPR``. Range ``[-1, 1]``. Unlike accuracy, TSS is insensitive
  to class imbalance and is the headline metric in operational
  geomagnetic-storm forecasting.
* :func:`hss`  - Heidke Skill Score = ``2(TP·TN - FP·FN) / ((TP+FN)(FN+TN) + (TP+FP)(FP+TN))``.
  Scores agreement against the climatological random baseline.
* :func:`brier_score` - mean squared error of the predicted probability
  against the 0/1 outcome. Lower is better; perfect = 0.
* :func:`expected_calibration_error` - weighted gap between predicted
  confidence and observed accuracy across equal-width probability bins.
* :func:`reliability_bins` - the bin-by-bin (confidence, accuracy, count)
  used to draw the reliability diagram and to compute ECE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix


def _contingency(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tp), int(fp), int(fn), int(tn)


def tss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True Skill Statistic = sensitivity + specificity - 1.

    Equal to ``TPR - FPR``. The standard operational score for
    geomagnetic-storm forecasting (Bloomfield et al. 2012).
    """
    tp, fp, fn, tn = _contingency(y_true, y_pred)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tpr - fpr


def hss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Heidke Skill Score against a random-chance baseline."""
    tp, fp, fn, tn = _contingency(y_true, y_pred)
    num = 2.0 * (tp * tn - fp * fn)
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return num / den if den > 0 else 0.0


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error between predicted probability and outcome."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y_true) ** 2))


@dataclass
class ReliabilityBins:
    """Container for the reliability-diagram primitives."""

    bin_edges: np.ndarray
    bin_centres: np.ndarray
    mean_predicted: np.ndarray
    fraction_positive: np.ndarray
    counts: np.ndarray


def reliability_bins(
    y_true: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
) -> ReliabilityBins:
    """Equal-width bins over [0, 1]; returns the data needed both to draw
    the reliability diagram and to compute ECE."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bin_id = np.clip(np.digitize(p, edges, right=False) - 1, 0, n_bins - 1)

    mean_pred = np.zeros(n_bins)
    frac_pos = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_id == b
        counts[b] = int(mask.sum())
        if counts[b]:
            mean_pred[b] = p[mask].mean()
            frac_pos[b] = y_true[mask].mean()

    return ReliabilityBins(
        bin_edges=edges,
        bin_centres=centres,
        mean_predicted=mean_pred,
        fraction_positive=frac_pos,
        counts=counts,
    )


def expected_calibration_error(
    y_true: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> float:
    """ECE = sum_b (n_b / N) * |conf_b - acc_b|.

    The standard one-number summary of how far a reliability curve sits
    from the diagonal. Range [0, 1]; smaller is better.
    """
    rb = reliability_bins(y_true, p, n_bins=n_bins)
    n = rb.counts.sum()
    if n == 0:
        return 0.0
    gaps = np.abs(rb.mean_predicted - rb.fraction_positive)
    return float(np.sum(rb.counts * gaps) / n)


def risk_coverage_curve(
    y_true: np.ndarray,
    p: np.ndarray,
    *,
    score: str = "tss",
    n_points: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep an abstention margin around 0.5 and return arrays of
    ``(coverage, score_on_covered, threshold)``.

    The "confidence" of a probabilistic binary classifier is taken as
    ``|p - 0.5|``: predictions near the decision boundary are the
    least confident. We sweep a margin ``m`` from 0 (cover everything)
    upwards, keeping only samples with ``|p - 0.5| >= m``, predicting
    those with the 0.5 threshold, and computing ``score`` on the
    covered subset.

    Returns
    -------
    coverage : array shape (n_points,)
        Fraction of samples kept at each margin.
    score_arr : array shape (n_points,)
        The chosen ``score`` ("tss" or "hss") on the covered subset.
        ``NaN`` where coverage is zero.
    thresholds : array shape (n_points,)
        The margin ``m`` at each grid point.
    """
    y_true = np.asarray(y_true)
    p = np.asarray(p, dtype=float)
    conf = np.abs(p - 0.5)
    margins = np.linspace(0.0, 0.5, n_points)

    score_fn = {"tss": tss, "hss": hss}[score]
    coverage = np.zeros(n_points)
    score_arr = np.full(n_points, np.nan)

    for i, m in enumerate(margins):
        mask = conf >= m
        coverage[i] = mask.mean()
        if mask.sum() > 0 and len(np.unique(y_true[mask])) > 1:
            score_arr[i] = score_fn(y_true[mask], (p[mask] >= 0.5).astype(int))
    return coverage, score_arr, margins
