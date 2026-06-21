"""Plot helpers for the HelioGuard master notebook.

These are pure formatting wrappers around matplotlib - kept out of the
notebook so the narrative reads cleanly. None of them save to disk:
the notebook owns the save decisions.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .metrics import reliability_bins, risk_coverage_curve


def reliability_diagram(
    y_true: np.ndarray,
    probas: dict[str, np.ndarray],
    *,
    n_bins: int = 10,
    ax: plt.Axes | None = None,
    title: str = "Reliability diagram",
) -> plt.Axes:
    """Overlay one curve per probability vector in ``probas``.

    Parameters
    ----------
    y_true
        Binary ground truth.
    probas
        Mapping ``label -> predicted probabilities`` - typically the
        ``{"raw": ..., "platt": ..., "isotonic": ..., "conformal": ...}``
        comparison from §9 of the notebook.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="perfect")
    for name, p in probas.items():
        rb = reliability_bins(y_true, p, n_bins=n_bins)
        keep = rb.counts > 0
        ax.plot(
            rb.mean_predicted[keep],
            rb.fraction_positive[keep],
            "o-",
            label=name,
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.3)
    return ax


def risk_coverage_plot(
    y_true: np.ndarray,
    p: np.ndarray,
    *,
    score: str = "tss",
    ax: plt.Axes | None = None,
    label: str | None = None,
    title: str = "Risk-coverage curve",
) -> plt.Axes:
    """Plot the score-on-covered-subset as we sweep the abstention margin."""
    coverage, score_arr, _ = risk_coverage_curve(y_true, p, score=score)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(coverage, score_arr, "o-", label=label or score.upper())
    ax.set_xlabel("Coverage (fraction of samples predicted)")
    ax.set_ylabel(f"{score.upper()} on covered subset")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    if label:
        ax.legend(frameon=False)
    return ax
