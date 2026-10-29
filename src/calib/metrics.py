"""Evaluation metrics, implemented from scratch with numpy.

Conventions: y in {0, 1} is the true label, p in [0, 1] the predicted P(y = 1).
"""

from dataclasses import dataclass

import numpy as np


def _check(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y.ndim != 1 or y.shape != p.shape:
        raise ValueError(f"y and p must be 1-D arrays of equal length, got {y.shape} and {p.shape}")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0 and 1")
    return y, p


def accuracy(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> float:
    """Share of correct predictions when predicting 1 iff p > threshold.

    The strict inequality matches scikit-learn's LR (decision_function > 0) and
    RF (argmax, ties go to class 0).
    """
    y, p = _check(y, p)
    return float(np.mean((p > threshold) == (y == 1)))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    """Mean negative log-likelihood: -(1/N) sum_i [y_i log p_i + (1 - y_i) log(1 - p_i)].

    p is first clipped to [eps, 1 - eps], so that a confident wrong prediction
    (p = 0 for a positive) costs -log(eps) ~ 34.5 instead of infinity.
    """
    y, p = _check(y, p)
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Brier score: (1/N) sum_i (p_i - y_i)^2."""
    y, p = _check(y, p)
    return float(np.mean((p - y) ** 2))


@dataclass
class ReliabilityBins:
    """Per-bin statistics of a reliability diagram (one entry per bin, empty bins included).

    For an empty bin, mean_pred and frac_pos are NaN and count is 0; plots drop them.
    """

    edges: np.ndarray      # n_bins + 1 bin edges
    mean_pred: np.ndarray  # mean predicted probability in the bin
    frac_pos: np.ndarray   # observed fraction of positives in the bin
    count: np.ndarray      # number of predictions in the bin


def reliability_bins(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10, strategy: str = "uniform"
) -> ReliabilityBins:
    """Group predictions into bins and compare mean prediction with observed frequency.

    strategy="uniform":  edges 0, 1/n_bins, ..., 1.
    strategy="quantile": edges at the quantiles of p, so bins hold ~equal counts
                         (ties in p can make some bins empty).
    Bin k is [edges[k], edges[k+1]); the last bin also includes its right edge.
    """
    y, p = _check(y, p)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    # Interior edges only: values below edges[1] go to bin 0, values >= edges[-2] to the last bin.
    bin_of = np.searchsorted(edges[1:-1], p, side="right")
    count = np.bincount(bin_of, minlength=n_bins)
    sum_pred = np.bincount(bin_of, weights=p, minlength=n_bins)
    sum_pos = np.bincount(bin_of, weights=y, minlength=n_bins)

    nonempty = count > 0
    mean_pred = np.full(n_bins, np.nan)
    frac_pos = np.full(n_bins, np.nan)
    mean_pred[nonempty] = sum_pred[nonempty] / count[nonempty]
    frac_pos[nonempty] = sum_pos[nonempty] / count[nonempty]
    return ReliabilityBins(edges=edges, mean_pred=mean_pred, frac_pos=frac_pos, count=count)


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10, strategy: str = "uniform") -> float:
    """Expected calibration error: sum_bins (count_b / N) * |frac_pos_b - mean_pred_b|.

    Empty bins have weight 0.
    """
    bins = reliability_bins(y, p, n_bins=n_bins, strategy=strategy)
    nonempty = bins.count > 0
    gaps = np.abs(bins.frac_pos[nonempty] - bins.mean_pred[nonempty])
    return float(np.sum(bins.count[nonempty] * gaps) / np.sum(bins.count))


def average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks 1..N of x; tied values all receive the mean of the ranks they occupy."""
    x = np.asarray(x)
    n = len(x)
    order = np.argsort(x, kind="stable")
    x_sorted = x[order]
    # Tie groups are the runs of equal values in x_sorted: [starts[g], ends[g]).
    starts = np.r_[0, np.flatnonzero(np.diff(x_sorted)) + 1]
    ends = np.r_[starts[1:], n]
    # The group occupies ranks starts+1, ..., ends, whose mean is (starts + 1 + ends) / 2.
    group_rank = (starts + 1 + ends) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.repeat(group_rank, ends - starts)
    return ranks


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic.

    AUC = P(p_pos > p_neg) + 0.5 * P(p_pos = p_neg)
        = (R_pos - n_pos (n_pos + 1) / 2) / (n_pos * n_neg),
    where R_pos is the sum of the (average, for ties) ranks of the positives.
    """
    y, p = _check(y, p)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc needs both classes")
    rank_sum_pos = average_ranks(p)[y == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
