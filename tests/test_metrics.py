import numpy as np
import pytest
from sklearn import metrics as skm

from calib.metrics import accuracy, average_ranks, brier, ece, log_loss, reliability_bins, roc_auc


def _random_data(seed: int, n: int = 2000, ties: bool = False):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.01, 0.99, size=n)
    if ties:
        p = np.round(p, 1)  # only ~11 distinct values
    y = (rng.random(n) < p).astype(int)
    return y, p


@pytest.mark.parametrize("seed", range(5))
def test_log_loss_and_brier_match_sklearn(seed):
    y, p = _random_data(seed)
    assert log_loss(y, p) == pytest.approx(skm.log_loss(y, p), abs=1e-10)
    assert brier(y, p) == pytest.approx(skm.brier_score_loss(y, p), abs=1e-10)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("ties", [False, True])
def test_roc_auc_matches_sklearn(seed, ties):
    y, p = _random_data(seed, ties=ties)
    assert roc_auc(y, p) == pytest.approx(skm.roc_auc_score(y, p), abs=1e-10)


def test_average_ranks_with_ties():
    np.testing.assert_array_equal(average_ranks(np.array([10, 20, 10, 30, 20, 10])),
                                  [2, 4.5, 2, 6, 4.5, 2])


def test_roc_auc_extremes():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.3, 0.4])) == 1.0
    assert roc_auc(y, np.array([0.4, 0.3, 0.2, 0.1])) == 0.0
    assert roc_auc(y, np.full(4, 0.5)) == 0.5


def test_accuracy_threshold_is_strict():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.5, 0.51, 0.9, 0.2])
    assert accuracy(y, p) == 1.0  # p = 0.5 is predicted as class 0


def test_log_loss_clips_extreme_predictions():
    y = np.array([1, 0])
    p = np.array([0.0, 0.0])  # first prediction is confidently wrong
    assert np.isfinite(log_loss(y, p))
    assert log_loss(y, p) == pytest.approx(-np.log(1e-15) / 2, rel=1e-6)


def test_ece_near_zero_for_calibrated_predictions():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=1_000_000)
    y = (rng.random(p.size) < p).astype(int)  # P(y = 1 | p) = p exactly
    assert ece(y, p) < 0.003
    assert ece(y, p, strategy="quantile") < 0.003


def test_ece_of_overconfident_predictions():
    # Always predicts 0.9, but only half are positive: one bin, gap 0.4.
    y = np.array([1, 0] * 50)
    p = np.full(100, 0.9)
    assert ece(y, p) == pytest.approx(0.4)


def test_reliability_bins_hand_example():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.05, 0.15, 0.15, 0.95, 1.0])  # 1.0 belongs to the last bin
    bins = reliability_bins(y, p, n_bins=10)
    assert bins.count.sum() == 5
    np.testing.assert_array_equal(bins.count[[0, 1, 9]], [1, 2, 2])
    np.testing.assert_allclose(bins.mean_pred[[0, 1, 9]], [0.05, 0.15, 0.975])
    np.testing.assert_allclose(bins.frac_pos[[0, 1, 9]], [0.0, 0.5, 1.0])
    assert np.isnan(bins.mean_pred[5]) and bins.count[5] == 0


def test_reliability_bins_quantile_counts_balanced():
    y, p = _random_data(0, n=1000)
    bins = reliability_bins(y, p, n_bins=10, strategy="quantile")
    assert bins.count.sum() == 1000
    assert bins.count.min() >= 99 and bins.count.max() <= 101
    # Each bin's mean prediction lies within its edges.
    assert np.all(bins.mean_pred >= bins.edges[:-1]) and np.all(bins.mean_pred <= bins.edges[1:])
