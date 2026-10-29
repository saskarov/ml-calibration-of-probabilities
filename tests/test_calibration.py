import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from calib.calibration import IsotonicCalibrator, PlattScaler, sigmoid


def _scores_and_labels(seed: int, n: int = 1000, ties: bool = False):
    rng = np.random.default_rng(seed)
    s = rng.normal(scale=2.0, size=n)
    if ties:
        s = np.round(s, 1)  # many repeated scores
    y = (rng.random(n) < sigmoid(1.5 * s - 0.5)).astype(int)
    return s, y


# ---------------------------------------------------------------- sigmoid


def test_sigmoid_is_stable_for_extreme_inputs():
    z = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
    with np.errstate(over="raise", invalid="raise"):  # (underflow to 0 is harmless)
        p = sigmoid(z)
    np.testing.assert_allclose(p, [0.0, 1 / (1 + np.exp(50)), 0.5, 1 / (1 + np.exp(-50)), 1.0])


# ---------------------------------------------------------------- Platt


@pytest.mark.parametrize("seed", range(5))
def test_platt_matches_unregularized_logistic_regression(seed):
    s, y = _scores_and_labels(seed)
    platt = PlattScaler().fit(s, y, smooth_targets=False)
    # C = inf means no regularization (`penalty=None` is deprecated since scikit-learn 1.8).
    lr = LogisticRegression(C=np.inf, solver="newton-cg", tol=1e-12, max_iter=1000)
    lr.fit(s.reshape(-1, 1), y)
    assert platt.a_ == pytest.approx(lr.coef_[0, 0], abs=1e-5)
    assert platt.b_ == pytest.approx(lr.intercept_[0], abs=1e-5)
    assert platt.n_iter_ < 100  # Newton converged before max_iter


@pytest.mark.parametrize("smooth_targets", [False, True])
def test_platt_recovers_known_sigmoid(smooth_targets):
    rng = np.random.default_rng(0)
    s = rng.normal(scale=2.0, size=50_000)
    y = (rng.random(s.size) < sigmoid(1.7 * s - 0.8)).astype(int)
    platt = PlattScaler().fit(s, y, smooth_targets=smooth_targets)
    assert platt.a_ == pytest.approx(1.7, abs=0.05)
    assert platt.b_ == pytest.approx(-0.8, abs=0.05)


def test_platt_smoothed_targets_keep_separable_fit_finite():
    s = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    y = np.array([0, 0, 0, 1, 1, 1])  # perfectly separated: the MLE has a -> infinity
    platt = PlattScaler().fit(s, y, smooth_targets=True)
    assert np.isfinite(platt.a_) and 0 < platt.a_ < 10
    p = platt.predict_proba(s)
    assert np.all((p > 0) & (p < 1))


def test_platt_is_monotone_increasing_for_informative_scores():
    s, y = _scores_and_labels(0)
    platt = PlattScaler().fit(s, y)
    grid = np.linspace(-10, 10, 101)
    assert platt.a_ > 0
    assert np.all(np.diff(platt.predict_proba(grid)) > 0)


# ---------------------------------------------------------------- isotonic


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("ties", [False, True])
def test_isotonic_output_is_non_decreasing(seed, ties):
    s, y = _scores_and_labels(seed, ties=ties)
    iso = IsotonicCalibrator().fit(s, y)
    assert np.all(np.diff(iso.block_value_) > 0)  # strictly increasing levels
    grid = np.linspace(s.min() - 1, s.max() + 1, 1000)
    assert np.all(np.diff(iso.predict_proba(grid)) >= 0)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("ties", [False, True])
def test_isotonic_matches_sklearn_on_fitted_scores(seed, ties):
    s, y = _scores_and_labels(seed, ties=ties)
    ours = IsotonicCalibrator().fit(s, y)
    ref = IsotonicRegression(out_of_bounds="clip").fit(s, y)
    np.testing.assert_allclose(ours.predict_proba(s), ref.predict(s), atol=1e-12)
    # Inside a block, and outside the fitted range, both are constant and must agree.
    inside = (ours.block_min_ + ours.block_max_) / 2
    outside = np.array([s.min() - 5.0, s.max() + 5.0])
    np.testing.assert_allclose(ours.predict_proba(inside), ref.predict(inside), atol=1e-12)
    np.testing.assert_allclose(ours.predict_proba(outside), ref.predict(outside), atol=1e-12)


def test_isotonic_with_sample_weights_matches_sklearn():
    s, y = _scores_and_labels(0, ties=True)
    w = np.random.default_rng(1).uniform(0.1, 3.0, size=s.size)
    ours = IsotonicCalibrator().fit(s, y, sample_weight=w)
    ref = IsotonicRegression(out_of_bounds="clip").fit(s, y, sample_weight=w)
    np.testing.assert_allclose(ours.predict_proba(s), ref.predict(s), atol=1e-12)


def test_isotonic_leaves_monotone_data_unchanged():
    s = np.array([0.1, 0.4, 0.5, 0.9, 1.3])
    y = np.array([0.0, 0.2, 0.3, 0.7, 1.0])
    iso = IsotonicCalibrator().fit(s, y)
    np.testing.assert_allclose(iso.predict_proba(s), y)
    assert iso.n_blocks_ == 5


def test_isotonic_hand_example():
    # Labels 0, 1, 0, 1 by score: the violation (1, 0) is pooled into 0.5.
    s = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([0, 1, 0, 1])
    iso = IsotonicCalibrator().fit(s, y)
    np.testing.assert_allclose(iso.predict_proba(s), [0.0, 0.5, 0.5, 1.0])
    assert iso.n_blocks_ == 3
    # Step-function lookup: between blocks -> lower block; beyond the ends -> end values.
    np.testing.assert_allclose(iso.predict_proba(np.array([0.0, 1.5, 3.5, 9.0])), [0.0, 0.0, 0.5, 1.0])


def test_isotonic_merges_tied_scores_before_pav():
    s = np.array([1.0, 1.0, 1.0, 2.0])
    y = np.array([0, 1, 1, 1])
    iso = IsotonicCalibrator().fit(s, y)
    np.testing.assert_allclose(iso.predict_proba(np.array([1.0, 2.0])), [2 / 3, 1.0])
