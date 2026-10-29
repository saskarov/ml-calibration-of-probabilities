import numpy as np
import pandas as pd
import pytest

from calib.data import DATASETS, Preprocessor, load_dataset, split_summary, stratified_split


# ---------------------------------------------------------------- stratified_split


def _imbalanced_labels(n: int = 1003, pos_share: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(n) < pos_share).astype(int)


@pytest.mark.parametrize("seed", range(5))
def test_split_disjoint_and_complete(seed):
    y = _imbalanced_labels(seed=seed)
    parts = stratified_split(y, rng=seed)
    all_idx = np.concatenate(parts)
    assert len(all_idx) == len(y)
    assert len(np.unique(all_idx)) == len(y)
    np.testing.assert_array_equal(np.sort(all_idx), np.arange(len(y)))


@pytest.mark.parametrize("fractions", [(0.6, 0.2, 0.2), (0.5, 0.5), (0.7, 0.1, 0.1, 0.1)])
def test_split_keeps_class_proportions(fractions):
    y = _imbalanced_labels()
    parts = stratified_split(y, fractions, rng=1)
    for c in np.unique(y):
        n_c = np.sum(y == c)
        for frac, idx in zip(fractions, parts):
            assert abs(np.sum(y[idx] == c) - frac * n_c) <= 1


def test_split_reproducible_and_seed_dependent():
    y = _imbalanced_labels()
    a = stratified_split(y, rng=42)
    b = stratified_split(y, rng=np.random.default_rng(42))
    c = stratified_split(y, rng=43)
    for x, z in zip(a, b):
        np.testing.assert_array_equal(x, z)
    assert not np.array_equal(a[0], c[0])


@pytest.mark.parametrize("fractions", [(0.6, 0.2), (0.6, 0.5, -0.1)])
def test_split_rejects_bad_fractions(fractions):
    with pytest.raises(ValueError):
        stratified_split(np.array([0, 1] * 10), fractions, rng=0)


# ---------------------------------------------------------------- Preprocessor


@pytest.fixture
def train_test_frames():
    train = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0],
        "const": [5.0, 5.0, 5.0, 5.0],
        "color": pd.Categorical(["red", "blue", "red", "green"]),
    })
    test = pd.DataFrame({
        "num": [10.0, -2.0],
        "const": [7.0, 5.0],
        "color": pd.Categorical(["blue", "purple"]),  # purple is unseen in training
    })
    return train, test


def test_preprocessor_uses_training_statistics(train_test_frames):
    train, test = train_test_frames
    pre = Preprocessor().fit(train)
    Z_train = pre.transform(train)
    Z_test = pre.transform(test)

    np.testing.assert_allclose(pre.mean_, [2.5, 5.0])
    np.testing.assert_allclose(pre.std_, [np.std([1, 2, 3, 4]), 1.0])  # constant column -> std 1
    np.testing.assert_allclose(Z_train[:, 0].mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(Z_train[:, 0].std(), 1.0)
    # Test rows are scaled with the training mean/std, not their own.
    np.testing.assert_allclose(Z_test[:, 0], (test["num"] - 2.5) / np.std([1, 2, 3, 4]))
    np.testing.assert_allclose(Z_test[:, 1], [2.0, 0.0])


def test_preprocessor_one_hot_and_unseen_categories(train_test_frames):
    train, test = train_test_frames
    pre = Preprocessor().fit(train)
    assert pre.feature_names_ == ["num", "const", "color=blue", "color=green", "color=red"]

    Z_train = pre.transform(train)
    np.testing.assert_array_equal(Z_train[:, 2:], [[0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0]])

    Z_test = pre.transform(test)
    np.testing.assert_array_equal(Z_test[:, 2:], [[1, 0, 0], [0, 0, 0]])  # unseen -> all zeros


def test_preprocessor_rejects_different_columns(train_test_frames):
    train, test = train_test_frames
    pre = Preprocessor().fit(train)
    with pytest.raises(ValueError):
        pre.transform(test[["color", "num", "const"]])


# ---------------------------------------------------------------- load_dataset (downloads on first run)


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_load_dataset(name):
    X, y = load_dataset(name)
    spec = DATASETS[name]
    assert X.shape == (spec["n_rows"], spec["n_features"])
    assert set(np.unique(y)) == {0, 1}
    assert not X.isna().any().any()
    assert X.attrs["n_raw"] - X.attrs["n_dropped"] == len(y)


def test_split_summary():
    y = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    splits = (np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7]), np.array([8, 9]))
    table = split_summary(y, splits)
    assert list(table["split"]) == ["train", "cal", "test"]
    assert list(table["n"]) == [6, 2, 2]
    assert list(table["n_pos"]) == [2, 1, 0]
    np.testing.assert_allclose(table["pos_share"], [2 / 6, 0.5, 0.0])
