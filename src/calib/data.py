"""Dataset loading, stratified train/calibration/test split and preprocessing.

Only `fetch_openml` comes from scikit-learn; the split and the preprocessing are
implemented here with numpy/pandas.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from calib.paths import DATA_DIR

logger = logging.getLogger(__name__)

# OpenML version, label strings and the expected values checked at load time.
# `n_rows` is after cleaning; `pos_share` is the share of the positive class.
DATASETS: dict[str, dict] = {
    "spambase": {
        "version": 1,
        "labels": {"0", "1"},
        "positive": "1",
        "n_raw": 4601,
        "n_rows": 4601,
        "n_features": 57,
        "pos_share": 0.394,
    },
    "adult": {
        "version": 2,
        "labels": {"<=50K", ">50K"},
        "positive": ">50K",
        "n_raw": 48842,
        "n_rows": 45222,
        "n_features": 14,
        "pos_share": 0.248,
    },
}


def fetch_raw(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Download (or read from the `data/` cache) the raw OpenML data: (X, label strings).

    Checks the raw shape and the label strings. No rows are dropped here.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    spec = DATASETS[name]

    # parser="liac-arff": the default pandas ARFF parser of scikit-learn 1.9
    # crashes with pandas 3 on Spambase (integer-like category labels).
    bunch = fetch_openml(
        name,
        version=spec["version"],
        as_frame=True,
        data_home=DATA_DIR,
        parser="liac-arff",
    )
    X: pd.DataFrame = bunch.data
    labels = bunch.target.astype(str)
    assert X.shape == (spec["n_raw"], spec["n_features"]), f"{name}: unexpected shape {X.shape}"

    # Verify the label strings instead of assuming them.
    found = set(labels.unique())
    if found != spec["labels"]:
        raise ValueError(f"{name}: expected labels {spec['labels']}, found {found}")
    return X, labels


def missing_mask(X: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame marking missing cells: NaN (as in OpenML Adult v2) or a literal '?'."""
    return X.isna() | (X.astype(str) == "?")


def load_dataset(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load a dataset from OpenML (cached in `data/`) and return (X, y), y in {0, 1}.

    Rows with any missing value are dropped (only Adult has them). The raw and
    dropped row counts are logged and stored in `X.attrs` for the report.
    """
    spec = DATASETS[name]
    X, labels = fetch_raw(name)

    keep = ~missing_mask(X).any(axis=1).to_numpy()
    n_dropped = int((~keep).sum())
    X = X.loc[keep].reset_index(drop=True)
    labels = labels.loc[keep].reset_index(drop=True)

    # Numeric columns as float64; categorical columns keep their pandas dtype.
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype(np.float64)

    y = (labels == spec["positive"]).to_numpy().astype(np.int64)

    pos_share = float(y.mean())
    assert len(y) == spec["n_rows"], f"{name}: expected {spec['n_rows']} rows, got {len(y)}"
    assert abs(pos_share - spec["pos_share"]) < 0.005, f"{name}: positive share {pos_share:.4f}"

    X.attrs = {"name": name, "n_raw": spec["n_raw"], "n_dropped": n_dropped}
    logger.info(
        "%s: %d raw rows, %d dropped (missing values), %d kept, %d features, positive share %.4f",
        name, spec["n_raw"], n_dropped, len(y), X.shape[1], pos_share,
    )
    return X, y


def stratified_split(
    y: np.ndarray,
    fractions: tuple[float, ...] = (0.6, 0.2, 0.2),
    *,
    rng: np.random.Generator | int,
) -> tuple[np.ndarray, ...]:
    """Split indices 0..n-1 into disjoint parts with (almost) equal class shares.

    For each class c with n_c examples, the indices of class c are shuffled and
    cut at positions round(n_c * (f_1 + ... + f_k)), k = 1, ..., K-1. Each part
    therefore receives n_c * f_k examples of class c up to rounding (error < 1
    sample per class). The parts of all classes are then concatenated.

    Returns one sorted index array per fraction, e.g. (idx_train, idx_cal, idx_test).
    """
    y = np.asarray(y)
    fractions_arr = np.asarray(fractions, dtype=float)
    if np.any(fractions_arr <= 0) or not np.isclose(fractions_arr.sum(), 1.0):
        raise ValueError(f"fractions must be positive and sum to 1, got {fractions}")
    rng = np.random.default_rng(rng)  # accepts an int seed or an existing Generator

    parts: list[list[np.ndarray]] = [[] for _ in fractions_arr]
    for c in np.unique(y):
        idx_c = rng.permutation(np.flatnonzero(y == c))
        cuts = np.round(np.cumsum(fractions_arr)[:-1] * len(idx_c)).astype(int)
        for k, piece in enumerate(np.split(idx_c, cuts)):
            parts[k].append(piece)

    return tuple(np.sort(np.concatenate(p)) for p in parts)


SPLIT_NAMES = ("train", "cal", "test")


def split_summary(y: np.ndarray, splits: tuple[np.ndarray, ...]) -> pd.DataFrame:
    """One row per split: size, number of positives and positive share."""
    rows = []
    for split_name, idx in zip(SPLIT_NAMES, splits):
        rows.append({
            "split": split_name,
            "n": len(idx),
            "n_pos": int(y[idx].sum()),
            "pos_share": float(y[idx].mean()),
        })
    return pd.DataFrame(rows)


def _as_str(column: pd.Series) -> np.ndarray:
    """Categorical column as a numpy array of strings (for one-hot comparison)."""
    return column.astype(str).to_numpy(dtype=str)


class Preprocessor:
    """Standardize numeric columns and one-hot encode categorical columns.

    Fitted on the training set only:
      numeric:     x' = (x - mean_train) / std_train   (std_train = 0 is replaced by 1)
      categorical: one 0/1 column per category seen in training; a category not seen
                   in training is encoded as all zeros.
    Output columns: numeric columns first, then the one-hot blocks, in `feature_names_`.
    """

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        self.columns_ = list(X.columns)
        self.numeric_cols_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.categorical_cols_ = [c for c in X.columns if c not in self.numeric_cols_]

        numeric = X[self.numeric_cols_].to_numpy(dtype=np.float64)
        if np.isnan(numeric).any():
            raise ValueError("numeric columns contain NaN")
        self.mean_ = numeric.mean(axis=0)
        std = numeric.std(axis=0)
        std[std == 0] = 1.0
        self.std_ = std

        self.categories_ = {c: np.unique(_as_str(X[c])) for c in self.categorical_cols_}
        self.feature_names_ = list(self.numeric_cols_) + [
            f"{c}={v}" for c in self.categorical_cols_ for v in self.categories_[c]
        ]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if list(X.columns) != self.columns_:
            raise ValueError("columns differ from those seen in fit")

        numeric = X[self.numeric_cols_].to_numpy(dtype=np.float64)
        if np.isnan(numeric).any():
            raise ValueError("numeric columns contain NaN")
        blocks = [(numeric - self.mean_) / self.std_]
        for c in self.categorical_cols_:
            values = _as_str(X[c])
            blocks.append((values[:, None] == self.categories_[c][None, :]).astype(np.float64))
        return np.hstack(blocks)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)


def dataset_summary(name: str, seed: int = 0) -> dict:
    """Facts about one dataset for the report's dataset table.

    `n_encoded` is the width of the encoded matrix when the Preprocessor is fitted
    on the training split of `seed` (as in the experiments); it can vary slightly
    with the seed if a rare category is absent from a training split.
    """
    X_raw, _ = fetch_raw(name)
    X, y = load_dataset(name)
    idx_train, _, _ = stratified_split(y, rng=seed)
    pre = Preprocessor().fit(X.iloc[idx_train])
    return {
        "dataset": name,
        "openml_version": DATASETS[name]["version"],
        "n_raw": len(X_raw),
        "n_rows_with_missing": int(missing_mask(X_raw).any(axis=1).sum()),
        "n_rows": len(y),
        "n_features": X.shape[1],
        "n_numeric": len(pre.numeric_cols_),
        "n_categorical": len(pre.categorical_cols_),
        "n_encoded": len(pre.feature_names_),
        "n_pos": int(y.sum()),
        "pos_share": float(y.mean()),
        "n_duplicate_rows": int(X.duplicated().sum()),
    }
