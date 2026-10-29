"""Experiments: main comparison, calibration-set size, and train vs held-out calibration.

Protocol for one (dataset, seed):
  1. Stratified 60/20/20 split into train / calibration / test (seeded).
  2. Preprocessor fitted on train; LR and RF trained and tuned (5-fold CV) on train.
  3. Calibrators fitted on calibration-set scores (or train scores, in the
     train-vs-held-out experiment); everything is evaluated on test only.
Models are deterministic given the seed, so every experiment retrains them and
obtains the same base models.
"""

import logging
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from calib.calibration import IsotonicCalibrator, PlattScaler
from calib.data import Preprocessor, load_dataset, stratified_split
from calib.metrics import accuracy, brier, ece, log_loss, roc_auc
from calib.models import chosen_hyperparams, get_probs, get_scores, train_logreg, train_rf
from calib.paths import RESULTS_DIR

logger = logging.getLogger(__name__)

MODELS = {"logreg": train_logreg, "rf": train_rf}
METRICS = {"accuracy": accuracy, "log_loss": log_loss, "brier": brier, "ece": ece, "auc": roc_auc}
METHODS = ("uncalibrated", "platt", "isotonic")


@lru_cache(maxsize=None)
def _load(dataset: str) -> tuple[pd.DataFrame, np.ndarray]:
    """load_dataset, cached in memory so the loop over seeds reads each dataset once."""
    return load_dataset(dataset)


@dataclass
class BaseModelOutputs:
    """Scores of one trained base model on the three splits."""

    model: str
    hyperparams: dict
    s_train: np.ndarray  # scores fed to the calibrators (see models.get_scores)
    s_cal: np.ndarray
    s_test: np.ndarray
    p_test: np.ndarray   # uncalibrated test probabilities (predict_proba)


@dataclass
class SeedData:
    """Labels of the three splits and the outputs of both base models, for one seed."""

    dataset: str
    seed: int
    y_train: np.ndarray
    y_cal: np.ndarray
    y_test: np.ndarray
    outputs: list[BaseModelOutputs]


def fit_base_models(dataset: str, seed: int) -> SeedData:
    """Split, preprocess (fitted on train), train LR and RF on train, and score all splits."""
    X, y = _load(dataset)
    idx_train, idx_cal, idx_test = stratified_split(y, rng=seed)

    pre = Preprocessor().fit(X.iloc[idx_train])
    X_train = pre.transform(X.iloc[idx_train])
    X_cal = pre.transform(X.iloc[idx_cal])
    X_test = pre.transform(X.iloc[idx_test])
    y_train = y[idx_train]

    outputs = []
    for name, train in MODELS.items():
        model = train(X_train, y_train, seed)
        outputs.append(BaseModelOutputs(
            model=name,
            hyperparams=chosen_hyperparams(model),
            s_train=get_scores(model, X_train),
            s_cal=get_scores(model, X_cal),
            s_test=get_scores(model, X_test),
            p_test=get_probs(model, X_test),
        ))
    return SeedData(dataset, seed, y_train, y[idx_cal], y[idx_test], outputs)


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    """All metrics on one set of predictions, plus how many predictions are exactly 0 or 1.

    `n_exact_01_wrong` counts those that are also wrong: each costs -log(1e-15) ~ 34.5
    in the (clipped) log-loss.
    """
    row = {name: metric(y, p) for name, metric in METRICS.items()}
    extreme = (p == 0) | (p == 1)
    row["n_exact_01"] = int(extreme.sum())
    row["n_exact_01_wrong"] = int(np.sum(extreme & (p != y)))
    return row


@dataclass
class CalibrationResult:
    """Test predictions of the three methods for one base model, and the fitted calibrators."""

    preds: dict[str, np.ndarray]
    platt: PlattScaler
    isotonic: IsotonicCalibrator


def calibrate(out: BaseModelOutputs, s_fit: np.ndarray, y_fit: np.ndarray) -> CalibrationResult:
    """Fit Platt and isotonic on (s_fit, y_fit) and predict on the test scores of `out`."""
    platt = PlattScaler().fit(s_fit, y_fit)
    isotonic = IsotonicCalibrator().fit(s_fit, y_fit)
    preds = {
        "uncalibrated": out.p_test,
        "platt": platt.predict_proba(out.s_test),
        "isotonic": isotonic.predict_proba(out.s_test),
    }
    return CalibrationResult(preds, platt, isotonic)


def _method_rows(result: CalibrationResult, y_test: np.ndarray, **keys) -> list[dict]:
    """One row per method: identifying `keys`, test metrics and the calibrator parameters."""
    rows = []
    for method, p in result.preds.items():
        row = {**keys, "method": method, **evaluate(y_test, p)}
        row["platt_a"] = result.platt.a_ if method == "platt" else np.nan
        row["platt_b"] = result.platt.b_ if method == "platt" else np.nan
        row["platt_n_iter"] = result.platt.n_iter_ if method == "platt" else np.nan
        row["iso_blocks"] = result.isotonic.n_blocks_ if method == "isotonic" else np.nan
        rows.append(row)
    return rows


def _run_seed(dataset: str, seed: int) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Main experiment for one seed: metric rows and arrays for plotting."""
    data = fit_base_models(dataset, seed)
    rows: list[dict] = []
    arrays: dict[str, np.ndarray] = {"y_cal": data.y_cal, "y_test": data.y_test}
    for out in data.outputs:
        result = calibrate(out, out.s_cal, data.y_cal)  # calibrators see the calibration set only
        rows += _method_rows(result, data.y_test, dataset=dataset, seed=seed, model=out.model,
                             **out.hyperparams)
        m = out.model
        arrays[f"{m}_s_cal"] = out.s_cal
        arrays[f"{m}_s_test"] = out.s_test
        for method, p in result.preds.items():
            arrays[f"{m}_p_{method}"] = p
        arrays[f"{m}_platt_ab"] = np.array([result.platt.a_, result.platt.b_])
        arrays[f"{m}_iso_min"] = result.isotonic.block_min_
        arrays[f"{m}_iso_max"] = result.isotonic.block_max_
        arrays[f"{m}_iso_value"] = result.isotonic.block_value_
    return rows, arrays


def _hyperparams_last(df: pd.DataFrame) -> pd.DataFrame:
    """Move the tuned-hyperparameter columns to the end of the table."""
    hyper = [c for c in ("C", "min_samples_leaf") if c in df.columns]
    return df[[c for c in df.columns if c not in hyper] + hyper]


def run_single(dataset: str, seed: int) -> list[dict]:
    """Main experiment for one (dataset, seed): one row per (model, method), metrics on test."""
    rows, _ = _run_seed(dataset, seed)
    return rows


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean and standard deviation (over seeds, ddof=1) of every metric, per group."""
    columns = list(METRICS) + ["n_exact_01", "n_exact_01_wrong"]
    summary = df.groupby(by, sort=False)[columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def run_main(datasets=("spambase", "adult"), seeds=range(10)) -> pd.DataFrame:
    """Main experiment over all datasets and seeds.

    Saves results/metrics_all.csv (one row per dataset, seed, model, method),
    results/metrics_summary.csv (mean and std over seeds) and, for seed 0,
    results/preds_{dataset}_seed0.npz (test predictions and fitted maps, for plotting).
    """
    rows: list[dict] = []
    for dataset in datasets:
        for seed in seeds:
            start = time.perf_counter()
            seed_rows, arrays = _run_seed(dataset, seed)
            rows += seed_rows
            if seed == 0:
                np.savez_compressed(RESULTS_DIR / f"preds_{dataset}_seed0.npz", **arrays)
            logger.info("main: %s seed %d done in %.1fs", dataset, seed, time.perf_counter() - start)

    df = _hyperparams_last(pd.DataFrame(rows))
    df.to_csv(RESULTS_DIR / "metrics_all.csv", index=False)
    summarize(df, ["dataset", "model", "method"]).to_csv(
        RESULTS_DIR / "metrics_summary.csv", index=False
    )
    return df


def run_calibration_size(
    dataset: str = "adult",
    sizes=(50, 100, 200, 500, 1000, 2000, 5000),
    seeds=range(10),
) -> pd.DataFrame:
    """Test metrics of Platt and isotonic fitted on stratified subsamples of the calibration set.

    For each seed and size n, a stratified subsample of about n calibration examples
    is drawn with its own generator, seeded by (seed, n). The base models are the same
    as in the main experiment. Rows with method "uncalibrated" (cal_size = 0) give the
    reference. Saves results/cal_size.csv.
    """
    rows: list[dict] = []
    for seed in seeds:
        start = time.perf_counter()
        data = fit_base_models(dataset, seed)
        n_cal = len(data.y_cal)
        for out in data.outputs:
            keys = {"dataset": dataset, "seed": seed, "model": out.model}
            rows.append({**keys, "cal_size": 0, "method": "uncalibrated",
                         **evaluate(data.y_test, out.p_test)})
            for size in sizes:
                if size >= n_cal:
                    raise ValueError(f"size {size} >= calibration set size {n_cal}")
                rng = np.random.default_rng([seed, size])
                sub, _ = stratified_split(data.y_cal, (size / n_cal, 1 - size / n_cal), rng=rng)
                result = calibrate(out, out.s_cal[sub], data.y_cal[sub])
                for method in ("platt", "isotonic"):
                    rows.append({**keys, "cal_size": len(sub), "method": method,
                                 **evaluate(data.y_test, result.preds[method])})
        logger.info("cal_size: %s seed %d done in %.1fs", dataset, seed, time.perf_counter() - start)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "cal_size.csv", index=False)
    return df


def run_train_vs_heldout(datasets=("spambase", "adult"), seeds=range(10)) -> pd.DataFrame:
    """Calibrators fitted on training-set scores vs on calibration-set scores, evaluated on test.

    Column `fit_on` is "train" or "cal". The training scores come from models that were
    fitted on those same examples, so they are optimistic (e.g. RF probabilities near 0/1);
    a calibrator fitted on them learns the wrong correction. Saves results/train_vs_heldout.csv.
    """
    rows: list[dict] = []
    for dataset in datasets:
        for seed in seeds:
            start = time.perf_counter()
            data = fit_base_models(dataset, seed)
            for out in data.outputs:
                for fit_on, s_fit, y_fit in (("train", out.s_train, data.y_train),
                                             ("cal", out.s_cal, data.y_cal)):
                    result = calibrate(out, s_fit, y_fit)
                    new_rows = _method_rows(result, data.y_test, dataset=dataset, seed=seed,
                                            model=out.model, fit_on=fit_on)
                    # The uncalibrated row does not depend on fit_on: keep it once.
                    rows += [r for r in new_rows
                             if not (fit_on == "train" and r["method"] == "uncalibrated")]
            logger.info("train_vs_heldout: %s seed %d done in %.1fs",
                        dataset, seed, time.perf_counter() - start)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "train_vs_heldout.csv", index=False)
    return df
