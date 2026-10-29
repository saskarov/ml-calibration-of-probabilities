import numpy as np
import pandas as pd
import pytest

from calib.experiments import evaluate, summarize


def test_evaluate_counts_exact_zero_one_predictions():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.0, 1.0, 0.0, 0.3, 0.8])  # three exact 0/1, one of them wrong
    row = evaluate(y, p)
    assert row["n_exact_01"] == 3
    assert row["n_exact_01_wrong"] == 1
    assert set(row) >= {"accuracy", "log_loss", "brier", "ece", "auc"}


def test_summarize_mean_and_std_over_seeds():
    df = pd.DataFrame({
        "model": ["rf"] * 3,
        "seed": [0, 1, 2],
        "accuracy": [0.8, 0.9, 1.0],
        "log_loss": [0.3, 0.3, 0.3],
        "brier": [0.1, 0.2, 0.3],
        "ece": [0.0, 0.0, 0.0],
        "auc": [0.9, 0.9, 0.9],
        "n_exact_01": [0, 0, 3],
        "n_exact_01_wrong": [0, 0, 0],
    })
    s = summarize(df, ["model"])
    assert s.loc[0, "accuracy_mean"] == pytest.approx(0.9)
    assert s.loc[0, "accuracy_std"] == pytest.approx(0.1)  # sample std, ddof = 1
    assert s.loc[0, "n_exact_01_mean"] == pytest.approx(1.0)
