import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from calib.models import RF_SCORE_CLIP, chosen_hyperparams, get_probs, get_scores


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] + 0.5 * rng.normal(size=300) > 0).astype(int)
    return X, y


def test_logreg_scores_are_logits(toy_data):
    X, y = toy_data
    model = LogisticRegression().fit(X, y)
    s = get_scores(model, X)
    np.testing.assert_allclose(1 / (1 + np.exp(-s)), get_probs(model, X), atol=1e-12)
    assert chosen_hyperparams(model) == {"C": 1.0}


def test_rf_scores_are_clipped_logits(toy_data):
    X, y = toy_data
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    p = get_probs(model, X)
    s = get_scores(model, X)
    assert np.all(np.isfinite(s))
    bound = np.log((1 - RF_SCORE_CLIP) / RF_SCORE_CLIP)
    assert np.all(np.abs(s) <= bound + 1e-9)
    # On training data many votes are unanimous, so the clip is exercised.
    assert np.any(np.isclose(np.abs(s), bound))
    # The score is a strictly increasing function of p, so it preserves the ranking.
    order = np.argsort(p, kind="stable")
    assert np.all(np.diff(s[order]) >= 0)
    np.testing.assert_allclose(1 / (1 + np.exp(-s)), np.clip(p, RF_SCORE_CLIP, 1 - RF_SCORE_CLIP))
