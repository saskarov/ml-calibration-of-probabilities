"""Base models: Logistic Regression and Random Forest (scikit-learn), tuned by CV.

Hyperparameters are chosen by 5-fold stratified cross-validation on the training
set only, using log-loss as the criterion. The returned model is the best
configuration refit on the whole training set.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

LOGREG_C_GRID = np.logspace(-3, 2, 6)  # 0.001, 0.01, ..., 100
RF_MIN_SAMPLES_LEAF_GRID = [1, 3, 10]
N_FOLDS = 5
RF_SCORE_CLIP = 1e-6


def _tune(estimator, param_grid: dict, X: np.ndarray, y: np.ndarray, seed: int):
    """Grid search with 5-fold stratified CV (shuffled with `seed`), scored by log-loss.

    "neg_log_loss" is scikit-learn's scorer name for minus the log-loss; it is used
    only to rank hyperparameters inside GridSearchCV.
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    search = GridSearchCV(estimator, param_grid, scoring="neg_log_loss", cv=cv, refit=True)
    search.fit(X, y)
    return search.best_estimator_


def train_logreg(X: np.ndarray, y: np.ndarray, seed: int) -> LogisticRegression:
    """L2-regularized logistic regression; C tuned over 10^-3, ..., 10^2.

    The lbfgs solver is deterministic; `seed` only fixes the CV folds.
    """
    model = LogisticRegression(max_iter=5000)
    return _tune(model, {"C": LOGREG_C_GRID}, X, y, seed)


def train_rf(X: np.ndarray, y: np.ndarray, seed: int) -> RandomForestClassifier:
    """Random forest with 500 trees and sqrt(d) features per split; min_samples_leaf tuned.

    Trained in parallel, but predicts with one thread: predict_proba sums the trees'
    probabilities, and with several threads the order of the additions (hence the
    floating-point rounding) changes from run to run. Rounding differences of ~1e-17
    can break ties between scores and change the isotonic fit, so the same seed would
    not always give identical results.
    """
    model = RandomForestClassifier(
        n_estimators=500, max_features="sqrt", n_jobs=-1, random_state=seed
    )
    best = _tune(model, {"min_samples_leaf": RF_MIN_SAMPLES_LEAF_GRID}, X, y, seed)
    return best.set_params(n_jobs=1)


def chosen_hyperparams(model) -> dict:
    """The tuned hyperparameter of a model returned by train_logreg / train_rf."""
    if isinstance(model, LogisticRegression):
        return {"C": float(model.C)}
    if isinstance(model, RandomForestClassifier):
        return {"min_samples_leaf": int(model.min_samples_leaf)}
    raise TypeError(f"unsupported model {type(model).__name__}")


def get_probs(model, X: np.ndarray) -> np.ndarray:
    """Uncalibrated P(y = 1 | x) as output by the model."""
    return model.predict_proba(X)[:, 1]


def get_scores(model, X: np.ndarray) -> np.ndarray:
    """1-D real-valued score s(x) fed to the calibrators.

    LR: the logit w.x + b (decision_function), so that predict_proba = sigmoid(s).
    RF: s = logit(clip(p, 1e-6, 1 - 1e-6)), where p is the average tree vote.

    Why the logit for RF: Platt fits p = sigmoid(a*s + b). With s = logit(p_RF),
    a = 1, b = 0 is the identity map, so Platt can leave a calibrated model
    unchanged, and a > 1 stretches the probabilities away from 0.5, which is the
    correction RF needs (its averaged votes are pushed towards the middle).
    The clip keeps the logit finite when all trees agree (p = 0 or 1).
    Isotonic regression depends only on the order of the scores, so the choice
    does not affect it.
    """
    if isinstance(model, LogisticRegression):
        return model.decision_function(X)
    if isinstance(model, RandomForestClassifier):
        p = np.clip(get_probs(model, X), RF_SCORE_CLIP, 1 - RF_SCORE_CLIP)
        return np.log(p) - np.log1p(-p)
    raise TypeError(f"unsupported model {type(model).__name__}")
