"""Calibration maps from a 1-D score s to a probability, implemented from scratch.

Both maps are fitted on held-out (calibration-set) scores and are monotone
non-decreasing in s, so they preserve the ranking of the predictions.
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function 1 / (1 + e^(-z)).

    Uses e = exp(-|z|) <= 1 so that nothing overflows:
      z >= 0: 1 / (1 + e),   z < 0: e / (1 + e).
    """
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


class PlattScaler:
    """Platt scaling: p = sigmoid(a * s + b).

    This is Platt's (1999) p = 1 / (1 + exp(A * s + B)) with A = -a, B = -b.

    Fitting minimizes the cross-entropy between targets t_i and p_i = sigmoid(a s_i + b):
        L(a, b) = -sum_i [t_i log p_i + (1 - t_i) log(1 - p_i)]
                = sum_i [log(1 + e^(z_i)) - t_i z_i],   z_i = a s_i + b,
    which is convex in theta = (a, b). It is minimized by Newton's method:
        gradient  g = sum_i (p_i - t_i) [s_i, 1]
        Hessian   H = sum_i p_i (1 - p_i) [s_i, 1][s_i, 1]^T + 1e-12 I
        step      Delta = -H^{-1} g,
    with backtracking (halving the step) until L does not increase, stopping when
    the step taken has norm < tol or after max_iter iterations.

    Targets: with smooth_targets=True, Platt's regularized targets
        t_+ = (N_+ + 1) / (N_+ + 2) for positives,  t_- = 1 / (N_- + 2) for negatives,
    which are the posterior means under a uniform prior and keep (a, b) finite
    even if the calibration data are perfectly separated. With smooth_targets=False,
    t = y and the fit equals an unregularized one-feature logistic regression.
    """

    def fit(
        self,
        s: np.ndarray,
        y: np.ndarray,
        smooth_targets: bool = True,
        max_iter: int = 100,
        tol: float = 1e-10,
    ) -> "PlattScaler":
        s = np.asarray(s, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_pos = y.sum()
        n_neg = len(y) - n_pos

        if smooth_targets:
            t = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
        else:
            t = y

        def loss(theta: np.ndarray) -> float:
            z = theta[0] * s + theta[1]
            return float(np.sum(np.logaddexp(0.0, z) - t * z))  # log(1 + e^z) computed stably

        # Platt's initialization: a = 0 and b = the log-odds of the (smoothed) class prior.
        theta = np.array([0.0, np.log((n_pos + 1) / (n_neg + 1))])
        current_loss = loss(theta)
        n_iter = 0
        for n_iter in range(1, max_iter + 1):
            p = sigmoid(theta[0] * s + theta[1])
            r = p - t
            w = p * (1 - p)
            g = np.array([np.sum(r * s), np.sum(r)])
            H = np.array([[np.sum(w * s * s), np.sum(w * s)],
                          [np.sum(w * s), np.sum(w)]]) + 1e-12 * np.eye(2)
            delta = -np.linalg.solve(H, g)

            # Backtracking line search: halve the step until the loss does not increase.
            step = 1.0
            new_loss = loss(theta + delta)
            while new_loss > current_loss and step > 1e-10:
                step /= 2
                new_loss = loss(theta + step * delta)
            if new_loss > current_loss:  # no step decreases the loss: we are at the minimum
                break

            theta = theta + step * delta
            current_loss = new_loss
            if np.linalg.norm(step * delta) < tol:
                break

        self.a_, self.b_ = float(theta[0]), float(theta[1])
        self.n_iter_ = n_iter
        return self

    def predict_proba(self, s: np.ndarray) -> np.ndarray:
        return sigmoid(self.a_ * np.asarray(s, dtype=np.float64) + self.b_)


class IsotonicCalibrator:
    """Isotonic regression by Pool Adjacent Violators (PAV).

    Finds the non-decreasing function f of the score minimizing the weighted squared error
        sum_i w_i (y_i - f(s_i))^2,
    whose solution is piecewise constant: the sorted scores are partitioned into
    contiguous blocks, and each block's value is the weighted mean of its labels.

    PAV: points sorted by score are pushed onto a stack of blocks
    (weight, weighted sum of labels, min score, max score). After each push, while
    the top block's mean is <= the mean of the block below, the two are merged.
    Merging equal means too (not only strict violations) leaves blocks with strictly
    increasing means, so `n_blocks_` counts the distinct output levels.

    Prediction is the step function: a score s gets the value of the last block whose
    min score is <= s; scores below the first block get the first value. A score between
    two blocks therefore gets the lower block's value, and scores above the last block
    get the last value.
    """

    def fit(self, s: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
            ) -> "IsotonicCalibrator":
        s = np.asarray(s, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        w = np.ones_like(s) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)

        # 1. Stable sort by score.
        order = np.argsort(s, kind="stable")
        s, y, w = s[order], y[order], w[order]

        # 2. Merge tied scores: one point per distinct score, with summed weight
        #    and weighted label sum (its value is the weighted mean label).
        starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
        point_score = s[starts]
        point_weight = np.add.reduceat(w, starts)
        point_wsum = np.add.reduceat(w * y, starts)

        # 3. PAV with a stack of blocks [weight, weighted_sum, score_min, score_max].
        stack: list[list[float]] = []
        for score, weight, wsum in zip(point_score, point_weight, point_wsum):
            stack.append([weight, wsum, score, score])
            while len(stack) >= 2 and stack[-1][1] / stack[-1][0] <= stack[-2][1] / stack[-2][0]:
                top = stack.pop()
                below = stack[-1]
                below[0] += top[0]
                below[1] += top[1]
                below[3] = top[3]

        # 4. Store the block boundaries and values.
        blocks = np.array(stack)
        self.block_weight_ = blocks[:, 0]
        self.block_value_ = blocks[:, 1] / blocks[:, 0]
        self.block_min_ = blocks[:, 2]
        self.block_max_ = blocks[:, 3]
        self.n_blocks_ = len(blocks)
        return self

    def predict_proba(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        idx = np.searchsorted(self.block_min_, s, side="right") - 1
        idx = np.clip(idx, 0, self.n_blocks_ - 1)
        return self.block_value_[idx]
