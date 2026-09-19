"""Signal estimates in the moving, whitened frame of the search distribution."""

from __future__ import annotations

from typing import cast

import numpy as np

from .state import JSONObject


def ranking_noise(samples: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Exact conditional Fisher-gradient powers under uniformly permuted utilities.

    For centered weights, E[w_i w_j] gives Var(sum_i w_i a_i) =
    ||w||^2 / (n-1) * sum_i ||a_i - mean(a)||^2. This includes correlations
    from mirroring, orthogonality, unequal lengths, and tied rank utilities.
    """
    dim, count = samples.shape
    gram = samples.T @ samples
    radii = np.diag(gram)
    mean = float(np.sum((samples - samples.mean(axis=1, keepdims=True)) ** 2))
    scale = float(np.sum((radii - radii.mean()) ** 2) / (2 * dim))
    shape = 0.5 * ((1 - 1 / dim) * (radii @ radii) - (np.sum(gram**2) - radii.sum() ** 2 / dim) / count)
    if dim == 1:
        shape = 0.0
    return float(weights @ weights) / (count - 1) * np.maximum([mean, scale, shape], 0.0)


class RateAdaptation:
    """Compare independent block signals with their random-ranking reference.

    Paths keep their coefficients in the moving factor frame. For the covariance
    block this is affine-invariant SPD parallel transport along the exponential
    covariance step. For the mean it is an isometric frame identification, not
    full joint-Gaussian Levi-Civita transport.
    """

    def __init__(self, dim: int) -> None:
        self.paths = [np.zeros(dim), np.zeros(1), np.zeros((dim, dim))]
        self.reference = np.zeros(3)
        self.multipliers = np.ones(3)
        self.signal = np.ones(3)
        self.steps = 0

    def update(
        self,
        gradients: list[np.ndarray],
        noise: np.ndarray,
        rates: np.ndarray,
        ceilings: np.ndarray,
        evidence_half_life: float,
        rate_half_life: float,
        signal_threshold: float,
        scale_recovery: float,
    ) -> None:
        base = rates / self.multipliers
        self.steps += 1
        decay = np.exp2(-1 / evidence_half_life)
        innovation = 1 - decay**2
        for i, gradient in enumerate(gradients):
            if noise[i] <= 0:
                continue
            normalized = gradient / np.sqrt(noise[i])
            self.reference[i] = decay**2 * self.reference[i] + innovation
            if i == 1:
                # S3's scale lesson: measure movement power, without cancellation
                # between expansion and contraction. Use the natural tangent to
                # remove finite-step dependence on the learning rate exactly.
                self.paths[i] *= decay**2
                self.paths[i] += innovation * normalized**2
                power = float(self.paths[i][0])
            else:
                self.paths[i] *= decay
                self.paths[i] += np.sqrt(innovation) * normalized
                power = float(np.sum(self.paths[i] ** 2))
            self.signal[i] = power / self.reference[i]
            feedback = float(np.clip(self.signal[i] / signal_threshold - 1, -1, 1))
            recovery = scale_recovery if i == 1 and feedback > 0 else 1.0
            self.multipliers[i] *= np.exp2(feedback / (rate_half_life * recovery))
        upper = np.maximum(ceilings / base, 1.0)
        self.multipliers = np.clip(self.multipliers, np.minimum(1.0, 1e-3 * upper), upper)

    def save(self) -> JSONObject:
        return {
            "paths": [path.tolist() for path in self.paths],
            "reference": [float(value) for value in self.reference],
            "multipliers": [float(value) for value in self.multipliers],
            "signal": [float(value) for value in self.signal],
            "steps": self.steps,
        }

    def load(self, state: JSONObject, permutation: list[int]) -> None:
        paths = state["paths"]
        assert isinstance(paths, list)
        self.paths = [
            np.asarray(paths[0], dtype=float)[permutation],
            np.asarray(paths[1], dtype=float),
            np.asarray(paths[2], dtype=float).reshape(len(permutation), len(permutation))[
                np.ix_(permutation, permutation)
            ],
        ]
        self.reference = np.array(state["reference"], dtype=float)
        self.multipliers = np.array(state["multipliers"], dtype=float)
        self.signal = np.array(state["signal"], dtype=float)
        self.steps = cast(int, state["steps"])
