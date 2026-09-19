"""Compact exact exponential updates and Gaussian relative entropy."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, qr


def gradient_spectrum(samples: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Q, v with Z diag(w) Z.T = Q diag(v) Q.T.

    Mirrored pairs share an outer product. A single orthogonal sampling block
    already provides its eigenvectors; other inputs use a projected eigensolve.
    """
    dim, count = samples.shape
    half = count // 2
    if count % 2 == 0 and np.array_equal(samples[:, half:], -samples[:, :half]):
        samples = samples[:, :half]
        weights = weights[:half] + weights[half:]
    lengths = np.linalg.norm(samples, axis=0)
    if samples.shape[1] <= dim and np.all(lengths > 0):
        basis = samples / lengths
        if np.allclose(basis.T @ basis, np.eye(len(lengths)), rtol=0, atol=1e-12):
            return basis, weights * lengths**2
    basis, projected = qr(samples, mode="economic")
    values, rotation = eigh((projected * weights) @ projected.T)
    return basis @ rotation, values


def gaussian_kl(mean_step: np.ndarray, eigenvalues: np.ndarray, complement: float, dim: int) -> float:
    """KL(new || old) for a whitened mean step and log-covariance spectrum."""
    with np.errstate(over="ignore", invalid="ignore"):
        covariance = np.sum(np.expm1(eigenvalues) - eigenvalues)
        if eigenvalues.size < dim:
            covariance += (dim - eigenvalues.size) * (np.expm1(complement) - complement)
        return 0.5 * float(mean_step @ mean_step + covariance)


def trust_fraction(mean_step: np.ndarray, eigenvalues: np.ndarray, complement: float, dim: int, limit: float) -> float:
    """Largest fraction of an exponential-coordinate step inside a KL ball."""
    if gaussian_kl(mean_step, eigenvalues, complement, dim) <= limit:
        return 1.0
    lower, upper = 0.0, 1.0
    for _ in range(40):
        middle = (lower + upper) / 2
        if gaussian_kl(middle * mean_step, middle * eigenvalues, middle * complement, dim) <= limit:
            lower = middle
        else:
            upper = middle
    return lower
