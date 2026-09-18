"""Deterministic landscapes and explicitly seeded observation noise."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

NOISE_MODELS = ("clean", "additive", "heteroscedastic", "outliers")


def seed_for(master: int, *identifiers: int) -> int:
    return int(np.random.SeedSequence([master, *identifiers]).generate_state(1)[0])


def random_rotation(dim: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    return q * np.where(np.diag(r) < 0, -1.0, 1.0)


@dataclass(frozen=True)
class Problem:
    name: str
    clean: Callable[[np.ndarray], float]
    x0: np.ndarray
    optimum: float = 0.0

    def gap(self, x: np.ndarray) -> float:
        return max(float(self.clean(x)) - self.optimum, 0.0)


def bbob(function: int, dimension: int, instance: int) -> Problem:
    import cocoex

    function_object = cocoex.BareProblem("bbob", function, dimension, instance)
    return Problem(str(function_object), function_object, np.zeros(dimension), float(function_object.best_value()))


class ObservationNoise:
    """Fixed random variates indexed by objective call, independent of candidate location.

    Separate instances with the same seed give paired streams even for unequal populations.
    Assessment calls never consume this stream. Values are in objective units.
    """

    def __init__(self, model: str, reference_gap: float, budget: int, seed: int) -> None:
        self.model = model
        self.scale = max(reference_gap, 1e-12)
        streams = np.random.SeedSequence(seed).spawn(3)
        self.normal = np.random.default_rng(streams[0]).standard_normal(budget)
        self.outlier = np.random.default_rng(streams[1]).random(budget) < 0.01
        self.tail = np.random.default_rng(streams[2]).standard_t(3, budget)

    def __call__(self, value: float, gap: float, index: int) -> float:
        if self.model == "clean":
            return value
        if self.model == "additive":
            return value + 0.01 * self.scale * self.normal[index]
        if self.model == "heteroscedastic":
            return value + (0.1 * gap + 0.001 * self.scale) * self.normal[index]
        if self.model == "outliers":
            return value + self.scale * (0.01 * self.normal[index] + self.outlier[index] * self.tail[index])
        if self.model == "constant":
            return 0.0
        if self.model == "random":
            return float(self.normal[index])
        raise ValueError(f"Unknown noise model: {self.model}")


def _as_matrix(points: np.ndarray) -> np.ndarray:
    x = np.asarray(points, dtype=float)
    return x[:, None] if x.ndim == 1 else x


def sphere(points: np.ndarray) -> np.ndarray:
    return np.sum(_as_matrix(points) ** 2, axis=0)


def ackley(points: np.ndarray) -> np.ndarray:
    x = _as_matrix(points)
    return (
        -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2, axis=0))) - np.exp(np.mean(np.cos(2 * np.pi * x), axis=0)) + 20 + np.e
    )


def centered_rosenbrock(points: np.ndarray) -> np.ndarray:
    x = _as_matrix(points)
    if len(x) == 1:
        return x[0] ** 2
    shifted = x + 1
    return np.sum(100 * (shifted[1:] - shifted[:-1] ** 2) ** 2 + (1 - shifted[:-1]) ** 2, axis=0)


def ellipsoid(points: np.ndarray) -> np.ndarray:
    x = _as_matrix(points)
    return np.sum((1e6 ** np.linspace(0, 1, len(x)))[:, None] * x**2, axis=0)


def cigar(points: np.ndarray) -> np.ndarray:
    x = _as_matrix(points)
    return x[0] ** 2 + 1e6 * np.sum(x[1:] ** 2, axis=0)


OBJECTIVES = {
    "sphere": sphere,
    "ackley": ackley,
    "rosenbrock": centered_rosenbrock,
    "ellipsoid": ellipsoid,
    "cigar": cigar,
}


def translated_objective(
    name: str, dimension: int, seed: np.random.SeedSequence, offset_scale: float, noise_scale: float
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], np.ndarray, bool]:
    """Legacy half-normal experiment used by tuning/workbench; preserve its RNG splitting."""
    offset_seed, noise_seed, rotation_seed = seed.spawn(3)
    offset = np.random.default_rng(offset_seed).uniform(-offset_scale, offset_scale, dimension)
    noise_rng = np.random.default_rng(noise_seed)
    rotated = name in {"ellipsoid", "cigar"} and dimension > 1
    rotation = random_rotation(dimension, np.random.default_rng(rotation_seed)) if rotated else np.eye(dimension)

    def clean(points: np.ndarray) -> np.ndarray:
        return OBJECTIVES[name](rotation.T @ (_as_matrix(points) - offset[:, None]))

    def noisy(points: np.ndarray) -> np.ndarray:
        base = clean(points)
        return base + noise_scale * (1 + base) * np.abs(noise_rng.standard_normal(base.shape))

    return clean, noisy, offset, rotated
