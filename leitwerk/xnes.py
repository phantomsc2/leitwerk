"""Core xNES distribution update implementation.

The class maintains the search distribution in factored form
`scale_global * scale_shape`,
generates mirrored orthogonal samples, and applies adaptive exponential updates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.linalg import cond, norm
from scipy.linalg import qr

from ._adaptation import RateAdaptation, ranking_noise
from ._geometry import gaussian_kl, gradient_spectrum, trust_fraction


@dataclass(frozen=True, slots=True)
class XNESLearningRates:
    """Initial rates and separated, random-ranking-calibrated adaptation.

    Rates adapt independently by default. Set ``adaptive=False`` and
    ``max_kl=float('inf')`` for the fixed-rate exponential update.
    """

    eta_mean: float = 1.0
    """Mean learning rate."""
    eta_scale_global: float = 0.5
    """Covariance scale learning rate."""
    eta_scale_shape: float = 0.1
    """Covariance shape learning rate."""
    adaptive: bool = True
    """Adapt the three initial rates from temporal natural-gradient consistency."""
    max_kl: float = 1.0
    """Maximum isolated KL per block; not a bound on the combined Gaussian KL."""
    signal_threshold: float = 2.0
    """Signal power required to increase a rate, relative to random rankings (one)."""
    evidence_half_life: float = 32.0
    """Generations for a mean/shape signal's memory to halve; scale power decays twice as fast."""
    rate_half_life: float = 10.0
    """Minimum generations to halve a rate under maximal negative feedback."""
    scale_recovery: float = 5.0
    """How many times slower the scale rate increases than decreases."""

    def __post_init__(self) -> None:
        if self.eta_mean <= 0.0:
            raise ValueError("eta_mean must be > 0.")
        if self.eta_scale_global <= 0.0:
            raise ValueError("eta_scale_global must be > 0.")
        if self.eta_scale_shape <= 0.0:
            raise ValueError("eta_scale_shape must be > 0.")
        if not self.max_kl > 0:
            raise ValueError("max_kl must be > 0.")
        if not np.isfinite(self.signal_threshold) or self.signal_threshold <= 1:
            raise ValueError("signal_threshold must be finite and > 1.")
        for name in ("evidence_half_life", "rate_half_life", "scale_recovery"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0.")


class XNES:
    """Exponential Natural Evolution Strategies distribution state.

    The distribution is stored in factored form as
    ``scale_global * scale_shape`` and updated
    from ranked standardized samples.

    Args:
        mean0: Initial mean vector.
        scale0: Initial scale, either scalar, diagonal vector, or full matrix.

    Raises:
        ValueError: If the supplied shapes are inconsistent, the scale matrix is
            not positive with finite determinant.
    """

    def __init__(self, mean0: np.ndarray, scale0: np.ndarray | float) -> None:
        self.mean = np.array(mean0, dtype=float, copy=True)
        self.scale_global: float
        self.scale_shape: np.ndarray
        self.adaptation = RateAdaptation(self.dim)
        self.last_kl = 0.0
        self.step_fractions = np.ones(3)
        self.block_kl = np.zeros(3)
        self.effective_rates = np.zeros(3)

        if self.dim == 0:
            self.scale_global = 1.0
            self.scale_shape = np.eye(0)
            return

        scale_matrix0 = _normalize_scale_matrix(scale0, self.dim)

        sign, logdet = np.linalg.slogdet(scale_matrix0)
        if sign <= 0 or not np.isfinite(logdet):
            msg = "Scale matrix must have a positive finite determinant."
            raise ValueError(msg)

        self.scale_global = max(float(np.exp(logdet / self.dim)), 1e-30)
        self.scale_shape = scale_matrix0 / self.scale_global

    @property
    def dim(self) -> int:
        """Dimension of the search space."""

        return int(self.mean.size)

    @property
    def axis_ratio(self) -> float:
        """Current principal-axis ratio of the scale transform."""

        if self.dim <= 1:
            return 1.0
        try:
            return float(cond(self.scale_shape))
        except np.linalg.LinAlgError:
            return np.inf

    @property
    def scale(self) -> np.ndarray:
        """Current full scale matrix `scale_global * scale_shape`."""

        return self.scale_global * self.scale_shape

    @property
    def scale_marginal(self) -> np.ndarray:
        """Current marginal per-dimension standard deviations in latent space."""

        scale = self.scale
        return np.sqrt(np.maximum(np.einsum("ij,ij->i", scale, scale), 0.0))

    def transform(self, samples: np.ndarray) -> np.ndarray:
        """Map standardized samples `z` into current distribution coordinates."""

        z = _validated_samples(samples, self.dim)
        return self.mean[:, None] + self.scale @ z

    def sample(
        self,
        num_samples: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample a mirrored standardized batch.

        Args:
            num_samples: Optional batch size. Values below two are clamped, and
                odd values are rounded up to keep mirrored pairs.
            rng: Optional NumPy random generator.

        Returns:
            Standardized samples `z`.
        """

        if self.dim == 0:
            n = int(num_samples) if num_samples is not None else 4
            return np.zeros((0, n))

        n = _default_sample_count(num_samples, self.dim)
        n_half = n // 2
        rng = rng or np.random.default_rng()

        z_half = np.empty((self.dim, n_half))
        for start in range(0, n_half, self.dim):
            end = min(start + self.dim, n_half)
            k = end - start
            raw = rng.standard_normal((self.dim, k))
            lengths = norm(raw, axis=0)
            basis, triangular = qr(raw, mode="economic")
            basis *= np.where(np.diag(triangular) < 0, -1.0, 1.0)
            z_half[:, start:end] = basis * lengths

        return np.hstack([z_half, -z_half])

    def update(
        self,
        samples: np.ndarray,
        ranking: Sequence[int | Sequence[int]],
        learning_rates: XNESLearningRates | None = None,
        eps: float = 1e-10,
    ) -> XNESStatus:
        """Apply one exponential update from ranked standardized samples.

        Args:
            samples: Standardized sample matrix with shape `(dim, n)`.
            ranking: Indices from best to worst, optionally grouped into tied ranks.
                Members of a tied group receive their average utility.
            learning_rates: Initial rates, signal calibration, and isolated KL budget per block.
            eps: Numerical stopping threshold.

        Returns:
            An `XNESStatus` indicating whether the update succeeded or hit a numerical or convergence stop condition.

        Raises:
            ValueError: If sample shapes are inconsistent, samples are not
                finite, or the ranking is not a valid permutation.
        """

        if self.dim == 0:
            return XNESStatus.SCALE_NORM_MIN

        learning_rates = learning_rates or XNESLearningRates()

        samples = _validated_samples(samples, self.dim)
        n = samples.shape[1]
        d = self.dim
        weights = _rank_weights(ranking, n)
        if not np.any(weights):
            self.step_fractions = np.ones(3)
            self.block_kl = np.zeros(3)
            self.effective_rates = np.zeros(3)
            self.last_kl = 0.0
            return XNESStatus.OK
        grad_mean = samples @ weights
        basis, values = gradient_spectrum(samples, weights)
        grad_scale_global = float(np.sum(values) / d)
        rates = np.array(
            [
                learning_rates.eta_mean,
                learning_rates.eta_scale_global,
                learning_rates.eta_scale_shape * _default_eta_scale_shape(d),
            ]
        )
        if learning_rates.adaptive:
            rates *= self.adaptation.multipliers
        local_mean_step = rates[0] * grad_mean
        limit = learning_rates.max_kl
        empty = np.empty(0)
        self.step_fractions = np.array(
            [
                min(1.0, np.sqrt(2 * limit / max(float(local_mean_step @ local_mean_step), 1e-300))),
                trust_fraction(empty, empty, rates[1] * grad_scale_global, d, limit),
                trust_fraction(empty, rates[2] * (values - grad_scale_global), -rates[2] * grad_scale_global, d, limit),
            ]
        )
        executed = rates * self.step_fractions
        self.effective_rates = executed.copy()
        local_mean_step = executed[0] * grad_mean
        scale_log = executed[1] * grad_scale_global
        shape_values = executed[2] * (values - grad_scale_global)
        shape_complement = -executed[2] * grad_scale_global
        self.block_kl = np.array(
            [
                0.5 * float(local_mean_step @ local_mean_step),
                gaussian_kl(empty, empty, scale_log, d),
                gaussian_kl(empty, shape_values, shape_complement, d),
            ]
        )
        self.last_kl = gaussian_kl(local_mean_step, shape_values + scale_log, shape_complement + scale_log, d)
        mean_step = self.scale @ local_mean_step
        self.mean += mean_step

        scale_global_log_step = 0.5 * scale_log

        self.scale_global *= float(np.exp(scale_global_log_step))
        if not np.isfinite(self.scale_global):
            return XNESStatus.SCALE_GLOBAL_INF

        if self.scale_global < eps:
            return XNESStatus.SCALE_GLOBAL_MIN
        if self.scale_global > 1.0 / eps:
            return XNESStatus.SCALE_GLOBAL_MAX

        shape_step = 0.5 * executed[2]
        self.scale_shape += ((self.scale_shape @ basis) * np.expm1(shape_step * values)) @ basis.T
        self.scale_shape *= np.exp(-shape_step * grad_scale_global)

        if learning_rates.adaptive:
            shape_gradient = (basis * values) @ basis.T - grad_scale_global * np.eye(d)
            self.adaptation.update(
                [grad_mean, np.array([np.sqrt(d / 2) * grad_scale_global]), shape_gradient / np.sqrt(2)],
                ranking_noise(samples, weights),
                rates,
                np.array([1.0, 1.0, 1.5 * _default_eta_scale_shape(d)]),
                learning_rates.evidence_half_life,
                learning_rates.rate_half_life,
                learning_rates.signal_threshold,
                learning_rates.scale_recovery,
            )

        sign, logdet = np.linalg.slogdet(self.scale_shape)
        if sign <= 0 or not np.isfinite(logdet):
            return XNESStatus.SCALE_INF
        self.scale_shape *= np.exp(-logdet / d)

        if not np.all(np.isfinite(self.mean)):
            return XNESStatus.MEAN_INF
        if not np.all(np.isfinite(self.scale_shape)):
            return XNESStatus.SCALE_INF

        scale = self.scale
        if not np.all(np.isfinite(scale)):
            return XNESStatus.SCALE_INF
        if norm(scale, 2) < eps:
            return XNESStatus.SCALE_NORM_MIN
        if norm(mean_step, 2) < eps:
            return XNESStatus.MEAN_STEP_MIN

        cond_scale = self.axis_ratio
        if not np.isfinite(cond_scale):
            return XNESStatus.SCALE_COND_INF

        if cond_scale > 1.0 / eps:
            return XNESStatus.SCALE_COND_MAX
        return XNESStatus.OK


class XNESStatus(Enum):
    """Outcome of one `XNES.update` step."""

    OK = auto()
    SCALE_GLOBAL_MIN = auto()
    SCALE_GLOBAL_MAX = auto()
    SCALE_GLOBAL_INF = auto()
    MEAN_INF = auto()
    SCALE_INF = auto()
    SCALE_COND_INF = auto()
    SCALE_COND_MAX = auto()
    MEAN_STEP_MIN = auto()
    SCALE_NORM_MIN = auto()

    @property
    def is_ok(self) -> bool:
        """Whether the update succeeded and sampling can continue."""

        return self is type(self).OK

    @property
    def is_completion(self) -> bool:
        """Whether the update reached a non-error stopping condition."""

        return self in (
            type(self).SCALE_GLOBAL_MIN,
            type(self).MEAN_STEP_MIN,
            type(self).SCALE_NORM_MIN,
        )

    @property
    def is_error(self) -> bool:
        """Whether the update hit an error stopping condition."""

        return not self.is_ok and not self.is_completion

    @property
    def is_terminal(self) -> bool:
        """Whether the update requested a restart."""

        return not self.is_ok


def _default_eta_scale_shape(dim: int) -> float:
    if dim <= 0:
        return 1.0
    return float(0.6 * (3.0 + np.log(dim)) / (dim * np.sqrt(dim)))


def _normalize_scale_matrix(scale: np.ndarray | float, dim: int) -> np.ndarray:
    scale0 = np.asarray(scale, dtype=float)
    if scale0.ndim == 0:
        return np.diag(np.repeat(scale0, dim))
    if scale0.ndim == 1:
        return np.diag(scale0)
    if scale0.shape != (dim, dim):
        msg = f"Expected scale shape {(dim, dim)}, got {scale0.shape}"
        raise ValueError(msg)
    return scale0


def _validated_samples(samples: np.ndarray, dim: int) -> np.ndarray:
    z = np.asarray(samples, dtype=float)
    if z.ndim != 2:
        msg = "samples must have shape (dim, n)."
        raise ValueError(msg)
    if z.shape[0] != dim:
        msg = f"Sample shape mismatch, expected {dim} rows, got {z.shape[0]}"
        raise ValueError(msg)
    if not np.all(np.isfinite(z)):
        msg = "samples must be finite."
        raise ValueError(msg)
    return z


def _default_sample_count(num_samples: int | None, dim: int) -> int:
    n = int(num_samples) if num_samples is not None else (4 + int(3 * np.log(dim)))
    if n <= 1:
        n = 2
    if n % 2 == 1:
        n += 1
    return n


def _utility_weights(sample_count: int) -> np.ndarray:
    w_pos = np.maximum(0.0, np.log(sample_count / 2 + 1) - np.log(np.arange(1, sample_count + 1)))
    w_sum = float(np.sum(w_pos))
    if w_sum <= 0.0:
        msg = "Invalid utility weights: positive weight sum must be > 0."
        raise ValueError(msg)
    w_pos /= w_sum
    return w_pos - (1.0 / sample_count)


def _rank_weights(ranking: Sequence[int | Sequence[int]], count: int) -> np.ndarray:
    groups = [[rank] if isinstance(rank, (int, np.integer)) else list(rank) for rank in ranking]
    indices = [index for group in groups for index in group]
    if sorted(indices) != list(range(count)) or any(not group for group in groups):
        raise ValueError("ranking must be a permutation matching sample count.")
    if len(groups) == 1:
        return np.zeros(count)
    utilities = _utility_weights(count)
    weights = np.empty(count)
    offset = 0
    for group in groups:
        weights[group] = np.mean(utilities[offset : offset + len(group)])
        offset += len(group)
    return weights - weights.mean()
