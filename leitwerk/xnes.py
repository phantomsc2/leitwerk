"""Core xNES distribution update implementation.

The class maintains the search distribution in factored form
`scale_global * scale_shape`,
generates mirrored orthogonal samples, and applies canonical xNES updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.linalg import cond, norm
from scipy.linalg import expm, qr


@dataclass(frozen=True, slots=True)
class XNESLearningRates:
    """Learning-rate multipliers used by xNES updates."""

    eta_mean: float = 1.0
    """Mean learning rate."""
    eta_scale_global: float = 0.5
    """Covariance scale learning rate."""
    eta_scale_shape: float = 0.25
    """Covariance shape learning rate."""

    def __post_init__(self) -> None:
        if self.eta_mean <= 0.0:
            raise ValueError("eta_mean must be > 0.")
        if self.eta_scale_global <= 0.0:
            raise ValueError("eta_scale_global must be > 0.")
        if self.eta_scale_shape <= 0.0:
            raise ValueError("eta_scale_shape must be > 0.")


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
            basis, _ = qr(raw, mode="economic")
            z_half[:, start:end] = basis * lengths

        return np.hstack([z_half, -z_half])

    def update(
        self,
        samples: np.ndarray,
        ranking: list[int],
        learning_rates: XNESLearningRates | None = None,
        eps: float = 1e-10,
    ) -> XNESStatus:
        """Apply one xNES update from ranked standardized samples.

        Args:
            samples: Standardized sample matrix with shape `(dim, n)`.
            ranking: Permutation of sample indices ordered from best to worst.
            learning_rates: xNES-specific learning rates.
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
        if len(ranking) != n or sorted(ranking) != list(range(n)):
            msg = "ranking must be a permutation matching sample count."
            raise ValueError(msg)
        w_active = _utility_weights(n)
        z_sorted = samples[:, ranking]

        grad_mean = z_sorted @ w_active
        grad_M = (z_sorted * w_active) @ z_sorted.T
        grad_scale_global = float(np.trace(grad_M) / d)
        grad_scale_shape = grad_M - grad_scale_global * np.eye(d)

        mean_step = learning_rates.eta_mean * self.scale_global * (self.scale_shape @ grad_mean)
        self.mean += mean_step

        scale_global_log_step = 0.5 * learning_rates.eta_scale_global * grad_scale_global
        scale_global_log_step = float(np.clip(scale_global_log_step, -50.0, 50.0))

        self.scale_global *= float(np.exp(scale_global_log_step))
        if not np.isfinite(self.scale_global):
            return XNESStatus.SCALE_GLOBAL_INF

        if self.scale_global < eps:
            return XNESStatus.SCALE_GLOBAL_MIN
        if self.scale_global > 1.0 / eps:
            return XNESStatus.SCALE_GLOBAL_MAX

        eta_scale_shape_eff = learning_rates.eta_scale_shape * _default_eta_scale_shape(d)
        self.scale_shape = self.scale_shape @ expm(0.5 * eta_scale_shape_eff * grad_scale_shape)

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
