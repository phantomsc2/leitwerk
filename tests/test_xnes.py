from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from leitwerk import XNES, XNESLearningRates, XNESStatus
from leitwerk.xnes import _default_eta_scale_shape
from scipy.linalg import expm


def _ranking(scores: np.ndarray) -> list[int]:
    return sorted(range(scores.size), key=lambda idx: float(scores[idx]), reverse=True)


def test_xnes_rank_invariance_under_monotonic_transform() -> None:
    dim = 4
    n = 24
    steps = 6
    eta_mean = 1.0
    eta_scale_global = 0.8
    eta_scale_shape = 0.2

    learning_rates = XNESLearningRates(eta_mean, eta_scale_global, eta_scale_shape)
    xnes_a = XNES(np.zeros(dim), np.eye(dim))
    xnes_b = XNES(np.zeros(dim), np.eye(dim))

    rng_a = np.random.default_rng(5)
    rng_b = np.random.default_rng(5)

    for _ in range(steps):
        z_a = xnes_a.sample(n, rng_a)
        x_a = xnes_a.transform(z_a)
        z_b = xnes_b.sample(n, rng_b)
        assert np.allclose(z_a, z_b)
        assert np.allclose(x_a, xnes_a.transform(z_a))

        raw_scores = -np.sum(x_a**2, axis=0) + 1e-12 * np.arange(n)
        transformed_scores = 3.0 * raw_scores + 2.0

        ranking_raw = _ranking(raw_scores)
        ranking_transformed = _ranking(transformed_scores)
        assert ranking_raw == ranking_transformed

        xnes_a.update(z_a, ranking_raw, learning_rates)
        xnes_b.update(z_b, ranking_transformed, learning_rates)

    assert np.allclose(xnes_a.mean, xnes_b.mean)
    assert np.allclose(xnes_a.scale, xnes_b.scale)


def test_xnes_linear_invariance_with_stress_values() -> None:
    n = 30
    steps = 5
    eta_mean = 1.0
    eta_scale_global = 0.7
    eta_scale_shape = 0.3

    mean = np.array([1e10, -1e10, 5e9], dtype=float)
    scale = np.array(
        [
            [1.0e-10, 2.0e-11, -1.0e-11],
            [0.0, 3.0e-10, 2.0e-11],
            [0.0, 0.0, 8.0e-11],
        ],
        dtype=float,
    )

    transform = np.array(
        [
            [2.0e4, 4.0e2, 0.0],
            [0.0, 3.0e-4, 8.0],
            [5.0, -2.0, 4.0e1],
        ],
        dtype=float,
    )
    shift = np.array([-2.0e9, 4.0e9, -3.0e8], dtype=float)

    learning_rates = XNESLearningRates(eta_mean, eta_scale_global, eta_scale_shape)
    xnes_x = XNES(mean, scale)
    xnes_y = XNES(transform @ mean + shift, transform @ scale)

    rng_x = np.random.default_rng(9)
    rng_y = np.random.default_rng(9)
    score_projection = np.array([1.7, -0.2, 3.3], dtype=float)

    for _ in range(steps):
        z_x = xnes_x.sample(n, rng_x)
        z_y = xnes_y.sample(n, rng_y)
        assert np.allclose(z_x, z_y)

        scores = score_projection @ z_x + 1e-12 * np.arange(n)
        ranking = _ranking(scores)

        status_x = xnes_x.update(z_x, ranking, learning_rates)
        status_y = xnes_y.update(z_y, ranking, learning_rates)
        assert status_x == status_y

        assert np.all(np.isfinite(xnes_x.mean))
        assert np.isfinite(xnes_x.scale_global)
        assert np.all(np.isfinite(xnes_x.scale))

        assert np.all(np.isfinite(xnes_y.mean))
        assert np.isfinite(xnes_y.scale_global)
        assert np.all(np.isfinite(xnes_y.scale))

        assert np.allclose(xnes_y.mean, transform @ xnes_x.mean + shift, rtol=1e-10, atol=3e-3)
        assert np.allclose(xnes_y.scale, transform @ xnes_x.scale, rtol=1e-10, atol=1e-12)


def test_xnes_converges_in_one_dimension_on_quadratic_objective() -> None:
    xnes = XNES(np.array([5.0]), np.array([[4.0]]))
    rng = np.random.default_rng(7)

    for _ in range(40):
        z = xnes.sample(12, rng)
        x = xnes.transform(z)
        ranking = _ranking(-(x[0] ** 2))
        status = xnes.update(z, ranking)
        assert status is XNESStatus.OK

    assert abs(float(xnes.mean[0])) < 1e-3
    assert float(xnes.scale_global) < 1e-2
    assert np.isclose(float(xnes.scale[0, 0]), float(xnes.scale_global))


def test_xnes_eta_scale_shape_scales_dimension_dependent_shape_rate() -> None:
    xnes = XNES(np.zeros(3), np.eye(3))

    samples = np.array(
        [
            [1.0, 0.0, -1.0, 0.5],
            [0.5, -1.0, 0.0, 1.0],
            [-0.5, 1.0, 0.5, -1.0],
        ],
        dtype=float,
    )
    ranking = [0, 1, 2, 3]

    n = samples.shape[1]
    d = xnes.dim
    w_pos = np.maximum(0.0, np.log(n / 2 + 1) - np.log(np.arange(1, n + 1)))
    w_pos /= float(np.sum(w_pos))
    w_active = w_pos - (1.0 / n)
    z_sorted = samples[:, ranking]
    grad_M = (z_sorted * w_active) @ z_sorted.T
    grad_sigma = float(np.trace(grad_M) / d)
    grad_B_shape = grad_M - grad_sigma * np.eye(d)

    expected_B = expm(0.5 * 0.2 * _default_eta_scale_shape(d) * grad_B_shape)
    sign, logdet = np.linalg.slogdet(expected_B)
    assert sign > 0
    expected_B *= np.exp(-logdet / d)

    status = xnes.update(
        samples,
        ranking,
        XNESLearningRates(eta_mean=1e-12, eta_scale_global=1e-12, eta_scale_shape=0.2),
    )
    assert status is XNESStatus.MEAN_STEP_MIN
    assert np.allclose(xnes.mean, np.zeros(3))
    assert np.isclose(xnes.scale_global, 1.0)
    assert np.allclose(xnes.scale_shape, expected_B)


@pytest.mark.parametrize("field", ["eta_mean", "eta_scale_global", "eta_scale_shape"])
def test_xnes_learning_rates_require_positive_values(field: str) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be > 0"):
        XNESLearningRates(**{field: 0.0})

    learning_rates = XNESLearningRates()
    with pytest.raises(FrozenInstanceError):
        setattr(learning_rates, field, -1.0)


def test_xnes_status_classification_partitions_statuses() -> None:
    ok = {status for status in XNESStatus if status.is_ok}
    completion = {status for status in XNESStatus if status.is_completion}
    error = {status for status in XNESStatus if status.is_error}
    terminal = {status for status in XNESStatus if status.is_terminal}

    assert ok == {XNESStatus.OK}
    assert completion == {
        XNESStatus.SCALE_GLOBAL_MIN,
        XNESStatus.MEAN_STEP_MIN,
        XNESStatus.SCALE_NORM_MIN,
    }
    assert error == set(XNESStatus) - ok - completion
    assert terminal == completion | error
