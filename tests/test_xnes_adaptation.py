from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest
from leitwerk import XNES, XNESLearningRates
from leitwerk._adaptation import RateAdaptation, ranking_noise
from leitwerk._geometry import gradient_spectrum
from leitwerk.xnes import _default_eta_scale_shape, _rank_weights, _utility_weights
from scipy.linalg import expm


@pytest.mark.parametrize("dim,count,mirrored", [(20, 8, True), (3, 16, True), (9, 6, False), (3, 10, False)])
def test_compact_exponential_matches_dense(dim: int, count: int, mirrored: bool) -> None:
    rng = np.random.default_rng(14)
    xnes = XNES(np.zeros(dim), 1.0)
    samples = xnes.sample(count, rng) if mirrored else rng.normal(size=(dim, count))
    weights = _utility_weights(count)[rng.permutation(count)]
    basis, values = gradient_spectrum(samples, weights)
    gradient = (samples * weights) @ samples.T
    assert np.allclose((basis * values) @ basis.T, gradient, atol=1e-12)
    compact = np.eye(dim) + (basis * np.expm1(0.3 * values)) @ basis.T
    assert np.allclose(compact, expm(0.3 * gradient), atol=1e-12)


def test_rank_deficient_external_samples() -> None:
    samples = np.array([[1, 1, 0, -1], [2, 2, 0, -2], [0, 0, 0, 0]], dtype=float)
    weights = _utility_weights(4)
    basis, values = gradient_spectrum(samples, weights)
    assert np.allclose((basis * values) @ basis.T, (samples * weights) @ samples.T)


@pytest.mark.parametrize("dim,count", [(20, 8), (3, 16)])
def test_fixed_rate_baseline_matches_dense_update(dim: int, count: int) -> None:
    rng = np.random.default_rng(9)
    initial = np.diag(np.linspace(0.5, 2, dim))
    xnes = XNES(np.ones(dim), initial)
    samples = xnes.sample(count, rng)
    ranking = rng.permutation(count).tolist()
    sorted_samples = samples[:, ranking]
    weights = _utility_weights(count)
    gradient = (sorted_samples * weights) @ sorted_samples.T
    scale_gradient = np.trace(gradient) / dim
    shape_gradient = gradient - scale_gradient * np.eye(dim)
    expected_mean = xnes.mean + initial @ (sorted_samples @ weights)
    expected_scale = initial @ expm(
        0.5 * (0.5 * scale_gradient * np.eye(dim) + 0.25 * _default_eta_scale_shape(dim) * shape_gradient)
    )
    xnes.update(samples, ranking, XNESLearningRates(1.0, 0.5, 0.25, adaptive=False, max_kl=np.inf))
    np.testing.assert_allclose(xnes.mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(xnes.scale, expected_scale, atol=1e-12)
    assert xnes.adaptation.steps == 0


def test_trust_limit_matches_actual_gaussian_kl() -> None:
    rng = np.random.default_rng(12)
    xnes = XNES(np.ones(8), np.diag(np.arange(1, 9)))
    old_mean, old_scale = xnes.mean.copy(), xnes.scale.copy()
    samples = xnes.sample(6, rng)
    xnes.update(samples, list(range(6)), XNESLearningRates(8, 8, 8, max_kl=0.03))
    relative = np.linalg.solve(old_scale, xnes.scale)
    displacement = np.linalg.solve(old_scale, xnes.mean - old_mean)
    kl = 0.5 * (np.sum(relative**2) + displacement @ displacement - 8 - 2 * np.linalg.slogdet(relative)[1])
    assert xnes.last_kl == pytest.approx(kl)
    assert np.all(xnes.block_kl <= 0.03 + 1e-10)
    assert xnes.block_kl[0] == pytest.approx(0.03)
    assert np.all((xnes.step_fractions > 0) & (xnes.step_fractions <= 1))


def test_trust_limits_do_not_couple_mean_scale_and_shape() -> None:
    samples = XNES(np.zeros(8), 1.0).sample(6, np.random.default_rng(13))
    base = XNES(np.zeros(8), 1.0)
    large_mean = XNES(np.zeros(8), 1.0)
    large_scale = XNES(np.zeros(8), 1.0)
    for optimizer, rates in (
        (base, XNESLearningRates(0.1, 0.2, 0.3, max_kl=0.01)),
        (large_mean, XNESLearningRates(100, 0.2, 0.3, max_kl=0.01)),
        (large_scale, XNESLearningRates(0.1, 100, 0.3, max_kl=0.01)),
    ):
        optimizer.update(samples, list(range(6)), rates)
    np.testing.assert_allclose(base.scale, large_mean.scale)
    np.testing.assert_allclose(base.mean, large_scale.mean)
    np.testing.assert_allclose(base.scale_shape, large_scale.scale_shape)


def test_rates_distinguish_persistent_and_alternating_signals() -> None:
    consistent, alternating = RateAdaptation(2), RateAdaptation(2)
    base = np.array([0.2, 0.2, 0.02])
    gradients = [np.ones(2), np.ones(1), np.diag([1.0, -1.0])]
    for step in range(100):
        for controller, sign in ((consistent, 1), (alternating, (-1) ** step)):
            controller.update(
                [sign * g for g in gradients],
                np.ones(3),
                base * controller.multipliers,
                np.ones(3),
                16,
                5,
                2,
                5,
            )
    assert np.all(consistent.multipliers[[0, 2]] > 1)
    assert np.all(alternating.multipliers[[0, 2]] < 0.1)
    # Scale measures excess movement power, not signed directional persistence.
    assert consistent.multipliers[1] == pytest.approx(alternating.multipliers[1])
    assert np.all(np.isfinite(consistent.signal))


@pytest.mark.parametrize("mirrored", [False, True])
def test_random_ranking_reference_matches_all_permutations(mirrored: bool) -> None:
    samples = np.random.default_rng(11).normal(size=(3, 4))
    if mirrored:
        samples[:, 2:] = -samples[:, :2]
    weights = _utility_weights(4)
    powers = []
    for order in permutations(range(4)):
        w = weights[list(order)]
        matrix = (samples * w) @ samples.T
        scale = np.trace(matrix) / 3
        shape = matrix - scale * np.eye(3)
        powers.append([np.sum((samples @ w) ** 2), 1.5 * scale**2, np.sum(shape**2) / 2])
    np.testing.assert_allclose(ranking_noise(samples, weights), np.mean(powers, axis=0), atol=1e-12)


def test_tied_utilities_and_constant_objective_do_not_create_motion() -> None:
    weights = _rank_weights([[0, 2], [1, 3]], 4)
    assert weights[0] == weights[2]
    assert weights[1] == weights[3]
    assert weights.sum() == pytest.approx(0)
    optimizer = XNES(np.ones(3), 2.0)
    rng = np.random.default_rng(10)
    for _ in range(10):
        samples = optimizer.sample(8, rng)
        optimizer.update(samples, [list(range(8))])
    np.testing.assert_array_equal(optimizer.mean, np.ones(3))
    np.testing.assert_array_equal(optimizer.scale, 2 * np.eye(3))
    assert optimizer.adaptation.steps == 0


def test_scale_rate_recovery_is_slower_than_other_blocks() -> None:
    controller = RateAdaptation(2)
    base = np.full(3, 0.01)
    for _ in range(10):
        controller.update(
            [np.full(2, 10.0), np.full(1, 10.0), np.diag([10.0, -10.0])],
            np.ones(3),
            base * controller.multipliers,
            np.ones(3),
            32,
            10,
            2,
            5,
        )
    np.testing.assert_allclose(controller.multipliers, [2.0, 2.0**0.2, 2.0])


def test_zero_shape_signal_does_not_throttle_one_dimensional_controller() -> None:
    xnes = XNES(np.ones(1), 1.0)
    rng = np.random.default_rng(2)
    for _ in range(10):
        samples = xnes.sample(6, rng)
        xnes.update(samples, list(range(6)))
    assert xnes.adaptation.multipliers[2] == 1


def test_adaptive_updates_are_equivariant_under_rotation_of_whitened_frame() -> None:
    rng = np.random.default_rng(21)
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    first = XNES(np.ones(4), 1.0)
    second = XNES(np.ones(4), rotation.T)
    for _ in range(20):
        samples = first.sample(10, rng)
        ranking = np.argsort(np.sum(first.transform(samples) ** 2, axis=0)).tolist()
        first.update(samples, ranking)
        second.update(rotation @ samples, ranking)
        assert np.allclose(first.mean, second.mean)
        assert np.allclose(first.scale @ first.scale.T, second.scale @ second.scale.T)
        assert np.allclose(first.adaptation.multipliers, second.adaptation.multipliers)
