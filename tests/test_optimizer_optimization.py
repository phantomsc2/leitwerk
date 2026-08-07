from __future__ import annotations

import numpy as np
import pytest
from leitwerk import XNESLearningRates, XNESStatus

from ._optimizer_helpers import (
    _initialized_optimizer,
    _make_identity_schema,
    _optimizer,
    _read_mean,
    _read_scale,
    _read_status,
    _run_function_optimization,
)


def test_optimizer_improves_sphere() -> None:
    def sphere(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    initial, final = _run_function_optimization(
        sphere,
        init_mean=3.0,
        init_scale=2.0,
        dim=4,
        batch_size=28,
        evaluations=1400,
    )
    assert final < 0.15 * initial


def test_restart_on_conditioning_failure() -> None:
    schema = _make_identity_schema("RestartSchema", x=(0.0, 1.0), y=(0.0, 1.0))
    optimizer = _initialized_optimizer(schema, batch_size=10)

    state = optimizer.save()
    assert isinstance(state, dict)
    state["scale"] = [[1e-10, 0.0], [0.0, 1e10]]

    restored = _optimizer(schema, batch_size=10)
    restored.load(state)

    for _ in range(10):
        params = restored.ask()
        restored.tell(-(params.x**2 + params.y**2))

    conditioned = restored.save()
    assert np.allclose(_read_mean(conditioned), np.array([0.0, 0.0]))
    assert float(np.linalg.cond(_read_scale(conditioned))) < 1e14


def test_successful_termination_restarts_from_final_mean_with_fresh_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = _make_identity_schema("RestartSuccess", x=(2.0, 1.5), y=(-1.0, 0.7))
    optimizer = _initialized_optimizer(schema, batch_size=4)
    final_mean = np.array([4.25, -3.5])

    def fake_update(
        samples: np.ndarray,
        ranking: list[int],
        learning_rates: XNESLearningRates | None = None,
    ) -> XNESStatus:
        del samples, ranking, learning_rates
        optimizer._xnes.mean = final_mean.copy()
        optimizer._xnes.scale_global = 6.0
        optimizer._xnes.scale_shape = np.array([[1.5, 0.2], [0.0, 0.5]])
        return XNESStatus.MEAN_STEP_MIN

    monkeypatch.setattr(optimizer._xnes, "update", fake_update)

    for _ in range(3):
        optimizer.ask()
        report = optimizer.tell(0.0)
        assert report.completed_batch is False

    optimizer.ask()
    report = optimizer.tell(0.0)

    assert report.completed_batch is True
    assert report.status is XNESStatus.MEAN_STEP_MIN
    assert report.restarted is True

    state = optimizer.save()
    assert np.allclose(_read_mean(state), final_mean)
    assert np.allclose(_read_scale(state), np.diag([1.5, 0.7]))

    status = _read_status(state)
    assert status["num_samples"] == 4
    assert status["num_batches"] == 1
    assert status["num_restarts"] == 1
    assert status["batch_progress"] == pytest.approx(0.0)


def test_failed_termination_restarts_from_schema_mean_with_fresh_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    schema = _make_identity_schema("RestartFailure", x=(2.0, 1.5), y=(-1.0, 0.7))
    optimizer = _initialized_optimizer(schema, batch_size=4)

    def fake_update(
        samples: np.ndarray,
        ranking: list[int],
        learning_rates: XNESLearningRates | None = None,
    ) -> XNESStatus:
        del samples, ranking, learning_rates
        optimizer._xnes.mean = np.array([4.25, -3.5])
        optimizer._xnes.scale_global = 6.0
        optimizer._xnes.scale_shape = np.array([[1.5, 0.2], [0.0, 0.5]])
        return XNESStatus.SCALE_COND_MAX

    monkeypatch.setattr(optimizer._xnes, "update", fake_update)

    for _ in range(3):
        optimizer.ask()
        report = optimizer.tell(0.0)
        assert report.completed_batch is False

    optimizer.ask()
    report = optimizer.tell(0.0)

    assert report.completed_batch is True
    assert report.status is XNESStatus.SCALE_COND_MAX
    assert report.restarted is True

    state = optimizer.save()
    assert np.allclose(_read_mean(state), np.array([2.0, -1.0]))
    assert np.allclose(_read_scale(state), np.diag([1.5, 0.7]))

    status = _read_status(state)
    assert status["num_samples"] == 4
    assert status["num_batches"] == 1
    assert status["num_restarts"] == 1
    assert status["batch_progress"] == pytest.approx(0.0)
