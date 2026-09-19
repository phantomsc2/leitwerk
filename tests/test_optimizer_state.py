from __future__ import annotations

import json
from typing import cast

import numpy as np
import pytest
from leitwerk import Optimizer, SchemaDiff
from leitwerk.state import JSONObject

from ._optimizer_helpers import (
    _assert_same_status,
    _initialized_optimizer,
    _initialized_state,
    _make_identity_schema,
    _optimizer,
    _read_batch,
    _read_mean,
    _read_results,
    _read_scale,
    _read_schema_names,
    _read_status,
)


def test_state_save_load_roundtrip() -> None:
    schema_a = _make_identity_schema("RoundtripA", alpha=(2.0, 1.5), beta=(-1.0, 2.0))
    opt_a = _optimizer(schema_a, batch_size=20)

    for _ in range(37):
        params = opt_a.ask()
        opt_a.tell(-(params.alpha**2 + params.beta**2))

    state = opt_a.save()
    assert isinstance(state, dict)

    schema_b = _make_identity_schema("RoundtripB", beta=(-1.0, 2.0), alpha=(2.0, 1.5))
    opt_b = _optimizer(schema_b, batch_size=20)
    load_result = opt_b.load(state)
    assert load_result == SchemaDiff(added=[], removed=[], changed=[], unchanged=["beta", "alpha"])

    loaded = opt_b.save()
    permutation = [_read_schema_names(state).index(name) for name in _read_schema_names(loaded)]
    assert np.allclose(_read_mean(loaded), _read_mean(state)[permutation])
    assert np.allclose(_read_scale(loaded), _read_scale(state)[np.ix_(permutation, permutation)])
    assert _read_results(loaded) == _read_results(state)
    _assert_same_status(_read_status(loaded), _read_status(state))
    np.testing.assert_allclose(opt_b._xnes.adaptation.paths[0], opt_a._xnes.adaptation.paths[0][permutation])
    np.testing.assert_allclose(
        opt_b._xnes.adaptation.paths[2], opt_a._xnes.adaptation.paths[2][np.ix_(permutation, permutation)]
    )


def test_adaptive_checkpoint_continues_the_same_trajectory() -> None:
    schema = _make_identity_schema("AdaptiveCheckpoint", x=(2.0, 1.0), y=(-3.0, 2.0))
    original = _optimizer(schema, batch_size=8)
    for _ in range(83):
        params = original.ask()
        original.tell(-(params.x**2 + 3 * params.y**2))
    checkpoint = json.loads(json.dumps(original.save(), allow_nan=False))
    restored = _optimizer(schema, batch_size=8)
    restored.load(checkpoint)
    assert restored._xnes.adaptation.steps > 5
    for _ in range(80):
        first, second = original.ask(), restored.ask()
        assert first.x == pytest.approx(second.x)
        assert first.y == pytest.approx(second.y)
        original.tell(-(first.x**2 + 3 * first.y**2))
        restored.tell(-(second.x**2 + 3 * second.y**2))
    np.testing.assert_allclose(original._xnes.adaptation.multipliers, restored._xnes.adaptation.multipliers)
    np.testing.assert_allclose(original._xnes.scale, restored._xnes.scale)


def test_schema_change_resets_adaptation() -> None:
    schema = _make_identity_schema("BeforeChange", x=(2.0, 1.0))
    original = _optimizer(schema, batch_size=8)
    for _ in range(48):
        params = original.ask()
        original.tell(-(params.x**2))
    changed = _optimizer(_make_identity_schema("AfterChange", x=(2.0, 1.0), y=(1.0, 2.0)))
    changed.load(original.save())
    assert changed._xnes.adaptation.steps == 0
    np.testing.assert_array_equal(changed._xnes.adaptation.multipliers, np.ones(3))


def test_status_block_exposes_basic_diagnostics_and_roundtrips() -> None:
    schema = _make_identity_schema("BasicDiagnostics", alpha=(2.0, 1.5), beta=(-1.0, 1.5))
    optimizer = _initialized_optimizer(schema, batch_size=4)

    initial_status = _read_status(optimizer.save())
    assert initial_status["num_samples"] == 0
    assert initial_status["num_batches"] == 0
    assert initial_status["num_restarts"] == 0
    assert initial_status["num_parameters"] == 2
    assert initial_status["axis_ratio"] == pytest.approx(1.0)
    assert initial_status["scale_global"] == pytest.approx(1.5)
    assert initial_status["batch_progress"] == pytest.approx(0.0)
    assert initial_status["batch_size"] == 4
    assert "settings" not in optimizer.save()

    params = optimizer.ask()
    optimizer.tell(-(params.alpha**2 + params.beta**2))
    progressed_state = optimizer.save()
    progressed_status = _read_status(progressed_state)

    assert progressed_status["num_samples"] == 1
    assert progressed_status["num_batches"] == 0
    assert progressed_status["num_restarts"] == 0
    assert progressed_status["num_parameters"] == 2
    assert progressed_status["axis_ratio"] == pytest.approx(1.0)
    assert progressed_status["scale_global"] == pytest.approx(1.5)
    assert progressed_status["batch_progress"] == pytest.approx(1.0)
    assert progressed_status["batch_size"] == 4

    restored = _initialized_optimizer(schema, batch_size=4)
    restored.load(progressed_state)
    _assert_same_status(_read_status(restored.save()), progressed_status)


def test_load_rejects_nonfinite_mean() -> None:
    schema = _make_identity_schema("BadMeanCheckpoint", x=(0.0, 1.0))
    state = _initialized_state(schema, batch_size=4)
    state["mean"] = [float("nan")]

    with pytest.raises(ValueError, match="checkpoint mean must contain only finite values"):
        _optimizer(schema, batch_size=4).load(state)


def test_load_rejects_misaligned_batch_shape() -> None:
    schema = _make_identity_schema("BadBatchCheckpoint", x=(0.0, 1.0))
    state = _initialized_state(schema, batch_size=4)
    state["batch"] = [1.0, -1.0]

    with pytest.raises(ValueError, match=r"checkpoint batch must have shape \(1, n\)"):
        _optimizer(schema, batch_size=4).load(state)


def test_load_rejects_nonfinite_results() -> None:
    schema = _make_identity_schema("BadResultsCheckpoint", x=(0.0, 1.0))
    state = _initialized_state(schema, batch_size=4)
    state["results"] = [[float("nan")], None, None, None]

    with pytest.raises(ValueError, match=r"checkpoint results\[0\] must contain only finite values"):
        _optimizer(schema, batch_size=4).load(state)


def test_load_rejects_bad_pending_context_matches_shape() -> None:
    schema = _make_identity_schema("BadContextCheckpoint", x=(0.0, 1.0))
    state = _initialized_state(schema, batch_size=4)
    state["pending_context_matches"] = cast(JSONObject, {1: 0})

    with pytest.raises(TypeError, match="checkpoint pending_context_matches keys must be strings"):
        _optimizer(schema, batch_size=4).load(state)


def test_load_cancels_pending_ask_and_replaces_state() -> None:
    schema = _make_identity_schema("PendingLoad", x=(0.0, 1.0))
    optimizer = _initialized_optimizer(schema, batch_size=4)
    state = optimizer.save()

    params = optimizer.ask()
    optimizer.load(state)

    with pytest.raises(RuntimeError, match="No pending ask"):
        optimizer.tell(params.x)

    restored_state = optimizer.save()
    assert _read_results(restored_state) == _read_results(state)
    assert np.allclose(_read_mean(restored_state), _read_mean(state))
    assert np.allclose(_read_scale(restored_state), _read_scale(state))
    assert np.allclose(_read_batch(restored_state), _read_batch(state))


def test_save_ignores_pending_ask_and_snapshots_committed_state() -> None:
    schema = _make_identity_schema("PendingSave", x=(0.0, 1.0))
    optimizer = _initialized_optimizer(schema, batch_size=4)
    initial_state = optimizer.save()

    params = optimizer.ask()
    saved_state = optimizer.save()

    assert _read_results(saved_state) == _read_results(initial_state)
    assert np.allclose(_read_mean(saved_state), _read_mean(initial_state))
    assert np.allclose(_read_scale(saved_state), _read_scale(initial_state))
    assert np.allclose(_read_batch(saved_state), _read_batch(initial_state))

    optimizer.tell(params.x)
    progressed_state = optimizer.save()
    assert sum(result is not None for result in _read_results(progressed_state)) == 1

    restored = _initialized_optimizer(schema, batch_size=4)
    restored.load(saved_state)

    with pytest.raises(RuntimeError, match="No pending ask"):
        restored.tell(params.x)

    restored_state = restored.save()
    assert _read_results(restored_state) == _read_results(initial_state)
    assert np.allclose(_read_mean(restored_state), _read_mean(initial_state))
    assert np.allclose(_read_scale(restored_state), _read_scale(initial_state))
    assert np.allclose(_read_batch(restored_state), _read_batch(initial_state))


def test_load_discards_unsaved_local_progress_at_idle_boundary() -> None:
    schema = _make_identity_schema("DiscardUnsavedProgress", x=(0.0, 1.0))
    optimizer = _initialized_optimizer(schema, batch_size=4)
    initial_state = optimizer.save()

    params = optimizer.ask()
    optimizer.tell(params.x)
    progressed_state = optimizer.save()
    assert sum(result is not None for result in _read_results(progressed_state)) == 1

    optimizer.load(initial_state)
    restored_state = optimizer.save()
    assert _read_results(restored_state) == _read_results(initial_state)
    assert np.allclose(_read_mean(restored_state), _read_mean(initial_state))
    assert np.allclose(_read_scale(restored_state), _read_scale(initial_state))
    assert np.allclose(_read_batch(restored_state), _read_batch(initial_state))


def test_runtime_batch_size_and_seed_do_not_roundtrip_through_state() -> None:
    schema = _make_identity_schema("RuntimeConfig", x=(4.0, 2.0))
    baseline = _optimizer(schema, batch_size=6, seed=101)

    for _ in range(4):
        params = baseline.ask()
        baseline.tell(-(params.x**2))

    state = baseline.save()
    assert isinstance(state, dict)
    assert "settings" not in state
    assert _read_status(state)["batch_size"] == 6

    restored = Optimizer(schema, batch_size=4, seed=202)
    restored.load(state)
    assert restored.batch_size == 4
    assert restored.seed == 202
    assert _read_status(restored.save())["batch_size"] == 6

    for _ in range(2):
        params = restored.ask()
        restored.tell(-(params.x**2))

    assert _read_status(restored.save())["batch_size"] == 4


def test_save_load_preserves_optimizer_state() -> None:
    schema = _make_identity_schema("PreserveState", x=(2.0, 1.5), y=(-1.0, 0.7))
    direct = _initialized_optimizer(schema, batch_size=12)

    recreated_state = direct.save()
    for _ in range(40):
        direct_params = direct.ask()
        direct.tell(-(direct_params.x**2 + 0.5 * direct_params.y**2))

        recreated = _optimizer(schema, batch_size=12)
        recreated.load(recreated_state)
        recreated_params = recreated.ask()
        recreated.tell(-(recreated_params.x**2 + 0.5 * recreated_params.y**2))
        recreated_state = recreated.save()

    direct_state = direct.save()
    assert np.allclose(_read_mean(direct_state), _read_mean(recreated_state))
    assert np.allclose(_read_scale(direct_state), _read_scale(recreated_state))
    for actual, expected in zip(_read_results(direct_state), _read_results(recreated_state), strict=True):
        if actual is None or expected is None:
            assert actual is expected
        else:
            assert actual == pytest.approx(expected)
