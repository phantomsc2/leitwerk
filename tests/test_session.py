from __future__ import annotations

import json
from dataclasses import make_dataclass
from pathlib import Path
from typing import Any

import leitwerk.session as session_module
import numpy as np
import pytest
from leitwerk import OptimizerSession, SchemaDiff, parameter

from ._optimizer_helpers import _TEST_SEED


def _make_schema(schema_name: str, **parameters: tuple[float, float]) -> type[Any]:
    return make_dataclass(
        schema_name,
        [(field_name, float, parameter(mean=mean, scale=scale)) for field_name, (mean, scale) in parameters.items()],
        frozen=True,
        slots=True,
    )


def _read_state(path: Path) -> dict[str, object]:
    state = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(state, dict)
    return state


def _read_batch_size(path: Path) -> int:
    status = _read_state(path)["status"]
    assert isinstance(status, dict)
    return int(status["batch_size"])


def _read_pending_context_matches(state: dict[str, object]) -> dict[str, int]:
    pending = state["pending_context_matches"]
    assert isinstance(pending, dict)
    return {str(key): int(value) for key, value in pending.items()}


def _read_factor_state(path: Path, group: str, key: str) -> dict[str, object]:
    factors = _read_state(path)["factors"]
    assert isinstance(factors, dict)
    group_state = factors[group]
    assert isinstance(group_state, dict)
    state = group_state[key]
    assert isinstance(state, dict)
    return state


def _read_scale(state: dict[str, object]) -> np.ndarray:
    scale = state["scale"]
    assert isinstance(scale, list)
    return np.asarray(scale, dtype=float)


class TestSessionPersistence:
    def test_session_flush_persists_initial_state_and_restores(self, tmp_path: Path) -> None:
        schema = _make_schema("SessionParams", beta=(-1.0, 2.0), alpha=(2.0, 1.5))
        path = tmp_path / "session.json"

        session = OptimizerSession(path, schema, batch_size=6, seed=_TEST_SEED)

        assert session.restored is False
        assert session.dirty is False
        assert session.schema_diff == SchemaDiff(added=["beta", "alpha"], removed=[], changed=[], unchanged=[])
        assert session.batch_size == 6
        assert session.seed == _TEST_SEED
        assert session.mean().__class__ is schema
        assert session.scale_marginal.__class__ is schema
        assert session.scale_marginal.alpha == 1.5
        assert session.scale_marginal.beta == 2.0

        session.flush()
        assert session.dirty is False
        assert path.exists()
        assert "settings" not in _read_state(path)

        restored = OptimizerSession(path, schema)

        assert restored.restored is True
        assert restored.dirty is False
        assert restored.batch_size is None
        assert restored.seed is None
        assert restored.schema_diff == SchemaDiff(added=[], removed=[], changed=[], unchanged=["beta", "alpha"])
        assert restored.mean() == session.mean()
        assert restored.scale_marginal == session.scale_marginal

    def test_session_tell_auto_flushes_committed_progress(self, tmp_path: Path) -> None:
        schema = _make_schema("TellSessionParams", x=(2.0, 1.5), y=(-1.0, 0.7))
        path = tmp_path / "session.json"
        session = OptimizerSession(path, schema, batch_size=4, seed=_TEST_SEED)

        params = session.ask()
        report = session.tell(-(params.x**2 + 0.5 * params.y**2))

        assert report.completed_batch is False
        assert session.dirty is False
        assert path.exists()

        restored = OptimizerSession(path, schema, batch_size=4, seed=_TEST_SEED)
        assert restored.ask() == session.ask()

    def test_session_reports_schema_diff_on_restore(self, tmp_path: Path) -> None:
        base_schema = _make_schema("BaseSessionParams", y=(-1.0, 0.7), x=(2.0, 1.5))
        path = tmp_path / "session.json"

        OptimizerSession(path, base_schema, batch_size=4, seed=_TEST_SEED).flush()

        changed_schema = _make_schema("ChangedSessionParams", z=(3.0, 2.0), x=(2.0, 1.5), y=(-1.0, 0.7))
        restored = OptimizerSession(path, changed_schema, batch_size=4, seed=_TEST_SEED)

        assert restored.restored is True
        assert restored.schema_diff == SchemaDiff(added=["z"], removed=[], changed=[], unchanged=["x", "y"])
        assert restored.mean().__class__ is changed_schema

    def test_session_runtime_batch_size_and_seed_only_affect_future_batches(self, tmp_path: Path) -> None:
        schema = _make_schema("SessionSettings", x=(2.0, 1.5))
        path = tmp_path / "session.json"
        session = OptimizerSession(path, schema, batch_size=6, seed=_TEST_SEED)
        for _ in range(4):
            params = session.ask()
            session.tell(-(params.x**2))

        restored = OptimizerSession(path, schema, batch_size=4, seed=999)
        assert restored.restored is True
        assert restored.batch_size == 4
        assert restored.seed == 999
        assert _read_batch_size(path) == 6

        for _ in range(2):
            params = restored.ask()
            restored.tell(-(params.x**2))

        assert _read_batch_size(path) == 4


class TestSessionContext:
    def test_session_ask_with_context_persists_factor_state(self, tmp_path: Path) -> None:
        schema = _make_schema("FactorSessionParams", x=(2.0, 2.0))
        path = tmp_path / "session.json"
        session = OptimizerSession(path, schema, batch_size=4, seed=_TEST_SEED)

        params = session.ask({"opponent": "Sharpy", "map": "Goldenaura"})
        session.tell(-(params.x**2))

        state = _read_state(path)
        expected_scale = np.array([2.0 / np.sqrt(3.0)])
        assert state["mirror_projection"] == []
        assert "factor_count" not in state
        base_scale = _read_scale(state)
        assert np.allclose(np.diag(base_scale), expected_scale)
        opponent_scale = _read_scale(_read_factor_state(path, "opponent", '"Sharpy"'))
        map_scale = _read_scale(_read_factor_state(path, "map", '"Goldenaura"'))
        assert np.allclose(np.diag(opponent_scale), expected_scale)
        assert np.allclose(np.diag(map_scale), expected_scale)
        total_covariance = base_scale @ base_scale.T + opponent_scale @ opponent_scale.T + map_scale @ map_scale.T
        assert np.allclose(total_covariance, np.diag([4.0]))

    def test_late_context_splits_learned_baseline_scale_with_new_factors(self, tmp_path: Path) -> None:
        schema = _make_schema("LateContextParams", x=(2.0, 2.0))
        path = tmp_path / "session.json"
        session = OptimizerSession(path, schema, batch_size=4, seed=_TEST_SEED)

        params = session.ask()
        session.tell(-(params.x**2))
        session._optimizer._xnes.scale_global = 0.5

        params = session.ask({"opponent": "Sharpy", "map": "Goldenaura"})
        session.tell(-(params.x**2))

        state = _read_state(path)
        expected_scale = np.array([0.5 / np.sqrt(3.0)])
        assert np.allclose(np.diag(_read_scale(state)), expected_scale)
        assert np.allclose(
            np.diag(_read_scale(_read_factor_state(path, "opponent", '"Sharpy"'))),
            expected_scale,
        )
        assert np.allclose(
            np.diag(_read_scale(_read_factor_state(path, "map", '"Goldenaura"'))),
            expected_scale,
        )

    def test_session_mean_with_missing_context_value_does_not_create_state(self, tmp_path: Path) -> None:
        schema = _make_schema("MissingFactorMeanParams", x=(2.0, 1.5))
        path = tmp_path / "session.json"
        session = OptimizerSession(path, schema, batch_size=4, seed=_TEST_SEED)

        assert session.mean({"opponent": "Unknown"}) == session.mean()
        session.flush()

        state = _read_state(path)
        assert "factors" not in state

    def test_session_restores_context_factor_state(self, tmp_path: Path) -> None:
        schema = _make_schema("RestoreFactorParams", x=(2.0, 1.5))
        path = tmp_path / "session.json"
        session = OptimizerSession(path, schema, batch_size=4, seed=_TEST_SEED)

        first = session.ask({"opponent": "Sharpy"})
        session.tell(-(first.x**2))

        restored = OptimizerSession(path, schema, batch_size=4, seed=_TEST_SEED)
        assert restored.ask({"opponent": "Sharpy"}).__class__ is schema

    def test_session_chooses_mirror_projection_from_known_context_cardinalities(self, tmp_path: Path) -> None:
        schema = _make_schema("MirrorProjectionParams", x=(2.0, 1.5))
        path = tmp_path / "session.json"
        session = OptimizerSession(path, schema, batch_size=4, seed=_TEST_SEED)

        for opponent, map_name in [
            ("A", "Goldenaura"),
            ("B", "SiteDelta"),
            ("C", "Goldenaura"),
            ("C", "SiteDelta"),
        ]:
            params = session.ask({"opponent": opponent, "map": map_name})
            session.tell(-(params.x**2))

        assert "mirror_projection" not in _read_state(path)

        params = session.ask({"opponent": "A", "map": "Goldenaura"})
        session.tell(-(params.x**2))

        state = _read_state(path)
        assert state["mirror_projection"] == ["map"]
        [mirror_context] = _read_pending_context_matches(state)
        assert json.loads(mirror_context) == {
            "projection": ["map"],
            "values": {"map": "Goldenaura"},
        }

    def test_session_rejects_non_object_context(self, tmp_path: Path) -> None:
        schema = _make_schema("BadSessionContextParams", x=(2.0, 1.5))
        session = OptimizerSession(tmp_path / "session.json", schema, batch_size=4, seed=_TEST_SEED)

        with pytest.raises(TypeError, match="JSON object"):
            session.ask("opponent:Sharpy")  # type: ignore[arg-type]

    def test_session_flush_preserves_unknown_top_level_keys_and_factors(self, tmp_path: Path) -> None:
        schema = _make_schema("PreserveFactorStorageParams", x=(2.0, 1.5))
        path = tmp_path / "session.json"
        session = OptimizerSession(path, schema, batch_size=4, seed=_TEST_SEED)
        session.flush()
        state = _read_state(path)
        state["custom"] = {"keep": True}
        state["factors"] = {"legacy": {}}
        path.write_text(json.dumps(state), encoding="utf-8")

        restored = OptimizerSession(path, schema, batch_size=4, seed=_TEST_SEED)
        restored.flush()

        saved = _read_state(path)
        assert saved["custom"] == {"keep": True}
        assert saved["factors"] == {"legacy": {}}


class TestSessionFailureHandling:
    def test_session_failed_tell_flush_marks_dirty_and_blocks_ask(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        schema = _make_schema("DirtySessionParams", x=(2.0, 1.5))
        path = tmp_path / "session.json"
        session = OptimizerSession(path, schema, batch_size=4, seed=_TEST_SEED)

        def fail_write(path: Path, payload: dict[str, object]) -> None:
            del path, payload
            raise OSError("disk full")

        monkeypatch.setattr(session_module, "_write_json_atomically", fail_write)

        params = session.ask()
        with pytest.raises(OSError, match="disk full"):
            session.tell(-(params.x**2))

        assert session.dirty is True
        with pytest.raises(RuntimeError, match=r"flush\(\) before ask"):
            session.ask()

    def test_session_flush_recovers_dirty_state_after_transient_write_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        schema = _make_schema("DirtyRecoveryParams", x=(2.0, 1.5))
        path = tmp_path / "session.json"
        session = OptimizerSession(path, schema, batch_size=4, seed=_TEST_SEED)
        real_write = session_module._write_json_atomically

        def fail_write(path: Path, payload: dict[str, object]) -> None:
            del path, payload
            raise OSError("disk full")

        monkeypatch.setattr(session_module, "_write_json_atomically", fail_write)

        params = session.ask()
        with pytest.raises(OSError, match="disk full"):
            session.tell(-(params.x**2))

        assert session.dirty is True

        monkeypatch.setattr(session_module, "_write_json_atomically", real_write)
        session.flush()

        assert session.dirty is False
        assert path.exists()
        assert session.ask().__class__ is schema

    def test_session_rejects_non_object_checkpoint(self, tmp_path: Path) -> None:
        schema = _make_schema("InvalidCheckpointParams", x=(0.0, 1.0))
        path = tmp_path / "session.json"
        path.write_text("[]", encoding="utf-8")

        with pytest.raises(TypeError, match="JSON object"):
            OptimizerSession(path, schema)
