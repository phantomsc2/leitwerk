from __future__ import annotations

import json
from dataclasses import make_dataclass
from pathlib import Path
from typing import Any

import leitwerk.session as session_module
import numpy as np
import pytest
from leitwerk import XNES, OptimizerSession, SchemaDiff, XNESLearningRates, XNESStatus, parameter

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
        assert session.mean.__class__ is schema
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
        assert restored.mean == session.mean
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
        assert restored.mean.__class__ is changed_schema

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

    def test_session_learning_rates_are_forwarded_and_not_persisted(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        schema = _make_schema("SessionLearningRates", x=(2.0, 1.5))
        path = tmp_path / "session.json"
        learning_rates = XNESLearningRates(eta_mean=0.5, eta_scale_global=0.4, eta_scale_shape=0.3)
        received: list[XNESLearningRates | None] = []

        def fake_update(
            self: XNES,
            samples: np.ndarray,
            ranking: list[int],
            learning_rates: XNESLearningRates | None = None,
        ) -> XNESStatus:
            del self, samples, ranking
            received.append(learning_rates)
            return XNESStatus.OK

        monkeypatch.setattr(XNES, "update", fake_update)
        session = OptimizerSession(path, schema, batch_size=4, learning_rates=learning_rates)
        for _ in range(4):
            session.ask()
            session.tell(0.0)

        assert received == [learning_rates]
        assert "learning_rates" not in _read_state(path)


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
