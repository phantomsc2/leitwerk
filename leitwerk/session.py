"""Filesystem-backed optimizer session wrapper."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Generic, TypeVar, cast

import numpy as np

from .optimizer import Optimizer, OptimizerReport
from .schema import SchemaDiff
from .state import JSONLike, JSONObject
from .xnes import XNESLearningRates

T = TypeVar("T")


class OptimizerSession(Generic[T]):
    """Persisted optimizer workflow around an in-memory `Optimizer`."""

    def __init__(
        self,
        path: str | Path,
        schema: type[T] | Mapping[str, object],
        batch_size: int | None = None,
        seed: int | None = None,
        learning_rates: XNESLearningRates | None = None,
    ) -> None:

        session_path = Path(path)
        optimizer = Optimizer(schema, batch_size=batch_size, seed=seed, learning_rates=learning_rates)

        restored = False
        schema_diff = _fresh_schema_diff(cast(Mapping[str, object], optimizer.save()["schema"]))
        if session_path.exists():
            state = json.loads(session_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                msg = "Persisted optimizer state must be a JSON object."
                raise TypeError(msg)
            restored = True
            schema_diff = optimizer.load(cast(JSONObject, state))

        self._path = session_path
        self._optimizer = optimizer
        self._dirty = False
        self._restored = restored
        self._schema_diff = schema_diff

    @property
    def restored(self) -> bool:
        """Whether this session loaded an existing checkpoint."""
        return self._restored

    @property
    def dirty(self) -> bool:
        """Whether committed optimizer state exists that has not been durably flushed."""
        return self._dirty

    @property
    def batch_size(self) -> int | None:
        """Configured sample count for the next freshly drawn batch."""
        return self._optimizer.batch_size

    @property
    def seed(self) -> int | None:
        """Configured root seed used for future batch sampling."""
        return self._optimizer.seed

    @property
    def schema_diff(self) -> SchemaDiff:
        """Difference against the restored schema, or an empty baseline on fresh sessions."""
        return self._schema_diff

    @property
    def mean(self) -> T:
        """Current optimizer mean parameters."""
        return self._optimizer.mean

    @property
    def scale_marginal(self) -> T:
        """Current optimizer scale-vector parameters."""
        return self._optimizer.scale_marginal

    def ask(self, context: JSONLike = None) -> T:
        """Reserve one sampled parameter set for evaluation."""
        self._require_clean()
        return self._optimizer.ask(context)

    def tell(self, result: float | Sequence[float] | np.ndarray) -> OptimizerReport:
        """Record one result and atomically persist the updated optimizer state."""
        report = self._optimizer.tell(result)
        self._dirty = True
        self.flush()
        return report

    def flush(self) -> None:
        """Persist the current committed optimizer state."""
        self._dirty = True
        _write_json_atomically(self._path, self._optimizer.save())
        self._dirty = False

    def _require_clean(self) -> None:
        if self._dirty:
            msg = "Session has unflushed committed state. Call flush() before ask()."
            raise RuntimeError(msg)


def _fresh_schema_diff(schema_json: Mapping[str, object]) -> SchemaDiff:
    return SchemaDiff(added=list(schema_json), removed=[], changed=[], unchanged=[])


def _write_json_atomically(path: Path, payload: JSONObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
