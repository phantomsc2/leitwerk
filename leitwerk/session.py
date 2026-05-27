"""Filesystem-backed optimizer session wrapper."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from hashlib import blake2b
from itertools import combinations
from math import log
from pathlib import Path
from typing import Generic, TypeVar, cast

import numpy as np

from .optimizer import Optimizer, OptimizerReport
from .schema import SchemaDiff
from .state import JSONLike, JSONObject

T = TypeVar("T")

_FACTORS_KEY = "factors"
_MIRROR_PROJECTION_KEY = "mirror_projection"


class OptimizerSession(Generic[T]):
    """Persisted optimizer workflow around an in-memory `Optimizer`."""

    def __init__(
        self,
        path: str | Path,
        schema: type[T] | Mapping[str, object],
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> None:

        session_path = Path(path)
        optimizer = Optimizer(schema, batch_size=batch_size, seed=seed)

        restored = False
        state_base: JSONObject = {}
        factors_state: Mapping[str, object] = {}
        mirror_projection: tuple[str, ...] | None = None
        schema_diff = _fresh_schema_diff(cast(Mapping[str, object], optimizer.save()["schema"]))
        if session_path.exists():
            state = json.loads(session_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                msg = "Persisted optimizer state must be a JSON object."
                raise TypeError(msg)
            restored = True
            state_base = dict(cast(JSONObject, state))
            factors_state = _read_factors_state(state.get(_FACTORS_KEY, {}))
            mirror_projection = _read_mirror_projection(state.get(_MIRROR_PROJECTION_KEY))
            schema_diff = optimizer.load(cast(JSONObject, state))

        self._path = session_path
        self._schema = schema
        self._optimizer = optimizer
        self._factors = self._load_factors(factors_state)
        self._state_base = state_base
        self._mirror_projection = mirror_projection
        self._pending_factors: tuple[tuple[str, str], ...] = ()
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

    def mean(self, context: Mapping[str, JSONLike] | None = None) -> T:
        """Current optimizer mean parameters."""
        latent = self._optimizer.mean_latent
        for name, key, _ in _context_items(context):
            factor = self._factors.get(name, {}).get(key)
            if factor is not None:
                latent += factor.mean_latent
        return self._optimizer.decode(latent)

    @property
    def scale_marginal(self) -> T:
        """Current optimizer scale-vector parameters."""
        return self._optimizer.scale_marginal

    def ask(self, context: Mapping[str, JSONLike] | None = None) -> T:
        """Reserve one sampled parameter set for evaluation."""
        self._require_clean()
        context_items = _context_items(context)
        active_factors: list[tuple[str, str]] = []
        for name, key, _ in context_items:
            self._factor(name, key)
            active_factors.append((name, key))
        if context is not None and self._mirror_projection is None:
            self._mirror_projection = self._choose_mirror_projection(context_items)
        mirror_context = _mirror_context(context_items, self._mirror_projection)
        latent = self._optimizer.ask_latent(mirror_context)
        for name, key in active_factors:
            factor = self._factor(name, key)
            latent += factor.ask_latent(mirror_context)
        self._pending_factors = tuple(active_factors)
        return self._optimizer.decode(latent)

    def tell(self, result: float | Sequence[float] | np.ndarray) -> OptimizerReport:
        """Record one result and atomically persist the updated optimizer state."""
        report = self._optimizer.tell(result)
        for name, key in self._pending_factors:
            self._factors[name][key].tell(result)
        self._pending_factors = ()
        if report.completed_batch:
            self._mirror_projection = None
        self._dirty = True
        self.flush()
        return report

    def flush(self) -> None:
        """Persist the current committed optimizer state."""
        self._dirty = True
        payload = self._save_state()
        _write_json_atomically(self._path, payload)
        self._dirty = False

    def _require_clean(self) -> None:
        if self._dirty:
            msg = "Session has unflushed committed state. Call flush() before ask()."
            raise RuntimeError(msg)

    def _factor(self, name: str, key: str) -> Optimizer[T]:
        group = self._factors.setdefault(name, {})
        factor = group.get(key)
        if factor is None:
            factor = self._new_factor(name, key)
            group[key] = factor
        return factor

    def _new_factor(self, name: str, key: str) -> Optimizer[T]:
        factor = Optimizer(
            self._schema,
            batch_size=self._optimizer.batch_size,
            seed=_factor_seed(self._optimizer.seed, name, key),
        )
        factor.load(_factor_initial_state(factor.save()))
        return factor

    def _load_factors(self, factors_state: Mapping[str, object]) -> dict[str, dict[str, Optimizer[T]]]:
        factors: dict[str, dict[str, Optimizer[T]]] = {}
        for name, values in factors_state.items():
            if not isinstance(name, str):
                msg = "factor group names must be strings."
                raise TypeError(msg)
            value_states = _read_factor_value_states(values, name)
            factors[name] = {}
            for key, state in value_states.items():
                factor = Optimizer(
                    self._schema,
                    batch_size=self._optimizer.batch_size,
                    seed=_factor_seed(self._optimizer.seed, name, key),
                )
                factor.load(state)
                factors[name][key] = factor
        return factors

    def _choose_mirror_projection(self, context_items: tuple[tuple[str, str, JSONLike], ...]) -> tuple[str, ...]:
        target = _current_batch_size(self._optimizer) / 2
        projections = _factor_group_projections(tuple(name for name, _, _ in context_items))
        return min(
            projections,
            key=lambda projection: (
                abs(log(_projection_cardinality(projection, self._factors) / target)),
                projection,
            ),
        )

    def _save_state(self) -> JSONObject:
        payload = dict(self._state_base)
        payload.update(self._optimizer.save())
        factors = self._save_factors()
        if self._has_factor_storage:
            payload[_FACTORS_KEY] = factors
        if self._mirror_projection is None:
            payload.pop(_MIRROR_PROJECTION_KEY, None)
        else:
            payload[_MIRROR_PROJECTION_KEY] = list(self._mirror_projection)
        return payload

    @property
    def _has_factor_storage(self) -> bool:
        return _FACTORS_KEY in self._state_base or any(self._factors.values())

    def _save_factors(self) -> JSONObject:
        saved: JSONObject = {}
        for name, values in _read_factors_state(self._state_base.get(_FACTORS_KEY, {})).items():
            saved[name] = cast(JSONObject, values)
        for name, values in sorted(self._factors.items()):
            saved[name] = {key: factor.save() for key, factor in sorted(values.items())}
        return saved


def _fresh_schema_diff(schema_json: Mapping[str, object]) -> SchemaDiff:
    return SchemaDiff(added=list(schema_json), removed=[], changed=[], unchanged=[])


def _context_items(context: Mapping[str, JSONLike] | None) -> tuple[tuple[str, str, JSONLike], ...]:
    if context is None:
        return ()
    if not isinstance(context, Mapping):
        msg = "context must be a JSON object."
        raise TypeError(msg)
    items: list[tuple[str, str, JSONLike]] = []
    for name in sorted(context):
        if not isinstance(name, str):
            msg = "factor group names must be strings."
            raise TypeError(msg)
        value = context[name]
        if value is None:
            continue
        items.append((name, _factor_key(value), value))
    return tuple(items)


def _factor_key(value: JSONLike) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        msg = "factor values must be JSON-serializable."
        raise TypeError(msg) from exc


def _mirror_context(
    context_items: tuple[tuple[str, str, JSONLike], ...],
    projection: tuple[str, ...] | None,
) -> Mapping[str, JSONLike] | None:
    if projection is None:
        return None
    values_by_name = {name: value for name, _, value in context_items}
    projection_json: list[JSONLike] = [name for name in projection]
    values: dict[str, JSONLike] = {name: values_by_name[name] for name in projection if name in values_by_name}
    return {
        "projection": projection_json,
        "values": values,
    }


def _factor_group_projections(names: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    return tuple(combination for size in range(len(names) + 1) for combination in combinations(names, size))


def _projection_cardinality(
    projection: tuple[str, ...],
    factors: Mapping[str, Mapping[str, object]],
) -> int:
    cardinality = 1
    for name in projection:
        cardinality *= max(len(factors.get(name, {})), 1)
    return cardinality


def _current_batch_size(optimizer: Optimizer[T]) -> int:
    status = optimizer.save()["status"]
    if not isinstance(status, Mapping):
        msg = "optimizer status is invalid."
        raise TypeError(msg)
    return int(cast(int | float | str, status["batch_size"]))


def _factor_initial_state(state: JSONObject) -> JSONObject:
    mean = state["mean"]
    if not isinstance(mean, list):
        msg = "fresh optimizer state is invalid."
        raise TypeError(msg)
    out = dict(state)
    out["mean"] = [0.0] * len(mean)
    out["batch"] = []
    out["results"] = []
    out["pending_context_matches"] = {}
    return out


def _factor_seed(seed: int | None, name: str, key: str) -> int | None:
    if seed is None:
        return None
    text = f"{int(seed)}\0{name}\0{key}".encode()
    return int.from_bytes(blake2b(text, digest_size=8).digest(), "big")


def _read_factors_state(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        msg = "factors must be a JSON object."
        raise TypeError(msg)
    return cast(Mapping[str, object], value)


def _read_mirror_projection(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        msg = "mirror_projection must be a sequence of strings."
        raise TypeError(msg)
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            msg = "mirror_projection must be a sequence of strings."
            raise TypeError(msg)
        out.append(item)
    return tuple(out)


def _read_factor_value_states(value: object, name: str) -> dict[str, JSONObject]:
    if not isinstance(value, Mapping):
        msg = f"factor group {name!r} must be a JSON object."
        raise TypeError(msg)
    out: dict[str, JSONObject] = {}
    for key, state in value.items():
        if not isinstance(key, str):
            msg = f"factor group {name!r} keys must be strings."
            raise TypeError(msg)
        if not isinstance(state, dict):
            msg = f"factor state {name!r}/{key!r} must be a JSON object."
            raise TypeError(msg)
        out[key] = cast(JSONObject, state)
    return out


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
