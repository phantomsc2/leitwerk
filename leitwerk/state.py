"""IO-agnostic optimizer state serialization and restoration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

from .schema.parameter import Parameter
from .schema.spec import SchemaDiff, SchemaSpec

JSONScalar = str | int | float | bool | None
JSONObject = dict[str, "JSONValue"]
JSONValue = JSONObject | list["JSONValue"] | JSONScalar
JSONLikeObject = Mapping[str, "JSONLike"]
JSONLike = JSONLikeObject | Sequence["JSONLike"] | JSONScalar


@dataclass(frozen=True, slots=True)
class RestoredOptimizerState:
    schema_diff: SchemaDiff
    mean: np.ndarray
    scale: np.ndarray
    batch: np.ndarray
    results: list[tuple[float, ...] | None]
    pending_context_matches: dict[str, int]
    num_samples: int
    num_batches: int
    num_restarts: int


def serialize_optimizer_state(
    *,
    status: Mapping[str, JSONValue],
    mean: np.ndarray,
    scale: np.ndarray,
    schema_state: Mapping[str, Mapping[str, object]],
    batch: np.ndarray,
    results: Sequence[tuple[float, ...] | None],
    pending_context_matches: Mapping[str, int],
) -> JSONObject:
    """Serialize optimizer state into a JSON-compatible mapping."""
    return {
        "status": dict(status),
        "results": _serialize_results(results),
        "pending_context_matches": dict(pending_context_matches),
        "schema": {name: dict(spec) for name, spec in schema_state.items()},
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "batch": batch.tolist(),
    }


def restore_optimizer_state(
    state: JSONObject,
    schema: SchemaSpec[object],
) -> RestoredOptimizerState:
    """Deserialize and reconcile optimizer state against the current schema."""
    state_obj = _require_object(state, "checkpoint state")
    schema_json = _require_object(_require_field(state_obj, "schema"), "checkpoint schema")
    saved_schema = _deserialize_schema(schema_json)
    saved_names = list(saved_schema)
    saved_dim = len(saved_names)
    mean = _as_finite_vector(_require_field(state_obj, "mean"), saved_dim, "checkpoint mean")
    scale = _as_finite_matrix(_require_field(state_obj, "scale"), (saved_dim, saved_dim), "checkpoint scale")
    batch = _as_batch_matrix(_require_field(state_obj, "batch"), saved_dim)
    results = _deserialize_results(_require_field(state_obj, "results"))
    pending_context_matches = _deserialize_pending_context_matches(_require_field(state_obj, "pending_context_matches"))
    num_samples, num_batches, num_restarts = _validate_status(_require_field(state_obj, "status"))

    schema_diff = schema.diff(saved_schema)
    mean, scale = _reconcile_distribution_state(saved_names, schema_diff.unchanged, mean, scale, schema)
    if batch.shape[1] != 0:
        batch = _reconcile_batch_state(saved_names, schema_diff, batch, results, schema)

    return RestoredOptimizerState(
        schema_diff=schema_diff,
        mean=mean,
        scale=scale,
        batch=batch,
        results=results,
        pending_context_matches=pending_context_matches,
        num_samples=num_samples,
        num_batches=num_batches,
        num_restarts=num_restarts,
    )


def _deserialize_schema(schema_json: Mapping[str, object]) -> dict[str, Parameter]:
    schema: dict[str, Parameter] = {}
    for name, spec in schema_json.items():
        if not isinstance(name, str):
            msg = "checkpoint schema keys must be strings."
            raise TypeError(msg)
        try:
            spec_obj = _require_object(spec, f"checkpoint schema entry {name!r}")
            mean = spec_obj["mean"]
            scale = spec_obj["scale"]
            min_value = spec_obj["min"]
            max_value = spec_obj["max"]
            schema[name] = Parameter(
                mean=None if mean is None else _coerce_float_like(mean, f"checkpoint schema entry {name!r}.mean"),
                scale=_coerce_float_like(scale, f"checkpoint schema entry {name!r}.scale"),
                min=None
                if min_value is None
                else _coerce_float_like(min_value, f"checkpoint schema entry {name!r}.min"),
                max=None
                if max_value is None
                else _coerce_float_like(max_value, f"checkpoint schema entry {name!r}.max"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"checkpoint schema entry {name!r} is invalid."
            raise ValueError(msg) from exc
    return schema


def _reconcile_distribution_state(
    saved_names: list[str],
    unchanged_names: list[str],
    mean: np.ndarray,
    scale: np.ndarray,
    schema: SchemaSpec[object],
) -> tuple[np.ndarray, np.ndarray]:
    reconciled_mean, reconciled_scale = schema.initial_distribution()

    saved_index = {name: idx for idx, name in enumerate(saved_names)}
    current_index = schema.index_by_name()
    shared_indices = [(current_index[name], saved_index[name]) for name in unchanged_names]
    for current_idx, saved_idx in shared_indices:
        reconciled_mean[current_idx] = float(mean[saved_idx])

    if shared_indices:
        shared_current_indices, shared_saved_indices = zip(*shared_indices, strict=True)
        reconciled_scale[np.ix_(shared_current_indices, shared_current_indices)] = scale[
            np.ix_(shared_saved_indices, shared_saved_indices)
        ]

    return reconciled_mean, reconciled_scale


def _reconcile_batch_state(
    saved_names: list[str],
    schema_diff: SchemaDiff,
    batch: np.ndarray,
    results: list[tuple[float, ...] | None],
    schema: SchemaSpec[object],
) -> np.ndarray:
    sample_count = batch.shape[1]
    reconciled_batch = np.zeros((schema.dim, sample_count), dtype=float)
    completed_mask = np.zeros(sample_count, dtype=bool)
    for idx, result in enumerate(results[:sample_count]):
        completed_mask[idx] = result is not None
    pending_mask = ~completed_mask
    mirror_index = _mirror_indices(sample_count)
    mirror_pending_mask = pending_mask & pending_mask[mirror_index]

    saved_index = {name: idx for idx, name in enumerate(saved_names)}
    current_index = schema.index_by_name()
    for name in schema_diff.unchanged:
        current_idx = current_index[name]
        saved_idx = saved_index[name]
        reconciled_batch[current_idx, :] = batch[saved_idx, :]
    for name in schema_diff.changed:
        current_idx = current_index[name]
        saved_idx = saved_index[name]
        reconciled_batch[current_idx, mirror_pending_mask] = batch[saved_idx, mirror_pending_mask]

    return reconciled_batch


def _serialize_results(results: Sequence[tuple[float, ...] | None]) -> list[list[float] | None]:
    return [None if row is None else list(row) for row in results]


def _deserialize_results(result_rows: object) -> list[tuple[float, ...] | None]:
    if not isinstance(result_rows, Sequence) or isinstance(result_rows, (str, bytes)):
        msg = "checkpoint results must be a sequence."
        raise TypeError(msg)
    rows = cast(Sequence[object], result_rows)
    results: list[tuple[float, ...] | None] = []
    for idx, row in enumerate(rows):
        if row is None:
            results.append(None)
            continue
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            msg = f"checkpoint results[{idx}] must be null or a sequence of numbers."
            raise TypeError(msg)
        try:
            values = tuple(float(value) for value in row)
        except (TypeError, ValueError) as exc:
            msg = f"checkpoint results[{idx}] must contain only numeric values."
            raise TypeError(msg) from exc
        if not values:
            msg = f"checkpoint results[{idx}] cannot be empty."
            raise ValueError(msg)
        if not np.all(np.isfinite(values)):
            msg = f"checkpoint results[{idx}] must contain only finite values."
            raise ValueError(msg)
        results.append(values)
    return results


def _as_batch_matrix(batch_json: object, dim: int) -> np.ndarray:
    batch = _as_finite_array(batch_json, "checkpoint batch")
    if batch.ndim == 1 and batch.size == 0:
        return np.zeros((dim, 0), dtype=float)
    if batch.ndim != 2 or batch.shape[0] != dim:
        msg = f"checkpoint batch must have shape ({dim}, n)."
        raise ValueError(msg)
    if batch.shape[1] % 2 == 1:
        msg = "checkpoint batch sample count must be even."
        raise ValueError(msg)
    return batch


def _mirror_indices(sample_count: int) -> np.ndarray:
    mirror_index = np.arange(sample_count)
    half = sample_count // 2
    if half:
        mirror_index[:half] += half
        mirror_index[half:] -= half
    return mirror_index


def _deserialize_pending_context_matches(context_json: object) -> dict[str, int]:
    pending_context_matches = _require_object(context_json, "checkpoint pending_context_matches")
    restored: dict[str, int] = {}
    for context, sample_index in pending_context_matches.items():
        context_name = _require_string(context, "checkpoint pending_context_matches keys")
        restored[context_name] = _coerce_int_like(
            sample_index,
            f"checkpoint pending_context_matches[{context_name!r}]",
        )
    return restored


def _validate_status(status_json: object) -> tuple[int, int, int]:
    status = _require_object(status_json, "checkpoint status")
    num_samples = _coerce_non_negative_int_like(
        _require_field(status, "num_samples"),
        "checkpoint status.num_samples",
    )
    num_batches = _coerce_non_negative_int_like(
        _require_field(status, "num_batches"),
        "checkpoint status.num_batches",
    )
    num_restarts = _coerce_non_negative_int_like(
        _require_field(status, "num_restarts"),
        "checkpoint status.num_restarts",
    )
    return num_samples, num_batches, num_restarts


def _require_field(state: Mapping[str, object], name: str) -> object:
    if name not in state:
        msg = f"checkpoint state is missing {name!r}."
        raise ValueError(msg)
    return state[name]


def _require_object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        msg = f"{name} must be a JSON object."
        raise TypeError(msg)
    return cast(Mapping[str, object], value)


def _as_finite_array(value: object, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must contain only numeric values."
        raise TypeError(msg) from exc
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)
    return array


def _as_finite_vector(value: object, size: int, name: str) -> np.ndarray:
    vector = _as_finite_array(value, name)
    if vector.ndim != 1 or vector.shape[0] != size:
        msg = f"{name} must have shape ({size},)."
        raise ValueError(msg)
    return vector


def _as_finite_matrix(value: object, shape: tuple[int, int], name: str) -> np.ndarray:
    matrix = _as_finite_array(value, name)
    if shape == (0, 0) and matrix.ndim == 1 and matrix.size == 0:
        return np.zeros((0, 0), dtype=float)
    if matrix.ndim != 2 or matrix.shape != shape:
        msg = f"{name} must have shape {shape}."
        raise ValueError(msg)
    return matrix


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        msg = f"{name} must be strings."
        raise TypeError(msg)
    return value


def _coerce_int_like(value: object, name: str) -> int:
    if isinstance(value, bool):
        msg = f"{name} must be an integer."
        raise TypeError(msg)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return int(value)
    msg = f"{name} must be an integer."
    raise TypeError(msg)


def _coerce_non_negative_int_like(value: object, name: str) -> int:
    out = _coerce_int_like(value, name)
    if out < 0:
        msg = f"{name} must be non-negative."
        raise ValueError(msg)
    return out


def _coerce_float_like(value: object, name: str) -> float:
    if value is None or isinstance(value, bool):
        msg = f"{name} must be numeric."
        raise TypeError(msg)
    try:
        return float(cast(str | int | float, value))
    except (TypeError, ValueError) as exc:
        msg = f"{name} must be numeric."
        raise TypeError(msg) from exc
