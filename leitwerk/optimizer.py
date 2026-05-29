"""Schema-based optimizer wrapper built on top of the xNES update rule."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

import numpy as np

from .schema import SchemaDiff
from .schema.parser import parse_schema
from .schema.spec import SchemaSpec
from .state import JSONLike, JSONObject, restore_optimizer_state, serialize_optimizer_state
from .xnes import XNES, XNESStatus

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class OptimizerReport:
    """Outcome of one `Optimizer.tell` call."""

    completed_batch: bool
    matched_context: bool
    status: XNESStatus
    restarted: bool


@dataclass(frozen=True, slots=True)
class SampleReservation:
    sample_index: int
    context: str | None
    matched_context: bool


@dataclass(slots=True)
class _BatchState:
    batch: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))
    results: list[tuple[float, ...] | None] = field(default_factory=list)
    pending_context_matches: dict[str, int] = field(default_factory=dict)

    def reset(self, batch: np.ndarray) -> None:
        self.batch = batch
        self.results = [None] * self.batch.shape[1]
        self.pending_context_matches = {}

    def restore(
        self,
        batch: np.ndarray,
        results: list[tuple[float, ...] | None],
        pending_context_matches: dict[str, int],
    ) -> None:
        self.batch = batch
        sample_count = self.batch.shape[1]
        self.results = [None] * sample_count
        for idx, item in enumerate(results[:sample_count]):
            self.results[idx] = item
        self.pending_context_matches = {
            context: sample_index
            for context, sample_index in pending_context_matches.items()
            if 0 <= sample_index < sample_count
            and self.results[sample_index] is not None
            and self.results[self._mirror_index(sample_index)] is None
        }

    def reserve(self, context: str | None) -> SampleReservation | None:
        sample_index, matched_context = self._pick_sample(context)
        if sample_index is None:
            return None
        return SampleReservation(
            sample_index=sample_index,
            context=context,
            matched_context=matched_context,
        )

    def record_result(self, reservation: SampleReservation, result: tuple[float, ...]) -> bool:
        sample_index = reservation.sample_index
        if not 0 <= sample_index < len(self.results):
            msg = "Sample index out of range."
            raise RuntimeError(msg)
        if self.results[sample_index] is not None:
            msg = "Sample already has a recorded result."
            raise RuntimeError(msg)

        self.results[sample_index] = result
        self._update_pending_context_matches(sample_index, reservation.context)
        return self.is_complete()

    def is_complete(self) -> bool:
        return all(item is not None for item in self.results)

    def _pick_sample(self, context: str | None) -> tuple[int | None, bool]:
        if context is not None:
            pending_index = self.pending_context_matches.get(context)
            if pending_index is not None:
                mirror_index = self._mirror_index(pending_index)
                if self.results[mirror_index] is None:
                    return mirror_index, True

        sample_index = next((idx for idx, result in enumerate(self.results) if result is None), None)
        return sample_index, False

    def _mirror_index(self, sample_index: int) -> int:
        half = self.batch.shape[1] // 2
        return sample_index + half if sample_index < half else sample_index - half

    def _update_pending_context_matches(self, sample_index: int, context: str | None) -> None:
        mirror_index = self._mirror_index(sample_index)
        stale_contexts = [
            key for key, pending_index in self.pending_context_matches.items() if pending_index == mirror_index
        ]
        for key in stale_contexts:
            del self.pending_context_matches[key]

        if context is None:
            return

        if self.results[mirror_index] is None:
            self.pending_context_matches[context] = sample_index


class Optimizer(Generic[T]):
    """Schema-based xNES optimizer over dataclass or mapping schemas."""

    def __init__(
        self,
        schema: type[T] | Mapping[str, object],
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> None:
        self._batch_size = batch_size
        self._seed = seed
        self._schema: SchemaSpec[T] = cast(SchemaSpec[T], parse_schema(schema))
        self._batch_state = _BatchState()
        self._pending_reservation: SampleReservation | None = None
        self._xnes: XNES
        self._num_samples = 0
        self._num_batches = 0
        self._num_restarts = 0
        self._reset_distribution()

    def save(self) -> JSONObject:
        """Serialize the current optimizer state into a JSON-compatible mapping."""
        return serialize_optimizer_state(
            status=self._status(),
            mean=self._xnes.mean,
            scale=self._xnes.scale,
            schema_state=self._schema.schema_state(),
            batch=self._batch_state.batch,
            results=self._batch_state.results,
            pending_context_matches=self._batch_state.pending_context_matches,
        )

    def load(self, state: JSONObject) -> SchemaDiff:
        """Restore optimizer state from a previous snapshot."""
        self._pending_reservation = None
        restored = restore_optimizer_state(state, cast(SchemaSpec[object], self._schema))
        self._xnes = XNES(restored.mean, restored.scale)
        self._num_samples = restored.num_samples
        self._num_batches = restored.num_batches
        self._num_restarts = restored.num_restarts

        if self._schema.dim == 0:
            self._batch_state.reset(np.zeros((0, 0), dtype=float))
        elif restored.batch.shape[1] == 0:
            self._sample_batch()
        else:
            self._batch_state.restore(
                restored.batch,
                restored.results,
                restored.pending_context_matches,
            )
        return restored.schema_diff

    def _reset_distribution(self, mean: np.ndarray | None = None) -> None:
        reset_mean, scale = self._schema.initial_distribution()
        if mean is not None:
            reset_mean = np.array(mean, dtype=float, copy=True)
        self._xnes = XNES(reset_mean, scale)
        if self._schema.dim == 0:
            self._batch_state.reset(np.zeros((0, 0), dtype=float))
            self._pending_reservation = None
        else:
            self._sample_batch()

    def ask(self, context: JSONLike = None) -> T:
        """Reserve one sampled parameter set for one evaluation."""
        return self.decode(self.ask_latent(context))

    def ask_latent(self, context: JSONLike = None) -> np.ndarray:
        """Reserve one latent sampled vector for one evaluation."""
        self._require_idle("ask")
        if self._schema.dim == 0:
            self._pending_reservation = SampleReservation(
                sample_index=0,
                context=_normalize_context(context),
                matched_context=False,
            )
            return np.array(self._xnes.mean, dtype=float, copy=True)

        self._pending_reservation = self._reserve(_normalize_context(context))
        return self._latent_for(self._pending_reservation)

    def decode(self, values: np.ndarray) -> T:
        """Decode one latent vector into the schema's runtime shape."""
        return self._schema.build_params(np.asarray(values, dtype=float))

    @property
    def mean(self) -> T:
        """Current mean parameters in the schema's runtime shape.

        For transformed parameters this is the transformed latent mean, used as
        a convenient center rather than the exact expected value.
        """
        return self.decode(self._xnes.mean)

    @property
    def mean_latent(self) -> np.ndarray:
        """Current latent mean vector."""
        return np.array(self._xnes.mean, dtype=float, copy=True)

    @property
    def scale_marginal(self) -> T:
        """Current scale-vector parameters in the schema's runtime shape."""
        return self._schema.build_scales(self._xnes.scale_marginal)

    @property
    def batch_size(self) -> int | None:
        """Configured sample count for the next freshly drawn batch."""
        return self._batch_size

    @property
    def seed(self) -> int | None:
        """Configured root seed used for future batch sampling."""
        return self._seed

    def tell(self, result: float | Sequence[float] | np.ndarray) -> OptimizerReport:
        """Submit the objective result for the pending sample."""
        reservation = self._require_pending()
        if self._schema.dim == 0:
            _normalize_result(result)
            self._num_samples += 1
            self._pending_reservation = None
            return OptimizerReport(False, reservation.matched_context, XNESStatus.OK, False)

        report = self._tell_reservation(reservation, result)
        self._pending_reservation = None
        return report

    def _tell_reservation(
        self,
        reservation: SampleReservation,
        result: float | Sequence[float] | np.ndarray,
    ) -> OptimizerReport:
        completed_batch = self._batch_state.record_result(reservation, _normalize_result(result))
        self._num_samples += 1
        if completed_batch:
            status, restarted = self._complete_batch()
            return OptimizerReport(True, reservation.matched_context, status, restarted)
        return OptimizerReport(False, reservation.matched_context, XNESStatus.OK, False)

    def _latent_for(self, reservation: SampleReservation) -> np.ndarray:
        return self._xnes.transform(self._batch_state.batch[:, [reservation.sample_index]])[:, 0]

    def _sample_batch(self) -> None:
        batch = self._xnes.sample(self._batch_size, self._batch_rng())
        self._pending_reservation = None
        self._batch_state.reset(batch)

    def _batch_rng(self) -> np.random.Generator:
        if self._seed is None:
            return np.random.Generator(np.random.PCG64())
        batch_seed = np.random.SeedSequence(int(self._seed), spawn_key=(self._num_batches,))
        return np.random.Generator(np.random.PCG64(batch_seed))

    def _complete_batch(self) -> tuple[XNESStatus, bool]:
        self._num_batches += 1
        status = self._xnes.update(self._batch_state.batch, self._ranking())
        restarted = status.is_terminal
        if restarted:
            self._num_restarts += 1
            restart_mean = self._xnes.mean if status.is_completion else None
            self._reset_distribution(restart_mean)
        else:
            self._sample_batch()
        return status, restarted

    def _status(self) -> JSONObject:
        batch_size = len(self._batch_state.results)
        completed = sum(result is not None for result in self._batch_state.results)
        return {
            "num_samples": self._num_samples,
            "num_batches": self._num_batches,
            "num_restarts": self._num_restarts,
            "num_parameters": self._xnes.dim,
            "axis_ratio": self._xnes.axis_ratio,
            "scale_global": self._xnes.scale_global,
            "batch_progress": completed,
            "batch_size": batch_size,
        }

    def _reserve(self, context: str | None) -> SampleReservation:
        reservation = self._batch_state.reserve(context)
        if reservation is None:
            if not self._batch_state.is_complete():
                msg = "No sample available."
                raise RuntimeError(msg)
            self._complete_batch()
            reservation = self._batch_state.reserve(context)
            if reservation is None:
                msg = "No sample available."
                raise RuntimeError(msg)
        return reservation

    def _ranking(self) -> list[int]:
        assert self._batch_state.is_complete()
        results = [cast(tuple[float, ...], item) for item in self._batch_state.results]
        return sorted(range(len(results)), key=lambda idx: results[idx], reverse=True)

    def _require_idle(self, action: str) -> None:
        if self._pending_reservation is not None:
            msg = f"Pending ask must be told before calling {action}()."
            raise RuntimeError(msg)

    def _require_pending(self) -> SampleReservation:
        if self._pending_reservation is None:
            msg = "No pending ask."
            raise RuntimeError(msg)
        return self._pending_reservation


def _normalize_result(result: float | Sequence[float] | np.ndarray) -> tuple[float, ...]:
    if isinstance(result, (int, float)):
        return (float(result),)
    if isinstance(result, np.ndarray):
        values = tuple(float(item) for item in result.reshape(-1))
        if not values:
            msg = "Sequence result cannot be empty."
            raise ValueError(msg)
        return values
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        values = tuple(float(item) for item in result)
        if not values:
            msg = "Sequence result cannot be empty."
            raise ValueError(msg)
        return values
    msg = f"Unsupported result type: {type(result)!r}"
    raise TypeError(msg)


def _normalize_context(context: JSONLike) -> str | None:
    if context is None or isinstance(context, str):
        return context
    try:
        return json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        msg = "context must be a string or JSON-serializable value."
        raise TypeError(msg) from exc
