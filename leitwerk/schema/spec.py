"""Normalized schema model shared across schema input modes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np

from .parameter import Parameter

T = TypeVar("T")
SchemaPath = tuple[str, ...]
BuildFn = Callable[[Mapping[SchemaPath, float]], Any]


@dataclass(frozen=True, slots=True)
class SchemaDiff:
    """Difference between persisted and current schema definitions.

    Attributes:
        added: Current schema leaf names absent from the loaded state.
        removed: Loaded schema leaf names absent from the current schema.
        changed: Shared leaf names whose persisted transform compatibility differs.
        unchanged: Shared leaf names whose persisted transform compatibility matches.
    """

    added: list[str]
    removed: list[str]
    changed: list[str]
    unchanged: list[str]


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Internal normalized description of one optimized schema leaf."""

    name: str
    path: SchemaPath
    parameter: Parameter
    mean0: float
    scale0: float


@dataclass(frozen=True, slots=True)
class SchemaSpec(Generic[T]):
    """Internal parsed schema-tree description used by `Optimizer`."""

    fields: tuple[FieldSpec, ...]
    instantiate: Callable[[Mapping[SchemaPath, float]], T]
    instantiate_scale: Callable[[Mapping[SchemaPath, float]], T]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(field_spec.name for field_spec in self.fields)

    @property
    def dim(self) -> int:
        return len(self.fields)

    def schema_state(self) -> dict[str, dict[str, object]]:
        return {field_spec.name: field_spec.parameter.state_spec() for field_spec in self.fields}

    def diff(self, saved_schema: Mapping[str, Parameter]) -> SchemaDiff:
        current_schema = {field_spec.name: field_spec.parameter for field_spec in self.fields}
        current_names = self.names
        saved_names = list(saved_schema)
        added = [name for name in current_names if name not in saved_schema]
        removed = [name for name in saved_names if name not in current_schema]
        changed = [
            name
            for name in current_names
            if name in saved_schema
            and saved_schema[name].reconciliation_key() != current_schema[name].reconciliation_key()
        ]
        unchanged = [
            name
            for name in current_names
            if name in saved_schema
            and saved_schema[name].reconciliation_key() == current_schema[name].reconciliation_key()
        ]
        return SchemaDiff(added=added, removed=removed, changed=changed, unchanged=unchanged)

    def index_by_name(self) -> dict[str, int]:
        return {field_spec.name: idx for idx, field_spec in enumerate(self.fields)}

    def initial_distribution(self) -> tuple[np.ndarray, np.ndarray]:
        mean = np.array([field_spec.mean0 for field_spec in self.fields], dtype=float)
        scale_diag = np.array([field_spec.scale0 for field_spec in self.fields], dtype=float)
        return mean, np.diag(scale_diag)

    def build_params(self, values: np.ndarray) -> T:
        leaf_values = {
            field_spec.path: float(field_spec.parameter.to_user_space(float(value)))
            for field_spec, value in zip(self.fields, values, strict=True)
        }
        return self.instantiate(leaf_values)

    def build_scales(self, values: np.ndarray) -> T:
        leaf_values = {field_spec.path: float(value) for field_spec, value in zip(self.fields, values, strict=True)}
        return self.instantiate_scale(leaf_values)
