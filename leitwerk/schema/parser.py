"""Schema parser dispatch and shared parser helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import is_dataclass
from typing import Any, overload

from .parameter import Parameter
from .spec import BuildFn, FieldSpec, SchemaPath, SchemaSpec, T


@overload
def parse_schema(model: type[T]) -> SchemaSpec[T]: ...


@overload
def parse_schema(model: Mapping[str, object]) -> SchemaSpec[dict[str, Any]]: ...


def parse_schema(model: object) -> SchemaSpec[Any]:
    """Parse a supported schema input into the normalized internal form."""
    if isinstance(model, type) and is_dataclass(model):
        from .dataclass import parse_dataclass_schema

        return parse_dataclass_schema(model)

    if isinstance(model, Mapping):
        from .mapping import parse_mapping_schema

        return parse_mapping_schema(model)

    msg = "leitwerk schema must be a dataclass type or mapping."
    raise TypeError(msg)


def build_field_spec(path: SchemaPath, parameter: Parameter) -> FieldSpec:
    name = path_name(path)
    mean0, scale0 = parameter.initial_state(name)
    return FieldSpec(name=name, path=path, parameter=parameter, mean0=mean0, scale0=scale0)


def build_scalar_builder(path: SchemaPath) -> BuildFn:
    def build_leaf(values: Mapping[SchemaPath, float]) -> float:
        return float(values[path])

    return build_leaf


def build_constant_builder(value: object) -> BuildFn:
    def build_leaf(values: Mapping[SchemaPath, float]) -> object:
        return value

    return build_leaf


def build_zero_builder() -> BuildFn:
    def build_leaf(values: Mapping[SchemaPath, float]) -> float:
        return 0.0

    return build_leaf


def field_name(field_spec: FieldSpec) -> str:
    return field_spec.name


def path_name(path: SchemaPath) -> str:
    return ".".join(path)
