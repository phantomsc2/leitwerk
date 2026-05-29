"""Mapping schema parsing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

from .parameter import Parameter
from .parser import build_constant_builder, build_field_spec, build_scalar_builder, build_zero_builder, path_name
from .spec import BuildFn, FieldSpec, SchemaPath, SchemaSpec

ParsedEntry = tuple[str, tuple[FieldSpec, ...], BuildFn, BuildFn]
ParsedNode = tuple[tuple[FieldSpec, ...], BuildFn, BuildFn]


def parse_mapping_schema(model: Mapping[str, object]) -> SchemaSpec[dict[str, Any]]:
    """Parse and validate a nested mapping schema."""
    field_specs, build_root, build_scale_root = _parse_mapping_node(model, (), set())
    return SchemaSpec(
        fields=field_specs,
        instantiate=cast(Callable[[Mapping[SchemaPath, float]], dict[str, Any]], build_root),
        instantiate_scale=cast(Callable[[Mapping[SchemaPath, float]], dict[str, Any]], build_scale_root),
    )


def _parse_mapping_node(
    model: Mapping[str, object],
    prefix: SchemaPath,
    seen_names: set[str],
) -> ParsedNode:
    parsed_fields = tuple(_parse_mapping_entry(key, value, prefix, seen_names) for key, value in model.items())
    field_specs = tuple(field_spec for _, child_specs, _, _ in parsed_fields for field_spec in child_specs)
    child_builders = tuple((name, build) for name, _, build, _ in parsed_fields)
    child_scale_builders = tuple((name, build) for name, _, _, build in parsed_fields)

    def build_node(values: Mapping[SchemaPath, float]) -> dict[str, Any]:
        return {name: build(values) for name, build in child_builders}

    def build_scale_node(values: Mapping[SchemaPath, float]) -> dict[str, Any]:
        return {name: build(values) for name, build in child_scale_builders}

    return field_specs, build_node, build_scale_node


def _parse_mapping_entry(
    key: object,
    value: object,
    prefix: SchemaPath,
    seen_names: set[str],
) -> ParsedEntry:
    if not isinstance(key, str):
        msg = "leitwerk schema mapping keys must be strings."
        raise TypeError(msg)

    path = prefix + (key,)
    _register_name(path, seen_names)
    if isinstance(value, Parameter):
        field_spec = build_field_spec(path, value)
        build = build_scalar_builder(path)
        return key, (field_spec,), build, build

    if isinstance(value, Mapping):
        child_field_specs, build_node, build_scale_node = _parse_mapping_node(
            cast(Mapping[str, object], value),
            path,
            seen_names,
        )
        return key, child_field_specs, build_node, build_scale_node

    return key, (), build_constant_builder(value), build_zero_builder()


def _register_name(path: SchemaPath, seen_names: set[str]) -> None:
    name = path_name(path)
    if name in seen_names:
        msg = f"leitwerk schema mapping path '{name}' is shadowed by another key."
        raise ValueError(msg)
    seen_names.add(name)
