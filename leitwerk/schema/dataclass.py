"""Dataclass schema parsing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import MISSING, Field, fields, is_dataclass
from typing import Annotated, Any, cast, get_args, get_origin, get_type_hints

from .parameter import PARAMETER_METADATA_KEY, Parameter
from .parser import build_constant_builder, build_field_spec, build_scalar_builder, build_zero_builder, path_name
from .spec import BuildFn, FieldSpec, SchemaPath, SchemaSpec, T

ParsedField = tuple[tuple[FieldSpec, ...], BuildFn, BuildFn]


def parse_dataclass_schema(model_type: type[T]) -> SchemaSpec[T]:
    """Parse and validate a dataclass schema tree."""
    if not isinstance(model_type, type) or not is_dataclass(model_type):
        msg = "leitwerk schema must be a dataclass type."
        raise TypeError(msg)

    field_specs, instantiate, instantiate_scale = _parse_dataclass_type(model_type, ())
    return SchemaSpec(
        fields=field_specs,
        instantiate=cast(Callable[[Mapping[SchemaPath, float]], T], instantiate),
        instantiate_scale=cast(Callable[[Mapping[SchemaPath, float]], T], instantiate_scale),
    )


def _parse_dataclass_type(model_type: type[Any], prefix: SchemaPath) -> ParsedField:
    dataclass_fields = tuple(fields(model_type))
    type_hints = get_type_hints(model_type, include_extras=True)
    parsed_fields = tuple(
        _parse_dataclass_field(
            type_hints.get(field.name),
            field,
            prefix + (field.name,),
        )
        for field in dataclass_fields
    )
    field_specs = tuple(field_spec for child_specs, _, _ in parsed_fields for field_spec in child_specs)
    constructor = cast(Callable[..., Any], model_type)
    child_builders = tuple(
        (field.name, build) for field, (_, build, _) in zip(dataclass_fields, parsed_fields, strict=True)
    )
    child_scale_builders = tuple(
        (field.name, build) for field, (_, _, build) in zip(dataclass_fields, parsed_fields, strict=True)
    )

    def instantiate(values: Mapping[SchemaPath, float]) -> Any:
        kwargs = {name: build(values) for name, build in child_builders}
        return constructor(**kwargs)

    def instantiate_scale(values: Mapping[SchemaPath, float]) -> Any:
        kwargs = {name: build(values) for name, build in child_scale_builders}
        return constructor(**kwargs)

    return field_specs, instantiate, instantiate_scale


def _parse_dataclass_field(
    annotation: Any,
    dataclass_field: Field[Any],
    path: SchemaPath,
) -> ParsedField:
    name = path_name(path)
    if not dataclass_field.init:
        msg = f"leitwerk schema field '{name}' must be init=True"
        raise TypeError(msg)

    runtime_annotation = _runtime_annotation(annotation)
    annotation_parameter = _annotation_parameter(annotation, path)
    if annotation_parameter is not None:
        return _parse_annotated_parameter_field(runtime_annotation, annotation_parameter, path)

    field_parameter = _field_parameter(dataclass_field)
    if field_parameter is not None:
        return _parse_parameter_field(runtime_annotation, field_parameter, path)

    if isinstance(runtime_annotation, type) and is_dataclass(runtime_annotation):
        return _parse_dataclass_type(runtime_annotation, path)

    constant_build = _constant_builder(dataclass_field)
    if constant_build is not None:
        return (), constant_build, build_zero_builder()

    msg = (
        f"leitwerk schema field '{name}' must be declared as float = parameter(...), "
        "annotated as Annotated[float, Parameter(...)], have a default value, or be a dataclass type"
    )
    raise TypeError(msg)


def _field_parameter(parameter_field: Field[Any]) -> Parameter | None:
    value = parameter_field.metadata.get(PARAMETER_METADATA_KEY)
    if isinstance(value, Parameter):
        return value
    return None


def _runtime_annotation(annotation: Any) -> Any:
    if get_origin(annotation) is Annotated:
        return get_args(annotation)[0]
    return annotation


def _annotation_parameter(annotation: Any, path: SchemaPath) -> Parameter | None:
    name = path_name(path)
    if get_origin(annotation) is not Annotated:
        return None

    _, *metadata = get_args(annotation)
    parameters = [item for item in metadata if isinstance(item, Parameter)]
    if len(parameters) > 1:
        msg = f"leitwerk schema field '{name}' must include exactly one Parameter(...) metadata value"
        raise TypeError(msg)
    return parameters[0] if parameters else None


def _parse_annotated_parameter_field(
    annotation: Any,
    field_parameter: Parameter,
    path: SchemaPath,
) -> ParsedField:
    name = path_name(path)
    if annotation is not float:
        msg = f"leitwerk schema field '{name}' must be annotated as Annotated[float, Parameter(...)]"
        raise TypeError(msg)

    field_spec = build_field_spec(path, field_parameter)
    build = build_scalar_builder(path)
    return (field_spec,), build, build


def _parse_parameter_field(
    annotation: Any,
    field_parameter: Parameter,
    path: SchemaPath,
) -> ParsedField:
    name = path_name(path)
    if annotation is not float:
        msg = f"leitwerk schema field '{name}' must be annotated as float when using parameter(...)"
        raise TypeError(msg)

    field_spec = build_field_spec(path, field_parameter)
    build = build_scalar_builder(path)
    return (field_spec,), build, build


def _constant_builder(dataclass_field: Field[Any]) -> BuildFn | None:
    if dataclass_field.default is not MISSING:
        return build_constant_builder(dataclass_field.default)
    if dataclass_field.default_factory is MISSING:
        return None

    factory = cast(Callable[[], object], dataclass_field.default_factory)

    def build_default(values: Mapping[SchemaPath, float]) -> object:
        return factory()

    return build_default
