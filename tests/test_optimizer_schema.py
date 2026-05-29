from __future__ import annotations

from dataclasses import dataclass, make_dataclass
from typing import Annotated, Any, cast

import numpy as np
import pytest
from leitwerk import Optimizer, OptimizerReport, Parameter, SchemaDiff, XNESStatus, parameter

from ._optimizer_helpers import (
    _initialized_optimizer,
    _initialized_state,
    _make_identity_schema,
    _make_mapping_schema,
    _make_schema,
    _optimizer,
    _parameter_spec,
    _read_batch,
    _read_batch_latent_points,
    _read_mean,
    _read_pending_context_matches,
    _read_results,
    _read_scale,
    _read_schema,
    _read_schema_names,
    _read_status,
    _softplus_inverse,
)


def test_schema_state_is_human_readable_parameter_specs() -> None:
    schema = _make_schema(
        "ReadableState",
        alpha=Parameter(mean=1.5, scale=0.25, max=3.0),
        beta=Parameter(mean=2.0, scale=3.0, min=0.0),
        gamma=Parameter(scale=1.0, min=0.0, max=1.0),
    )
    state = _initialized_state(schema, batch_size=4)

    assert _read_schema(state) == {
        "alpha": _parameter_spec(mean=1.5, scale=0.25, max=3.0),
        "beta": _parameter_spec(mean=2.0, scale=3.0, min=0.0),
        "gamma": _parameter_spec(mean=None, scale=1.0, min=0.0, max=1.0),
    }


def test_load_reconciles_added_and_removed_parameters() -> None:
    base_schema = _make_identity_schema("BaseSchema", x=(2.0, 1.5), y=(-1.0, 0.7))
    base = _optimizer(base_schema, batch_size=4)

    for _ in range(5):
        params = base.ask()
        base.tell(-(params.x**2 + 0.5 * params.y**2))

    base_state = base.save()
    base_mean = _read_mean(base_state)
    base_scale = _read_scale(base_state)
    base_batch = _read_batch(base_state)
    base_batch_latent_points = _read_batch_latent_points(base_state)
    base_results = _read_results(base_state)
    assert any(result is not None for result in base_results)

    added_schema = _make_identity_schema("AddedSchema", x=(2.0, 1.5), y=(-1.0, 0.7), z=(3.0, 2.0))
    added = _optimizer(added_schema, batch_size=4)
    load_result = added.load(base_state)
    assert load_result == SchemaDiff(added=["z"], removed=[], changed=[], unchanged=["x", "y"])

    added_state = added.save()
    assert _read_schema_names(added_state) == ["x", "y", "z"]
    assert np.allclose(_read_mean(added_state), np.array([base_mean[0], base_mean[1], 3.0]))
    assert np.allclose(
        _read_scale(added_state),
        np.array(
            [
                [base_scale[0, 0], base_scale[0, 1], 0.0],
                [base_scale[1, 0], base_scale[1, 1], 0.0],
                [0.0, 0.0, 2.0],
            ]
        ),
    )
    assert np.allclose(_read_batch(added_state), np.vstack([base_batch, np.zeros((1, base_batch.shape[1]))]))
    assert np.allclose(
        _read_batch_latent_points(added_state),
        np.vstack([base_batch_latent_points, np.full((1, base_batch_latent_points.shape[1]), 3.0)]),
    )
    assert _read_results(added_state) == base_results

    added.ask()
    added.tell(-1.0)
    added_partial_state = added.save()
    added_mean = _read_mean(added_partial_state)
    added_scale = _read_scale(added_partial_state)
    added_batch = _read_batch(added_partial_state)
    added_batch_latent_points = _read_batch_latent_points(added_partial_state)
    added_results = _read_results(added_partial_state)
    assert any(result is not None for result in added_results)

    removed_schema = _make_identity_schema("RemovedSchema", x=(2.0, 1.5), y=(-1.0, 0.7))
    removed = _optimizer(removed_schema, batch_size=4)
    load_result = removed.load(added_partial_state)
    assert load_result == SchemaDiff(added=[], removed=["z"], changed=[], unchanged=["x", "y"])

    removed_state = removed.save()
    assert _read_schema_names(removed_state) == ["x", "y"]
    assert np.allclose(_read_mean(removed_state), added_mean[:2])
    assert np.allclose(_read_scale(removed_state), added_scale[:2, :2])
    assert np.allclose(_read_batch(removed_state), added_batch[:2, :])
    assert np.allclose(_read_batch_latent_points(removed_state), added_batch_latent_points[:2, :])
    assert _read_results(removed_state) == added_results


def test_load_reconciles_bounds_changes_and_selectively_preserves_batch() -> None:
    base_schema = _make_schema(
        "BaseChangedSchema",
        x=Parameter(mean=0.25, scale=1.0, min=0.0),
        y=Parameter(mean=-1.0, scale=0.7),
    )
    base = _optimizer(base_schema, batch_size=4)

    for _ in range(5):
        params = base.ask()
        base.tell(-((params.x - 0.2) ** 2 + 0.5 * params.y**2))

    base_state = base.save()
    base_mean = _read_mean(base_state)
    base_scale = _read_scale(base_state)
    base_batch = _read_batch(base_state)
    base_results = _read_results(base_state)
    completed_mask = np.array([result is not None for result in base_results], dtype=bool)
    pending_mask = ~completed_mask
    half = base_batch.shape[1] // 2
    mirror_index = np.arange(base_batch.shape[1])
    mirror_index[:half] += half
    mirror_index[half:] -= half
    mirror_pending_mask = pending_mask & pending_mask[mirror_index]
    assert completed_mask.any()
    assert pending_mask.any()

    changed_schema = _make_schema(
        "ChangedSchema",
        x=Parameter(mean=0.25, scale=1.0, min=0.0, max=1.0),
        y=Parameter(mean=-1.0, scale=0.7),
    )
    changed = _optimizer(changed_schema, batch_size=4)
    load_result = changed.load(base_state)
    assert load_result == SchemaDiff(added=[], removed=[], changed=["x"], unchanged=["y"])

    changed_state = changed.save()
    fresh_state = _initialized_state(changed_schema, batch_size=4)
    changed_mean = _read_mean(changed_state)
    changed_scale = _read_scale(changed_state)
    changed_batch = _read_batch(changed_state)

    assert np.allclose(changed_mean, np.array([_read_mean(fresh_state)[0], base_mean[1]]))
    assert np.allclose(
        changed_scale,
        np.array(
            [
                [_read_scale(fresh_state)[0, 0], 0.0],
                [0.0, base_scale[1, 1]],
            ]
        ),
    )
    assert np.allclose(changed_batch[1, :], base_batch[1, :])
    assert np.allclose(changed_batch[0, completed_mask], 0.0)
    assert np.allclose(changed_batch[0, mirror_pending_mask], base_batch[0, mirror_pending_mask])
    assert np.allclose(changed_batch[0, ~mirror_pending_mask], 0.0)
    assert _read_results(changed_state) == base_results
    assert _read_pending_context_matches(changed_state) == _read_pending_context_matches(base_state)


def test_mean_and_scale_changes_do_not_trigger_schema_changeover() -> None:
    base_schema = _make_identity_schema("StrictBase", x=(2.0, 1.5), y=(-1.0, 0.5))
    base_state = _initialized_state(base_schema, batch_size=4)

    changed_schema = _make_schema(
        "StrictChanged",
        x=Parameter(mean=4.0, scale=1.5),
        y=Parameter(mean=-1.0, scale=2.0),
    )
    changed = _optimizer(changed_schema, batch_size=4)
    load_result = changed.load(base_state)
    assert load_result == SchemaDiff(added=[], removed=[], changed=[], unchanged=["x", "y"])

    changed_state = changed.save()
    assert _read_schema(changed_state) == {
        "x": _parameter_spec(mean=4.0, scale=1.5),
        "y": _parameter_spec(mean=-1.0, scale=2.0),
    }
    assert np.allclose(_read_mean(changed_state), _read_mean(base_state))
    assert np.allclose(_read_scale(changed_state), _read_scale(base_state))
    assert np.allclose(_read_batch(changed_state), _read_batch(base_state))
    assert _read_results(changed_state) == _read_results(base_state)


def test_schema_order_follows_dataclass_field_traversal() -> None:
    first_schema = _make_schema(
        "FirstSchema",
        zeta=Parameter(mean=3.0, scale=1.0, min=0.0),
        alpha=Parameter(mean=1.0, scale=2.0),
        mean=Parameter(mean=2.0, scale=3.0, min=-1.0, max=5.0),
    )
    second_schema = _make_schema(
        "SecondSchema",
        mean=Parameter(mean=2.0, scale=3.0, min=-1.0, max=5.0),
        zeta=Parameter(mean=3.0, scale=1.0, min=0.0),
        alpha=Parameter(mean=1.0, scale=2.0),
    )

    state_first = _initialized_optimizer(first_schema, batch_size=18).save()
    state_second = _initialized_optimizer(second_schema, batch_size=18).save()

    first_names = _read_schema_names(state_first)
    second_names = _read_schema_names(state_second)
    assert first_names == ["zeta", "alpha", "mean"]
    assert second_names == ["mean", "zeta", "alpha"]

    second_index = {name: idx for idx, name in enumerate(second_names)}
    permutation = [second_index[name] for name in first_names]
    assert np.allclose(_read_mean(state_first), _read_mean(state_second)[permutation])


def test_nested_schema_flattens_leaf_names_and_rebuilds_dataclasses() -> None:
    combat = make_dataclass(
        "CombatParameters",
        [
            ("retreat_threshold", Annotated[float, Parameter(mean=-1.0, scale=2.0)]),
            ("attack_threshold", Annotated[float, Parameter(mean=2.0, scale=3.0, min=0.0)]),
        ],
        frozen=True,
        slots=True,
    )
    mining = make_dataclass(
        "MiningParameters",
        [("gas_priority", Annotated[float, Parameter(mean=0.5, scale=1.0, min=0.0, max=1.0)])],
        frozen=True,
        slots=True,
    )
    schema = make_dataclass(
        "NestedParameters",
        [
            ("mining", mining),
            ("alpha", Annotated[float, Parameter(mean=1.5, scale=0.25, max=3.0)]),
            ("combat", combat),
        ],
        frozen=True,
        slots=True,
    )

    expected_names = ["mining.gas_priority", "alpha", "combat.retreat_threshold", "combat.attack_threshold"]
    optimizer = _optimizer(schema, batch_size=6)
    assert _read_schema_names(optimizer.save()) == expected_names

    mean = cast(Any, optimizer.mean)
    assert mean.__class__ is schema
    mean = cast(Any, mean)
    mean_combat = cast(Any, mean.combat)
    mean_mining = cast(Any, mean.mining)
    assert mean_combat.__class__ is combat
    assert mean_mining.__class__ is mining
    mean_combat = cast(Any, mean_combat)
    mean_mining = cast(Any, mean_mining)
    assert mean.alpha == 1.5
    assert mean_combat.attack_threshold == 2.0
    assert mean_combat.retreat_threshold == -1.0
    assert mean_mining.gas_priority == 0.5

    scale_marginal = cast(Any, optimizer.scale_marginal)
    assert scale_marginal.__class__ is schema
    scale_marginal = cast(Any, scale_marginal)
    scale_combat = cast(Any, scale_marginal.combat)
    scale_mining = cast(Any, scale_marginal.mining)
    assert scale_combat.__class__ is combat
    assert scale_mining.__class__ is mining
    scale_combat = cast(Any, scale_combat)
    scale_mining = cast(Any, scale_mining)
    assert scale_marginal.alpha == 0.25
    assert scale_combat.attack_threshold == pytest.approx(3.0)
    assert scale_combat.retreat_threshold == 2.0
    assert scale_mining.gas_priority == 1.0

    params = cast(Any, optimizer.ask())
    assert params.__class__ is schema
    params = cast(Any, params)
    params_combat = cast(Any, params.combat)
    params_mining = cast(Any, params.mining)
    assert params_combat.__class__ is combat
    assert params_mining.__class__ is mining
    params_combat = cast(Any, params_combat)
    params_mining = cast(Any, params_mining)
    assert isinstance(params.alpha, float)
    assert isinstance(params_combat.retreat_threshold, float)
    assert isinstance(params_mining.gas_priority, float)
    assert params_combat.attack_threshold > 0.0
    assert 0.0 < params_mining.gas_priority < 1.0
    assert params.alpha < 3.0


def test_dataclass_schema_passes_through_constant_fields_and_zeros_scale() -> None:
    combat = make_dataclass(
        "ConstantCombatParameters",
        [
            ("attack_threshold", float, parameter(mean=2.0, scale=3.0, min=0.0)),
            ("retreat_threshold", float, 1.0),
        ],
        frozen=True,
        slots=True,
    )
    schema = make_dataclass(
        "MixedConstantParameters",
        [
            ("combat", combat),
            ("gas_priority", float, parameter(mean=0.5, scale=0.25, min=0.0, max=1.0)),
            ("label", str, "fixed"),
        ],
        frozen=True,
        slots=True,
    )

    optimizer = _initialized_optimizer(schema, batch_size=6)
    assert _read_schema_names(optimizer.save()) == ["combat.attack_threshold", "gas_priority"]

    mean = cast(Any, optimizer.mean)
    assert mean.__class__ is schema
    mean = cast(Any, mean)
    mean_combat = cast(Any, mean.combat)
    assert mean_combat.__class__ is combat
    mean_combat = cast(Any, mean_combat)
    assert mean_combat.attack_threshold == 2.0
    assert mean_combat.retreat_threshold == 1.0
    assert mean.gas_priority == 0.5
    assert mean.label == "fixed"

    scale_marginal = cast(Any, optimizer.scale_marginal)
    assert scale_marginal.__class__ is schema
    scale_marginal = cast(Any, scale_marginal)
    scale_combat = cast(Any, scale_marginal.combat)
    assert scale_combat.__class__ is combat
    scale_combat = cast(Any, scale_combat)
    assert scale_combat.attack_threshold == pytest.approx(3.0)
    assert scale_combat.retreat_threshold == 0.0
    assert scale_marginal.gas_priority == 0.25
    assert scale_marginal.label == 0.0

    params = cast(Any, optimizer.ask())
    assert params.__class__ is schema
    params = cast(Any, params)
    assert params.combat.retreat_threshold == 1.0
    assert params.label == "fixed"
    assert params.combat.attack_threshold > 0.0
    assert 0.0 < params.gas_priority < 1.0


def test_all_constant_dataclass_schema_is_noop_optimizer() -> None:
    @dataclass(frozen=True, slots=True)
    class ConstantParameters:
        fixed: float = 1.0
        label: str = "hold"

    optimizer = _optimizer(ConstantParameters, batch_size=4)

    assert optimizer.mean == ConstantParameters()
    scale_marginal = optimizer.scale_marginal
    assert scale_marginal.fixed == 0.0
    assert scale_marginal.label == 0.0
    assert optimizer.ask() == ConstantParameters()
    assert optimizer.tell(1.0) == OptimizerReport(False, False, XNESStatus.OK, False)


def test_parameter_helper_builds_dataclass_fields() -> None:
    schema = make_dataclass(
        "HelperParameters",
        [
            ("x", float, parameter(mean=1.0, scale=0.5, min=0.0)),
            ("y", float, parameter(max=2.0)),
        ],
        frozen=True,
        slots=True,
    )

    optimizer = _optimizer(schema, batch_size=4)
    assert _read_schema_names(optimizer.save()) == ["x", "y"]


def test_nested_mapping_schema_flattens_leaf_names_and_rebuilds_plain_dicts() -> None:
    schema = {
        "mining": {
            "gas_priority": Parameter(mean=0.5, scale=1.0, min=0.0, max=1.0),
        },
        "alpha": Parameter(mean=1.5, scale=0.25, max=3.0),
        "combat": {
            "retreat_threshold": Parameter(mean=-1.0, scale=2.0),
            "attack_threshold": Parameter(mean=2.0, scale=3.0, min=0.0),
        },
    }

    expected_names = ["mining.gas_priority", "alpha", "combat.retreat_threshold", "combat.attack_threshold"]
    optimizer = _initialized_optimizer(schema, batch_size=6)
    assert _read_schema_names(optimizer.save()) == expected_names

    mean = optimizer.mean
    assert isinstance(mean, dict)
    assert isinstance(mean["combat"], dict)
    assert isinstance(mean["mining"], dict)
    assert mean["alpha"] == 1.5
    assert mean["combat"]["attack_threshold"] == 2.0
    assert mean["combat"]["retreat_threshold"] == -1.0
    assert mean["mining"]["gas_priority"] == 0.5

    scale_marginal = optimizer.scale_marginal
    assert isinstance(scale_marginal, dict)
    assert isinstance(scale_marginal["combat"], dict)
    assert isinstance(scale_marginal["mining"], dict)
    assert scale_marginal["alpha"] == 0.25
    assert scale_marginal["combat"]["attack_threshold"] == pytest.approx(3.0)
    assert scale_marginal["combat"]["retreat_threshold"] == 2.0
    assert scale_marginal["mining"]["gas_priority"] == 1.0

    params = optimizer.ask()
    assert isinstance(params, dict)
    assert isinstance(params["combat"], dict)
    assert isinstance(params["mining"], dict)
    assert isinstance(params["alpha"], float)
    assert isinstance(params["combat"]["attack_threshold"], float)
    assert isinstance(params["mining"]["gas_priority"], float)
    assert params["combat"]["attack_threshold"] > 0.0
    assert 0.0 < params["mining"]["gas_priority"] < 1.0
    assert params["alpha"] < 3.0


def test_mapping_schema_passes_through_constant_leaf_values_and_zeros_scale() -> None:
    marker: list[str] = ["shared"]
    schema = {
        "x": Parameter(mean=1.0, scale=0.5),
        "fixed": 1.0,
        "nested": {
            "marker": marker,
            "y": Parameter(mean=2.0, scale=3.0),
        },
    }

    optimizer = _initialized_optimizer(schema, batch_size=6)
    assert _read_schema_names(optimizer.save()) == ["x", "nested.y"]

    mean = optimizer.mean
    assert mean["x"] == 1.0
    assert mean["fixed"] == 1.0
    assert cast(dict[str, object], mean["nested"])["marker"] is marker
    assert cast(dict[str, object], mean["nested"])["y"] == 2.0

    scale_marginal = optimizer.scale_marginal
    assert scale_marginal == {"x": 0.5, "fixed": 0.0, "nested": {"marker": 0.0, "y": 3.0}}

    params = optimizer.ask()
    assert params["fixed"] == 1.0
    assert cast(dict[str, object], params["nested"])["marker"] is marker
    assert isinstance(params["x"], float)
    assert isinstance(cast(dict[str, object], params["nested"])["y"], float)


def test_mapping_schema_order_follows_traversal_and_respects_insertion_order() -> None:
    first_schema = _make_mapping_schema(
        zeta=Parameter(mean=3.0, scale=1.0, min=0.0),
        alpha=Parameter(mean=1.0, scale=2.0),
        branch={"mean": Parameter(mean=2.0, scale=3.0, min=-1.0, max=5.0)},
    )
    second_schema = _make_mapping_schema(
        branch={"mean": Parameter(mean=2.0, scale=3.0, min=-1.0, max=5.0)},
        zeta=Parameter(mean=3.0, scale=1.0, min=0.0),
        alpha=Parameter(mean=1.0, scale=2.0),
    )

    state_first = _initialized_optimizer(first_schema, batch_size=18).save()
    state_second = _initialized_optimizer(second_schema, batch_size=18).save()

    first_names = _read_schema_names(state_first)
    second_names = _read_schema_names(state_second)
    assert first_names == ["zeta", "alpha", "branch.mean"]
    assert second_names == ["branch.mean", "zeta", "alpha"]

    second_index = {name: idx for idx, name in enumerate(second_names)}
    permutation = [second_index[name] for name in first_names]
    assert np.allclose(_read_mean(state_first), _read_mean(state_second)[permutation])


def test_mapping_schema_state_save_load_roundtrip() -> None:
    schema_a = {
        "beta": Parameter(mean=-1.0, scale=2.0),
        "block": {
            "alpha": Parameter(mean=2.0, scale=1.5),
        },
    }
    opt_a = _optimizer(schema_a, batch_size=20)

    for _ in range(37):
        params = opt_a.ask()
        block = cast(dict[str, object], params["block"])
        opt_a.tell(-(cast(float, block["alpha"]) ** 2 + cast(float, params["beta"]) ** 2))

    state = opt_a.save()
    assert isinstance(state, dict)

    schema_b = {
        "block": {
            "alpha": Parameter(mean=2.0, scale=1.5),
        },
        "beta": Parameter(mean=-1.0, scale=2.0),
    }
    opt_b = _optimizer(schema_b, batch_size=20)
    load_result = opt_b.load(state)
    assert load_result == SchemaDiff(added=[], removed=[], changed=[], unchanged=["block.alpha", "beta"])

    loaded = opt_b.save()
    state_names = _read_schema_names(state)
    permutation = [state_names.index(name) for name in _read_schema_names(loaded)]
    assert np.allclose(_read_mean(loaded), _read_mean(state)[permutation])
    assert np.allclose(_read_scale(loaded), _read_scale(state)[np.ix_(permutation, permutation)])
    assert _read_results(loaded) == _read_results(state)


def test_all_constant_mapping_schema_is_noop_optimizer() -> None:
    schema = {
        "fixed": 1.0,
        "nested": {
            "label": "hold",
        },
    }
    optimizer = _optimizer(schema, batch_size=4)

    assert optimizer.mean == schema
    assert optimizer.scale_marginal == {"fixed": 0.0, "nested": {"label": 0.0}}
    state = optimizer.save()
    assert _read_schema(state) == {}
    assert _read_mean(state).shape == (0,)
    assert _read_scale(state).shape == (0,)
    assert _read_batch(state).shape == (0,)
    assert _read_results(state) == []
    assert _read_status(state)["num_parameters"] == 0
    assert _read_status(state)["batch_size"] == 0

    with pytest.raises(RuntimeError, match="No pending ask"):
        optimizer.tell(1.0)

    assert optimizer.ask(context={"case": 1}) == schema
    with pytest.raises(RuntimeError, match="Pending ask"):
        optimizer.ask()

    assert optimizer.tell(1.0) == OptimizerReport(False, False, XNESStatus.OK, False)
    progressed_state = optimizer.save()
    progressed_status = _read_status(progressed_state)
    assert progressed_status["num_samples"] == 1
    assert progressed_status["num_batches"] == 0
    assert progressed_status["num_restarts"] == 0
    assert progressed_status["batch_progress"] == 0.0
    assert progressed_status["batch_size"] == 0

    restored = _optimizer(schema, batch_size=4)
    assert restored.load(progressed_state) == SchemaDiff(added=[], removed=[], changed=[], unchanged=[])
    assert restored.ask() == schema


def test_load_treats_parameters_changed_to_constants_as_removed() -> None:
    base_schema = _make_identity_schema("LearnedToConstantBase", x=(2.0, 1.5), y=(-1.0, 0.7))
    base_state = _initialized_state(base_schema, batch_size=4)

    schema = {
        "x": 10.0,
        "y": Parameter(mean=-1.0, scale=0.7),
    }
    optimizer = _optimizer(schema, batch_size=4)
    assert optimizer.load(base_state) == SchemaDiff(added=[], removed=["x"], changed=[], unchanged=["y"])

    state = optimizer.save()
    assert _read_schema_names(state) == ["y"]
    assert optimizer.mean["x"] == 10.0
    assert optimizer.scale_marginal["x"] == 0.0


def test_mapping_schema_rejects_leaf_name_shadowing() -> None:
    with pytest.raises(ValueError, match=r"shadowed by another key"):
        Optimizer({"a.b": Parameter(), "a": {"b": Parameter()}})


def test_mapping_schema_rejects_branch_name_shadowing() -> None:
    with pytest.raises(ValueError, match=r"shadowed by another key"):
        Optimizer({"a.b": {"c": Parameter()}, "a": {"b": {"d": Parameter()}}})


def test_schema_requires_parameter_annotated_float_fields() -> None:
    bad_schema = make_dataclass("BadSchema", [("x", float)], frozen=True, slots=True)

    with pytest.raises(TypeError, match=r"float = parameter\(\.\.\.\)|Annotated\[float, Parameter"):
        Optimizer(bad_schema)


def test_schema_requires_float_fields_for_parameter_helper() -> None:
    bad_schema = make_dataclass("BadHelperSchema", [("x", int, parameter())], frozen=True, slots=True)

    with pytest.raises(TypeError, match=r"annotated as float when using parameter\(\.\.\.\)"):
        Optimizer(bad_schema)


@pytest.mark.parametrize(
    ("parameter", "message"),
    [
        (Parameter(mean=0.0, scale=0.0), r"scale must be a positive finite float"),
        (Parameter(mean=1.0, min=1.0, max=1.0), r"min < max"),
        (Parameter(mean=2.0, min=1.0, max=2.0), r"min < mean < max"),
        (Parameter(mean=1.0, min=1.0), r"mean > min"),
        (Parameter(mean=1.0, max=1.0), r"mean < max"),
    ],
)
def test_schema_rejects_invalid_parameter_domains(parameter: Parameter, message: str) -> None:
    schema = _make_schema("InvalidSchema", x=parameter)

    with pytest.raises(ValueError, match=message):
        Optimizer(schema)


def test_omitted_mean_uses_canonical_latent_zero_center() -> None:
    schema = _make_schema(
        "CanonicalMeans",
        unbounded=Parameter(),
        lower=Parameter(min=2.0),
        upper=Parameter(max=5.0),
        interval=Parameter(min=2.0, max=6.0),
    )
    optimizer = _initialized_optimizer(schema, batch_size=6)

    state = optimizer.save()
    assert _read_schema(state) == {
        "interval": _parameter_spec(mean=None, scale=1.0, min=2.0, max=6.0),
        "lower": _parameter_spec(mean=None, scale=1.0, min=2.0),
        "unbounded": _parameter_spec(mean=None, scale=1.0),
        "upper": _parameter_spec(mean=None, scale=1.0, max=5.0),
    }
    assert np.allclose(_read_mean(state), np.zeros(4))

    mean = optimizer.mean
    assert mean.interval == pytest.approx(4.0)
    assert mean.lower == pytest.approx(2.0 + np.log(2.0))
    assert mean.unbounded == pytest.approx(0.0)
    assert mean.upper == pytest.approx(5.0 - np.log(2.0))


def test_legacy_parameter_constructors_are_not_exposed() -> None:
    assert not hasattr(Parameter, "unbounded")
    assert not hasattr(Parameter, "between")
    assert not hasattr(Parameter, "above")
    assert not hasattr(Parameter, "below")
    assert not hasattr(Parameter, "above_exponential")
    assert not hasattr(Parameter, "below_exponential")


def test_ask_returns_typed_sample() -> None:
    schema = _make_identity_schema("TypedSample", b=(2.0, 1.0), a=(-1.0, 1.0))
    params = _initialized_optimizer(schema, batch_size=6).ask()

    assert params.__class__ is schema
    assert isinstance(params.a, float)
    assert isinstance(params.b, float)


def test_mean_returns_current_user_space_means_without_context() -> None:
    schema = _make_schema(
        "BestParams",
        b=Parameter(scale=1.0, min=0.0, max=1.0),
        a=Parameter(scale=1.0, min=-2.0),
    )
    optimizer = _initialized_optimizer(schema, batch_size=6)

    mean = optimizer.mean
    assert mean.__class__ is schema
    assert mean.a == pytest.approx(-2.0 + np.log(2.0))
    assert mean.b == pytest.approx(0.5)


def test_transformed_parameters_use_latent_state_but_emit_user_space_values() -> None:
    schema = _make_schema(
        "TransformedParams",
        a=Parameter(mean=1.2, scale=3.4),
        b=Parameter(mean=2.9, scale=1.0, min=2.3, max=3.4),
        c=Parameter(mean=3.0, scale=1.0, min=0.0),
        d=Parameter(mean=3.0, scale=1.0, max=4.0),
    )
    optimizer = _optimizer(schema, batch_size=6)

    state = optimizer.save()
    expected_latent_mean = np.array(
        [
            1.2,
            np.log((2.9 - 2.3) / (3.4 - 2.9)),
            _softplus_inverse(3.0),
            -_softplus_inverse(1.0),
        ]
    )
    assert np.allclose(_read_mean(state), expected_latent_mean)
    assert np.allclose(np.diag(_read_scale(state)), np.array([3.4, 1.0, 1.0, 1.0]))

    mean = optimizer.mean
    assert mean.a == pytest.approx(1.2)
    assert mean.b == pytest.approx(2.9)
    assert mean.c == pytest.approx(3.0)
    assert mean.d == pytest.approx(3.0)

    for _ in range(6):
        params = optimizer.ask()
        assert 2.3 < params.b < 3.4
        assert params.c > 0.0
        assert params.d < 4.0
        optimizer.tell(0.0)

    reloaded = _optimizer(schema, batch_size=6)
    reloaded.load(state)
    reloaded_mean = reloaded.mean
    assert reloaded_mean.a == pytest.approx(mean.a)
    assert reloaded_mean.b == pytest.approx(mean.b)
    assert reloaded_mean.c == pytest.approx(mean.c)
    assert reloaded_mean.d == pytest.approx(mean.d)
