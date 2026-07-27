from pathlib import Path

import numpy as np

from elc_rl.controller_parameters import (
    PARAMETER_ORDER,
    derive_physics_controller_initials,
    load_physics_controller_parameter_space,
)
from elc_rl.physics_motor_model import load_physics_motor_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _space():
    return load_physics_controller_parameter_space(PROJECT_ROOT)


def test_physics_parameter_order_and_initials_are_model_derived():
    space = _space()
    assert space.names == PARAMETER_ORDER
    assert space.metadata["profile"] == "physics_v1"
    assert np.allclose(
        space.initial,
        derive_physics_controller_initials(PROJECT_ROOT),
        rtol=1e-12,
        atol=0.0,
    )
    assert all(
        spec.source_kind in {
            "mentor_physics_model_derived",
            "approved_dobc_structure",
        }
        for spec in space.specs
    )


def test_parameter_metadata_matches_the_mentor_model():
    config = load_physics_motor_config(PROJECT_ROOT)
    space = _space()
    assert space.metadata["physics_model"]["model_id"] == config.payload["model_id"]
    assert space.metadata["physics_model"]["primary_training_plant"]
    assert not space.metadata["physics_model"]["measured_frf_used_as_training_plant"]
    assert space.metadata["controller_design"]["target_crossover_hz"] == (
        config.target_crossovers_hz
    )
    assert all(
        spec.sample_period_s == config.sample_period_s
        for spec in space.specs
    )


def test_normalized_mapping_round_trip_and_bounded_action():
    space = _space()
    normalized_initial = space.normalize(space.initial)
    assert np.all((-1.0 <= normalized_initial) & (normalized_initial <= 1.0))
    assert np.allclose(space.denormalize(normalized_initial), space.initial)

    grid = np.stack([space.lower, space.initial, space.upper])
    assert np.allclose(space.denormalize(space.normalize(grid)), grid)
    updated = space.apply_action(space.initial, np.ones(11))
    assert np.all(updated >= space.lower)
    assert np.all(updated <= space.upper)


def test_all_eleven_parameters_remain_simulation_only():
    space = _space()
    assert len(space.specs) == 11
    assert all(spec.original_value is None for spec in space.specs)
    assert all(spec.hardware_status.startswith("simulation_only") for spec in space.specs)
    assert space.metadata["safety_policy"]["direct_hardware_use_allowed"] is False
