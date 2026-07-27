from pathlib import Path

import numpy as np

from elc_rl.controller_parameters import load_physics_controller_parameter_space
from elc_rl.physics_motor_model import (
    MODEL_PARAMETER_NAMES,
    load_physics_motor_config,
    load_physics_motor_ensemble,
    simulate_scenario,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_mentor_model_values_and_units_are_preserved():
    config = load_physics_motor_config(PROJECT_ROOT)
    motor = config.nominal
    assert motor.inductance_h == 0.002707
    assert motor.resistance_ohm == 4.993
    assert motor.inertia_kg_m2 == 0.00221
    assert motor.torque_constant_nm_per_a == 0.1633
    assert config.sample_period_s == 200e-6
    assert np.isclose(motor.electrical_time_constant_s, 0.002707 / 4.993)
    assert np.isclose(motor.mechanical_time_constant_s, 0.00221 / 0.0237)


def test_physics_ensemble_is_coherent_and_inside_declared_uncertainty():
    config = load_physics_motor_config(PROJECT_ROOT)
    ensemble = load_physics_motor_ensemble(PROJECT_ROOT)
    assert ensemble["parameters"].shape == (56, len(MODEL_PARAMETER_NAMES))
    assert np.count_nonzero(ensemble["role"] == "train") == 40
    assert np.count_nonzero(ensemble["role"] == "validation") == 16
    nominal_index = int(np.flatnonzero(ensemble["is_nominal"] == 1)[0])
    assert np.array_equal(ensemble["parameters"][nominal_index], config.nominal.as_array())
    relative = np.abs(ensemble["parameters"] / config.nominal.as_array() - 1.0)
    limits = np.asarray(
        [config.uncertainty_fraction[name] for name in MODEL_PARAMETER_NAMES]
    )
    assert np.all(relative <= limits[None, :] + 1e-12)


def test_all_11_physics_initials_are_active_and_match_design():
    space = load_physics_controller_parameter_space(PROJECT_ROOT)
    expected = np.asarray(
        [
            62.13670362077854,
            780.8328464532967,
            0.04944681764341488,
            3.4056021000320973,
            36.52161528088719,
            0.0006775230105303097,
            1.0,
            0.002,
            7.614564064852501,
            14044.89042327615,
            0.0001514869388013989,
        ]
    )
    assert np.allclose(space.initial, expected, rtol=1e-12, atol=0.0)
    assert np.all(space.initial > 0.0)
    assert all(spec.sample_period_s == 200e-6 for spec in space.specs)
    assert space.metadata["physics_model"]["primary_training_plant"]


def test_nominal_discrete_scenarios_respect_simulation_envelope():
    config = load_physics_motor_config(PROJECT_ROOT)
    space = load_physics_controller_parameter_space(PROJECT_ROOT)
    for scenario in ("current", "speed", "position", "disturbance"):
        trace = simulate_scenario(config, config.nominal, space.initial, scenario)
        assert not trace.terminated
        assert np.isfinite(trace.output).all()
        assert np.max(np.abs(trace.voltage_v)) <= config.limits["voltage_v"] + 1e-12
        assert np.max(np.abs(trace.current_a)) <= config.limits["hard_current_a"] + 1e-6
        assert np.max(np.abs(trace.speed_rad_s)) < config.limits["hard_speed_rad_s"]


def test_approved_dobc_sign_reduces_resisting_load_disturbance():
    config = load_physics_motor_config(PROJECT_ROOT)
    space = load_physics_controller_parameter_space(PROJECT_ROOT)
    disabled = space.initial.copy()
    disabled[6] = 0.0
    without_dobc = simulate_scenario(
        config, config.nominal, disabled, "disturbance"
    )
    with_dobc = simulate_scenario(
        config, config.nominal, space.initial, "disturbance"
    )
    start = int(config.scenarios["disturbance_start_s"] / config.sample_period_s)
    target = config.scenarios["speed_reference_rad_s"]
    peak_without = np.max(np.abs(target - without_dobc.output[start:]))
    peak_with = np.max(np.abs(target - with_dobc.output[start:]))
    assert peak_with < 0.5 * peak_without


def test_encoder_stress_and_scenario_override_are_seed_reproducible():
    config = load_physics_motor_config(PROJECT_ROOT)
    space = load_physics_controller_parameter_space(PROJECT_ROOT)
    override = {"load_torque_step_nm": 0.08165}
    first = simulate_scenario(
        config,
        config.nominal,
        space.initial,
        "disturbance",
        encoder_effects=True,
        seed=20260724,
        scenario_overrides=override,
    )
    second = simulate_scenario(
        config,
        config.nominal,
        space.initial,
        "disturbance",
        encoder_effects=True,
        seed=20260724,
        scenario_overrides=override,
    )
    assert np.array_equal(first.output, second.output)
    assert np.array_equal(first.current_reference_a, second.current_reference_a)
    assert np.max(first.load_torque_nm) == 0.08165
    assert not first.terminated
    assert np.isfinite(first.output).all()
