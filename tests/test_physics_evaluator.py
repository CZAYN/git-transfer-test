from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from elc_rl.physics_evaluator import (
    PHYSICS_FRICTION_CONTEXT_PARAMETER_NAMES,
    PhysicsControllerEvaluator,
    get_physics_controller_evaluator,
    get_physics_time_domain_evaluator,
)
from elc_rl.physics_motor_model import MODEL_PARAMETER_NAMES
from elc_rl.tuning_env import PIDTuningEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_physics_frequency_baseline_is_safe_over_56_models():
    evaluator = get_physics_controller_evaluator(PROJECT_ROOT)
    report = evaluator.audit(evaluator.space.initial)
    assert report["backend"] == "physics"
    assert report["evaluated_model_count"] == 56
    assert report["safety"]["safe"]
    expected = {
        "current_reference": 400.0,
        "speed_train": 40.0,
        "position_surrogate": 10.0,
    }
    for split, target in expected.items():
        summary = report["splits"][split]
        assert summary["stable_fraction"] == 1.0
        assert np.isclose(summary["crossover_hz_median"], target, rtol=0.16)
    assert report["safety"]["current_to_speed_crossover_ratio"] >= 4.0
    assert report["safety"]["speed_to_position_crossover_ratio"] >= 3.0


def test_physics_nonlinear_training_report_is_safe_and_finite():
    frequency = get_physics_controller_evaluator(PROJECT_ROOT)
    evaluator = get_physics_time_domain_evaluator(PROJECT_ROOT)
    sampled = frequency.sample_training_indices(np.random.default_rng(20260722))
    report = evaluator.train(evaluator.space.initial, sampled)
    assert report["backend"] == "physics"
    assert report["safety"]["safe"]
    for split in ("current_reference", "speed_train", "position_surrogate"):
        summary = report["splits"][split]
        assert summary["stable_fraction"] == 1.0
        assert summary["voltage_limit_ratio_worst"] <= 1.001
        assert summary["current_limit_ratio_worst"] <= 1.001
        assert summary["speed_limit_ratio_worst"] <= 1.001
        assert np.isfinite(summary["friction_torque_peak_nm_worst"])


def test_default_environment_uses_physics_and_one_coherent_motor():
    environment = PIDTuningEnv(
        PROJECT_ROOT,
        stage="joint",
        max_episode_steps=2,
        audit_interval=8,
        initial_perturbation=0.0,
    )
    observation, info = environment.reset(seed=20260722, options={"perturb": False})
    assert environment.backend == "physics"
    assert info["backend"] == "physics"
    assert len(info["sampled_model_ids"]) == 1
    assert info["fast_safe"] and info["audit_safe"]
    assert environment.observation_space.contains(observation)
    assert observation["sampled_frf"].shape == (96,)
    transition = environment.step(np.zeros(11, dtype=np.float32))
    assert environment.observation_space.contains(transition[0])
    assert not transition[2]


def test_sampled_physics_frf_is_finite_deterministic_and_task_independent():
    evaluator = get_physics_controller_evaluator(PROJECT_ROOT)
    sampled = evaluator.sample_training_indices(np.random.default_rng(20260730))
    first = evaluator.sampled_frf_vector(sampled)
    second = evaluator.sampled_frf_vector(sampled)
    assert first.shape == (96,)
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
    friction = evaluator.friction_context_vector(sampled)
    repeated_friction = evaluator.friction_context_vector(sampled)
    assert friction.shape == (6,)
    assert np.isfinite(friction).all()
    assert np.array_equal(friction, repeated_friction)
    assert np.max(np.abs(friction)) <= 1.0 + 1e-12
    assert np.any(np.abs(friction) > 0.0)


def test_active_lugre_context_exposes_the_sampled_model_uncertainty():
    evaluator = PhysicsControllerEvaluator(PROJECT_ROOT)
    payload = json.loads(json.dumps(evaluator.config.payload))
    payload["friction_model"]["active"] = "lugre"
    payload["friction_model"]["parameter_status"]["coulomb_friction_nm"] = (
        "synthetic_test_fixture"
    )
    payload["friction_model"]["parameter_status"]["static_friction_nm"] = (
        "synthetic_test_fixture"
    )
    payload["nominal_parameters"]["coulomb_friction_nm"] = 0.04
    payload["nominal_parameters"]["static_friction_nm"] = 0.07
    payload["uncertainty_fraction"]["coulomb_friction_nm"] = 0.1
    payload["uncertainty_fraction"]["static_friction_nm"] = 0.1
    nominal = replace(
        evaluator.config.nominal,
        coulomb_friction_nm=0.04,
        static_friction_nm=0.07,
    )
    evaluator.config = replace(
        evaluator.config,
        payload=payload,
        nominal=nominal,
    )
    evaluator.config.validate()

    sampled = np.asarray([int(evaluator.training_indices[1])], dtype=np.int64)
    deviations = np.asarray([0.5, -0.5, 0.25, 0.75, -0.75, 1.0])
    row = nominal.as_array().copy()
    for name, deviation in zip(
        PHYSICS_FRICTION_CONTEXT_PARAMETER_NAMES, deviations
    ):
        index = MODEL_PARAMETER_NAMES.index(name)
        uncertainty = evaluator.config.uncertainty_fraction[name]
        row[index] *= 1.0 + uncertainty * deviation
    evaluator.ensemble = {
        key: np.asarray(value).copy() for key, value in evaluator.ensemble.items()
    }
    evaluator.ensemble["parameters"][sampled[0]] = row

    context = evaluator.friction_context_vector(sampled)

    assert np.allclose(context, deviations, rtol=0.0, atol=1e-12)
