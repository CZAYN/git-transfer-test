from pathlib import Path

import numpy as np

from elc_rl.physics_evaluator import (
    get_physics_controller_evaluator,
    get_physics_time_domain_evaluator,
)
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
    transition = environment.step(np.zeros(11, dtype=np.float32))
    assert environment.observation_space.contains(transition[0])
    assert not transition[2]

