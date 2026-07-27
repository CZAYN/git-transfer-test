from pathlib import Path

import numpy as np

from elc_rl.tuning_env import (
    PIDTuningEnv,
    STAGE_INDICES,
    STAGE_ORDER,
    TIME_METRIC_NAMES,
    time_stage_cost,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_reset_is_seed_deterministic_and_observation_is_valid():
    environment = PIDTuningEnv(
        PROJECT_ROOT,
        stage="joint",
        initial_perturbation=0.02,
    )
    first_observation, first_info = environment.reset(seed=1234)
    second_observation, second_info = environment.reset(seed=1234)
    assert environment.observation_space.contains(first_observation)
    for key in first_observation:
        assert np.array_equal(first_observation[key], second_observation[key])
    assert first_info["sampled_model_ids"] == second_info["sampled_model_ids"]
    assert first_observation["time_metrics"].shape == (len(TIME_METRIC_NAMES),)
    assert first_info["stage_cost"] == (
        first_info["frequency_stage_cost"] + first_info["time_stage_cost"]
    )
    assert first_info["fast_time_safe"]
    assert first_info["audit_time_safe"]


def test_dobc_time_cost_prefers_the_configured_nominal_gain():
    environment = PIDTuningEnv(
        PROJECT_ROOT,
        stage="dobc",
        initial_perturbation=0.0,
    )
    environment.reset(seed=20260715, options={"perturb": False})
    sampled = environment._sampled_indices
    assert sampled is not None
    baseline = environment.parameter_space.initial.copy()
    candidate = baseline.copy()
    candidate[6] = 0.5
    baseline_report = environment.time_evaluator.train(baseline, sampled)
    candidate_report = environment.time_evaluator.train(candidate, sampled)
    assert time_stage_cost(baseline_report, "dobc") < time_stage_cost(
        candidate_report, "dobc"
    )


def test_stage_action_mask_only_updates_active_parameters():
    for stage in STAGE_ORDER[:-1]:
        environment = PIDTuningEnv(
            PROJECT_ROOT,
            stage=stage,
            initial_perturbation=0.0,
            audit_interval=16,
        )
        environment.reset(seed=7, options={"perturb": False})
        before = environment.parameters
        _, _, _, _, _ = environment.step(np.ones(11, dtype=np.float32))
        after = environment.parameters
        active = np.zeros(11, dtype=bool)
        active[list(STAGE_INDICES[stage])] = True
        assert np.array_equal(after[~active], before[~active])
        assert np.any(after[active] != before[active])


def test_custom_stage_base_parameters_are_preserved_on_reset():
    reference = PIDTuningEnv(PROJECT_ROOT, initial_perturbation=0.0)
    base = reference.parameter_space.initial.copy()
    base[6] = 0.5
    environment = PIDTuningEnv(
        PROJECT_ROOT,
        stage="dobc",
        initial_perturbation=0.0,
        base_parameters=base,
    )
    environment.reset(seed=77, options={"perturb": False})
    assert np.allclose(environment.parameters, base, rtol=1e-12, atol=0.0)


def test_episode_truncation_and_periodic_audit():
    environment = PIDTuningEnv(
        PROJECT_ROOT,
        stage="joint",
        max_episode_steps=3,
        audit_interval=2,
        initial_perturbation=0.0,
    )
    environment.reset(seed=11, options={"perturb": False})
    zero = np.zeros(11, dtype=np.float32)
    _, _, terminated, truncated, first_info = environment.step(zero)
    assert not terminated and not truncated
    assert not first_info["audit_performed"]
    _, _, terminated, truncated, second_info = environment.step(zero)
    assert not terminated and not truncated
    assert second_info["audit_performed"]
    assert second_info["audit_safe"]
    _, _, terminated, truncated, _ = environment.step(zero)
    assert not terminated and truncated


def test_every_stage_can_reset_and_step():
    for stage in STAGE_ORDER:
        environment = PIDTuningEnv(
            PROJECT_ROOT,
            stage=stage,
            max_episode_steps=2,
            audit_interval=2,
        )
        observation, info = environment.reset(seed=99)
        assert environment.observation_space.contains(observation)
        assert info["fast_safe"]
        transition = environment.step(environment.action_space.sample())
        assert environment.observation_space.contains(transition[0])
        assert isinstance(transition[1], float)
        assert isinstance(transition[2], bool)
        assert isinstance(transition[3], bool)
