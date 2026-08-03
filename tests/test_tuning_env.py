from pathlib import Path
import shutil

import numpy as np

from elc_rl.tuning_env import (
    OBSERVATION_KEYS,
    PIDTuningEnv,
    STAGE_INDICES,
    STAGE_ORDER,
    TIME_METRIC_NAMES,
    time_stage_cost,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PURE_PHYSICS_RUNTIME_FILES = (
    "config/motor_physics.json",
    "data/processed/controller_parameter_space.json",
    "data/processed/physics_motor_ensemble.npz",
    "data/processed/physics_motor_ensemble_manifest.json",
)


def test_reset_is_seed_deterministic_and_observation_is_valid():
    environment = PIDTuningEnv(
        PROJECT_ROOT,
        stage="joint",
        initial_perturbation=0.02,
    )
    first_observation, first_info = environment.reset(seed=1234)
    second_observation, second_info = environment.reset(seed=1234)
    assert environment.observation_space.contains(first_observation)
    assert set(first_observation) == set(OBSERVATION_KEYS)
    assert set(environment.observation_space.spaces) == set(OBSERVATION_KEYS)
    assert "frf_context" not in first_observation
    assert first_observation["sampled_frf"].shape == (96,)
    friction_context = first_observation["friction_context"]
    assert friction_context.shape == (6,)
    assert np.isfinite(friction_context).all()
    assert np.max(np.abs(friction_context)) <= 1.0 + 1e-6
    assert np.any(np.abs(friction_context) > 0.0)
    flattened_size = sum(
        int(np.prod(space.shape))
        for space in environment.observation_space.spaces.values()
    )
    assert flattened_size == 179
    for key in first_observation:
        assert np.array_equal(first_observation[key], second_observation[key])
    assert first_info["sampled_model_ids"] == second_info["sampled_model_ids"]
    assert first_observation["time_metrics"].shape == (len(TIME_METRIC_NAMES),)
    assert first_info["stage_cost"] == (
        first_info["frequency_stage_cost"] + first_info["time_stage_cost"]
    )
    assert first_info["fast_time_safe"]
    assert first_info["audit_time_safe"]


def test_environment_runs_without_measured_frf_artifacts(tmp_path):
    for relative in PURE_PHYSICS_RUNTIME_FILES:
        source = PROJECT_ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    assert not (tmp_path / "data/processed/frf_tasks.npz").exists()
    environment = PIDTuningEnv(
        tmp_path,
        stage="joint",
        initial_perturbation=0.0,
    )
    observation, info = environment.reset(seed=20260730, options={"perturb": False})
    assert environment.observation_space.contains(observation)
    assert set(observation) == set(OBSERVATION_KEYS)
    assert info["backend"] == "physics"


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


def test_audit_schedule_continues_across_episode_resets():
    environment = PIDTuningEnv(
        PROJECT_ROOT,
        stage="joint",
        max_episode_steps=2,
        audit_interval=3,
        initial_perturbation=0.0,
    )
    environment.reset(seed=21, options={"perturb": False})
    zero = np.zeros(11, dtype=np.float32)
    environment.step(zero)
    _, _, _, truncated, second_info = environment.step(zero)
    assert truncated
    assert not second_info["audit_performed"]

    environment.reset(options={"perturb": False})
    _, _, _, _, third_info = environment.step(zero)
    assert third_info["total_step"] == 3
    assert third_info["audit_performed"]

    environment.reset(seed=21, options={"perturb": False})
    _, _, _, _, reseeded_info = environment.step(zero)
    assert reseeded_info["total_step"] == 1
    assert not reseeded_info["audit_performed"]


def test_exported_environment_state_restores_exact_next_transition():
    first = PIDTuningEnv(
        PROJECT_ROOT,
        stage="joint",
        max_episode_steps=8,
        audit_interval=3,
        initial_perturbation=0.02,
        worker_rank=2,
    )
    first.reset(seed=314159)
    first.step(np.full(11, 0.05, dtype=np.float32))
    state = first.export_state()
    assert state["schema_version"] == 3
    expected = first.step(np.full(11, -0.03, dtype=np.float32))

    restored = PIDTuningEnv(
        PROJECT_ROOT,
        stage="joint",
        max_episode_steps=8,
        audit_interval=3,
        initial_perturbation=0.02,
        worker_rank=2,
    )
    restored.reset(seed=1)
    restored.restore_state(state)
    actual = restored.step(np.full(11, -0.03, dtype=np.float32))

    for key in expected[0]:
        assert np.array_equal(expected[0][key], actual[0][key])
    assert expected[1:4] == actual[1:4]
    assert expected[4]["worker_rank"] == actual[4]["worker_rank"] == 2
    assert expected[4]["total_step"] == actual[4]["total_step"]
    assert expected[4]["stage_cost"] == actual[4]["stage_cost"]
    assert np.array_equal(expected[4]["parameters"], actual[4]["parameters"])


def test_legacy_environment_state_is_rejected():
    environment = PIDTuningEnv(PROJECT_ROOT, initial_perturbation=0.0)
    environment.reset(seed=9, options={"perturb": False})
    state = environment.export_state()
    state["schema_version"] = 2
    with np.testing.assert_raises_regex(
        ValueError,
        "unsupported environment state schema",
    ):
        environment.restore_state(state)


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
