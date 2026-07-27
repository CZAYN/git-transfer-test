import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from elc_rl.physics_motor_model import (
    MODEL_PARAMETER_NAMES,
    load_physics_motor_config,
    load_physics_motor_ensemble,
)
from elc_rl.physics_test_dataset import (
    FINAL_TEST_ENSEMBLE_RELATIVE_PATH,
    FINAL_TEST_MANIFEST_RELATIVE_PATH,
    build_physics_test_ensemble,
    construct_physics_test_ensemble,
    load_final_test_spec,
    load_physics_test_ensemble,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_final_test_split_is_sealed_and_has_expected_roles():
    ensemble = load_physics_test_ensemble(PROJECT_ROOT)
    manifest = json.loads(
        (PROJECT_ROOT / FINAL_TEST_MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    assert ensemble["parameters"].shape == (24, len(MODEL_PARAMETER_NAMES))
    assert np.count_nonzero(ensemble["test_group"] == "in_distribution") == 16
    assert np.count_nonzero(ensemble["test_group"] == "ood") == 8
    assert np.all(ensemble["final_test_only"] == 1)
    assert np.all(ensemble["active_for_training"] == 0)
    assert np.all(ensemble["active_for_validation"] == 0)
    assert manifest["status"] == "sealed_unconsumed"
    assert not manifest["candidate_evaluated_during_construction"]
    assert manifest["isolation"]["active_for_training_count"] == 0
    assert manifest["isolation"]["active_for_validation_count"] == 0
    assert manifest["test_ensemble_sha256"] == _sha256(
        PROJECT_ROOT / FINAL_TEST_ENSEMBLE_RELATIVE_PATH
    )


def test_test_models_do_not_overlap_train_or_validation_models():
    test = load_physics_test_ensemble(PROJECT_ROOT)
    source = load_physics_motor_ensemble(PROJECT_ROOT)
    overlap = np.all(
        np.isclose(
            test["parameters"][:, None, :],
            source["parameters"][None, :, :],
            rtol=0.0,
            atol=1e-14,
        ),
        axis=2,
    )
    assert not np.any(overlap)
    assert np.min(test["normalized_min_distance_to_train_validation"][:16]) >= 0.02
    assert len(set(str(value) for value in test["model_id"])) == 24


def test_in_distribution_models_use_only_declared_training_uncertainty():
    config = load_physics_motor_config(PROJECT_ROOT)
    ensemble = load_physics_test_ensemble(PROJECT_ROOT)
    values = ensemble["parameters"][ensemble["test_group"] == "in_distribution"]
    nominal = config.nominal.as_array()
    uncertainty = np.asarray(
        [config.uncertainty_fraction[name] for name in MODEL_PARAMETER_NAMES]
    )
    relative = np.abs(values / nominal[None, :] - 1.0)
    assert np.all(relative <= uncertainty[None, :] + 1e-12)
    assert not np.any(
        np.all(np.isclose(values, nominal[None, :], rtol=0.0, atol=1e-14), axis=1)
    )


def test_every_ood_model_exits_the_training_uncertainty_box():
    config = load_physics_motor_config(PROJECT_ROOT)
    ensemble = load_physics_test_ensemble(PROJECT_ROOT)
    values = ensemble["parameters"][ensemble["test_group"] == "ood"]
    nominal = config.nominal.as_array()
    uncertainty = np.asarray(
        [config.uncertainty_fraction[name] for name in MODEL_PARAMETER_NAMES]
    )
    normalized = np.abs((values / nominal[None, :] - 1.0) / uncertainty[None, :])
    assert np.all(np.any(normalized > 1.0 + 1e-12, axis=1))


def test_test_set_construction_is_array_reproducible_without_rewriting_files():
    sealed = load_physics_test_ensemble(PROJECT_ROOT)
    rebuilt = construct_physics_test_ensemble(PROJECT_ROOT)
    assert set(sealed) == set(rebuilt)
    for name in sealed:
        assert np.array_equal(sealed[name], rebuilt[name])


def test_training_runtime_does_not_import_or_name_final_test_artifacts():
    forbidden = (
        "physics_test_dataset",
        "final_test_evaluator",
        "physics_motor_test_v1",
        "final_test_spec_v1",
    )
    runtime_files = (
        PROJECT_ROOT / "src" / "elc_rl" / "tuning_env.py",
        PROJECT_ROOT / "src" / "elc_rl" / "physics_evaluator.py",
        PROJECT_ROOT / "src" / "elc_rl" / "transition_dataset.py",
        PROJECT_ROOT / "scripts" / "train_sac_smoke.py",
        PROJECT_ROOT / "src" / "elc_rl" / "sac_training.py",
        PROJECT_ROOT / "scripts" / "train_sac.py",
        PROJECT_ROOT / "scripts" / "select_final_candidate.py",
        PROJECT_ROOT / "scripts" / "check_server_runtime.py",
        PROJECT_ROOT / "scripts" / "run_optimizer_baseline.py",
        PROJECT_ROOT / "scripts" / "generate_transitions.py",
    )
    for path in runtime_files:
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path.name


def test_final_test_thresholds_are_fixed_before_candidate_evaluation():
    spec = load_final_test_spec(PROJECT_ROOT)
    thresholds = spec["acceptance_thresholds"]
    assert thresholds["hard_all_24_models"]["closed_loop_stable"]
    assert thresholds["hard_all_24_models"]["maximum_current_limit_ratio"] == 1.001
    assert thresholds["minimum_phase_margin_deg"] == {
        "current": 20.0,
        "speed": 20.0,
        "position": 25.0,
    }
    assert not spec["isolation_policy"][
        "evaluate_candidate_before_training_is_locked"
    ]


def test_sealed_test_set_refuses_an_accidental_rebuild():
    with pytest.raises(FileExistsError, match="already exist"):
        build_physics_test_ensemble(PROJECT_ROOT)
