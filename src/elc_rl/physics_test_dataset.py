"""Sealed final-test motor ensemble, intentionally isolated from RL training.

This module constructs and validates test-only motor parameters.  It does not
evaluate controllers and is not imported by the Gymnasium environment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .physics_motor_model import (
    MODEL_PARAMETER_NAMES,
    load_physics_motor_config,
    load_physics_motor_ensemble,
)


FINAL_TEST_SPEC_RELATIVE_PATH = Path("config") / "final_test_spec_v1.json"
FINAL_TEST_ENSEMBLE_RELATIVE_PATH = (
    Path("data") / "processed" / "physics_motor_test_v1.npz"
)
FINAL_TEST_MANIFEST_RELATIVE_PATH = (
    Path("data") / "processed" / "physics_motor_test_v1_manifest.json"
)
SOURCE_ENSEMBLE_RELATIVE_PATH = (
    Path("data") / "processed" / "physics_motor_ensemble_v1.npz"
)
SOURCE_MANIFEST_RELATIVE_PATH = (
    Path("data") / "processed" / "physics_motor_ensemble_v1_manifest.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_final_test_spec(project_root: Path) -> dict[str, Any]:
    path = Path(project_root).resolve() / FINAL_TEST_SPEC_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload["schema_version"]) != 1:
        raise ValueError("unsupported final-test specification schema")
    if payload["test_suite_id"] != "physics_motor_final_test_v1":
        raise ValueError("unexpected final-test suite identifier")
    if payload["isolation_policy"]["evaluate_candidate_before_training_is_locked"]:
        raise ValueError("final-test isolation policy must forbid pre-training evaluation")
    if not payload["isolation_policy"]["final_report_write_once"]:
        raise ValueError("final-test report must be write-once")
    return payload


def _stratified_uniform(
    rng: np.random.Generator, rows: int, columns: int
) -> np.ndarray:
    values = np.empty((rows, columns), dtype=np.float64)
    for column in range(columns):
        strata = (rng.permutation(rows) + rng.random(rows)) / rows
        values[:, column] = 2.0 * strata - 1.0
    return values


def _normalized_chebyshev_distance(
    candidates: np.ndarray,
    references: np.ndarray,
    nominal: np.ndarray,
    uncertainty: np.ndarray,
) -> np.ndarray:
    scale = nominal * uncertainty
    difference = np.abs(
        (candidates[:, None, :] - references[None, :, :]) / scale[None, None, :]
    )
    return np.min(np.max(difference, axis=2), axis=1)


def _minimum_internal_distance(
    candidates: np.ndarray, nominal: np.ndarray, uncertainty: np.ndarray
) -> float:
    scale = nominal * uncertainty
    difference = np.abs(
        (candidates[:, None, :] - candidates[None, :, :]) / scale[None, None, :]
    )
    pairwise = np.max(difference, axis=2)
    np.fill_diagonal(pairwise, np.inf)
    return float(np.min(pairwise))


def construct_physics_test_ensemble(project_root: Path) -> dict[str, np.ndarray]:
    """Construct deterministic arrays in memory without exposing any candidate."""

    root = Path(project_root).resolve()
    spec = load_final_test_spec(root)
    config = load_physics_motor_config(root)
    source = load_physics_motor_ensemble(root)
    source_parameters = np.asarray(source["parameters"], dtype=np.float64)
    nominal = config.nominal.as_array()
    uncertainty = np.asarray(
        [config.uncertainty_fraction[name] for name in MODEL_PARAMETER_NAMES],
        dtype=np.float64,
    )

    id_spec = spec["in_distribution_test"]
    id_count = int(id_spec["model_count"])
    minimum_distance = float(
        id_spec["minimum_normalized_chebyshev_distance_from_train_validation"]
    )
    rng = np.random.default_rng(int(id_spec["seed"]))
    id_parameters: np.ndarray | None = None
    id_distances: np.ndarray | None = None
    construction_attempt = 0
    for construction_attempt in range(1, 101):
        unit = _stratified_uniform(rng, id_count, nominal.size)
        candidate = nominal[None, :] * (1.0 + unit * uncertainty[None, :])
        distances = _normalized_chebyshev_distance(
            candidate, source_parameters, nominal, uncertainty
        )
        internal_distance = _minimum_internal_distance(
            candidate, nominal, uncertainty
        )
        if np.all(distances >= minimum_distance) and internal_distance >= minimum_distance:
            id_parameters = candidate
            id_distances = distances
            break
    if id_parameters is None or id_distances is None:
        raise RuntimeError("failed to construct a non-overlapping in-distribution test set")

    ood_rows: list[np.ndarray] = []
    ood_ids: list[str] = []
    for corner in spec["ood_test"]["corner_models"]:
        multipliers = np.ones(nominal.size, dtype=np.float64)
        for name, value in corner["multipliers"].items():
            if name not in MODEL_PARAMETER_NAMES:
                raise ValueError(f"unknown OOD motor parameter: {name}")
            multiplier = float(value)
            if not 0.5 <= multiplier <= 1.5:
                raise ValueError(f"unsafe OOD multiplier for {name}: {multiplier}")
            multipliers[MODEL_PARAMETER_NAMES.index(name)] = multiplier
        ood_rows.append(nominal * multipliers)
        ood_ids.append(str(corner["model_id"]))
    ood_parameters = np.asarray(ood_rows, dtype=np.float64)
    if ood_parameters.shape != (
        int(spec["ood_test"]["model_count"]),
        len(MODEL_PARAMETER_NAMES),
    ):
        raise ValueError("OOD corner-model count does not match the test specification")
    ood_distances = _normalized_chebyshev_distance(
        ood_parameters, source_parameters, nominal, uncertainty
    )

    parameters = np.vstack([id_parameters, ood_parameters])
    model_id = np.asarray(
        [f"test_id_{index + 1:02d}" for index in range(id_count)] + ood_ids
    )
    test_group = np.asarray(
        ["in_distribution"] * id_count + ["ood"] * len(ood_ids)
    )
    scenario_profile = np.asarray(
        ["standard"] * id_count + ["ood_stress"] * len(ood_ids)
    )
    result = {
        "schema_version": np.asarray(1, dtype=np.int16),
        "test_suite_id": np.asarray(spec["test_suite_id"]),
        "parameter_names": np.asarray(MODEL_PARAMETER_NAMES),
        "parameters": parameters,
        "model_id": model_id,
        "test_group": test_group,
        "scenario_profile": scenario_profile,
        "final_test_only": np.ones(parameters.shape[0], dtype=np.int8),
        "active_for_training": np.zeros(parameters.shape[0], dtype=np.int8),
        "active_for_validation": np.zeros(parameters.shape[0], dtype=np.int8),
        "normalized_min_distance_to_train_validation": np.concatenate(
            [id_distances, ood_distances]
        ),
        "construction_attempt": np.asarray(construction_attempt, dtype=np.int16),
    }
    validate_physics_test_ensemble(result, source_parameters, config.nominal.as_array())
    return result


def validate_physics_test_ensemble(
    ensemble: dict[str, np.ndarray],
    source_parameters: np.ndarray,
    nominal: np.ndarray,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "test_suite_id",
        "parameter_names",
        "parameters",
        "model_id",
        "test_group",
        "scenario_profile",
        "final_test_only",
        "active_for_training",
        "active_for_validation",
        "normalized_min_distance_to_train_validation",
        "construction_attempt",
    }
    missing = required.difference(ensemble)
    if missing:
        raise ValueError(f"final-test ensemble is missing arrays: {sorted(missing)}")
    if int(ensemble["schema_version"]) != 1:
        raise ValueError("unexpected final-test ensemble schema")
    if str(ensemble["test_suite_id"].item()) != "physics_motor_final_test_v1":
        raise ValueError("unexpected final-test suite identifier")
    if tuple(ensemble["parameter_names"].tolist()) != MODEL_PARAMETER_NAMES:
        raise ValueError("final-test parameter order is invalid")
    parameters = np.asarray(ensemble["parameters"], dtype=np.float64)
    if parameters.shape != (24, len(MODEL_PARAMETER_NAMES)):
        raise ValueError("final-test parameter matrix must have shape (24, 11)")
    if not np.isfinite(parameters).all() or np.any(parameters <= 0.0):
        raise ValueError("final-test parameters must be finite and positive")
    if np.count_nonzero(ensemble["test_group"] == "in_distribution") != 16:
        raise ValueError("final-test suite must contain 16 in-distribution models")
    if np.count_nonzero(ensemble["test_group"] == "ood") != 8:
        raise ValueError("final-test suite must contain 8 OOD models")
    if not np.all(ensemble["final_test_only"] == 1):
        raise ValueError("every final-test model must be marked test-only")
    if np.any(ensemble["active_for_training"] != 0):
        raise ValueError("final-test model leaked into training eligibility")
    if np.any(ensemble["active_for_validation"] != 0):
        raise ValueError("final-test model leaked into validation eligibility")
    if len(set(str(value) for value in ensemble["model_id"])) != 24:
        raise ValueError("final-test model identifiers must be unique")
    exact_overlap = np.any(
        np.all(
            np.isclose(
                parameters[:, None, :],
                np.asarray(source_parameters)[None, :, :],
                rtol=0.0,
                atol=1e-14,
            ),
            axis=2,
        )
    )
    if exact_overlap:
        raise ValueError("final-test ensemble overlaps the training/validation ensemble")
    nominal_overlap = np.any(
        np.all(np.isclose(parameters, nominal[None, :], rtol=0.0, atol=1e-14), axis=1)
    )
    if nominal_overlap:
        raise ValueError("nominal motor must not be included in the final-test set")
    distances = np.asarray(
        ensemble["normalized_min_distance_to_train_validation"], dtype=np.float64
    )
    if distances.shape != (24,) or not np.isfinite(distances).all():
        raise ValueError("invalid train/validation separation distances")
    return {
        "model_count": 24,
        "in_distribution_models": 16,
        "ood_models": 8,
        "exact_overlap_count": 0,
        "minimum_normalized_distance_to_train_validation": float(np.min(distances)),
        "minimum_in_distribution_distance_to_train_validation": float(
            np.min(distances[:16])
        ),
    }


def build_physics_test_ensemble(
    project_root: Path, *, overwrite: bool = False
) -> dict[str, Any]:
    """Write and seal the test-only ensemble; refuse accidental rebuilds."""

    root = Path(project_root).resolve()
    output = root / FINAL_TEST_ENSEMBLE_RELATIVE_PATH
    manifest_path = root / FINAL_TEST_MANIFEST_RELATIVE_PATH
    final_output_dir = root / "outputs" / "final_test_v1"
    if overwrite and (
        (final_output_dir / "final_test_report.json").exists()
        or (final_output_dir / "FINAL_TEST_CONSUMED.json").exists()
    ):
        raise PermissionError("a consumed final-test suite can never be rebuilt")
    if (output.exists() or manifest_path.exists()) and not overwrite:
        raise FileExistsError(
            "final-test artifacts already exist; rebuilding requires explicit overwrite=True"
        )
    ensemble = construct_physics_test_ensemble(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **ensemble)

    source = load_physics_motor_ensemble(root)
    source_parameters = np.asarray(source["parameters"], dtype=np.float64)
    validation = validate_physics_test_ensemble(
        ensemble, source_parameters, load_physics_motor_config(root).nominal.as_array()
    )
    train_mask = source["role"] == "train"
    validation_mask = source["role"] == "validation"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "test_suite_id": "physics_motor_final_test_v1",
        "status": "sealed_unconsumed",
        "candidate_evaluated_during_construction": False,
        "test_ensemble_path": str(FINAL_TEST_ENSEMBLE_RELATIVE_PATH).replace("\\", "/"),
        "test_ensemble_sha256": _sha256(output),
        "test_spec_path": str(FINAL_TEST_SPEC_RELATIVE_PATH).replace("\\", "/"),
        "test_spec_sha256": _sha256(root / FINAL_TEST_SPEC_RELATIVE_PATH),
        "physics_config_sha256": _sha256(
            root / "config" / "motor_physics_v1.json"
        ),
        "frozen_training_validation_ensemble_path": str(
            SOURCE_ENSEMBLE_RELATIVE_PATH
        ).replace("\\", "/"),
        "frozen_training_validation_ensemble_sha256": _sha256(
            root / SOURCE_ENSEMBLE_RELATIVE_PATH
        ),
        "frozen_training_validation_manifest_sha256": _sha256(
            root / SOURCE_MANIFEST_RELATIVE_PATH
        ),
        "frozen_training_parameter_matrix_sha256": _matrix_sha256(
            source_parameters[train_mask]
        ),
        "frozen_validation_parameter_matrix_sha256": _matrix_sha256(
            source_parameters[validation_mask]
        ),
        "test_parameter_matrix_sha256": _matrix_sha256(ensemble["parameters"]),
        "in_distribution_seed": int(
            load_final_test_spec(root)["in_distribution_test"]["seed"]
        ),
        "construction_attempt": int(ensemble["construction_attempt"]),
        "parameter_names": list(MODEL_PARAMETER_NAMES),
        **validation,
        "isolation": {
            "active_for_training_count": int(
                np.count_nonzero(ensemble["active_for_training"])
            ),
            "active_for_validation_count": int(
                np.count_nonzero(ensemble["active_for_validation"])
            ),
            "final_test_only_count": int(
                np.count_nonzero(ensemble["final_test_only"])
            ),
            "final_report_exists_at_seal_time": bool(
                (root / "outputs" / "final_test_v1" / "final_test_report.json").exists()
            ),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_physics_test_ensemble(project_root: Path) -> dict[str, np.ndarray]:
    """Load the sealed test set and verify source/config hashes have not changed."""

    root = Path(project_root).resolve()
    output = root / FINAL_TEST_ENSEMBLE_RELATIVE_PATH
    manifest_path = root / FINAL_TEST_MANIFEST_RELATIVE_PATH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        output: manifest["test_ensemble_sha256"],
        root / FINAL_TEST_SPEC_RELATIVE_PATH: manifest["test_spec_sha256"],
        root / "config" / "motor_physics_v1.json": manifest[
            "physics_config_sha256"
        ],
        root / SOURCE_ENSEMBLE_RELATIVE_PATH: manifest[
            "frozen_training_validation_ensemble_sha256"
        ],
        root / SOURCE_MANIFEST_RELATIVE_PATH: manifest[
            "frozen_training_validation_manifest_sha256"
        ],
    }
    for path, expected in checks.items():
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"sealed final-test dependency changed: {path.name}")
    with np.load(output, allow_pickle=False) as archive:
        ensemble = {name: archive[name] for name in archive.files}
    source = load_physics_motor_ensemble(root)
    validation = validate_physics_test_ensemble(
        ensemble,
        source["parameters"],
        load_physics_motor_config(root).nominal.as_array(),
    )
    if _matrix_sha256(ensemble["parameters"]) != manifest["test_parameter_matrix_sha256"]:
        raise ValueError("sealed final-test parameter matrix hash mismatch")
    if validation["model_count"] != int(manifest["model_count"]):
        raise ValueError("final-test manifest model count mismatch")
    return ensemble
