"""Server-oriented formal SAC training for the physics-v1 tuning environment.

This module intentionally depends on the production environment and evaluators
directly.  It does not import the small pipeline-check training entry point or
consume any artifacts produced by that entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import tempfile
import time
from typing import Any, Iterable, Mapping

import gymnasium
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import torch

from .physics_evaluator import (
    get_physics_controller_evaluator,
    get_physics_time_domain_evaluator,
)
from .tuning_env import PIDTuningEnv, STAGE_ORDER, stage_cost_v2


TRAINING_INPUT_RELATIVE_PATHS = (
    "config/motor_physics_v1.json",
    "data/processed/controller_parameter_space_physics_v1.json",
    "data/processed/frf_tasks.npz",
    "data/processed/frf_tasks_manifest.json",
    "data/processed/physics_motor_ensemble_v1.npz",
    "data/processed/physics_motor_ensemble_v1_manifest.json",
    "src/elc_rl/__init__.py",
    "src/elc_rl/controller_parameters.py",
    "src/elc_rl/evaluation_utils.py",
    "src/elc_rl/physics_evaluator.py",
    "src/elc_rl/physics_motor_model.py",
    "src/elc_rl/sac_training.py",
    "src/elc_rl/task_dataset.py",
    "src/elc_rl/tuning_env.py",
    "scripts/train_sac.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _required_keys(payload: Mapping[str, Any], keys: Iterable[str], context: str) -> None:
    missing = sorted(set(keys) - set(payload))
    if missing:
        raise ValueError(f"{context} is missing keys: {missing}")


@dataclass(frozen=True)
class StageTrainingSpec:
    name: str
    total_timesteps: int


@dataclass(frozen=True)
class FormalTrainingConfig:
    path: Path
    payload: dict[str, Any]
    stages: tuple[StageTrainingSpec, ...]
    seeds: tuple[int, ...]

    @property
    def sha256(self) -> str:
        return sha256_file(self.path)

    @property
    def run_name(self) -> str:
        return str(self.payload["run_name"])

    @property
    def default_device(self) -> str:
        return str(self.payload["default_device"])


def load_formal_training_config(
    project_root: Path,
    config_path: Path | None = None,
) -> FormalTrainingConfig:
    root = Path(project_root).resolve()
    path = (
        root / "config" / "sac_training_v1.json"
        if config_path is None
        else Path(config_path).resolve()
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    _required_keys(
        payload,
        (
            "schema_version",
            "backend",
            "run_name",
            "default_device",
            "seeds",
            "stages",
            "environment",
            "sac",
            "checkpoint",
            "validation",
            "tensorboard",
            "runtime",
            "isolation",
        ),
        "formal training configuration",
    )
    if payload["schema_version"] != 1 or payload["backend"] != "physics_v1":
        raise ValueError("formal training configuration is not physics-v1 schema 1")

    raw_stages = payload["stages"]
    if not isinstance(raw_stages, list):
        raise ValueError("stages must be a list")
    stages = tuple(
        StageTrainingSpec(
            name=str(item["name"]),
            total_timesteps=int(item["total_timesteps"]),
        )
        for item in raw_stages
    )
    if tuple(stage.name for stage in stages) != STAGE_ORDER:
        raise ValueError(f"formal stages must exactly match {STAGE_ORDER}")
    if any(stage.total_timesteps <= 0 for stage in stages):
        raise ValueError("every formal stage needs positive total_timesteps")

    seeds = tuple(int(value) for value in payload["seeds"])
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("formal training requires at least three unique seeds")
    if any(value < 0 for value in seeds):
        raise ValueError("training seeds must be non-negative")

    environment = payload["environment"]
    _required_keys(
        environment,
        ("max_episode_steps", "audit_interval", "initial_perturbation"),
        "environment configuration",
    )
    if int(environment["max_episode_steps"]) <= 0:
        raise ValueError("max_episode_steps must be positive")
    if int(environment["audit_interval"]) <= 0:
        raise ValueError("audit_interval must be positive")
    if not 0.0 <= float(environment["initial_perturbation"]) <= 0.5:
        raise ValueError("initial_perturbation must be between 0 and 0.5")

    sac = payload["sac"]
    _required_keys(
        sac,
        (
            "policy",
            "learning_rate",
            "buffer_size",
            "learning_starts",
            "batch_size",
            "tau",
            "gamma",
            "train_frequency",
            "gradient_steps",
            "network_architecture",
        ),
        "SAC configuration",
    )
    if str(sac["policy"]) != "MultiInputPolicy":
        raise ValueError("physics-v1 formal training requires MultiInputPolicy")
    positive_integer_fields = (
        "buffer_size",
        "learning_starts",
        "batch_size",
        "train_frequency",
        "gradient_steps",
    )
    if any(int(sac[name]) <= 0 for name in positive_integer_fields):
        raise ValueError("SAC integer hyperparameters must be positive")
    if int(sac["buffer_size"]) < int(sac["batch_size"]):
        raise ValueError("buffer_size must not be smaller than batch_size")
    architecture = tuple(int(value) for value in sac["network_architecture"])
    if not architecture or any(value <= 0 for value in architecture):
        raise ValueError("network_architecture must contain positive widths")

    checkpoint = payload["checkpoint"]
    _required_keys(
        checkpoint,
        ("interval_timesteps", "save_replay_buffer", "keep_last"),
        "checkpoint configuration",
    )
    if int(checkpoint["interval_timesteps"]) <= 0:
        raise ValueError("checkpoint interval must be positive")
    if int(checkpoint["keep_last"]) <= 0:
        raise ValueError("checkpoint keep_last must be positive")

    validation = payload["validation"]
    _required_keys(
        validation,
        (
            "interval_timesteps",
            "candidate_pool_size",
            "periodic_candidate_limit",
            "stage_finalist_limit",
            "minimum_cost_improvement",
        ),
        "validation configuration",
    )
    if any(
        int(validation[name]) <= 0
        for name in (
            "interval_timesteps",
            "candidate_pool_size",
            "periodic_candidate_limit",
            "stage_finalist_limit",
        )
    ):
        raise ValueError("validation counts and intervals must be positive")
    if float(validation["minimum_cost_improvement"]) < 0.0:
        raise ValueError("minimum_cost_improvement must be non-negative")

    isolation = payload["isolation"]
    if (
        int(isolation.get("training_models", -1)) != 40
        or int(isolation.get("validation_models", -1)) != 16
        or bool(isolation.get("sealed_evaluation_available_during_training", True))
    ):
        raise ValueError("formal training data-isolation declaration is invalid")

    return FormalTrainingConfig(
        path=path,
        payload=payload,
        stages=stages,
        seeds=seeds,
    )


def build_training_input_manifest(
    project_root: Path,
    config: FormalTrainingConfig,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    relative_config = None
    try:
        relative_config = config.path.relative_to(root).as_posix()
    except ValueError:
        relative_config = str(config.path)
    paths = list(TRAINING_INPUT_RELATIVE_PATHS)
    if relative_config not in paths:
        paths.append(relative_config)

    files: list[dict[str, Any]] = []
    for name in paths:
        candidate = Path(name)
        path = candidate if candidate.is_absolute() else root / candidate
        if not path.is_file():
            raise FileNotFoundError(f"missing formal-training input: {path}")
        files.append(
            {
                "path": name.replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    fingerprint = hashlib.sha256(_canonical_json(files)).hexdigest()
    return {
        "schema_version": 1,
        "backend": "physics_v1",
        "files": files,
        "fingerprint": fingerprint,
    }


@dataclass
class CandidateRecord:
    stage: str
    fast_cost: float
    parameters: np.ndarray
    global_timestep: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "fast_cost": float(self.fast_cost),
            "parameters": np.asarray(self.parameters, dtype=np.float64).tolist(),
            "global_timestep": int(self.global_timestep),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CandidateRecord":
        parameters = np.asarray(payload["parameters"], dtype=np.float64)
        if parameters.shape != (11,) or not np.isfinite(parameters).all():
            raise ValueError("serialized candidate parameters are invalid")
        return cls(
            stage=str(payload["stage"]),
            fast_cost=float(payload["fast_cost"]),
            parameters=parameters,
            global_timestep=int(payload["global_timestep"]),
        )


class CandidatePool:
    def __init__(
        self,
        stage: str,
        maximum_size: int,
        records: Iterable[CandidateRecord] = (),
    ) -> None:
        if stage not in STAGE_ORDER:
            raise ValueError(f"invalid candidate-pool stage: {stage}")
        if maximum_size <= 0:
            raise ValueError("candidate-pool size must be positive")
        self.stage = stage
        self.maximum_size = int(maximum_size)
        self.records: list[CandidateRecord] = []
        for record in records:
            self.add(record)

    def add(self, record: CandidateRecord) -> None:
        if record.stage != self.stage:
            raise ValueError("candidate stage does not match pool stage")
        values = np.asarray(record.parameters, dtype=np.float64)
        if values.shape != (11,) or not np.isfinite(values).all():
            raise ValueError("candidate parameters must be finite shape-(11,) values")
        duplicate = next(
            (
                existing
                for existing in self.records
                if np.allclose(
                    values,
                    existing.parameters,
                    rtol=1e-12,
                    atol=1e-14,
                )
            ),
            None,
        )
        if duplicate is not None:
            if record.fast_cost < duplicate.fast_cost:
                duplicate.fast_cost = float(record.fast_cost)
                duplicate.global_timestep = int(record.global_timestep)
                self.records.sort(
                    key=lambda item: (item.fast_cost, item.global_timestep)
                )
            return
        self.records.append(
            CandidateRecord(
                stage=record.stage,
                fast_cost=float(record.fast_cost),
                parameters=values.copy(),
                global_timestep=int(record.global_timestep),
            )
        )
        self.records.sort(key=lambda item: (item.fast_cost, item.global_timestep))
        del self.records[self.maximum_size :]

    def to_payload(self) -> list[dict[str, Any]]:
        return [record.to_payload() for record in self.records]


class StopController:
    """Mutable stop flag set by the command-line signal handlers."""

    def __init__(self) -> None:
        self.requested = False
        self.signal_name: str | None = None

    def request(self, signal_name: str) -> None:
        self.requested = True
        self.signal_name = signal_name


class CandidateCollectorCallback(BaseCallback):
    def __init__(
        self,
        pool: CandidatePool,
        stop_controller: StopController,
    ) -> None:
        super().__init__(verbose=0)
        self.pool = pool
        self.stop_controller = stop_controller

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not bool(info.get("fast_safe", False)):
                continue
            self.pool.add(
                CandidateRecord(
                    stage=self.pool.stage,
                    fast_cost=float(info["stage_cost"]),
                    parameters=np.asarray(info["parameters"], dtype=np.float64),
                    global_timestep=int(self.model.num_timesteps),
                )
            )
        return not self.stop_controller.requested


def time_report_safe(report: Mapping[str, Any]) -> bool:
    splits = report.get("splits", {})
    stable = bool(
        splits
        and all(
            float(summary["stable_fraction"]) == 1.0
            for summary in splits.values()
        )
    )
    safety = report.get("safety")
    return bool(stable and (safety is None or bool(safety.get("safe", False))))


def audit_parameters(
    environment: PIDTuningEnv,
    parameters: np.ndarray,
    stage: str,
    *,
    full_time_domain: bool,
) -> dict[str, Any]:
    values = np.asarray(parameters, dtype=np.float64)
    environment.parameter_space.normalize(values)
    frequency = environment.evaluator.audit(values)
    time_domain = (
        environment.time_evaluator.full_audit(values)
        if full_time_domain
        else environment.time_evaluator.audit(values)
    )
    frequency_safe = bool(frequency["safety"]["safe"])
    time_safe = time_report_safe(time_domain)
    return {
        "safe": bool(frequency_safe and time_safe),
        "frequency_safe": frequency_safe,
        "time_safe": time_safe,
        "cost": stage_cost_v2(
            frequency,
            time_domain,
            stage,
            environment.position_target_hz,
        ),
        "parameters": values.tolist(),
        "frequency": frequency,
        "time_domain": time_domain,
    }


def _unique_parameter_sets(values: Iterable[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for raw in values:
        candidate = np.asarray(raw, dtype=np.float64)
        if not any(
            np.allclose(candidate, existing, rtol=1e-12, atol=1e-14)
            for existing in unique
        ):
            unique.append(candidate.copy())
    return unique


def validate_candidate_pool(
    environment: PIDTuningEnv,
    stage: str,
    stage_base_parameters: np.ndarray,
    pool: CandidatePool,
    candidate_limit: int,
) -> dict[str, Any]:
    candidates = _unique_parameter_sets(
        [
            np.asarray(stage_base_parameters, dtype=np.float64),
            *(record.parameters for record in pool.records[:candidate_limit]),
        ]
    )
    audits = [
        audit_parameters(
            environment,
            parameters,
            stage,
            full_time_domain=False,
        )
        for parameters in candidates
    ]
    safe = [report for report in audits if report["safe"]]
    selected = min(safe, key=lambda report: float(report["cost"])) if safe else None
    return {
        "schema_version": 1,
        "backend": "physics_v1",
        "stage": stage,
        "validation_kind": "runtime_training_validation",
        "candidate_count": len(audits),
        "safe_candidate_count": len(safe),
        "selected": selected,
        "candidates": audits,
    }


def select_stage_curriculum_parameters(
    environment: PIDTuningEnv,
    stage: str,
    stage_base_parameters: np.ndarray,
    pool: CandidatePool,
    periodic_candidate_limit: int,
    finalist_limit: int,
    minimum_improvement: float,
    *,
    full_time_domain: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    runtime = validate_candidate_pool(
        environment,
        stage,
        stage_base_parameters,
        pool,
        periodic_candidate_limit,
    )
    runtime_candidates = sorted(
        (
            item
            for item in runtime["candidates"]
            if bool(item["safe"])
        ),
        key=lambda item: float(item["cost"]),
    )
    finalist_values = _unique_parameter_sets(
        [
            np.asarray(stage_base_parameters, dtype=np.float64),
            *(
                np.asarray(item["parameters"], dtype=np.float64)
                for item in runtime_candidates[:finalist_limit]
            ),
        ]
    )
    full_audits = [
        audit_parameters(
            environment,
            parameters,
            stage,
            full_time_domain=full_time_domain,
        )
        for parameters in finalist_values
    ]
    baseline_audit = next(
        item
        for item in full_audits
        if np.allclose(
            np.asarray(item["parameters"], dtype=np.float64),
            np.asarray(stage_base_parameters, dtype=np.float64),
            rtol=1e-12,
            atol=1e-14,
        )
    )
    safe_finalists = [item for item in full_audits if bool(item["safe"])]
    best = (
        min(safe_finalists, key=lambda item: float(item["cost"]))
        if safe_finalists
        else baseline_audit
    )
    accepted = bool(
        best["safe"]
        and float(best["cost"])
        < float(baseline_audit["cost"]) - float(minimum_improvement)
    )
    selected = (
        np.asarray(best["parameters"], dtype=np.float64)
        if accepted
        else np.asarray(stage_base_parameters, dtype=np.float64)
    )
    report = {
        "schema_version": 1,
        "backend": "physics_v1",
        "stage": stage,
        "runtime_validation": runtime,
        "finalist_audit_scope": (
            "all_56_models" if full_time_domain else "runtime_validation_probe"
        ),
        "full_audit_finalists": full_audits,
        "baseline_full_audit": baseline_audit,
        "best_full_audit": best,
        "accepted": accepted,
        "selected_parameters": selected.tolist(),
    }
    return selected.copy(), report


def _system_manifest(device: str) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": __import__("stable_baselines3").__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "requested_device": device,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def _configure_randomness(seed: int, deterministic_algorithms: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(
        bool(deterministic_algorithms),
        warn_only=True,
    )


def _save_rng_state(path: Path) -> None:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    torch.save(state, path)


def _restore_rng_state(path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _save_model_atomic(model: SAC, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.zip")
    model.save(temporary)
    os.replace(temporary, path)


def _checkpoint_name(
    stage_index: int,
    stage: str,
    stage_steps: int,
    global_steps: int,
) -> str:
    return (
        f"stage_{stage_index + 1:02d}_{stage}"
        f"_s{stage_steps:09d}_g{global_steps:09d}"
    )


def _save_resume_checkpoint(
    model: SAC,
    run_dir: Path,
    state: dict[str, Any],
    config: FormalTrainingConfig,
) -> Path:
    checkpoint_root = run_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    name = _checkpoint_name(
        int(state["stage_index"]),
        str(state["stage"]),
        int(state["stage_timesteps_completed"]),
        int(state["global_timesteps_completed"]),
    )
    target = checkpoint_root / name
    if target.exists():
        suffix = 1
        while (checkpoint_root / f"{name}_r{suffix:02d}").exists():
            suffix += 1
        target = checkpoint_root / f"{name}_r{suffix:02d}"
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=checkpoint_root)
    )
    try:
        model.save(temporary / "model.zip")
        if bool(config.payload["checkpoint"]["save_replay_buffer"]):
            model.save_replay_buffer(temporary / "replay_buffer.pkl")
        _save_rng_state(temporary / "rng_state.pt")
        metadata = {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "stage": state["stage"],
            "stage_index": state["stage_index"],
            "stage_timesteps_completed": state["stage_timesteps_completed"],
            "global_timesteps_completed": state["global_timesteps_completed"],
            "config_sha256": state["config_sha256"],
            "input_fingerprint": state["input_fingerprint"],
        }
        _atomic_write_json(temporary / "checkpoint.json", metadata)
        (temporary / "COMPLETE").write_text("complete\n", encoding="utf-8")
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    state["latest_checkpoint"] = target.relative_to(run_dir).as_posix()
    state["updated_at_utc"] = utc_now()
    _atomic_write_json(run_dir / "trainer_state.json", state)
    _prune_resume_checkpoints(
        checkpoint_root,
        keep_last=int(config.payload["checkpoint"]["keep_last"]),
        protected=target,
    )
    return target


def _prune_resume_checkpoints(
    checkpoint_root: Path,
    *,
    keep_last: int,
    protected: Path,
) -> None:
    completed = sorted(
        (
            path
            for path in checkpoint_root.iterdir()
            if path.is_dir() and (path / "COMPLETE").is_file()
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    keep = set(completed[:keep_last]) | {protected}
    root = checkpoint_root.resolve()
    for path in completed:
        if path in keep:
            continue
        resolved = path.resolve()
        if resolved.parent != root:
            raise RuntimeError(f"refusing to prune unexpected checkpoint path: {resolved}")
        shutil.rmtree(resolved)


def _load_checkpoint(
    run_dir: Path,
    state: Mapping[str, Any],
    environment: PIDTuningEnv,
    device: str,
    tensorboard_log: str | None,
    expect_replay_buffer: bool,
) -> SAC:
    relative = state.get("latest_checkpoint")
    if not relative:
        raise ValueError("resume state does not reference a checkpoint")
    checkpoint = (run_dir / str(relative)).resolve()
    if checkpoint.parent != (run_dir / "checkpoints").resolve():
        raise ValueError("resume checkpoint is outside the run checkpoint directory")
    if not (checkpoint / "COMPLETE").is_file():
        raise ValueError("resume checkpoint is incomplete")
    model = SAC.load(
        checkpoint / "model.zip",
        env=environment,
        device=device,
        tensorboard_log=tensorboard_log,
    )
    replay_path = checkpoint / "replay_buffer.pkl"
    if expect_replay_buffer:
        if not replay_path.is_file():
            raise FileNotFoundError("resume checkpoint has no Replay Buffer")
        model.load_replay_buffer(replay_path)
    _restore_rng_state(checkpoint / "rng_state.pt")
    expected_steps = int(state["global_timesteps_completed"])
    if int(model.num_timesteps) != expected_steps:
        raise ValueError(
            "checkpoint timestep mismatch: "
            f"model={model.num_timesteps}, state={expected_steps}"
        )
    return model


def _new_model(
    environment: PIDTuningEnv,
    seed: int,
    device: str,
    tensorboard_log: str | None,
    effective_sac: Mapping[str, Any],
) -> SAC:
    sac = effective_sac
    return SAC(
        str(sac["policy"]),
        environment,
        learning_rate=float(sac["learning_rate"]),
        buffer_size=int(sac["buffer_size"]),
        learning_starts=int(sac["learning_starts"]),
        batch_size=int(sac["batch_size"]),
        tau=float(sac["tau"]),
        gamma=float(sac["gamma"]),
        train_freq=(int(sac["train_frequency"]), "step"),
        gradient_steps=int(sac["gradient_steps"]),
        policy_kwargs={
            "net_arch": [int(value) for value in sac["network_architecture"]]
        },
        tensorboard_log=tensorboard_log,
        seed=seed,
        device=device,
        verbose=0,
    )


def _effective_sac_parameters(
    config: FormalTrainingConfig,
    run_kind: str,
) -> dict[str, Any]:
    parameters = dict(config.payload["sac"])
    parameters["network_architecture"] = list(
        config.payload["sac"]["network_architecture"]
    )
    if run_kind == "engineering_check":
        parameters["buffer_size"] = min(int(parameters["buffer_size"]), 512)
        parameters["learning_starts"] = min(
            int(parameters["learning_starts"]),
            64,
        )
        parameters["batch_size"] = min(int(parameters["batch_size"]), 64)
    return parameters


def _effective_stage_steps(
    config: FormalTrainingConfig,
    engineering_steps_per_stage: int | None,
) -> dict[str, int]:
    if engineering_steps_per_stage is None:
        return {
            stage.name: int(stage.total_timesteps)
            for stage in config.stages
        }
    if engineering_steps_per_stage <= 0:
        raise ValueError("engineering-check steps must be positive")
    return {
        stage.name: int(engineering_steps_per_stage)
        for stage in config.stages
    }


def _initial_state(
    config: FormalTrainingConfig,
    input_manifest: Mapping[str, Any],
    seed: int,
    run_kind: str,
    effective_steps: Mapping[str, int],
    initial_parameters: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "backend": "physics_v1",
        "status": "running",
        "run_kind": run_kind,
        "seed": int(seed),
        "config_sha256": config.sha256,
        "input_fingerprint": input_manifest["fingerprint"],
        "effective_stage_timesteps": dict(effective_steps),
        "stage_index": 0,
        "stage": STAGE_ORDER[0],
        "stage_timesteps_completed": 0,
        "global_timesteps_completed": 0,
        "curriculum_parameters": np.asarray(
            initial_parameters, dtype=np.float64
        ).tolist(),
        "stage_base_parameters": np.asarray(
            initial_parameters, dtype=np.float64
        ).tolist(),
        "candidate_pool": [],
        "completed_stages": [],
        "latest_checkpoint": None,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "interruption": None,
        "accumulated_wall_time_s": 0.0,
    }


def _verify_resume_state(
    state: Mapping[str, Any],
    config: FormalTrainingConfig,
    input_manifest: Mapping[str, Any],
    seed: int,
    run_kind: str,
    effective_steps: Mapping[str, int],
) -> None:
    expected = {
        "backend": "physics_v1",
        "seed": int(seed),
        "config_sha256": config.sha256,
        "input_fingerprint": input_manifest["fingerprint"],
        "run_kind": run_kind,
        "effective_stage_timesteps": dict(effective_steps),
    }
    mismatches = {
        key: (state.get(key), value)
        for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatches:
        raise ValueError(f"resume state is incompatible: {mismatches}")
    if state.get("status") == "completed":
        return
    stage_index = int(state["stage_index"])
    if not 0 <= stage_index <= len(STAGE_ORDER):
        raise ValueError("resume stage index is invalid")
    if stage_index == len(STAGE_ORDER):
        if state["stage"] not in {STAGE_ORDER[-1], "completed"}:
            raise ValueError("post-stage resume state has an invalid stage name")
    elif state["stage"] != STAGE_ORDER[stage_index]:
        raise ValueError("resume stage name and index disagree")


def _write_periodic_validation(
    run_dir: Path,
    stage_index: int,
    stage: str,
    global_steps: int,
    report: Mapping[str, Any],
) -> Path:
    directory = run_dir / "validation_reports"
    path = directory / (
        f"stage_{stage_index + 1:02d}_{stage}_g{global_steps:09d}.json"
    )
    _atomic_write_json(path, report)
    return path


def _close_model_environment(model: SAC | None) -> None:
    if model is None:
        return
    environment = model.get_env()
    if environment is not None:
        environment.close()


def run_formal_training(
    project_root: Path,
    *,
    config_path: Path | None,
    seed: int,
    run_dir: Path,
    device: str | None = None,
    resume: bool = False,
    engineering_steps_per_stage: int | None = None,
    stop_controller: StopController | None = None,
) -> dict[str, Any]:
    """Run or resume one independent formal-training seed."""

    root = Path(project_root).resolve()
    output = Path(run_dir).resolve()
    config = load_formal_training_config(root, config_path)
    if seed not in config.seeds and engineering_steps_per_stage is None:
        raise ValueError(
            f"formal seed {seed} is not declared in the training configuration"
        )
    selected_device = config.default_device if device is None else str(device)
    if selected_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    run_kind = (
        "formal_training"
        if engineering_steps_per_stage is None
        else "engineering_check"
    )
    effective_steps = _effective_stage_steps(config, engineering_steps_per_stage)
    effective_sac = _effective_sac_parameters(config, run_kind)
    input_manifest = build_training_input_manifest(root, config)
    evaluator = get_physics_controller_evaluator(root)
    initial_parameters = evaluator.space.initial.copy()
    controller = stop_controller if stop_controller is not None else StopController()

    if resume:
        state_path = output / "trainer_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(f"missing resume state: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        _verify_resume_state(
            state,
            config,
            input_manifest,
            seed,
            run_kind,
            effective_steps,
        )
        if state["status"] == "completed":
            summary_path = output / "seed_summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(
                    "completed trainer state has no seed_summary.json"
                )
            return json.loads(summary_path.read_text(encoding="utf-8"))
        state["status"] = "running"
        state["interruption"] = None
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                f"new formal-training output directory is not empty: {output}"
            )
        output.mkdir(parents=True, exist_ok=True)
        state = _initial_state(
            config,
            input_manifest,
            seed,
            run_kind,
            effective_steps,
            initial_parameters,
        )
        manifest = {
            "schema_version": 1,
            "backend": "physics_v1",
            "run_kind": run_kind,
            "run_name": config.run_name,
            "seed": int(seed),
            "created_at_utc": utc_now(),
            "project_root_at_launch": str(root),
            "output_directory": str(output),
            "configuration_path": str(config.path),
            "configuration": config.payload,
            "effective_stage_timesteps": effective_steps,
            "effective_sac": effective_sac,
            "training_inputs": input_manifest,
            "system": _system_manifest(selected_device),
            "data_policy": "training and validation ensemble only",
            "hardware_use_allowed": False,
        }
        _atomic_write_json(output / "run_manifest.json", manifest)
        _atomic_write_json(output / "trainer_state.json", state)

    deterministic = bool(
        config.payload["runtime"]["torch_deterministic_algorithms"]
    )
    _configure_randomness(seed, deterministic)
    tensorboard_log = (
        str(output / str(config.payload["tensorboard"]["subdirectory"]))
        if bool(config.payload["tensorboard"]["enabled"])
        else None
    )
    checkpoint_interval = int(
        config.payload["checkpoint"]["interval_timesteps"]
    )
    validation_interval = int(
        config.payload["validation"]["interval_timesteps"]
    )
    model: SAC | None = None
    environment: PIDTuningEnv | None = None
    session_started = time.perf_counter()
    previous_wall_time = float(state.get("accumulated_wall_time_s", 0.0))

    try:
        while int(state["stage_index"]) < len(STAGE_ORDER):
            stage_index = int(state["stage_index"])
            stage = STAGE_ORDER[stage_index]
            total_stage_steps = int(effective_steps[stage])
            stage_base = np.asarray(
                state["stage_base_parameters"], dtype=np.float64
            )
            env_config = config.payload["environment"]
            environment = PIDTuningEnv(
                root,
                stage=stage,
                max_episode_steps=int(env_config["max_episode_steps"]),
                audit_interval=int(env_config["audit_interval"]),
                initial_perturbation=float(env_config["initial_perturbation"]),
                base_parameters=stage_base,
            )
            stage_seed = seed + 1009 * stage_index
            environment.action_space.seed(stage_seed)

            if model is None:
                if resume and int(state["global_timesteps_completed"]) > 0:
                    model = _load_checkpoint(
                        output,
                        state,
                        environment,
                        selected_device,
                        tensorboard_log,
                        expect_replay_buffer=bool(
                            config.payload["checkpoint"]["save_replay_buffer"]
                        ),
                    )
                    resume = False
                else:
                    model = _new_model(
                        environment,
                        seed,
                        selected_device,
                        tensorboard_log,
                        effective_sac,
                    )
            else:
                previous = model.get_env()
                model.set_env(environment, force_reset=True)
                if previous is not None:
                    previous.close()

            pool = CandidatePool(
                stage,
                int(config.payload["validation"]["candidate_pool_size"]),
                (
                    CandidateRecord.from_payload(item)
                    for item in state.get("candidate_pool", [])
                ),
            )
            completed = int(state["stage_timesteps_completed"])
            next_checkpoint = (
                (completed // checkpoint_interval) + 1
            ) * checkpoint_interval
            next_validation = (
                (completed // validation_interval) + 1
            ) * validation_interval

            while completed < total_stage_steps:
                event_step = min(
                    total_stage_steps,
                    next_checkpoint,
                    next_validation,
                )
                requested_steps = event_step - completed
                callback = CandidateCollectorCallback(pool, controller)
                before = int(model.num_timesteps)
                model.learn(
                    total_timesteps=requested_steps,
                    callback=callback,
                    reset_num_timesteps=False,
                    progress_bar=bool(config.payload["runtime"]["progress_bar"]),
                    tb_log_name=f"seed_{seed}",
                )
                learned = int(model.num_timesteps) - before
                if learned <= 0:
                    raise RuntimeError("SAC made no progress during a training chunk")
                completed += learned
                state["stage_timesteps_completed"] = completed
                state["global_timesteps_completed"] = int(model.num_timesteps)
                state["candidate_pool"] = pool.to_payload()
                state["updated_at_utc"] = utc_now()

                if completed >= next_validation and not controller.requested:
                    validation_report = validate_candidate_pool(
                        environment,
                        stage,
                        stage_base,
                        pool,
                        int(
                            config.payload["validation"][
                                "periodic_candidate_limit"
                            ]
                        ),
                    )
                    validation_report["global_timesteps"] = int(
                        model.num_timesteps
                    )
                    validation_report["stage_timesteps"] = completed
                    _write_periodic_validation(
                        output,
                        stage_index,
                        stage,
                        int(model.num_timesteps),
                        validation_report,
                    )
                    while next_validation <= completed:
                        next_validation += validation_interval

                checkpoint_due = (
                    completed >= next_checkpoint
                    or completed >= total_stage_steps
                    or controller.requested
                )
                if checkpoint_due:
                    _save_resume_checkpoint(model, output, state, config)
                    while next_checkpoint <= completed:
                        next_checkpoint += checkpoint_interval

                if controller.requested:
                    state["accumulated_wall_time_s"] = (
                        previous_wall_time + time.perf_counter() - session_started
                    )
                    state["status"] = "interrupted"
                    state["interruption"] = {
                        "signal": controller.signal_name,
                        "at_utc": utc_now(),
                    }
                    state["updated_at_utc"] = utc_now()
                    _atomic_write_json(output / "trainer_state.json", state)
                    return {
                        "schema_version": 1,
                        "status": "interrupted",
                        "run_kind": run_kind,
                        "seed": seed,
                        "stage": stage,
                        "stage_timesteps_completed": completed,
                        "global_timesteps_completed": int(model.num_timesteps),
                        "accumulated_wall_time_s": state[
                            "accumulated_wall_time_s"
                        ],
                        "run_dir": str(output),
                        "resume_command_required": True,
                    }

            selected, stage_report = select_stage_curriculum_parameters(
                environment,
                stage,
                stage_base,
                pool,
                int(config.payload["validation"]["periodic_candidate_limit"]),
                int(config.payload["validation"]["stage_finalist_limit"]),
                float(
                    config.payload["validation"]["minimum_cost_improvement"]
                ),
                full_time_domain=bool(
                    run_kind == "formal_training" or stage == "joint"
                ),
            )
            stage_report.update(
                {
                    "stage_index": stage_index,
                    "seed": seed,
                    "stage_seed": stage_seed,
                    "stage_timesteps": completed,
                    "global_timesteps": int(model.num_timesteps),
                    "candidate_pool_size": len(pool.records),
                }
            )
            stage_report_path = output / (
                f"stage_{stage_index + 1:02d}_{stage}_report.json"
            )
            _atomic_write_json(stage_report_path, stage_report)
            final_model_path = (
                output
                / "models"
                / f"stage_{stage_index + 1:02d}_{stage}_final.zip"
            )
            _save_model_atomic(model, final_model_path)

            state["completed_stages"].append(
                {
                    "stage": stage,
                    "stage_timesteps": completed,
                    "global_timesteps": int(model.num_timesteps),
                    "selected_parameters": selected.tolist(),
                    "accepted": bool(stage_report["accepted"]),
                    "report": stage_report_path.relative_to(output).as_posix(),
                    "model": final_model_path.relative_to(output).as_posix(),
                }
            )
            state["curriculum_parameters"] = selected.tolist()
            state["stage_index"] = stage_index + 1
            state["stage_timesteps_completed"] = 0
            state["candidate_pool"] = []
            state["updated_at_utc"] = utc_now()
            if int(state["stage_index"]) < len(STAGE_ORDER):
                state["stage"] = STAGE_ORDER[int(state["stage_index"])]
                state["stage_base_parameters"] = selected.tolist()
            _atomic_write_json(output / "trainer_state.json", state)

        if model is None:
            final_parameters_for_resume = np.asarray(
                state["curriculum_parameters"], dtype=np.float64
            )
            env_config = config.payload["environment"]
            environment = PIDTuningEnv(
                root,
                stage="joint",
                max_episode_steps=int(env_config["max_episode_steps"]),
                audit_interval=int(env_config["audit_interval"]),
                initial_perturbation=0.0,
                base_parameters=final_parameters_for_resume,
            )
            if int(state["global_timesteps_completed"]) <= 0:
                raise RuntimeError(
                    "completed stage state has no model checkpoint to finalize"
                )
            model = _load_checkpoint(
                output,
                state,
                environment,
                selected_device,
                tensorboard_log,
                expect_replay_buffer=bool(
                    config.payload["checkpoint"]["save_replay_buffer"]
                ),
            )

        final_parameters = np.asarray(
            state["curriculum_parameters"], dtype=np.float64
        )
        if environment is None or model is None:
            raise RuntimeError("formal training completed without a model")
        final_audit = None
        if state["completed_stages"]:
            last_stage = state["completed_stages"][-1]
            if last_stage["stage"] == "joint":
                last_report = json.loads(
                    (output / last_stage["report"]).read_text(encoding="utf-8")
                )
                if last_report.get("finalist_audit_scope") == "all_56_models":
                    selected_report = (
                        last_report["best_full_audit"]
                        if bool(last_report["accepted"])
                        else last_report["baseline_full_audit"]
                    )
                    if np.allclose(
                        np.asarray(
                            selected_report["parameters"], dtype=np.float64
                        ),
                        final_parameters,
                        rtol=1e-12,
                        atol=1e-14,
                    ):
                        final_audit = selected_report
        if final_audit is None:
            final_audit = audit_parameters(
                environment,
                final_parameters,
                "joint",
                full_time_domain=True,
            )
        eligible = bool(run_kind == "formal_training" and final_audit["safe"])
        accumulated_wall_time = (
            previous_wall_time + time.perf_counter() - session_started
        )
        candidate_path = output / "seed_candidate.npz"
        _atomic_save_npz(
            candidate_path,
            parameter_names=np.asarray(environment.parameter_space.names),
            parameters=final_parameters,
            normalized_parameters=environment.parameter_space.normalize(
                final_parameters
            ),
            seed=np.asarray(seed, dtype=np.int64),
            training_complete=np.asarray(True),
            eligible_for_selection=np.asarray(eligible),
            input_fingerprint=np.asarray(input_manifest["fingerprint"]),
        )
        _atomic_write_json(output / "seed_candidate_audit.json", final_audit)
        summary = {
            "schema_version": 1,
            "backend": "physics_v1",
            "status": "completed",
            "run_kind": run_kind,
            "seed": seed,
            "device": str(model.device),
            "total_timesteps": int(model.num_timesteps),
            "effective_stage_timesteps": effective_steps,
            "candidate": candidate_path.relative_to(output).as_posix(),
            "candidate_safe_over_all_56_models": bool(final_audit["safe"]),
            "candidate_joint_cost": float(final_audit["cost"]),
            "eligible_for_multi_seed_selection": eligible,
            "completed_stages": state["completed_stages"],
            "input_fingerprint": input_manifest["fingerprint"],
            "accumulated_wall_time_s": accumulated_wall_time,
            "mean_environment_steps_per_second": (
                float(model.num_timesteps) / accumulated_wall_time
                if accumulated_wall_time > 0.0
                else 0.0
            ),
            "completed_at_utc": utc_now(),
            "run_dir": str(output),
            "hardware_use_allowed": False,
        }
        _atomic_write_json(output / "seed_summary.json", summary)
        state["accumulated_wall_time_s"] = accumulated_wall_time
        state["status"] = "completed"
        state["stage"] = "completed"
        state["candidate_pool"] = []
        state["updated_at_utc"] = utc_now()
        _atomic_write_json(output / "trainer_state.json", state)
        return summary
    except BaseException as error:
        state["accumulated_wall_time_s"] = (
            previous_wall_time + time.perf_counter() - session_started
        )
        state["status"] = "failed"
        state["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "at_utc": utc_now(),
        }
        state["updated_at_utc"] = utc_now()
        if model is not None and int(state["stage_index"]) < len(STAGE_ORDER):
            state["global_timesteps_completed"] = int(model.num_timesteps)
            try:
                _save_resume_checkpoint(model, output, state, config)
            except BaseException as checkpoint_error:
                state["failure"]["checkpoint_error"] = str(checkpoint_error)
        _atomic_write_json(output / "trainer_state.json", state)
        raise
    finally:
        _close_model_environment(model)


def discover_seed_candidates(runs_root: Path) -> list[Path]:
    root = Path(runs_root).resolve()
    return sorted(root.glob("seed_*/seed_candidate.npz"))


def select_multi_seed_candidate(
    project_root: Path,
    candidate_paths: Iterable[Path],
    output_dir: Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"selection output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    evaluator = get_physics_controller_evaluator(root)
    time_evaluator = get_physics_time_domain_evaluator(root)
    position_target_hz = float(
        evaluator.space.metadata["position_design"]["target_crossover_hz"]
    )

    rows: list[dict[str, Any]] = []
    for raw_path in candidate_paths:
        path = Path(raw_path).resolve()
        with np.load(path, allow_pickle=False) as archive:
            parameter_names = tuple(
                str(value) for value in archive["parameter_names"]
            )
            parameters = np.asarray(archive["parameters"], dtype=np.float64)
            seed = int(np.asarray(archive["seed"]).item())
            complete = bool(np.asarray(archive["training_complete"]).item())
            eligible = bool(
                np.asarray(archive["eligible_for_selection"]).item()
            )
            fingerprint = str(np.asarray(archive["input_fingerprint"]).item())
        if parameter_names != evaluator.space.names:
            raise ValueError(f"candidate parameter order is invalid: {path}")
        evaluator.space.normalize(parameters)
        if not complete or not eligible:
            raise ValueError(
                f"candidate is not eligible for formal selection: {path}"
            )
        frequency = evaluator.audit(parameters)
        time_domain = time_evaluator.full_audit(parameters)
        safe = bool(
            frequency["safety"]["safe"] and time_report_safe(time_domain)
        )
        cost = stage_cost_v2(
            frequency,
            time_domain,
            "joint",
            position_target_hz,
        )
        audit = {
            "schema_version": 1,
            "backend": "physics_v1",
            "candidate": str(path),
            "seed": seed,
            "input_fingerprint": fingerprint,
            "safe": safe,
            "joint_cost": cost,
            "parameter_names": list(evaluator.space.names),
            "parameters": parameters.tolist(),
            "frequency": frequency,
            "time_domain": time_domain,
            "hardware_use_allowed": False,
        }
        audit_path = output / "audits" / f"seed_{seed}_audit.json"
        _atomic_write_json(audit_path, audit)
        rows.append(
            {
                "seed": seed,
                "candidate": str(path),
                "candidate_sha256": sha256_file(path),
                "input_fingerprint": fingerprint,
                "safe": safe,
                "joint_cost": float(cost),
                "parameters": parameters.tolist(),
                "audit": audit_path.relative_to(output).as_posix(),
            }
        )

    if len(rows) < 3:
        raise ValueError("multi-seed selection requires at least three candidates")
    seeds = [int(row["seed"]) for row in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError("multi-seed selection received duplicate seeds")
    fingerprints = {row["input_fingerprint"] for row in rows}
    if len(fingerprints) != 1:
        raise ValueError("candidate runs used different training inputs")
    safe_rows = [row for row in rows if row["safe"]]
    if not safe_rows:
        raise RuntimeError("no candidate is safe over all 56 audit models")
    ranked = sorted(
        rows,
        key=lambda row: (
            not bool(row["safe"]),
            float(row["joint_cost"]),
            int(row["seed"]),
        ),
    )
    selected = ranked[0]
    selected_parameters = np.asarray(selected["parameters"], dtype=np.float64)
    final_candidate = output / "final_candidate.npz"
    _atomic_save_npz(
        final_candidate,
        parameter_names=np.asarray(evaluator.space.names),
        parameters=selected_parameters,
        normalized_parameters=evaluator.space.normalize(selected_parameters),
        selected_seed=np.asarray(selected["seed"], dtype=np.int64),
        source_candidate_sha256=np.asarray(selected["candidate_sha256"]),
        input_fingerprint=np.asarray(selected["input_fingerprint"]),
        safe_over_all_56_models=np.asarray(True),
    )
    leaderboard = {
        "schema_version": 1,
        "backend": "physics_v1",
        "selection_policy": (
            "hard safety over all 56 training/validation models, "
            "then minimum joint validation cost"
        ),
        "candidate_count": len(rows),
        "safe_candidate_count": len(safe_rows),
        "ranking": ranked,
        "selected_seed": selected["seed"],
        "selected_source_candidate": selected["candidate"],
        "selected_joint_cost": selected["joint_cost"],
        "final_candidate": final_candidate.name,
        "created_at_utc": utc_now(),
        "hardware_use_allowed": False,
    }
    _atomic_write_json(output / "candidate_leaderboard.json", leaderboard)
    selected_audit = json.loads(
        (output / selected["audit"]).read_text(encoding="utf-8")
    )
    _atomic_write_json(output / "final_candidate_audit.json", selected_audit)
    return leaderboard
