from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from elc_rl.tuning_env import (  # noqa: E402
    PIDTuningEnv,
    STAGE_ORDER,
    stage_cost_v2,
)


DEFAULT_SEED = 20260715
DEFAULT_STEPS_PER_STAGE = 128


@dataclass
class CandidateRecord:
    cost: float
    parameters: np.ndarray


class CandidatePoolCallback(BaseCallback):
    """Keep a small fast-path candidate pool for later full auditing."""

    def __init__(self, max_candidates: int = 3) -> None:
        super().__init__(verbose=0)
        self.max_candidates = max_candidates
        self.candidates: list[CandidateRecord] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not bool(info.get("fast_safe", False)):
                continue
            parameters = np.asarray(info["parameters"], dtype=np.float64)
            cost = float(info["stage_cost"])
            if any(
                np.allclose(parameters, record.parameters, rtol=1e-12, atol=1e-14)
                for record in self.candidates
            ):
                continue
            self.candidates.append(CandidateRecord(cost, parameters.copy()))
            self.candidates.sort(key=lambda record: record.cost)
            del self.candidates[self.max_candidates :]
        return True


def _time_safe(report: dict[str, Any]) -> bool:
    stable = bool(
        report["splits"]
        and all(
            float(summary["stable_fraction"]) == 1.0
            for summary in report["splits"].values()
        )
    )
    return bool(
        stable
        and (
            "safety" not in report
            or bool(report["safety"].get("safe", False))
        )
    )


def _audit(
    environment: PIDTuningEnv, parameters: np.ndarray, stage: str
) -> dict[str, Any]:
    frequency_report = environment.evaluator.audit(parameters)
    time_report = environment.time_evaluator.audit(parameters)
    frequency_safe = bool(frequency_report["safety"]["safe"])
    time_safe = _time_safe(time_report)
    return {
        "safe": bool(frequency_safe and time_safe),
        "frequency_safe": frequency_safe,
        "time_safe": time_safe,
        "cost": stage_cost_v2(
            frequency_report,
            time_report,
            stage,
            environment.position_target_hz,
        ),
        "frequency": {
            "cost": frequency_report["cost"],
            "safety": frequency_report["safety"],
            "splits": frequency_report["splits"],
            "dobc": frequency_report["dobc"],
        },
        "time_domain": {
            "splits": time_report["splits"],
            "assumptions": time_report["assumptions"],
        },
    }


def _parameter_mapping(environment: PIDTuningEnv, values: np.ndarray) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(environment.parameter_space.names, values)
    }


def train_staged_sac(
    project_root: Path,
    *,
    steps_per_stage: int,
    seed: int,
    output_dir: Path,
    device: str,
    stages: tuple[str, ...] = STAGE_ORDER,
    save_replay_buffer: bool = False,
) -> dict[str, Any]:
    if steps_per_stage <= 0:
        raise ValueError("steps_per_stage must be positive")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if not stages or any(stage not in STAGE_ORDER for stage in stages):
        raise ValueError(f"stages must be a non-empty subset of {STAGE_ORDER}")
    if len(set(stages)) != len(stages):
        raise ValueError("stages must not contain duplicates")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    curriculum_parameters: np.ndarray | None = None
    model: SAC | None = None
    stage_reports: list[dict[str, Any]] = []
    final_checkpoint: Path | None = None

    for stage_index, stage in enumerate(stages):
        environment = PIDTuningEnv(
            project_root,
            stage=stage,
            max_episode_steps=32,
            audit_interval=64,
            initial_perturbation=0.02,
            base_parameters=curriculum_parameters,
        )
        stage_seed = seed + 1009 * stage_index
        environment.action_space.seed(stage_seed)
        baseline_parameters = environment.base_parameters
        baseline_audit = _audit(environment, baseline_parameters, stage)
        print(
            f"[{stage_index + 1}/{len(stages)}] {stage}: "
            f"baseline audit cost={baseline_audit['cost']:.6f}",
            flush=True,
        )

        if model is None:
            model = SAC(
                "MultiInputPolicy",
                environment,
                learning_rate=3e-4,
                buffer_size=10_000,
                learning_starts=64,
                batch_size=64,
                tau=0.005,
                gamma=0.98,
                train_freq=(1, "step"),
                gradient_steps=1,
                policy_kwargs={"net_arch": [64, 64]},
                seed=seed,
                device=device,
                verbose=0,
            )
        else:
            previous_vector_environment = model.get_env()
            model.set_env(environment)
            if previous_vector_environment is not None:
                previous_vector_environment.close()

        callback = CandidatePoolCallback(max_candidates=3)
        model.learn(
            total_timesteps=steps_per_stage,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        checkpoint = output_dir / f"stage_{stage_index + 1:02d}_{stage}.zip"
        model.save(checkpoint)
        final_checkpoint = checkpoint

        audited_candidates: list[tuple[np.ndarray, dict[str, Any], float | None]] = [
            (baseline_parameters, baseline_audit, None)
        ]
        for record in callback.candidates:
            audited_candidates.append(
                (record.parameters, _audit(environment, record.parameters, stage), record.cost)
            )
        safe_candidates = [item for item in audited_candidates if item[1]["safe"]]
        selected_parameters, selected_audit, selected_fast_cost = min(
            safe_candidates,
            key=lambda item: float(item[1]["cost"]),
        )
        accepted = bool(
            float(selected_audit["cost"]) < float(baseline_audit["cost"]) - 1e-12
        )
        curriculum_parameters = (
            selected_parameters.copy() if accepted else baseline_parameters.copy()
        )
        stage_report = {
            "stage": stage,
            "stage_seed": stage_seed,
            "steps": steps_per_stage,
            "checkpoint": checkpoint.name,
            "candidate_pool_size": len(callback.candidates),
            "baseline_parameters": _parameter_mapping(environment, baseline_parameters),
            "baseline_audit": baseline_audit,
            "selected_fast_cost": selected_fast_cost,
            "selected_parameters": _parameter_mapping(environment, selected_parameters),
            "selected_audit": selected_audit,
            "accepted": accepted,
            "curriculum_parameters": _parameter_mapping(
                environment, curriculum_parameters
            ),
        }
        (output_dir / f"stage_{stage_index + 1:02d}_{stage}_report.json").write_text(
            json.dumps(stage_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        stage_reports.append(stage_report)
        print(
            f"[{stage_index + 1}/{len(stages)}] {stage}: "
            f"selected audit cost={selected_audit['cost']:.6f}, accepted={accepted}",
            flush=True,
        )

    if model is None or curriculum_parameters is None or final_checkpoint is None:
        raise RuntimeError("SAC smoke training did not initialize")
    replay_buffer_path = None
    if save_replay_buffer:
        replay_buffer_path = output_dir / "replay_buffer.pkl"
        model.save_replay_buffer(replay_buffer_path)
    final_environment = PIDTuningEnv(
        project_root,
        stage="joint",
        initial_perturbation=0.0,
        base_parameters=curriculum_parameters,
    )
    observation, _ = final_environment.reset(seed=seed, options={"perturb": False})
    loaded = SAC.load(final_checkpoint, env=final_environment, device=device)
    predicted_action, _ = loaded.predict(observation, deterministic=True)
    if predicted_action.shape != (11,) or not np.isfinite(predicted_action).all():
        raise RuntimeError("saved SAC checkpoint failed deterministic prediction")
    final_audit = _audit(final_environment, curriculum_parameters, "joint")
    np.savez_compressed(
        output_dir / "final_candidate.npz",
        parameter_names=np.asarray(final_environment.parameter_space.names),
        parameters=curriculum_parameters,
        normalized_parameters=final_environment.parameter_space.normalize(
            curriculum_parameters
        ),
        simulation_audit_safe=np.asarray(final_audit["safe"]),
    )
    summary = {
        "schema_version": 1,
        "run_kind": "physics-aware GPU SAC pipeline smoke test; not converged final training",
        "backend": "physics_v1",
        "stage_sequence": list(stages),
        "seed": seed,
        "steps_per_stage": steps_per_stage,
        "total_timesteps": int(model.num_timesteps),
        "replay_buffer_size": int(model.replay_buffer.size()),
        "replay_buffer_saved": replay_buffer_path is not None,
        "replay_buffer_path": (
            None if replay_buffer_path is None else replay_buffer_path.name
        ),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "device": str(model.device),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
        "stages": stage_reports,
        "final_checkpoint": final_checkpoint.name,
        "checkpoint_reload_prediction_valid": True,
        "final_parameters": _parameter_mapping(
            final_environment, curriculum_parameters
        ),
        "final_audit": final_audit,
        "hardware_use_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    final_environment.close()
    model.get_env().close()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run staged SAC GPU smoke training.")
    parser.add_argument(
        "--steps-per-stage", type=int, default=DEFAULT_STEPS_PER_STAGE
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_ORDER,
        default=list(STAGE_ORDER),
    )
    parser.add_argument(
        "--save-replay-buffer",
        action="store_true",
        help="persist the large replay buffer; disabled by default for smoke runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    arguments = parser.parse_args()
    output_dir = (
        arguments.output_dir
        if arguments.output_dir is not None
        else PROJECT_ROOT / "outputs" / "sac_smoke_physics_v1"
    )
    result = train_staged_sac(
        PROJECT_ROOT,
        steps_per_stage=arguments.steps_per_stage,
        seed=arguments.seed,
        output_dir=output_dir,
        device=arguments.device,
        stages=tuple(arguments.stages),
        save_replay_buffer=arguments.save_replay_buffer,
    )
    print(
        json.dumps(
            {
                "total_timesteps": result["total_timesteps"],
                "device": result["device"],
                "gpu_name": result["gpu_name"],
                "replay_buffer_size": result["replay_buffer_size"],
                "final_audit_safe": result["final_audit"]["safe"],
                "final_audit_cost": result["final_audit"]["cost"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
