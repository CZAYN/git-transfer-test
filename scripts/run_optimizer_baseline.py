from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import scipy
from scipy.optimize import differential_evolution


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elc_rl.physics_evaluator import (  # noqa: E402
    get_physics_controller_evaluator,
    get_physics_time_domain_evaluator,
)
from elc_rl.tuning_env import combined_stage_cost  # noqa: E402


DEFAULT_SEED = 20260715


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
        and ("safety" not in report or bool(report["safety"].get("safe", False)))
    )


def run_optimizer_baseline(
    project_root: Path,
    *,
    seed: int,
    maxiter: int,
    popsize: int,
    output_dir: Path,
) -> dict[str, Any]:
    if maxiter <= 0 or popsize <= 0:
        raise ValueError("maxiter and popsize must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    frequency_evaluator = get_physics_controller_evaluator(project_root)
    time_evaluator = get_physics_time_domain_evaluator(project_root)
    space = frequency_evaluator.space
    position_target_hz = float(
        space.metadata["position_design"]["target_crossover_hz"]
    )
    rng = np.random.default_rng(seed)
    sampled_indices = frequency_evaluator.sample_training_indices(rng)
    evaluation_count = 0
    best_seen = float("inf")

    def fast_objective(normalized: np.ndarray) -> float:
        nonlocal evaluation_count, best_seen
        evaluation_count += 1
        try:
            parameters = space.denormalize(np.asarray(normalized, dtype=np.float64))
            frequency_report = frequency_evaluator.train(parameters, sampled_indices)
            time_report = time_evaluator.train(parameters, sampled_indices)
            cost = combined_stage_cost(
                frequency_report,
                time_report,
                "joint",
                position_target_hz,
            )
            safe = bool(frequency_report["safety"]["safe"]) and _time_safe(
                time_report
            )
            value = float(cost if safe else 1000.0 + cost)
        except (FloatingPointError, ValueError, OverflowError):
            value = 1e6
        best_seen = min(best_seen, value)
        return value

    generation = 0

    def progress(_candidate: np.ndarray, convergence: float) -> bool:
        nonlocal generation
        generation += 1
        print(
            f"generation={generation} evaluations={evaluation_count} "
            f"best_fast_cost={best_seen:.6f} convergence={convergence:.6g}",
            flush=True,
        )
        return False

    initial_normalized = space.normalize(space.initial)
    start = time.perf_counter()
    result = differential_evolution(
        fast_objective,
        bounds=[(-1.0, 1.0)] * 11,
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-3,
        atol=1e-4,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=seed,
        callback=progress,
        disp=False,
        polish=False,
        init="latinhypercube",
        x0=initial_normalized,
        workers=1,
        updating="immediate",
    )
    elapsed_s = time.perf_counter() - start

    def full_audit(parameters: np.ndarray) -> dict[str, Any]:
        frequency_report = frequency_evaluator.audit(parameters)
        time_report = time_evaluator.audit(parameters)
        frequency_safe = bool(frequency_report["safety"]["safe"])
        time_safe = _time_safe(time_report)
        return {
            "safe": bool(frequency_safe and time_safe),
            "frequency_safe": frequency_safe,
            "time_safe": time_safe,
            "cost": combined_stage_cost(
                frequency_report,
                time_report,
                "joint",
                position_target_hz,
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

    normalized_pool = [initial_normalized.copy(), np.asarray(result.x).copy()]
    order = np.argsort(np.asarray(result.population_energies))[:5]
    normalized_pool.extend(np.asarray(result.population)[order])
    unique_pool: list[np.ndarray] = []
    for values in normalized_pool:
        if not any(
            np.allclose(values, existing, rtol=1e-12, atol=1e-14)
            for existing in unique_pool
        ):
            unique_pool.append(np.asarray(values, dtype=np.float64).copy())

    audited_candidates = []
    for normalized in unique_pool:
        parameters = space.denormalize(np.clip(normalized, -1.0, 1.0))
        audited_candidates.append(
            {
                "fast_cost": fast_objective(normalized),
                "normalized_parameters": normalized.tolist(),
                "parameters": {
                    name: float(value) for name, value in zip(space.names, parameters)
                },
                "audit": full_audit(parameters),
            }
        )
    safe_candidates = [row for row in audited_candidates if row["audit"]["safe"]]
    selected = min(safe_candidates, key=lambda row: float(row["audit"]["cost"]))
    selected_parameters = np.asarray(
        [selected["parameters"][name] for name in space.names], dtype=np.float64
    )
    baseline = audited_candidates[0]

    sac_comparison = None
    sac_candidate_path = (
        project_root
        / "outputs"
        / "sac_smoke_physics"
        / "final_candidate.npz"
    )
    if sac_candidate_path.exists():
        with np.load(sac_candidate_path, allow_pickle=False) as data:
            sac_parameters = np.asarray(data["parameters"], dtype=np.float64)
        sac_comparison = {
            "parameters": {
                name: float(value) for name, value in zip(space.names, sac_parameters)
            },
            "audit": full_audit(sac_parameters),
        }

    report = {
        "schema_version": 1,
        "backend": "physics",
        "run_kind": (
            "deterministic differential-evolution simulation baseline; "
            "limited-budget, not final convergence"
        ),
        "algorithm": "scipy.optimize.differential_evolution best1bin",
        "scipy_version": scipy.__version__,
        "seed": seed,
        "maxiter": maxiter,
        "popsize_multiplier": popsize,
        "population_members": int(result.population.shape[0]),
        "sampled_model_ids": frequency_evaluator.model_ids(sampled_indices),
        "elapsed_s": elapsed_s,
        "objective_evaluations": evaluation_count,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_best_fast_cost": float(result.fun),
        "baseline": baseline,
        "audited_candidate_count": len(audited_candidates),
        "audited_candidates": audited_candidates,
        "selected": selected,
        "selected_improves_baseline": bool(
            float(selected["audit"]["cost"]) < float(baseline["audit"]["cost"])
        ),
        "sac_smoke_comparison": sac_comparison,
        "hardware_use_allowed": False,
    }
    (output_dir / "differential_evolution_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "differential_evolution_candidate.npz",
        parameter_names=np.asarray(space.names),
        parameters=selected_parameters,
        normalized_parameters=space.normalize(selected_parameters),
        simulation_audit_safe=np.asarray(selected["audit"]["safe"]),
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a traditional optimizer baseline for the 11D objective."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--maxiter", type=int, default=8)
    parser.add_argument("--popsize", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    arguments = parser.parse_args()
    output_dir = (
        arguments.output_dir
        if arguments.output_dir is not None
        else PROJECT_ROOT / "outputs" / "optimizer_baseline_physics"
    )
    baseline_report = run_optimizer_baseline(
        PROJECT_ROOT,
        seed=arguments.seed,
        maxiter=arguments.maxiter,
        popsize=arguments.popsize,
        output_dir=output_dir,
    )
    print(
        json.dumps(
            {
                "elapsed_s": baseline_report["elapsed_s"],
                "objective_evaluations": baseline_report["objective_evaluations"],
                "baseline_audit_cost": baseline_report["baseline"]["audit"]["cost"],
                "selected_audit_cost": baseline_report["selected"]["audit"]["cost"],
                "selected_audit_safe": baseline_report["selected"]["audit"]["safe"],
                "selected_improves_baseline": baseline_report[
                    "selected_improves_baseline"
                ],
                "sac_audit_cost": (
                    None
                    if baseline_report["sac_smoke_comparison"] is None
                    else baseline_report["sac_smoke_comparison"]["audit"]["cost"]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
