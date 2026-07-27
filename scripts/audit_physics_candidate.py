from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elc_rl.physics_evaluator import (  # noqa: E402
    get_physics_controller_evaluator,
    get_physics_time_domain_evaluator,
)
from elc_rl.tuning_env import stage_cost_v2  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit a physics-v1 11-D candidate over all 56 motor models."
    )
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    candidate_path = arguments.candidate.resolve()
    with np.load(candidate_path, allow_pickle=False) as archive:
        parameters = np.asarray(archive["parameters"], dtype=np.float64)
    frequency_evaluator = get_physics_controller_evaluator(PROJECT_ROOT)
    time_evaluator = get_physics_time_domain_evaluator(PROJECT_ROOT)
    frequency_evaluator.space.normalize(parameters)
    frequency = frequency_evaluator.audit(parameters, include_models=True)
    time_domain = time_evaluator.full_audit(parameters, include_models=True)
    safe = bool(frequency["safety"]["safe"] and time_domain["safety"]["safe"])
    report = {
        "schema_version": 1,
        "backend": "physics_v1",
        "candidate": str(candidate_path),
        "parameter_names": list(frequency_evaluator.space.names),
        "parameters": parameters.tolist(),
        "safe_over_all_56_models": safe,
        "joint_cost": stage_cost_v2(
            frequency,
            time_domain,
            "joint",
            float(
                frequency_evaluator.space.metadata["position_design"][
                    "target_crossover_hz"
                ]
            ),
        ),
        "frequency": frequency,
        "time_domain": time_domain,
        "hardware_use_allowed": False,
    }
    output = (
        arguments.output.resolve()
        if arguments.output is not None
        else candidate_path.with_name(f"{candidate_path.stem}_full_audit.json")
    )
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "safe_over_all_56_models": safe,
                "joint_cost": report["joint_cost"],
                "frequency_models": frequency["evaluated_model_count"],
                "time_domain_models": time_domain["evaluated_model_count"],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
