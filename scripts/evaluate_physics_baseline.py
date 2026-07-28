from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elc_rl.physics_evaluator import (  # noqa: E402
    compare_physics_to_measured_frf,
    get_physics_controller_evaluator,
    get_physics_time_domain_evaluator,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-time-audit",
        action="store_true",
        help="evaluate all 56 nonlinear models instead of the 4-model runtime audit",
    )
    args = parser.parse_args()
    frequency_evaluator = get_physics_controller_evaluator(PROJECT_ROOT)
    time_evaluator = get_physics_time_domain_evaluator(PROJECT_ROOT)
    parameters = frequency_evaluator.space.initial
    frequency = frequency_evaluator.audit(parameters)
    time_domain = (
        time_evaluator.full_audit(parameters)
        if args.full_time_audit
        else time_evaluator.audit(parameters)
    )
    comparison = compare_physics_to_measured_frf(PROJECT_ROOT)
    report = {
        "schema_version": 1,
        "backend": "physics",
        "time_audit_scope": "all_56_models" if args.full_time_audit else "runtime_4_models",
        "parameter_names": list(frequency_evaluator.space.names),
        "parameters": parameters.tolist(),
        "frequency": frequency,
        "time_domain": time_domain,
        "measured_frf_comparison": comparison,
    }
    output = PROJECT_ROOT / "outputs" / "physics_validation_report.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "frequency_safe": frequency["safety"]["safe"],
                "time_domain_safe": time_domain["safety"]["safe"],
                "frequency_models": frequency["evaluated_model_count"],
                "time_domain_models": time_domain["evaluated_model_count"],
                "report": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
