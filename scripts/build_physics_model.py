from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elc_rl.controller_parameters import (  # noqa: E402
    build_physics_controller_parameter_space,
    load_physics_controller_parameter_space,
)
from elc_rl.physics_motor_model import (  # noqa: E402
    build_physics_motor_ensemble,
    load_physics_motor_config,
)


if __name__ == "__main__":
    config = load_physics_motor_config(PROJECT_ROOT)
    ensemble = build_physics_motor_ensemble(PROJECT_ROOT)
    build_physics_controller_parameter_space(PROJECT_ROOT)
    space = load_physics_controller_parameter_space(PROJECT_ROOT)
    print(
        json.dumps(
            {
                "model_id": config.payload["model_id"],
                "sample_period_s": config.sample_period_s,
                "ensemble_models": int(ensemble["parameters"].shape[0]),
                "training_models": int(
                    (ensemble["active_for_training"] == 1).sum()
                ),
                "validation_models": int(
                    (ensemble["role"] == "validation").sum()
                ),
                "parameter_names": list(space.names),
                "initial_parameters": {
                    name: float(value)
                    for name, value in zip(space.names, space.initial)
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
