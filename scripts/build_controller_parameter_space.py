from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elc_rl.controller_parameters import (  # noqa: E402
    PHYSICS_PARAMETER_SPACE_JSON,
    PHYSICS_PARAMETER_SPACE_NPZ,
    build_physics_controller_parameter_space,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    payload = build_physics_controller_parameter_space(PROJECT_ROOT)
    json_name = PHYSICS_PARAMETER_SPACE_JSON
    npz_name = PHYSICS_PARAMETER_SPACE_NPZ
    json_path = PROJECT_ROOT / "data" / "processed" / json_name
    npz_path = PROJECT_ROOT / "data" / "processed" / npz_name
    summary = {
        "profile": "physics",
        "parameter_count": len(payload["parameters"]),
        "source_baselines": sum(
            parameter["source_kind"] == "excel_baseline"
            for parameter in payload["parameters"]
        ),
        "simulation_only_initials": sum(
            parameter["source_kind"] != "excel_baseline"
            for parameter in payload["parameters"]
        ),
        "json_sha256": _sha256(json_path),
        "npz_sha256": _sha256(npz_path),
    }
    print(json.dumps(summary, indent=2))
