from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elc_rl.physics_test_dataset import build_physics_test_ensemble  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and seal the candidate-free physics-v1 final-test set."
    )
    parser.add_argument(
        "--overwrite-unconsumed",
        action="store_true",
        help="rebuild only if no final-test report or consumption marker exists",
    )
    arguments = parser.parse_args()
    manifest = build_physics_test_ensemble(
        PROJECT_ROOT, overwrite=arguments.overwrite_unconsumed
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
