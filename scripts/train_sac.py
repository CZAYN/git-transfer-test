from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elc_rl.sac_training import (  # noqa: E402
    StopController,
    load_formal_training_config,
    run_formal_training,
)


def _install_signal_handlers(controller: StopController) -> None:
    def handle(signum: int, _frame: object) -> None:
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        controller.request(name)
        print(
            f"Received {name}; stopping after the current environment step "
            "and writing a resumable checkpoint.",
            flush=True,
        )

    signal.signal(signal.SIGINT, handle)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run or resume one independent seed of formal physics-v1 SAC training."
        )
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-code-migration",
        action="store_true",
        help=(
            "with --resume, permit only the audited boundary/resume hotfix source "
            "files to differ; configuration and data changes remain forbidden"
        ),
    )
    parser.add_argument(
        "--engineering-check-steps-per-stage",
        type=int,
        default=None,
        help=(
            "replace every formal stage budget with a small positive value; "
            "the resulting candidate is marked ineligible for final selection"
        ),
    )
    arguments = parser.parse_args()
    if arguments.allow_code_migration and not arguments.resume:
        parser.error("--allow-code-migration requires --resume")

    project_root = arguments.project_root.resolve()
    config = load_formal_training_config(project_root, arguments.config)
    seed = config.seeds[0] if arguments.seed is None else int(arguments.seed)
    output_dir = (
        project_root / "outputs" / config.run_name / f"seed_{seed}"
        if arguments.output_dir is None
        else arguments.output_dir.resolve()
    )
    stop_controller = StopController()
    _install_signal_handlers(stop_controller)
    result = run_formal_training(
        project_root,
        config_path=config.path,
        seed=seed,
        run_dir=output_dir,
        device=arguments.device,
        resume=arguments.resume,
        allow_code_migration=arguments.allow_code_migration,
        engineering_steps_per_stage=(
            arguments.engineering_check_steps_per_stage
        ),
        stop_controller=stop_controller,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_kind": result["run_kind"],
                "seed": result["seed"],
                "total_timesteps": result.get(
                    "total_timesteps",
                    result.get("global_timesteps_completed"),
                ),
                "run_dir": str(output_dir),
                "eligible_for_multi_seed_selection": result.get(
                    "eligible_for_multi_seed_selection",
                    False,
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
