"""One-way final evaluation over the sealed physics-v1 test ensemble.

Nothing in this module is imported by the training environment.  The CLI
wrapper enforces candidate locking and write-once report generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .controller_parameters import load_physics_controller_parameter_space
from .physics_evaluator import (
    PHYSICS_FREQUENCY_POINTS,
    _control_metrics,
    _disturbance_metrics,
    _evaluate_open_loop,
    _reference_metrics,
    physics_loop_transfers,
)
from .physics_motor_model import (
    MotorParameters,
    SimulationTrace,
    load_physics_motor_config,
    simulate_scenario,
)
from .physics_test_dataset import (
    load_final_test_spec,
    load_physics_test_ensemble,
)


def _limit_metrics(trace: SimulationTrace, limits: dict[str, Any]) -> dict[str, Any]:
    return {
        "terminated": bool(trace.terminated),
        "voltage_limit_ratio": float(
            np.max(np.abs(trace.voltage_v)) / float(limits["voltage_v"])
        ),
        "current_limit_ratio": float(
            np.max(np.abs(trace.current_a)) / float(limits["hard_current_a"])
        ),
        "speed_limit_ratio": float(
            np.max(np.abs(trace.speed_rad_s)) / float(limits["hard_speed_rad_s"])
        ),
        "saturation_count": int(trace.saturation_count),
    }


def _standard_time_metrics(
    project_root: Path,
    motor: MotorParameters,
    parameters: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    config = load_physics_motor_config(project_root)
    scenario_spec = load_final_test_spec(project_root)["test_scenarios"]["standard"]
    overrides = {
        "current_reference_a": float(scenario_spec["current_reference_a"]),
        "speed_reference_rad_s": float(scenario_spec["speed_reference_rad_s"]),
        "position_reference_rad": float(scenario_spec["position_reference_rad"]),
        "load_torque_step_nm": float(scenario_spec["load_torque_step_nm"]),
    }
    loops: dict[str, Any] = {}
    traces: list[SimulationTrace] = []
    for offset, loop in enumerate(("current", "speed", "position")):
        trace = simulate_scenario(
            config,
            motor,
            parameters,
            loop,
            encoder_effects=bool(scenario_spec["encoder_effects"]),
            seed=seed + offset,
            scenario_overrides=overrides,
        )
        traces.append(trace)
        loops[loop] = {
            **_reference_metrics(trace),
            **_control_metrics(trace),
            **_limit_metrics(trace, config.limits),
        }
    disturbance = simulate_scenario(
        config,
        motor,
        parameters,
        "disturbance",
        encoder_effects=bool(scenario_spec["encoder_effects"]),
        seed=seed + 3,
        scenario_overrides=overrides,
    )
    traces.append(disturbance)
    return {
        "loops": loops,
        "disturbance": {
            **_disturbance_metrics(
                disturbance, config.scenarios["disturbance_start_s"]
            ),
            **_limit_metrics(disturbance, config.limits),
        },
        "all_scenarios_finite": bool(
            all(np.isfinite(trace.output).all() for trace in traces)
        ),
    }


def _ood_stress_metrics(
    project_root: Path,
    motor: MotorParameters,
    parameters: np.ndarray,
    *,
    model_index: int,
) -> dict[str, Any]:
    config = load_physics_motor_config(project_root)
    stress = load_final_test_spec(project_root)["test_scenarios"]["ood_stress"]
    seed = int(stress["encoder_noise_seed_base"]) + model_index
    trace = simulate_scenario(
        config,
        motor,
        parameters,
        "disturbance",
        encoder_effects=bool(stress["encoder_effects"]),
        seed=seed,
        scenario_overrides={
            "load_torque_step_nm": float(stress["load_torque_step_nm"])
        },
    )
    return {
        **_disturbance_metrics(trace, config.scenarios["disturbance_start_s"]),
        **_limit_metrics(trace, config.limits),
        "encoder_effects": bool(stress["encoder_effects"]),
        "seed": seed,
    }


def _frequency_metrics(
    project_root: Path, motor: MotorParameters, parameters: np.ndarray
) -> dict[str, Any]:
    config = load_physics_motor_config(project_root)
    systems = physics_loop_transfers(config, motor, parameters)
    loops = {
        loop: _evaluate_open_loop(
            loop, systems[transfer_name], frequency_points=PHYSICS_FREQUENCY_POINTS
        )
        for loop, transfer_name in (
            ("current", "current_open"),
            ("speed", "speed_open"),
            ("position", "position_open"),
        )
    }
    current_speed = float(loops["current"]["crossover_hz"]) / float(
        loops["speed"]["crossover_hz"]
    )
    speed_position = float(loops["speed"]["crossover_hz"]) / float(
        loops["position"]["crossover_hz"]
    )
    return {
        "loops": loops,
        "current_to_speed_crossover_ratio": current_speed,
        "speed_to_position_crossover_ratio": speed_position,
    }


def _hard_pass(
    frequency: dict[str, Any],
    standard_time: dict[str, Any],
    thresholds: dict[str, Any],
    phase_thresholds: dict[str, float],
    stress: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for loop, metrics in frequency["loops"].items():
        if not bool(metrics["stable"]):
            failures.append(f"{loop}:unstable")
        if float(metrics["phase_margin_deg"]) < float(phase_thresholds[loop]):
            failures.append(f"{loop}:phase_margin")
        if float(metrics["gain_margin_db"]) < float(
            thresholds["minimum_gain_margin_db"]
        ):
            failures.append(f"{loop}:gain_margin")
        if float(metrics["sensitivity_peak"]) > float(
            thresholds["maximum_sensitivity_peak"]
        ):
            failures.append(f"{loop}:sensitivity_peak")
    if float(frequency["current_to_speed_crossover_ratio"]) < float(
        thresholds["minimum_current_to_speed_crossover_ratio"]
    ):
        failures.append("hierarchy:current_to_speed")
    if float(frequency["speed_to_position_crossover_ratio"]) < float(
        thresholds["minimum_speed_to_position_crossover_ratio"]
    ):
        failures.append("hierarchy:speed_to_position")

    scenarios = list(standard_time["loops"].items()) + [
        ("disturbance", standard_time["disturbance"])
    ]
    if stress is not None:
        scenarios.append(("ood_stress", stress))
    if not standard_time["all_scenarios_finite"]:
        failures.append("time_domain:non_finite")
    for name, metrics in scenarios:
        if bool(metrics["terminated"]):
            failures.append(f"{name}:terminated")
        for metric_name, threshold_name in (
            ("voltage_limit_ratio", "maximum_voltage_limit_ratio"),
            ("current_limit_ratio", "maximum_current_limit_ratio"),
            ("speed_limit_ratio", "maximum_speed_limit_ratio"),
        ):
            if float(metrics[metric_name]) > float(thresholds[threshold_name]):
                failures.append(f"{name}:{metric_name}")
    return not failures, failures


def _id_performance_pass(
    standard_time: dict[str, Any], thresholds: dict[str, Any]
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for loop, metrics in standard_time["loops"].items():
        if float(metrics["settling_time_s"]) > float(
            thresholds["maximum_settling_time_s"][loop]
        ):
            failures.append(f"{loop}:settling_time")
        if float(metrics["overshoot_ratio"]) > float(
            thresholds["maximum_overshoot_ratio"][loop]
        ):
            failures.append(f"{loop}:overshoot")
        if float(metrics["steady_state_error"]) > float(
            thresholds["maximum_steady_state_error"]
        ):
            failures.append(f"{loop}:steady_state_error")
    disturbance = standard_time["disturbance"]
    if float(disturbance["disturbance_peak"]) > float(
        thresholds["maximum_standard_disturbance_peak_rad_s"]
    ):
        failures.append("disturbance:peak")
    if float(disturbance["disturbance_recovery_time_s"]) > float(
        thresholds["maximum_standard_disturbance_recovery_s"]
    ):
        failures.append("disturbance:recovery")
    return not failures, failures


def evaluate_locked_final_candidate(
    project_root: Path, parameters: np.ndarray
) -> dict[str, Any]:
    """Consume the sealed suite in memory; the CLI controls one-time writing."""

    root = Path(project_root).resolve()
    space = load_physics_controller_parameter_space(root)
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (11,):
        raise ValueError("final candidate must have shape (11,)")
    space.normalize(values)
    ensemble = load_physics_test_ensemble(root)
    spec = load_final_test_spec(root)
    acceptance = spec["acceptance_thresholds"]
    hard_thresholds = acceptance["hard_all_24_models"]
    phase_thresholds = acceptance["minimum_phase_margin_deg"]
    id_thresholds = acceptance["in_distribution_performance"]

    rows: list[dict[str, Any]] = []
    for index in range(ensemble["parameters"].shape[0]):
        motor = MotorParameters.from_array(ensemble["parameters"][index])
        group = str(ensemble["test_group"][index])
        frequency = _frequency_metrics(root, motor, values)
        standard_time = _standard_time_metrics(
            root, motor, values, seed=20260725 + index * 10
        )
        stress = (
            _ood_stress_metrics(root, motor, values, model_index=index)
            if group == "ood"
            else None
        )
        hard_pass, hard_failures = _hard_pass(
            frequency,
            standard_time,
            hard_thresholds,
            phase_thresholds,
            stress,
        )
        performance_pass, performance_failures = (
            _id_performance_pass(standard_time, id_thresholds)
            if group == "in_distribution"
            else (True, [])
        )
        rows.append(
            {
                "model_id": str(ensemble["model_id"][index]),
                "test_group": group,
                "scenario_profile": str(ensemble["scenario_profile"][index]),
                "hard_pass": hard_pass,
                "hard_failures": hard_failures,
                "id_performance_pass": (
                    performance_pass if group == "in_distribution" else None
                ),
                "id_performance_failures": performance_failures,
                "frequency": frequency,
                "standard_time": standard_time,
                "ood_stress": stress,
            }
        )

    id_rows = [row for row in rows if row["test_group"] == "in_distribution"]
    ood_rows = [row for row in rows if row["test_group"] == "ood"]
    hard_pass_count = sum(bool(row["hard_pass"]) for row in rows)
    id_performance_pass_count = sum(
        bool(row["id_performance_pass"]) for row in id_rows
    )
    overall_pass = bool(
        hard_pass_count == 24 and id_performance_pass_count == len(id_rows)
    )
    return {
        "schema_version": 1,
        "test_suite_id": spec["test_suite_id"],
        "backend": "physics_v1",
        "parameter_names": list(space.names),
        "parameters": values.tolist(),
        "overall_pass": overall_pass,
        "summary": {
            "model_count": len(rows),
            "hard_pass_count": hard_pass_count,
            "hard_fail_count": len(rows) - hard_pass_count,
            "in_distribution_model_count": len(id_rows),
            "in_distribution_performance_pass_count": id_performance_pass_count,
            "ood_model_count": len(ood_rows),
            "ood_hard_pass_count": sum(bool(row["hard_pass"]) for row in ood_rows),
        },
        "acceptance_thresholds": acceptance,
        "models": rows,
        "policy": {
            "test_results_must_not_be_used_to_retrain_or_select_another_candidate": True,
            "hardware_use_allowed": False,
            "ood_performance_is_reported_but_not_a_soft-selection_signal": True,
        },
    }
