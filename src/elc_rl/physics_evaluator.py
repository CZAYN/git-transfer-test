"""Frequency- and time-domain evaluators for the physics motor backend."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation_utils import (
    GAIN_MARGIN_CAP_DB,
    dobc_metrics,
    interpolate_pair,
    summarize_loop_rows,
    zero_crossing_locations,
)
from .controller_parameters import load_physics_controller_parameter_space
from .physics_motor_model import (
    MotorParameters,
    PhysicsMotorConfig,
    SimulationTrace,
    load_physics_motor_config,
    load_physics_motor_ensemble,
    simulate_scenario,
)
from .task_dataset import FRFTask, load_frf_task


PHYSICS_FREQUENCY_POINTS = 1024
PHYSICS_TRAIN_FREQUENCY_POINTS = 320
PHYSICS_FREQUENCY_LIMITS_HZ = {
    "current": (0.2, 2250.0),
    "speed": (0.02, 800.0),
    "position": (0.01, 250.0),
}


def _trim(values: np.ndarray) -> np.ndarray:
    result = np.trim_zeros(np.asarray(values, dtype=np.float64), trim="f")
    return result if result.size else np.asarray([0.0])


def _tf(
    numerator: np.ndarray | list[float], denominator: np.ndarray | list[float]
) -> tuple[np.ndarray, np.ndarray]:
    num = _trim(np.asarray(numerator, dtype=np.float64))
    den = _trim(np.asarray(denominator, dtype=np.float64))
    if den[0] == 0.0 or not np.isfinite(np.concatenate([num, den])).all():
        raise ValueError("invalid transfer function")
    return num / den[0], den / den[0]


def _series(
    *systems: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    numerator = np.asarray([1.0])
    denominator = np.asarray([1.0])
    for num, den in systems:
        numerator = np.convolve(numerator, num)
        denominator = np.convolve(denominator, den)
    return _tf(numerator, denominator)


def _feedback(
    forward: tuple[np.ndarray, np.ndarray],
    feedback: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    forward_num, forward_den = forward
    feedback_num, feedback_den = (
        _tf([1.0], [1.0]) if feedback is None else feedback
    )
    numerator = np.convolve(forward_num, feedback_den)
    denominator = np.polyadd(
        np.convolve(forward_den, feedback_den),
        np.convolve(forward_num, feedback_num),
    )
    return _tf(numerator, denominator)


def _response(
    system: tuple[np.ndarray, np.ndarray], frequency_hz: np.ndarray
) -> np.ndarray:
    numerator, denominator = system
    s = 1j * 2.0 * np.pi * np.asarray(frequency_hz, dtype=np.float64)
    return np.polyval(numerator, s) / np.polyval(denominator, s)


def _pid_tf(
    pid: tuple[float, float, float], filter_time_s: float
) -> tuple[np.ndarray, np.ndarray]:
    kp, ki, kd = pid
    return _tf(
        [kp * filter_time_s + kd, kp + ki * filter_time_s, ki],
        [filter_time_s, 1.0, 0.0],
    )


def _pid_values(parameters: np.ndarray, loop: str) -> tuple[float, float, float]:
    indices = {"position": (0, 1, 2), "speed": (3, 4, 5), "current": (8, 9, 10)}[
        loop
    ]
    return tuple(float(parameters[index]) for index in indices)  # type: ignore[return-value]


def physics_loop_transfers(
    config: PhysicsMotorConfig,
    motor: MotorParameters,
    controller_parameters: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Construct internally consistent nested-loop linear transfer functions."""

    filters = config.derivative_filter_s
    electrical = _tf(
        [1.0],
        np.convolve(
            [motor.current_delay_s, 1.0],
            [motor.inductance_h, motor.resistance_ohm],
        ),
    )
    current_controller = _pid_tf(
        _pid_values(controller_parameters, "current"), filters["current"]
    )
    current_open = _series(current_controller, electrical)
    current_closed = _feedback(current_open)

    mechanical = _tf(
        [motor.torque_constant_nm_per_a],
        [motor.inertia_kg_m2, motor.viscous_friction_nm_s_per_rad],
    )
    speed_sensor = _tf([1.0], [motor.speed_measurement_delay_s, 1.0])
    speed_controller = _pid_tf(
        _pid_values(controller_parameters, "speed"), filters["speed"]
    )
    current_to_speed = _series(current_closed, mechanical)
    speed_forward = _series(speed_controller, current_to_speed)
    speed_open = _series(speed_forward, speed_sensor)
    speed_actual_closed = _feedback(speed_forward, speed_sensor)

    position_sensor = _tf([1.0], [motor.position_measurement_delay_s, 1.0])
    position_controller = _pid_tf(
        _pid_values(controller_parameters, "position"), filters["position"]
    )
    speed_to_position_feedback = _series(
        speed_actual_closed, _tf([1.0], [1.0, 0.0]), position_sensor
    )
    position_open = _series(position_controller, speed_to_position_feedback)
    return {
        "current_open": current_open,
        "current_closed": current_closed,
        "speed_open": speed_open,
        "speed_actual_closed": speed_actual_closed,
        "position_open": position_open,
        "electrical_plant": electrical,
        "speed_measurement_plant": _series(mechanical, speed_sensor),
        "position_measurement_plant": _series(
            _tf([1.0], [1.0, 0.0]), position_sensor
        ),
    }


@lru_cache(maxsize=32)
def _frequency_grid(loop: str, points: int) -> np.ndarray:
    lower, upper = PHYSICS_FREQUENCY_LIMITS_HZ[loop]
    grid = np.geomspace(lower, upper, points)
    grid.setflags(write=False)
    return grid


def _evaluate_open_loop(
    loop: str,
    system: tuple[np.ndarray, np.ndarray],
    *,
    frequency_points: int,
) -> dict[str, float | bool]:
    frequency_hz = _frequency_grid(loop, frequency_points)
    open_loop = _response(system, frequency_hz)
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(open_loop), 1e-300))
    phase_deg = np.rad2deg(np.unwrap(np.angle(open_loop)))
    log_frequency = np.log10(frequency_hz)

    gain_crossings = zero_crossing_locations(log_frequency, magnitude_db)
    if gain_crossings:
        crossover_hz = float(gain_crossings[0][0])
        phase_margin_deg = float(
            min(
            180.0 + interpolate_pair(phase_deg, index, fraction)
                for _, index, fraction in gain_crossings
            )
        )
    else:
        lower, upper = PHYSICS_FREQUENCY_LIMITS_HZ[loop]
        crossover_hz = float(lower if magnitude_db[0] < 0.0 else upper)
        phase_margin_deg = -180.0

    gain_margin_candidates: list[float] = []
    target = -180.0
    while target >= float(np.min(phase_deg)):
        for _, index, fraction in zero_crossing_locations(
            log_frequency, phase_deg - target
        ):
            gain_margin_candidates.append(
            -interpolate_pair(magnitude_db, index, fraction)
            )
        target -= 360.0
    gain_margin_db = float(
        min(gain_margin_candidates)
        if gain_margin_candidates
        else GAIN_MARGIN_CAP_DB
    )

    sensitivity = 1.0 / (1.0 + open_loop)
    complementary = open_loop / (1.0 + open_loop)
    low_frequency_gain = float(abs(complementary[0]))
    threshold = low_frequency_gain / np.sqrt(2.0)
    bandwidth_indices = np.flatnonzero(np.abs(complementary) <= threshold)
    bandwidth_hz = float(
        frequency_hz[bandwidth_indices[0]]
        if bandwidth_indices.size
        else frequency_hz[-1]
    )
    numerator, denominator = system
    characteristic = _trim(np.polyadd(denominator, numerator))
    poles = np.roots(characteristic / np.max(np.abs(characteristic)))
    finite = bool(
        np.isfinite(open_loop.real).all()
        and np.isfinite(open_loop.imag).all()
        and np.isfinite(poles.real).all()
        and np.isfinite(poles.imag).all()
    )
    maximum_real_pole = float(np.max(poles.real))
    pole_stable = bool(maximum_real_pole < -1e-8)
    stable = bool(
        finite and pole_stable and phase_margin_deg > 0.0 and gain_margin_db > 0.0
    )
    return {
        "stable": stable,
        "pade_stable": pole_stable,
        "maximum_real_pole": maximum_real_pole,
        "crossover_hz": crossover_hz,
        "phase_margin_deg": phase_margin_deg,
        "gain_margin_db": gain_margin_db,
        "bandwidth_hz": bandwidth_hz,
        "low_frequency_gain": low_frequency_gain,
        "sensitivity_peak": float(np.max(np.abs(sensitivity))),
        "complementary_peak": float(np.max(np.abs(complementary))),
    }


def _physics_split(loop: str, role: str) -> str:
    if role == "validation":
        return f"{loop}_validation"
    return {
        "current": "current_reference",
        "speed": "speed_train",
        "position": "position_surrogate",
    }[loop]


def _physics_cost_and_safety(
    summaries: dict[str, dict[str, float | int]],
    dobc: dict[str, float],
    targets: dict[str, float],
) -> tuple[dict[str, float], dict[str, Any]]:
    core = {
        "current": summaries["current_reference"],
        "speed": summaries["speed_train"],
        "position": summaries["position_surrogate"],
    }
    crossovers = {
        loop: float(summary["crossover_hz_median"])
        for loop, summary in core.items()
    }
    current_speed_ratio = crossovers["current"] / crossovers["speed"]
    speed_position_ratio = crossovers["speed"] / crossovers["position"]
    crossover_cost = float(
        sum(abs(np.log(crossovers[loop] / targets[loop])) for loop in core)
    )
    margin_targets = {"current": 55.0, "speed": 55.0, "position": 55.0}
    margin_cost = float(
        sum(
            max(0.0, margin_targets[loop] - float(core[loop]["phase_margin_deg_worst"]))
            / margin_targets[loop]
            for loop in core
        )
    )
    sensitivity_cost = float(
        sum(
            max(0.0, float(summary["sensitivity_peak_worst"]) - 1.5)
            for summary in core.values()
        )
    )
    hierarchy_cost = float(
        max(0.0, 4.0 - current_speed_ratio) / 4.0
        + max(0.0, 3.0 - speed_position_ratio) / 3.0
    )
    uncertainty_cost = float(
        sum(
            np.log(
                float(summary["crossover_hz_max"])
                / float(summary["crossover_hz_min"])
            )
            for summary in core.values()
        )
    )
    dobc_cost = float(
        dobc["ideal_0p1_to_10hz_residual_rms"]
        + 0.15 * dobc["aggressiveness_proxy"]
    )
    stable_core = all(float(summary["stable_fraction"]) == 1.0 for summary in core.values())
    margins_safe = all(
        float(core[loop]["phase_margin_deg_worst"]) >= threshold
        for loop, threshold in {"current": 35.0, "speed": 35.0, "position": 40.0}.items()
    ) and all(float(summary["gain_margin_db_worst"]) >= 3.0 for summary in core.values())
    peaks_safe = all(
        float(summary["sensitivity_peak_worst"]) <= 2.5
        for summary in core.values()
    )
    validation_summaries = [
        summaries[name]
        for name in ("current_validation", "speed_validation", "position_validation")
        if name in summaries
    ]
    validation_safe = all(
        float(summary["stable_fraction"]) == 1.0
        and float(summary["phase_margin_deg_worst"]) >= 20.0
        and float(summary["gain_margin_db_worst"]) >= 3.0
        and float(summary["sensitivity_peak_worst"]) <= 3.0
        for summary in validation_summaries
    )
    hierarchy_safe = current_speed_ratio >= 4.0 and speed_position_ratio >= 3.0
    safe = bool(
        stable_core
        and margins_safe
        and peaks_safe
        and validation_safe
        and hierarchy_safe
    )
    components = {
        "crossover": crossover_cost,
        "phase_margin": margin_cost,
        "sensitivity": sensitivity_cost,
        "bandwidth_hierarchy": hierarchy_cost,
        "ensemble_uncertainty": uncertainty_cost,
        "dobc_idealized": dobc_cost,
        "unsafe": 0.0 if safe else 100.0,
    }
    components["total"] = float(
        crossover_cost
        + 2.0 * margin_cost
        + sensitivity_cost
        + 3.0 * hierarchy_cost
        + 0.25 * uncertainty_cost
        + 0.2 * dobc_cost
        + components["unsafe"]
    )
    safety = {
        "safe": safe,
        "stable_core_ensemble": stable_core,
        "minimum_margins_satisfied": margins_safe,
        "sensitivity_peaks_satisfied": peaks_safe,
        "validation_and_robustness_satisfied": validation_safe,
        "bandwidth_hierarchy_satisfied": hierarchy_safe,
        "current_to_speed_crossover_ratio": current_speed_ratio,
        "speed_to_position_crossover_ratio": speed_position_ratio,
        "thresholds": {
            "minimum_current_to_speed_ratio": 4.0,
            "minimum_speed_to_position_ratio": 3.0,
            "minimum_gain_margin_db": 3.0,
            "maximum_sensitivity_peak": 2.5,
            "validation_minimum_phase_margin_deg": 20.0,
        },
    }
    return components, safety


class PhysicsControllerEvaluator:
    """Frequency evaluator over coherent randomized physical motor instances."""

    backend = "physics"

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.config = load_physics_motor_config(self.project_root)
        self.space = load_physics_controller_parameter_space(self.project_root)
        self.ensemble = load_physics_motor_ensemble(self.project_root)
        self.training_indices = np.flatnonzero(
            self.ensemble["active_for_training"] == 1
        ).astype(np.int64)
        self.audit_indices = np.flatnonzero(
            self.ensemble["active_for_audit"] == 1
        ).astype(np.int64)
        if self.training_indices.size != 40 or self.audit_indices.size != 56:
            raise ValueError("physics ensemble split sizes are invalid")

    def sample_training_indices(self, rng: np.random.Generator) -> np.ndarray:
        return np.asarray([int(rng.choice(self.training_indices))], dtype=np.int64)

    def validate_sampled_indices(self, indices: np.ndarray) -> np.ndarray:
        values = np.asarray(indices, dtype=np.int64)
        if values.shape != (1,):
            raise ValueError("physics requires one coherent model index per episode")
        if int(values[0]) not in set(self.training_indices.tolist()):
            raise ValueError("physics sampled model is not in the training split")
        return values

    def model_ids(self, indices: np.ndarray) -> tuple[str, ...]:
        return tuple(
            str(self.ensemble["model_id"][int(index)])
            for index in np.asarray(indices, dtype=np.int64)
        )

    def motor(self, index: int) -> MotorParameters:
        return MotorParameters.from_array(self.ensemble["parameters"][int(index)])

    def sampled_frf_vector(self, indices: np.ndarray, task: FRFTask) -> np.ndarray:
        values = self.validate_sampled_indices(indices)
        motor = self.motor(int(values[0]))
        systems = physics_loop_transfers(self.config, motor, self.space.initial)
        speed_amplitude_index = int(np.flatnonzero(task.speed_amplitudes_mA == 100.0)[0])
        frequencies = {
            "current": 10.0 ** task.current_frf[:, 0],
            "speed": 10.0 ** task.speed_frf[speed_amplitude_index, :, 0],
            "position": 10.0 ** task.position_frf[:, 0],
        }
        plant_names = {
            "current": "electrical_plant",
            "speed": "speed_measurement_plant",
            "position": "position_measurement_plant",
        }
        parts: list[np.ndarray] = []
        for loop in ("current", "speed", "position"):
            response = _response(systems[plant_names[loop]], frequencies[loop])
            features = np.column_stack(
                [
                    20.0 * np.log10(np.maximum(np.abs(response), 1e-300)) / 40.0,
                    np.rad2deg(np.unwrap(np.angle(response))) / 180.0,
                ]
            )
            parts.append(features.reshape(-1))
        vector = np.concatenate(parts).astype(np.float64)
        if vector.shape != (96,) or not np.isfinite(vector).all():
            raise ValueError("physics sampled FRF vector is invalid")
        return vector

    def _evaluate(
        self,
        parameters: np.ndarray,
        indices: np.ndarray,
        *,
        frequency_points: int,
        mode: str,
        include_models: bool,
    ) -> dict[str, Any]:
        values = np.asarray(parameters, dtype=np.float64)
        self.space.normalize(values)
        grouped: dict[str, list[dict[str, Any]]] = {}
        rows: list[dict[str, Any]] = []
        for raw_index in np.asarray(indices, dtype=np.int64):
            index = int(raw_index)
            motor = self.motor(index)
            systems = physics_loop_transfers(self.config, motor, values)
            role = str(self.ensemble["role"][index])
            for loop, transfer_name in (
                ("current", "current_open"),
                ("speed", "speed_open"),
                ("position", "position_open"),
            ):
                split = _physics_split(loop, role)
                row: dict[str, Any] = {
                    "model_id": str(self.ensemble["model_id"][index]),
                    "loop": loop,
                    "role": role,
                    "split": split,
                    **_evaluate_open_loop(
                        loop,
                        systems[transfer_name],
                        frequency_points=frequency_points,
                    ),
                }
                rows.append(row)
                grouped.setdefault(split, []).append(row)
        summaries = {
            split: summarize_loop_rows(split_rows)
            for split, split_rows in grouped.items()
        }
        required = {"current_reference", "speed_train", "position_surrogate"}
        if not required.issubset(summaries):
            raise ValueError("physics evaluation is missing training loop summaries")
        dobc = dobc_metrics(values, self.space)
        targets = self.config.target_crossovers_hz
        cost, safety = _physics_cost_and_safety(summaries, dobc, targets)
        report: dict[str, Any] = {
            "schema_version": 2,
            "backend": self.backend,
            "task_id": self.space.task_id,
            "evaluation_mode": mode,
            "parameter_names": list(self.space.names),
            "parameters": values.tolist(),
            "evaluated_model_count": int(len(indices)),
            "evaluated_model_ids": list(self.model_ids(indices)),
            "targets_hz": {
                "current_reference": targets["current"],
                "speed_train": targets["speed"],
                "position_surrogate": targets["position"],
            },
            "splits": summaries,
            "dobc": dobc,
            "cost": cost,
            "safety": safety,
            "semantics": {
                "motor": "mentor physics model with coherent per-episode uncertainty",
                "current": "electrical plant plus current delay and filtered PID",
                "speed": "closed current loop, Kt/(J*s+B), speed feedback lag and filtered PID",
                "position": "closed actual-speed loop, integrator, position lag and filtered PID",
                "dobc": self.config.payload["controller_design"]["dobc"]["structure"],
                "measured_frf": "validation context only; not fitted into these model parameters",
            },
        }
        if include_models:
            report["models"] = rows
        return report

    def train(
        self,
        parameters: np.ndarray,
        sampled_indices: np.ndarray,
        *,
        include_models: bool = False,
    ) -> dict[str, Any]:
        indices = self.validate_sampled_indices(sampled_indices)
        return self._evaluate(
            parameters,
            indices,
            frequency_points=PHYSICS_TRAIN_FREQUENCY_POINTS,
            mode="train",
            include_models=include_models,
        )

    def audit(
        self, parameters: np.ndarray, *, include_models: bool = False
    ) -> dict[str, Any]:
        return self._evaluate(
            parameters,
            self.audit_indices,
            frequency_points=PHYSICS_FREQUENCY_POINTS,
            mode="audit",
            include_models=include_models,
        )


def _reference_metrics(trace: SimulationTrace) -> dict[str, float | bool]:
    target = float(trace.reference[-1])
    if abs(target) <= 1e-12:
        raise ValueError("reference scenario has a zero target")
    normalized = trace.output / target
    error = 1.0 - normalized
    ten = np.flatnonzero(normalized >= 0.1)
    ninety = np.flatnonzero(normalized >= 0.9)
    rise_time = (
        float(trace.time_s[ninety[0]] - trace.time_s[ten[0]])
        if ten.size and ninety.size and ninety[0] >= ten[0]
        else float(trace.time_s[-1])
    )
    outside = np.flatnonzero(np.abs(error) > 0.02)
    settled = bool(not outside.size or outside[-1] < len(error) - 1)
    settling_time = (
        0.0
        if not outside.size
        else float(trace.time_s[min(int(outside[-1]) + 1, len(error) - 1)])
    )
    return {
        "rise_time_s": rise_time,
        "settling_time_s": settling_time,
        "settled": settled,
        "overshoot_ratio": max(0.0, float(np.max(normalized) - 1.0)),
        "steady_state_error": abs(float(error[-1])),
        "iae": float(np.trapezoid(np.abs(error), trace.time_s)),
        "rms_error": float(np.sqrt(np.mean(error**2))),
        "output_peak": float(np.max(np.abs(normalized))),
    }


def _control_metrics(trace: SimulationTrace) -> dict[str, float]:
    slew = np.diff(trace.primary_control) / np.diff(trace.time_s)
    return {
        "control_peak": float(np.max(np.abs(trace.primary_control))),
        "control_rms": float(np.sqrt(np.mean(trace.primary_control**2))),
        "control_slew_peak": float(np.max(np.abs(slew))) if slew.size else 0.0,
    }


def _disturbance_metrics(
    trace: SimulationTrace, disturbance_start_s: float
) -> dict[str, float | bool]:
    start = int(np.searchsorted(trace.time_s, disturbance_start_s))
    time_s = trace.time_s[start:] - trace.time_s[start]
    deviation = trace.output[start:] - trace.reference[start:]
    absolute = np.abs(deviation)
    peak_index = int(np.argmax(absolute))
    peak = float(absolute[peak_index])
    threshold = max(0.02 * peak, 1e-6)
    outside = np.flatnonzero(absolute[peak_index:] > threshold)
    recovered = bool(not outside.size or peak_index + outside[-1] < len(absolute) - 1)
    recovery_time = (
        0.0
        if not outside.size
        else float(
            time_s[min(peak_index + int(outside[-1]) + 1, len(time_s) - 1)]
            - time_s[peak_index]
        )
    )
    return {
        "disturbance_peak": peak,
        "disturbance_iae": float(np.trapezoid(absolute, time_s)),
        "disturbance_recovery_time_s": recovery_time,
        "disturbance_recovered": recovered,
    }


def _safe_ratio(value: float, baseline: float) -> float:
    return float(value / max(abs(baseline), 1e-12))


class PhysicsTimeDomainEvaluator:
    """Nonlinear discrete-time evaluator with limits, anti-windup and DOBC."""

    backend = "physics"

    def __init__(self, frequency_evaluator: PhysicsControllerEvaluator) -> None:
        self.frequency_evaluator = frequency_evaluator
        self.project_root = frequency_evaluator.project_root
        self.config = frequency_evaluator.config
        self.space = frequency_evaluator.space
        self.ensemble = frequency_evaluator.ensemble
        self._baseline_cache: dict[tuple[int, str], dict[str, Any]] = {}
        validation = np.flatnonzero(self.ensemble["role"] == "validation").astype(
            np.int64
        )
        nominal = int(np.flatnonzero(self.ensemble["is_nominal"] == 1)[0])
        validation_probe = validation[
            np.linspace(0, validation.size - 1, 3, dtype=np.int64)
        ]
        self.runtime_audit_indices = np.concatenate(
            [np.asarray([nominal], dtype=np.int64), validation_probe]
        )

    def _raw_scenario(
        self, index: int, scenario: str, parameters: np.ndarray
    ) -> dict[str, Any]:
        trace = simulate_scenario(
            self.config,
            self.frequency_evaluator.motor(index),
            parameters,
            scenario,
        )
        limits = self.config.limits
        metrics: dict[str, Any] = {
            "time_domain_stable": bool(not trace.terminated),
            "maximum_real_pole": 0.0 if not trace.terminated else 1.0,
            "derivative_filter_time_s": self.config.derivative_filter_s.get(
                "speed" if scenario == "disturbance" else scenario, 0.0
            ),
            "saturation_count": trace.saturation_count,
            "voltage_peak_v": float(np.max(np.abs(trace.voltage_v))),
            "current_peak_a": float(np.max(np.abs(trace.current_a))),
            "speed_peak_rad_s": float(np.max(np.abs(trace.speed_rad_s))),
            "voltage_limit_ratio": float(
                np.max(np.abs(trace.voltage_v)) / float(limits["voltage_v"])
            ),
            "current_limit_ratio": float(
                np.max(np.abs(trace.current_a)) / float(limits["hard_current_a"])
            ),
            "speed_limit_ratio": float(
                np.max(np.abs(trace.speed_rad_s)) / float(limits["hard_speed_rad_s"])
            ),
        }
        if scenario == "disturbance":
            metrics.update(
                _disturbance_metrics(
                    trace, self.config.scenarios["disturbance_start_s"]
                )
            )
        else:
            metrics.update(_reference_metrics(trace))
            metrics.update(_control_metrics(trace))
        return metrics

    def _baseline(self, index: int, scenario: str) -> dict[str, Any]:
        key = (index, scenario)
        if key not in self._baseline_cache:
            self._baseline_cache[key] = self._raw_scenario(
                index, scenario, self.space.initial
            )
        return self._baseline_cache[key]

    def _model_metrics(
        self, index: int, loop: str, parameters: np.ndarray
    ) -> dict[str, Any]:
        metrics = self._raw_scenario(index, loop, parameters)
        is_baseline = np.array_equal(
            np.asarray(parameters, dtype=np.float64), self.space.initial
        )
        if is_baseline:
            self._baseline_cache[(index, loop)] = dict(metrics)
            baseline = metrics
        else:
            baseline = self._baseline(index, loop)
        for name in ("control_peak", "control_rms", "control_slew_peak", "iae"):
            metrics[f"{name}_ratio_to_baseline"] = _safe_ratio(
                float(metrics[name]), float(baseline[name])
            )
        if loop == "speed":
            disturbance = self._raw_scenario(index, "disturbance", parameters)
            if is_baseline:
                self._baseline_cache[(index, "disturbance")] = dict(disturbance)
                disturbance_baseline = disturbance
            else:
                disturbance_baseline = self._baseline(index, "disturbance")
            metrics.update(
                {
                    key: value
                    for key, value in disturbance.items()
                    if key.startswith("disturbance_")
                }
            )
            for name in ("disturbance_peak", "disturbance_iae"):
                metrics[f"{name}_ratio_to_baseline"] = _safe_ratio(
                    float(metrics[name]), float(disturbance_baseline[name])
                )
        return metrics

    @staticmethod
    def _summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
        def values(name: str) -> np.ndarray:
            return np.asarray([float(row[name]) for row in rows], dtype=np.float64)

        summary: dict[str, float | int] = {
            "model_count": len(rows),
            "stable_count": int(sum(bool(row["time_domain_stable"]) for row in rows)),
            "stable_fraction": float(
                np.mean([bool(row["time_domain_stable"]) for row in rows])
            ),
            "rise_time_s_median": float(np.median(values("rise_time_s"))),
            "settling_time_s_worst": float(np.max(values("settling_time_s"))),
            "overshoot_ratio_worst": float(np.max(values("overshoot_ratio"))),
            "steady_state_error_worst": float(np.max(values("steady_state_error"))),
            "iae_ratio_to_baseline_median": float(
                np.median(values("iae_ratio_to_baseline"))
            ),
            "control_peak_ratio_to_baseline_worst": float(
                np.max(values("control_peak_ratio_to_baseline"))
            ),
            "control_rms_ratio_to_baseline_worst": float(
                np.max(values("control_rms_ratio_to_baseline"))
            ),
            "control_slew_ratio_to_baseline_worst": float(
                np.max(values("control_slew_peak_ratio_to_baseline"))
            ),
            "voltage_limit_ratio_worst": float(np.max(values("voltage_limit_ratio"))),
            "current_limit_ratio_worst": float(np.max(values("current_limit_ratio"))),
            "speed_limit_ratio_worst": float(np.max(values("speed_limit_ratio"))),
            "saturation_count_worst": int(np.max(values("saturation_count"))),
        }
        if "disturbance_peak" in rows[0]:
            summary.update(
                {
                    "disturbance_peak_ratio_to_baseline_worst": float(
                        np.max(values("disturbance_peak_ratio_to_baseline"))
                    ),
                    "disturbance_iae_ratio_to_baseline_median": float(
                        np.median(values("disturbance_iae_ratio_to_baseline"))
                    ),
                    "disturbance_recovery_time_s_worst": float(
                        np.max(values("disturbance_recovery_time_s"))
                    ),
                }
            )
        return summary

    def evaluate(
        self,
        parameters: np.ndarray,
        indices: np.ndarray,
        *,
        mode: str,
        include_models: bool = False,
    ) -> dict[str, Any]:
        values = np.asarray(parameters, dtype=np.float64)
        self.space.normalize(values)
        grouped: dict[str, list[dict[str, Any]]] = {}
        rows: list[dict[str, Any]] = []
        for raw_index in np.asarray(indices, dtype=np.int64):
            index = int(raw_index)
            role = str(self.ensemble["role"][index])
            for loop in ("current", "speed", "position"):
                split = _physics_split(loop, role)
                row = {
                    "model_id": str(self.ensemble["model_id"][index]),
                    "loop": loop,
                    "role": role,
                    "split": split,
                    **self._model_metrics(index, loop, values),
                }
                rows.append(row)
                grouped.setdefault(split, []).append(row)
        summaries = {
            split: self._summary(split_rows) for split, split_rows in grouped.items()
        }
        core_names = ("current_reference", "speed_train", "position_surrogate")
        core_safe = all(
            float(summaries[name]["stable_fraction"]) == 1.0
            and float(summaries[name]["current_limit_ratio_worst"]) <= 1.001
            and float(summaries[name]["speed_limit_ratio_worst"]) <= 1.001
            and float(summaries[name]["voltage_limit_ratio_worst"]) <= 1.001
            for name in core_names
        )
        validation_names = [name for name in summaries if name.endswith("_validation")]
        validation_safe = all(
            float(summaries[name]["stable_fraction"]) == 1.0
            and float(summaries[name]["current_limit_ratio_worst"]) <= 1.001
            and float(summaries[name]["speed_limit_ratio_worst"]) <= 1.001
            and float(summaries[name]["voltage_limit_ratio_worst"]) <= 1.001
            for name in validation_names
        )
        report: dict[str, Any] = {
            "schema_version": 2,
            "backend": self.backend,
            "task_id": self.space.task_id,
            "evaluation_mode": mode,
            "evaluated_model_count": int(len(indices)),
            "evaluated_model_ids": list(self.frequency_evaluator.model_ids(indices)),
            "splits": summaries,
            "safety": {
                "safe": bool(core_safe and validation_safe),
                "core_limits_and_termination_safe": core_safe,
                "validation_limits_and_termination_safe": validation_safe,
            },
            "assumptions": {
                "integration_step_s": self.config.sample_period_s,
                "controller": "three filtered PID controllers with conditional anti-windup",
                "actuator_limits": "derived simulation-validity envelope, not hardware ratings",
                "disturbance": "positive resisting load-torque step at the mechanical shaft",
                "dobc": self.config.payload["controller_design"]["dobc"]["structure"],
                "friction": self.config.payload["friction_model"],
                "encoder_effects_during_reward": False,
            },
        }
        if include_models:
            report["models"] = rows
        return report

    def train(
        self,
        parameters: np.ndarray,
        sampled_indices: np.ndarray,
        *,
        include_models: bool = False,
    ) -> dict[str, Any]:
        indices = self.frequency_evaluator.validate_sampled_indices(sampled_indices)
        return self.evaluate(
            parameters, indices, mode="train", include_models=include_models
        )

    def audit(
        self, parameters: np.ndarray, *, include_models: bool = False
    ) -> dict[str, Any]:
        return self.evaluate(
            parameters,
            self.runtime_audit_indices,
            mode="runtime_audit",
            include_models=include_models,
        )

    def full_audit(
        self, parameters: np.ndarray, *, include_models: bool = False
    ) -> dict[str, Any]:
        """Run the expensive nonlinear audit over all 56 fixed motor models."""

        return self.evaluate(
            parameters,
            self.frequency_evaluator.audit_indices,
            mode="full_audit",
            include_models=include_models,
        )


@lru_cache(maxsize=4)
def _cached_physics_controller_evaluator(project_root: str) -> PhysicsControllerEvaluator:
    return PhysicsControllerEvaluator(Path(project_root))


def get_physics_controller_evaluator(project_root: Path) -> PhysicsControllerEvaluator:
    return _cached_physics_controller_evaluator(str(Path(project_root).resolve()))


@lru_cache(maxsize=4)
def _cached_physics_time_evaluator(project_root: str) -> PhysicsTimeDomainEvaluator:
    return PhysicsTimeDomainEvaluator(
        get_physics_controller_evaluator(Path(project_root))
    )


def get_physics_time_domain_evaluator(project_root: Path) -> PhysicsTimeDomainEvaluator:
    return _cached_physics_time_evaluator(str(Path(project_root).resolve()))


def clear_physics_evaluator_caches() -> None:
    _cached_physics_time_evaluator.cache_clear()
    _cached_physics_controller_evaluator.cache_clear()


def build_physics_baseline_evaluation(project_root: Path) -> dict[str, Any]:
    evaluator = get_physics_controller_evaluator(project_root)
    frequency = evaluator.audit(evaluator.space.initial, include_models=True)
    time_domain = get_physics_time_domain_evaluator(project_root).full_audit(
        evaluator.space.initial, include_models=True
    )
    report = {
        "schema_version": 1,
        "backend": "physics",
        "frequency": frequency,
        "time_domain": time_domain,
    }
    output = Path(project_root) / "outputs" / "physics_baseline_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def compare_physics_to_measured_frf(project_root: Path) -> dict[str, Any]:
    """Report model mismatch without fitting physics parameters to the FRF data."""

    root = Path(project_root).resolve()
    evaluator = get_physics_controller_evaluator(root)
    task = load_frf_task(root)
    systems = physics_loop_transfers(
        evaluator.config, evaluator.config.nominal, evaluator.space.initial
    )
    plant_names = {
        "current": "electrical_plant",
        "speed": "speed_measurement_plant",
        "position": "position_measurement_plant",
    }

    def metrics(loop: str, features: np.ndarray) -> dict[str, float | int]:
        frequency_hz = 10.0 ** features[:, 0]
        measured_magnitude_db = features[:, 1]
        measured_phase_deg = np.rad2deg(np.arctan2(features[:, 2], features[:, 3]))
        quality = features[:, 4].astype(bool)
        predicted = _response(systems[plant_names[loop]], frequency_hz)
        predicted_magnitude_db = 20.0 * np.log10(
            np.maximum(np.abs(predicted), 1e-300)
        )
        predicted_phase_deg = np.rad2deg(np.angle(predicted))
        magnitude_error = predicted_magnitude_db - measured_magnitude_db
        phase_error = (
            predicted_phase_deg - measured_phase_deg + 180.0
        ) % 360.0 - 180.0
        valid = quality & np.isfinite(magnitude_error) & np.isfinite(phase_error)
        if not np.any(valid):
            raise ValueError(f"no valid measured FRF values for {loop}")
        magnitude_offset_db = float(np.median(magnitude_error[valid]))
        shape_error = magnitude_error[valid] - magnitude_offset_db
        return {
            "point_count": int(features.shape[0]),
            "valid_point_count": int(np.count_nonzero(valid)),
            "raw_magnitude_rmse_db": float(
                np.sqrt(np.mean(magnitude_error[valid] ** 2))
            ),
            "best_constant_magnitude_offset_db": magnitude_offset_db,
            "gain_aligned_magnitude_shape_rmse_db": float(
                np.sqrt(np.mean(shape_error**2))
            ),
            "circular_phase_rmse_deg": float(
                np.sqrt(np.mean(phase_error[valid] ** 2))
            ),
            "circular_phase_mae_deg": float(np.mean(np.abs(phase_error[valid]))),
        }

    speed_results: dict[str, Any] = {}
    for index, amplitude in enumerate(task.speed_amplitudes_mA):
        speed_results[f"{float(amplitude):g}_mA"] = {
            "role": str(task.speed_roles[index]),
            **metrics("speed", task.speed_frf[index, :, :5]),
        }
    report: dict[str, Any] = {
        "schema_version": 1,
        "backend": "physics",
        "model_id": evaluator.config.payload["model_id"],
        "comparison_type": "held_out_diagnostic_no_parameter_fitting",
        "current": metrics("current", task.current_frf),
        "speed_by_amplitude": speed_results,
        "position": metrics("position", task.position_frf),
        "interpretation": {
            "raw_magnitude": "includes any unknown signal-unit or experiment gain offset",
            "gain_aligned_shape": "removes one median dB offset but does not alter model dynamics",
            "phase": "shortest circular phase difference at each measured frequency",
            "pass_fail": "not assigned because measured-loop signal semantics and scaling are not fully confirmed",
            "training_role": "this report validates model mismatch; it does not turn measured FRFs into the training plant",
        },
    }
    output = root / "outputs" / "physics_frf_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
