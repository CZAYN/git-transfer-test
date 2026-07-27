"""Mentor-supplied physics motor model and deterministic simulation utilities.

The model is a single-axis, FOC-decoupled servo approximation.  It keeps the
electrical, mechanical, measurement-delay and saturation semantics explicit so
that the reinforcement-learning environment does not need to infer them from
measured frequency responses.  Measured FRFs remain independent validation
data.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


MODEL_PARAMETER_NAMES = (
    "inductance_h",
    "resistance_ohm",
    "current_delay_s",
    "friction_stiffness_nm_per_rad",
    "friction_damping_nm_s_per_rad",
    "viscous_friction_nm_s_per_rad",
    "stribeck_velocity_rad_s",
    "speed_measurement_delay_s",
    "inertia_kg_m2",
    "torque_constant_nm_per_a",
    "position_measurement_delay_s",
)

PHYSICS_CONFIG_RELATIVE_PATH = Path("config") / "motor_physics_v1.json"
PHYSICS_ENSEMBLE_RELATIVE_PATH = (
    Path("data") / "processed" / "physics_motor_ensemble_v1.npz"
)
PHYSICS_ENSEMBLE_MANIFEST_RELATIVE_PATH = (
    Path("data") / "processed" / "physics_motor_ensemble_v1_manifest.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MotorParameters:
    """One physical motor instance, expressed entirely in SI units."""

    inductance_h: float
    resistance_ohm: float
    current_delay_s: float
    friction_stiffness_nm_per_rad: float
    friction_damping_nm_s_per_rad: float
    viscous_friction_nm_s_per_rad: float
    stribeck_velocity_rad_s: float
    speed_measurement_delay_s: float
    inertia_kg_m2: float
    torque_constant_nm_per_a: float
    position_measurement_delay_s: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "MotorParameters":
        return cls(**{name: float(values[name]) for name in MODEL_PARAMETER_NAMES})

    @classmethod
    def from_array(cls, values: np.ndarray) -> "MotorParameters":
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(MODEL_PARAMETER_NAMES),):
            raise ValueError("motor parameter vector has an invalid shape")
        return cls(**dict(zip(MODEL_PARAMETER_NAMES, array.tolist())))

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [getattr(self, name) for name in MODEL_PARAMETER_NAMES], dtype=np.float64
        )

    def validate(self) -> None:
        values = self.as_array()
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("all physics motor parameters must be finite and positive")

    @property
    def electrical_time_constant_s(self) -> float:
        return self.inductance_h / self.resistance_ohm

    @property
    def mechanical_time_constant_s(self) -> float:
        return self.inertia_kg_m2 / self.viscous_friction_nm_s_per_rad


@dataclass(frozen=True)
class PhysicsMotorConfig:
    """Validated configuration plus its nominal motor and derived sections."""

    project_root: Path
    path: Path
    payload: dict[str, Any]
    nominal: MotorParameters

    @property
    def sample_period_s(self) -> float:
        return float(self.payload["sample_period_s"])

    @property
    def uncertainty_fraction(self) -> dict[str, float]:
        return {
            name: float(self.payload["uncertainty_fraction"][name])
            for name in MODEL_PARAMETER_NAMES
        }

    @property
    def limits(self) -> dict[str, Any]:
        return dict(self.payload["simulation_limits"])

    @property
    def scenarios(self) -> dict[str, float]:
        return {
            key: float(value) for key, value in self.payload["scenarios"].items()
        }

    @property
    def derivative_filter_s(self) -> dict[str, float]:
        return {
            loop: float(value)
            for loop, value in self.payload["controller_design"][
                "derivative_filter_s"
            ].items()
        }

    @property
    def target_crossovers_hz(self) -> dict[str, float]:
        design = self.payload["controller_design"]
        return {
            "current": float(design["current_crossover_hz"]),
            "speed": float(design["speed_crossover_hz"]),
            "position": float(design["position_crossover_hz"]),
        }

    def validate(self) -> None:
        if int(self.payload["schema_version"]) != 1:
            raise ValueError("unsupported physics motor configuration schema")
        if self.payload["model_id"] != "mentor_motor_physics_v1":
            raise ValueError("unexpected physics motor model_id")
        if self.sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be positive")
        self.nominal.validate()
        for name, fraction in self.uncertainty_fraction.items():
            if not 0.0 <= fraction <= 0.5:
                raise ValueError(f"invalid uncertainty fraction for {name}")
        targets = self.target_crossovers_hz
        if not targets["current"] > targets["speed"] > targets["position"] > 0.0:
            raise ValueError("controller crossover targets must be strictly nested")
        limits = self.limits
        if not (
            float(limits["hard_current_a"])
            >= float(limits["training_current_a"])
            > 0.0
        ):
            raise ValueError("current simulation limits are inconsistent")
        if not (
            float(limits["termination_speed_rad_s"])
            > float(limits["hard_speed_rad_s"])
            >= float(limits["training_speed_rad_s"])
            > 0.0
        ):
            raise ValueError("speed simulation limits are inconsistent")


@lru_cache(maxsize=4)
def _cached_config(project_root: str) -> PhysicsMotorConfig:
    root = Path(project_root)
    path = root / PHYSICS_CONFIG_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = PhysicsMotorConfig(
        project_root=root,
        path=path,
        payload=payload,
        nominal=MotorParameters.from_mapping(payload["nominal_parameters"]),
    )
    config.validate()
    return config


def load_physics_motor_config(project_root: Path) -> PhysicsMotorConfig:
    """Load the immutable physics-v1 configuration for one project root."""

    return _cached_config(str(Path(project_root).resolve()))


def clear_physics_motor_config_cache() -> None:
    _cached_config.cache_clear()


def build_physics_motor_ensemble(project_root: Path) -> dict[str, np.ndarray]:
    """Build a deterministic 40-train/16-validation physical model ensemble."""

    root = Path(project_root).resolve()
    config = load_physics_motor_config(root)
    ensemble_spec = config.payload["ensemble"]
    training_count = int(ensemble_spec["training_models"])
    validation_count = int(ensemble_spec["validation_models"])
    model_count = training_count + validation_count
    if training_count <= 0 or validation_count <= 0:
        raise ValueError("physics ensemble needs training and validation models")

    rng = np.random.default_rng(int(ensemble_spec["seed"]))
    nominal = config.nominal.as_array()
    uncertainty = np.asarray(
        [config.uncertainty_fraction[name] for name in MODEL_PARAMETER_NAMES],
        dtype=np.float64,
    )
    unit = np.empty((model_count, nominal.size), dtype=np.float64)
    for column in range(nominal.size):
        strata = (rng.permutation(model_count) + rng.random(model_count)) / model_count
        unit[:, column] = 2.0 * strata - 1.0
    parameters = nominal[None, :] * (1.0 + unit * uncertainty[None, :])
    parameters[0] = nominal

    role = np.asarray(
        ["train"] * training_count + ["validation"] * validation_count
    )
    model_id = np.asarray(
        [
            "physics_v1_nominal"
            if index == 0
            else f"physics_v1_{role[index]}_{index:03d}"
            for index in range(model_count)
        ]
    )
    result = {
        "schema_version": np.asarray(1, dtype=np.int16),
        "parameter_names": np.asarray(MODEL_PARAMETER_NAMES),
        "parameters": parameters,
        "model_id": model_id,
        "role": role,
        "is_nominal": (np.arange(model_count) == 0).astype(np.int8),
        "active_for_training": (role == "train").astype(np.int8),
        "active_for_audit": np.ones(model_count, dtype=np.int8),
    }
    output = root / PHYSICS_ENSEMBLE_RELATIVE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    manifest = {
        "schema_version": 1,
        "model_id": config.payload["model_id"],
        "config_path": str(PHYSICS_CONFIG_RELATIVE_PATH).replace("\\", "/"),
        "config_sha256": _sha256(config.path),
        "ensemble_path": str(PHYSICS_ENSEMBLE_RELATIVE_PATH).replace("\\", "/"),
        "model_count": model_count,
        "training_models": training_count,
        "validation_models": validation_count,
        "seed": int(ensemble_spec["seed"]),
        "parameter_names": list(MODEL_PARAMETER_NAMES),
        "uncertainty_fraction": config.uncertainty_fraction,
        "nominal_model_included": True,
        "measured_frf_used_for_fitting": False,
        "measured_frf_role": config.payload["provenance"]["measured_frf_role"],
    }
    manifest_path = root / PHYSICS_ENSEMBLE_MANIFEST_RELATIVE_PATH
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def load_physics_motor_ensemble(project_root: Path) -> dict[str, np.ndarray]:
    """Load and validate the generated physical-model ensemble."""

    root = Path(project_root).resolve()
    path = root / PHYSICS_ENSEMBLE_RELATIVE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run scripts/build_physics_model.py first"
        )
    with np.load(path, allow_pickle=False) as archive:
        ensemble = {name: archive[name] for name in archive.files}
    if tuple(ensemble["parameter_names"].tolist()) != MODEL_PARAMETER_NAMES:
        raise ValueError("physics ensemble parameter order is invalid")
    parameters = np.asarray(ensemble["parameters"], dtype=np.float64)
    if parameters.ndim != 2 or parameters.shape[1] != len(MODEL_PARAMETER_NAMES):
        raise ValueError("physics ensemble matrix has an invalid shape")
    if not np.isfinite(parameters).all() or np.any(parameters <= 0.0):
        raise ValueError("physics ensemble contains invalid values")
    if np.count_nonzero(ensemble["is_nominal"]) != 1:
        raise ValueError("physics ensemble must contain exactly one nominal model")
    return ensemble


@dataclass
class _FilteredPID:
    kp: float
    ki: float
    kd: float
    filter_time_s: float
    sample_period_s: float
    integral: float = 0.0
    derivative: float = 0.0
    previous_error: float = 0.0

    def update(self, error: float, lower: float, upper: float) -> tuple[float, float]:
        dt = self.sample_period_s
        raw_derivative = (error - self.previous_error) / dt
        alpha = dt / (self.filter_time_s + dt)
        self.derivative += alpha * (raw_derivative - self.derivative)
        candidate_integral = self.integral + self.ki * error * dt
        unsaturated = self.kp * error + candidate_integral + self.kd * self.derivative
        saturated = float(np.clip(unsaturated, lower, upper))
        drives_further_into_saturation = (
            (unsaturated > upper and error > 0.0)
            or (unsaturated < lower and error < 0.0)
        )
        if drives_further_into_saturation:
            unsaturated = self.kp * error + self.integral + self.kd * self.derivative
            saturated = float(np.clip(unsaturated, lower, upper))
        else:
            self.integral = candidate_integral
        self.previous_error = error
        return saturated, float(unsaturated)


@dataclass(frozen=True)
class SimulationTrace:
    scenario: str
    time_s: np.ndarray
    output: np.ndarray
    reference: np.ndarray
    primary_control: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    speed_rad_s: np.ndarray
    position_rad: np.ndarray
    current_reference_a: np.ndarray
    speed_reference_rad_s: np.ndarray
    load_torque_nm: np.ndarray
    saturation_count: int
    terminated: bool


def wrap_angle(angle_rad: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(angle_rad) + np.pi) % (2.0 * np.pi) - np.pi


def _controller_vector(parameters: np.ndarray) -> dict[str, tuple[float, float, float]]:
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (11,) or not np.isfinite(values).all():
        raise ValueError("controller parameter vector must be finite with shape (11,)")
    return {
        "position": tuple(values[0:3]),
        "speed": tuple(values[3:6]),
        "current": tuple(values[8:11]),
    }


def simulate_scenario(
    config: PhysicsMotorConfig,
    motor: MotorParameters,
    controller_parameters: np.ndarray,
    scenario: str,
    *,
    encoder_effects: bool = False,
    seed: int = 0,
    scenario_overrides: Mapping[str, float] | None = None,
) -> SimulationTrace:
    """Run one deterministic cascaded PIDF/DOBC scenario at the 200 us step."""

    if scenario not in {"current", "speed", "position", "disturbance"}:
        raise ValueError(f"unknown simulation scenario: {scenario}")
    motor.validate()
    gains = _controller_vector(controller_parameters)
    dt = config.sample_period_s
    scenario_spec = config.scenarios
    if scenario_overrides is not None:
        allowed_overrides = {
            "current_reference_a",
            "speed_reference_rad_s",
            "position_reference_rad",
            "load_torque_step_nm",
            "disturbance_start_s",
        }
        unknown = set(scenario_overrides).difference(allowed_overrides)
        if unknown:
            raise ValueError(f"unsupported scenario overrides: {sorted(unknown)}")
        for name, raw_value in scenario_overrides.items():
            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError(f"non-finite scenario override: {name}")
            if name in {"load_torque_step_nm", "disturbance_start_s"} and value < 0.0:
                raise ValueError(f"scenario override must be non-negative: {name}")
            scenario_spec[name] = value
    duration_key = {
        "current": "current_duration_s",
        "speed": "speed_duration_s",
        "position": "position_duration_s",
        "disturbance": "disturbance_duration_s",
    }[scenario]
    points = int(round(scenario_spec[duration_key] / dt)) + 1
    time_s = np.arange(points, dtype=np.float64) * dt
    filters = config.derivative_filter_s
    position_pid = _FilteredPID(*gains["position"], filters["position"], dt)
    speed_pid = _FilteredPID(*gains["speed"], filters["speed"], dt)
    current_pid = _FilteredPID(*gains["current"], filters["current"], dt)

    limits = config.limits
    current_limit = float(limits["hard_current_a"])
    speed_command_limit = float(limits["training_speed_rad_s"])
    voltage_limit = float(limits["voltage_v"])
    termination_speed = float(limits["termination_speed_rad_s"])
    dobc_gain = float(controller_parameters[6])
    dobc_time = float(controller_parameters[7])
    dobc_alpha = dt / (dobc_time + dt)
    nominal_inverse = config.nominal
    encoder_lsb = float(config.payload["encoder"]["position_lsb_rad"])
    rng = np.random.default_rng(seed)

    output = np.zeros(points, dtype=np.float64)
    reference = np.zeros(points, dtype=np.float64)
    primary_control = np.zeros(points, dtype=np.float64)
    voltage = np.zeros(points, dtype=np.float64)
    current = np.zeros(points, dtype=np.float64)
    speed = np.zeros(points, dtype=np.float64)
    position = np.zeros(points, dtype=np.float64)
    current_reference = np.zeros(points, dtype=np.float64)
    speed_reference = np.zeros(points, dtype=np.float64)
    load_torque = np.zeros(points, dtype=np.float64)

    i_a = 0.0
    omega = 0.0
    theta = 0.0
    applied_voltage = 0.0
    omega_feedback = 0.0
    theta_feedback = 0.0
    previous_theta_observed = 0.0
    previous_omega_feedback = 0.0
    estimated_load = 0.0
    saturation_count = 0
    terminated = False

    current_delay_alpha = dt / (motor.current_delay_s + dt)
    speed_delay_alpha = dt / (motor.speed_measurement_delay_s + dt)
    position_delay_alpha = dt / (motor.position_measurement_delay_s + dt)

    for index, now in enumerate(time_s):
        theta_feedback += position_delay_alpha * (theta - theta_feedback)
        if encoder_effects:
            theta_observed = theta_feedback + rng.uniform(-0.5, 0.5) * encoder_lsb
            theta_observed = float(np.round(theta_observed / encoder_lsb) * encoder_lsb)
            encoder_speed = (theta_observed - previous_theta_observed) / dt
            omega_feedback += speed_delay_alpha * (
                encoder_speed - omega_feedback
            )
        else:
            theta_observed = theta_feedback
            omega_feedback += speed_delay_alpha * (omega - omega_feedback)
        previous_theta_observed = theta_observed

        requested_current = scenario_spec["current_reference_a"]
        requested_speed = scenario_spec["speed_reference_rad_s"]
        requested_position = scenario_spec["position_reference_rad"]

        if scenario == "current":
            iq_reference = requested_current
            speed_command = 0.0
            reference[index] = requested_current
        else:
            if scenario == "position":
                position_error = float(wrap_angle(requested_position - theta_observed))
                speed_command, speed_unsaturated = position_pid.update(
                    position_error, -speed_command_limit, speed_command_limit
                )
                saturation_count += int(not np.isclose(speed_command, speed_unsaturated))
                reference[index] = requested_position
            else:
                speed_command = requested_speed
                reference[index] = requested_speed

            speed_error = speed_command - omega_feedback
            iq_pid, iq_unsaturated = speed_pid.update(
                speed_error, -current_limit, current_limit
            )
            saturation_count += int(not np.isclose(iq_pid, iq_unsaturated))
            measured_acceleration = (omega_feedback - previous_omega_feedback) / dt
            raw_load_estimate = nominal_inverse.torque_constant_nm_per_a * i_a - (
                nominal_inverse.inertia_kg_m2 * measured_acceleration
                + nominal_inverse.viscous_friction_nm_s_per_rad * omega_feedback
            )
            estimated_load += dobc_alpha * (raw_load_estimate - estimated_load)
            iq_reference_unsaturated = iq_pid + (
                dobc_gain / nominal_inverse.torque_constant_nm_per_a
            ) * estimated_load
            iq_reference = float(
                np.clip(iq_reference_unsaturated, -current_limit, current_limit)
            )
            saturation_count += int(
                not np.isclose(iq_reference, iq_reference_unsaturated)
            )

        current_error = iq_reference - i_a
        voltage_command, voltage_unsaturated = current_pid.update(
            current_error, -voltage_limit, voltage_limit
        )
        saturation_count += int(not np.isclose(voltage_command, voltage_unsaturated))
        applied_voltage += current_delay_alpha * (voltage_command - applied_voltage)

        di_dt = (
            applied_voltage - motor.resistance_ohm * i_a
        ) / motor.inductance_h
        i_a += dt * di_dt
        i_a = float(np.clip(i_a, -1.25 * current_limit, 1.25 * current_limit))

        active_load = 0.0
        if scenario == "disturbance" and now >= scenario_spec["disturbance_start_s"]:
            active_load = scenario_spec["load_torque_step_nm"]
        domega_dt = (
            motor.torque_constant_nm_per_a * i_a
            - motor.viscous_friction_nm_s_per_rad * omega
            - active_load
        ) / motor.inertia_kg_m2
        omega += dt * domega_dt
        theta += dt * omega

        current[index] = i_a
        speed[index] = omega
        position[index] = theta
        current_reference[index] = iq_reference
        speed_reference[index] = speed_command
        load_torque[index] = active_load
        voltage[index] = voltage_command
        primary_control[index] = {
            "current": voltage_command,
            "speed": iq_reference,
            "position": speed_command,
            "disturbance": iq_reference,
        }[scenario]
        output[index] = {
            "current": i_a,
            "speed": omega,
            "position": theta,
            "disturbance": omega,
        }[scenario]
        previous_omega_feedback = omega_feedback

        if abs(omega) > termination_speed or not np.isfinite(
            [i_a, omega, theta, applied_voltage]
        ).all():
            terminated = True
            if index + 1 < points:
                for target in (
                    output,
                    reference,
                    primary_control,
                    voltage,
                    current,
                    speed,
                    position,
                    current_reference,
                    speed_reference,
                    load_torque,
                ):
                    target[index + 1 :] = target[index]
            break

    return SimulationTrace(
        scenario=scenario,
        time_s=time_s,
        output=output,
        reference=reference,
        primary_control=primary_control,
        voltage_v=voltage,
        current_a=current,
        speed_rad_s=speed,
        position_rad=position,
        current_reference_a=current_reference,
        speed_reference_rad_s=speed_reference,
        load_torque_nm=load_torque,
        saturation_count=saturation_count,
        terminated=terminated,
    )
