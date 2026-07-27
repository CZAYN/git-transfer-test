"""Auditable 11-dimensional physics-v1 controller parameter space."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .physics_motor_model import load_physics_motor_config
from .task_dataset import EXPECTED_TASK_ID, load_frf_task


PARAMETER_ORDER = (
    "kppos",
    "kipos",
    "kdpos",
    "kpspeed",
    "kispeed",
    "kdspeed",
    "kgspeed",
    "tauspeed",
    "kpcurr",
    "kicurr",
    "kdcurr",
)

PHYSICS_PARAMETER_SPACE_JSON = "controller_parameter_space_physics_v1.json"
PHYSICS_PARAMETER_SPACE_NPZ = "controller_parameter_space_physics_v1.npz"
_BOUNDARY_EPS_FACTOR = 128.0


def _physical_boundary_tolerance(lower: float, upper: float) -> float:
    scale = max(1.0, abs(lower), abs(upper), abs(upper - lower))
    return _BOUNDARY_EPS_FACTOR * np.finfo(np.float64).eps * scale


@dataclass(frozen=True)
class ParameterSpec:
    """One physical parameter and its bounded RL representation."""

    name: str
    module: str
    initial: float
    lower: float
    upper: float
    transform: str
    action_step_fraction: float
    source_kind: str
    source: str
    original_value: float | None
    unit: str
    sample_period_s: float | None
    digital_initial: float | None
    training_stage: int
    hardware_status: str


@dataclass(frozen=True)
class ControllerParameterSpace:
    """Validated physical/normalized mapping for the 11 controller outputs."""

    task_id: str
    specs: tuple[ParameterSpec, ...]
    metadata: dict[str, Any]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    @property
    def initial(self) -> np.ndarray:
        return np.asarray([spec.initial for spec in self.specs], dtype=np.float64)

    @property
    def lower(self) -> np.ndarray:
        return np.asarray([spec.lower for spec in self.specs], dtype=np.float64)

    @property
    def upper(self) -> np.ndarray:
        return np.asarray([spec.upper for spec in self.specs], dtype=np.float64)

    @property
    def action_step_fraction(self) -> np.ndarray:
        return np.asarray(
            [spec.action_step_fraction for spec in self.specs], dtype=np.float64
        )

    def validate(self) -> None:
        if self.task_id != EXPECTED_TASK_ID:
            raise ValueError(f"unexpected task_id: {self.task_id}")
        if self.names != PARAMETER_ORDER:
            raise ValueError(f"unexpected parameter order: {self.names}")

        lower = self.lower
        initial = self.initial
        upper = self.upper
        if not np.isfinite(np.concatenate([lower, initial, upper])).all():
            raise ValueError("parameter space contains non-finite values")
        if not np.all(lower < upper):
            raise ValueError("every parameter must have a non-empty interval")
        if not np.all((lower <= initial) & (initial <= upper)):
            raise ValueError("initial parameter values must be inside their bounds")

        for spec in self.specs:
            if spec.transform not in {"linear", "log"}:
                raise ValueError(f"unsupported transform for {spec.name}: {spec.transform}")
            if spec.transform == "log" and spec.lower <= 0:
                raise ValueError(f"log-transformed lower bound is not positive: {spec.name}")
            if not 0 < spec.action_step_fraction <= 1:
                raise ValueError(f"invalid action step fraction: {spec.name}")
            if spec.source_kind != "excel_baseline" and spec.original_value is not None:
                raise ValueError(f"inferred parameter cannot claim an original value: {spec.name}")
            if spec.sample_period_s is not None and spec.sample_period_s <= 0:
                raise ValueError(f"invalid sample period: {spec.name}")

    def normalize(self, physical: np.ndarray) -> np.ndarray:
        """Map physical values to [-1, 1] using each declared transform."""

        values = np.asarray(physical, dtype=np.float64)
        if values.shape[-1:] != (len(self.specs),):
            raise ValueError(f"expected final dimension {len(self.specs)}, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("physical values must be finite")
        normalized = np.empty_like(values, dtype=np.float64)
        for index, spec in enumerate(self.specs):
            value = values[..., index]
            tolerance = _physical_boundary_tolerance(spec.lower, spec.upper)
            if np.any(
                (value < spec.lower - tolerance) | (value > spec.upper + tolerance)
            ):
                raise ValueError(f"physical value outside bounds: {spec.name}")
            value = np.clip(value, spec.lower, spec.upper)
            if spec.transform == "log":
                fraction = (np.log(value) - np.log(spec.lower)) / (
                    np.log(spec.upper) - np.log(spec.lower)
                )
            else:
                fraction = (value - spec.lower) / (spec.upper - spec.lower)
            normalized[..., index] = np.clip(2.0 * fraction - 1.0, -1.0, 1.0)
        return normalized

    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        """Map normalized values in [-1, 1] back to physical parameters."""

        values = np.asarray(normalized, dtype=np.float64)
        if values.shape[-1:] != (len(self.specs),):
            raise ValueError(f"expected final dimension {len(self.specs)}, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("normalized values must be finite")
        tolerance = _BOUNDARY_EPS_FACTOR * np.finfo(np.float64).eps
        if np.any((values < -1.0 - tolerance) | (values > 1.0 + tolerance)):
            raise ValueError("normalized values must stay inside [-1, 1]")
        values = np.clip(values, -1.0, 1.0)
        physical = np.empty_like(values, dtype=np.float64)
        for index, spec in enumerate(self.specs):
            fraction = (values[..., index] + 1.0) / 2.0
            if spec.transform == "log":
                transformed = np.exp(
                    np.log(spec.lower)
                    + fraction * (np.log(spec.upper) - np.log(spec.lower))
                )
            else:
                transformed = spec.lower + fraction * (
                    spec.upper - spec.lower
                )
            physical[..., index] = np.where(
                fraction <= 0.0,
                spec.lower,
                np.where(fraction >= 1.0, spec.upper, transformed),
            )
        return np.clip(physical, self.lower, self.upper)

    def apply_action(self, physical: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Apply a bounded normalized delta action to one physical parameter vector."""

        current = self.normalize(np.asarray(physical, dtype=np.float64))
        delta = np.asarray(action, dtype=np.float64)
        if delta.shape != (len(self.specs),):
            raise ValueError(f"expected action shape {(len(self.specs),)}, got {delta.shape}")
        if not np.isfinite(delta).all():
            raise ValueError("action values must be finite")
        if np.any((delta < -1.0) | (delta > 1.0)):
            raise ValueError("action values must stay inside [-1, 1]")
        updated = np.clip(current + delta * self.action_step_fraction, -1.0, 1.0)
        return self.denormalize(updated)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digital_value(name: str, analog_value: float, sample_period_s: float) -> float:
    if name.startswith("kp"):
        return analog_value
    if name.startswith("ki"):
        return analog_value * sample_period_s
    if name.startswith("kd"):
        return analog_value / sample_period_s
    raise KeyError(name)


def _parameter(
    *,
    name: str,
    module: str,
    initial: float,
    lower: float,
    upper: float,
    transform: str,
    action_step_fraction: float,
    source_kind: str,
    source: str,
    original_value: float | None,
    unit: str,
    sample_period_s: float | None,
    training_stage: int,
    hardware_status: str,
) -> dict[str, Any]:
    digital_initial = None
    if sample_period_s is not None and module in {"current", "speed", "position"}:
        digital_initial = _digital_value(name, initial, sample_period_s)
    return {
        "name": name,
        "module": module,
        "initial": float(initial),
        "lower": float(lower),
        "upper": float(upper),
        "transform": transform,
        "action_step_fraction": float(action_step_fraction),
        "source_kind": source_kind,
        "source": source,
        "original_value": None if original_value is None else float(original_value),
        "unit": unit,
        "sample_period_s": sample_period_s,
        "digital_initial": digital_initial,
        "training_stage": training_stage,
        "hardware_status": hardware_status,
    }



def _space_from_payload(payload: dict[str, Any]) -> ControllerParameterSpace:
    specs = tuple(ParameterSpec(**values) for values in payload["parameters"])
    metadata = {key: value for key, value in payload.items() if key != "parameters"}
    return ControllerParameterSpace(
        task_id=str(payload["task_id"]), specs=specs, metadata=metadata
    )



def derive_physics_controller_initials(project_root: Path) -> np.ndarray:
    """Derive the 11 physics-v1 initials from the mentor model and bandwidths."""

    config = load_physics_motor_config(project_root)
    motor = config.nominal
    design = config.payload["controller_design"]
    current_omega = 2.0 * np.pi * float(design["current_crossover_hz"])
    speed_omega = 2.0 * np.pi * float(design["speed_crossover_hz"])
    position_omega = 2.0 * np.pi * float(design["position_crossover_hz"])
    derivative_ratio = float(design["derivative_ratio_at_crossover"])
    position_integral_ratio = float(
        design["position_integral_ratio_at_crossover"]
    )

    current_delay_correction = np.sqrt(
        1.0 + (current_omega * motor.current_delay_s) ** 2
    )
    current_kp = motor.inductance_h * current_omega * current_delay_correction
    current_ki = motor.resistance_ohm * current_omega * current_delay_correction
    current_kd = derivative_ratio * current_kp / current_omega

    speed_delay_correction = np.sqrt(
        1.0 + (speed_omega * motor.speed_measurement_delay_s) ** 2
    )
    speed_kp = (
        motor.inertia_kg_m2
        * speed_omega
        * speed_delay_correction
        / motor.torque_constant_nm_per_a
    )
    speed_ki = speed_kp * (
        motor.viscous_friction_nm_s_per_rad / motor.inertia_kg_m2
    )
    speed_kd = derivative_ratio * speed_kp / speed_omega

    position_derivative_ratio = derivative_ratio
    position_kp = position_omega / np.sqrt(
        1.0
        + (position_integral_ratio - position_derivative_ratio) ** 2
    )
    position_ki = position_integral_ratio * position_omega * position_kp
    position_kd = position_derivative_ratio * position_kp / position_omega

    dobc = design["dobc"]
    return np.asarray(
        [
            position_kp,
            position_ki,
            position_kd,
            speed_kp,
            speed_ki,
            speed_kd,
            float(dobc["gain_initial"]),
            float(dobc["filter_time_initial_s"]),
            current_kp,
            current_ki,
            current_kd,
        ],
        dtype=np.float64,
    )


def build_physics_controller_parameter_space(project_root: Path) -> dict[str, Any]:
    """Build a separate 11-D space for the physics-v1 training backend."""

    root = Path(project_root).resolve()
    config = load_physics_motor_config(root)
    task = load_frf_task(root)
    initial = derive_physics_controller_initials(root)
    by_name = dict(zip(PARAMETER_ORDER, initial.tolist()))
    dt = config.sample_period_s
    design = config.payload["controller_design"]
    dobc = design["dobc"]

    def gain_parameter(
        name: str,
        module: str,
        *,
        stage: int,
        derivative: bool = False,
    ) -> dict[str, Any]:
        value = by_name[name]
        if derivative:
            lower, upper, transform = 0.0, value * 4.0, "linear"
            step = 0.05
        else:
            lower, upper, transform = value / 4.0, value * 4.0, "log"
            step = 0.06
        return _parameter(
            name=name,
            module=module,
            initial=value,
            lower=lower,
            upper=upper,
            transform=transform,
            action_step_fraction=step,
            source_kind="mentor_physics_model_derived",
            source=(
                "physics-v1 loop-shaping with the mentor motor model and fixed "
                "filtered-derivative semantics"
            ),
            original_value=None,
            unit=("native_analog_gain_s" if derivative else (
                "native_analog_gain_per_s" if name.startswith("ki") else "native_analog_gain"
            )),
            sample_period_s=dt,
            training_stage=stage,
            hardware_status="simulation_only_requires_measured_frf_and_hardware_validation",
        )

    parameters = [
        gain_parameter("kppos", "position", stage=3),
        gain_parameter("kipos", "position", stage=3),
        gain_parameter("kdpos", "position", stage=3, derivative=True),
        gain_parameter("kpspeed", "speed", stage=2),
        gain_parameter("kispeed", "speed", stage=2),
        gain_parameter("kdspeed", "speed", stage=2, derivative=True),
        _parameter(
            name="kgspeed",
            module="DOBC",
            initial=float(dobc["gain_initial"]),
            lower=float(dobc["gain_bounds"][0]),
            upper=float(dobc["gain_bounds"][1]),
            transform="linear",
            action_step_fraction=0.05,
            source_kind="approved_dobc_structure",
            source=str(dobc["structure"]),
            original_value=None,
            unit="dimensionless",
            sample_period_s=dt,
            training_stage=4,
            hardware_status="simulation_only_requires_disturbance_validation",
        ),
        _parameter(
            name="tauspeed",
            module="DOBC",
            initial=float(dobc["filter_time_initial_s"]),
            lower=float(dobc["filter_time_bounds_s"][0]),
            upper=float(dobc["filter_time_bounds_s"][1]),
            transform="log",
            action_step_fraction=0.06,
            source_kind="approved_dobc_structure",
            source=str(dobc["structure"]),
            original_value=None,
            unit="s",
            sample_period_s=dt,
            training_stage=4,
            hardware_status="simulation_only_requires_disturbance_validation",
        ),
        gain_parameter("kpcurr", "current", stage=1),
        gain_parameter("kicurr", "current", stage=1),
        gain_parameter("kdcurr", "current", stage=1, derivative=True),
    ]

    payload: dict[str, Any] = {
        "schema_version": 2,
        "profile": "physics_v1",
        "task_id": task.task_id,
        "parameter_order": list(PARAMETER_ORDER),
        "controller_convention": (
            "continuous filtered PIDF: C(s)=Kp+Ki/s+Kd*s/(Tf*s+1); "
            "implemented discretely at Ts=200 us with anti-windup"
        ),
        "digital_conversion": {
            "Kp_d": "Kp",
            "Ki_d": "Ki*Ts",
            "Kd_d": "Kd/Ts",
            "sample_period_s": dt,
            "implementation_note": "runtime uses physical continuous gains, not these display conversions",
        },
        "physics_model": {
            "model_id": config.payload["model_id"],
            "config_relative_path": str(config.path.relative_to(root)).replace("\\", "/"),
            "config_sha256": _sha256(config.path),
            "primary_training_plant": True,
            "measured_frf_used_as_training_plant": False,
        },
        "controller_design": {
            "target_crossover_hz": config.target_crossovers_hz,
            "derivative_filter_s": config.derivative_filter_s,
            "derivative_ratio_at_crossover": float(
                design["derivative_ratio_at_crossover"]
            ),
            "position_integral_ratio_at_crossover": float(
                design["position_integral_ratio_at_crossover"]
            ),
        },
        "position_design": {
            "status": "physics_model_derived",
            "target_crossover_hz": float(design["position_crossover_hz"]),
            "cascade_ratio_speed_to_position": float(
                design["speed_crossover_hz"] / design["position_crossover_hz"]
            ),
            "position_sample_period_s": dt,
            "continuous_rotation": True,
        },
        "dobc_design": {
            "status": "approved_simulation_structure",
            "structure": str(dobc["structure"]),
            "nominal_inverse_excludes_measurement_delay": bool(
                dobc["nominal_inverse_excludes_measurement_delay"]
            ),
        },
        "parameters": parameters,
        "safety_policy": {
            "direct_hardware_use_allowed": False,
            "simulation_limits_are_hardware_ratings": False,
            "required_before_hardware": [
                "confirm voltage/current/speed limits against drive and motor ratings",
                "confirm encoder resolution and feedback filtering",
                "compare physics-v1 FRFs with all measured three-loop FRFs",
                "validate candidates in HIL and bounded low-energy tests",
            ],
        },
    }
    space = _space_from_payload(payload)
    space.validate()
    output_dir = root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / PHYSICS_PARAMETER_SPACE_JSON
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / PHYSICS_PARAMETER_SPACE_NPZ,
        schema_version=np.asarray(2, dtype=np.int16),
        profile=np.asarray("physics_v1"),
        task_id=np.asarray(task.task_id),
        parameter_names=np.asarray(space.names),
        initial=space.initial,
        lower=space.lower,
        upper=space.upper,
        transform=np.asarray([spec.transform for spec in space.specs]),
        source_kind=np.asarray([spec.source_kind for spec in space.specs]),
        training_stage=np.asarray(
            [spec.training_stage for spec in space.specs], dtype=np.int16
        ),
        action_step_fraction=space.action_step_fraction,
    )
    return payload


def load_physics_controller_parameter_space(
    project_root: Path,
) -> ControllerParameterSpace:
    path = (
        Path(project_root).resolve()
        / "data"
        / "processed"
        / PHYSICS_PARAMETER_SPACE_JSON
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("profile") != "physics_v1":
        raise ValueError("controller parameter file is not the physics-v1 profile")
    space = _space_from_payload(payload)
    space.validate()
    return space
