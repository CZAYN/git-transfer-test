"""Gymnasium environment for staged three-loop PID and DOBC tuning."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .physics_evaluator import (
    get_physics_controller_evaluator,
    get_physics_time_domain_evaluator,
)


STAGE_ORDER = ("current", "speed", "position", "dobc", "joint")
STAGE_INDICES = {
    "current": (8, 9, 10),
    "speed": (3, 4, 5),
    "position": (0, 1, 2),
    "dobc": (6, 7),
    "joint": tuple(range(11)),
}
CORE_SPLITS = (
    "current_reference",
    "speed_train",
    "position_surrogate",
)
PER_LOOP_METRIC_NAMES = (
    "crossover_log_ratio",
    "phase_margin_norm",
    "gain_margin_norm",
    "bandwidth_log_ratio",
    "sensitivity_peak_norm",
    "stable_fraction",
)
METRIC_NAMES = tuple(
    f"{split}:{metric}"
    for split in CORE_SPLITS
    for metric in PER_LOOP_METRIC_NAMES
) + (
    "current_to_speed_ratio_norm",
    "speed_to_position_ratio_norm",
    "dobc_residual_rms",
    "dobc_aggressiveness",
    "total_cost_squashed",
)
TIME_TARGETS_S = {
    "current_reference": 0.005,
    "speed_train": 0.050,
    "position_surrogate": 0.300,
}
TIME_PER_LOOP_METRIC_NAMES = (
    "rise_time_normalized",
    "settling_time_normalized",
    "overshoot_normalized",
    "steady_state_error_normalized",
    "iae_ratio_to_baseline",
    "control_peak_ratio_to_soft_limit",
    "control_slew_ratio_to_soft_limit",
    "stable_fraction",
)
TIME_METRIC_NAMES = tuple(
    f"{split}:{metric}"
    for split in CORE_SPLITS
    for metric in TIME_PER_LOOP_METRIC_NAMES
) + (
    "speed_train:disturbance_peak_ratio_to_baseline",
    "speed_train:disturbance_iae_ratio_to_baseline",
    "speed_train:disturbance_recovery_time_normalized",
)
OBSERVATION_KEYS = (
    "sampled_frf",
    "friction_context",
    "parameter_state",
    "metrics",
    "time_metrics",
    "action_mask",
    "stage",
)


def _metric_vector(report: dict[str, Any], position_target_hz: float) -> np.ndarray:
    del position_target_hz
    targets = report["targets_hz"]
    values: list[float] = []
    for split in CORE_SPLITS:
        summary = report["splits"][split]
        target = targets[split]
        values.extend(
            [
                float(np.log(float(summary["crossover_hz_median"]) / target)),
                float(summary["phase_margin_deg_worst"]) / 180.0,
                float(summary["gain_margin_db_worst"]) / 120.0,
                float(np.log(float(summary["bandwidth_hz_median"]) / target)),
                float(summary["sensitivity_peak_worst"]) / 2.5,
                float(summary["stable_fraction"]),
            ]
        )
    safety = report["safety"]
    values.extend(
        [
            float(safety["current_to_speed_crossover_ratio"]) / 4.0,
            float(safety["speed_to_position_crossover_ratio"]) / 3.0,
            float(report["dobc"]["ideal_0p1_to_10hz_residual_rms"]),
            float(report["dobc"]["aggressiveness_proxy"]),
            float(np.tanh(float(report["cost"]["total"]) / 10.0)),
        ]
    )
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (len(METRIC_NAMES),) or not np.isfinite(vector).all():
        raise ValueError("environment metric vector is invalid")
    return np.clip(vector, -5.0, 5.0)


def _time_metric_vector(report: dict[str, Any]) -> np.ndarray:
    values: list[float] = []
    for split in CORE_SPLITS:
        summary = report["splits"][split]
        target_s = TIME_TARGETS_S[split]
        values.extend(
            [
                float(summary["rise_time_s_median"]) / target_s,
                float(summary["settling_time_s_worst"]) / target_s,
                float(summary["overshoot_ratio_worst"]) / 0.10,
                float(summary["steady_state_error_worst"]) / 0.02,
                float(summary["iae_ratio_to_baseline_median"]),
                float(summary["control_peak_ratio_to_baseline_worst"]) / 1.25,
                float(summary["control_slew_ratio_to_baseline_worst"]) / 1.50,
                float(summary["stable_fraction"]),
            ]
        )
    speed = report["splits"]["speed_train"]
    values.extend(
        [
            float(speed["disturbance_peak_ratio_to_baseline_worst"]),
            float(speed["disturbance_iae_ratio_to_baseline_median"]),
            float(speed["disturbance_recovery_time_s_worst"]) / 0.050,
        ]
    )
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (len(TIME_METRIC_NAMES),) or not np.isfinite(vector).all():
        raise ValueError("environment time-domain metric vector is invalid")
    return np.clip(vector, -5.0, 5.0)


def _loop_stage_cost(summary: dict[str, float | int], target_hz: float) -> float:
    crossover = float(summary["crossover_hz_median"])
    phase_margin = float(summary["phase_margin_deg_worst"])
    gain_margin = float(summary["gain_margin_db_worst"])
    sensitivity_peak = float(summary["sensitivity_peak_worst"])
    stable_fraction = float(summary["stable_fraction"])
    return float(
        abs(np.log(crossover / target_hz))
        + 2.0 * max(0.0, 55.0 - phase_margin) / 55.0
        + max(0.0, 6.0 - gain_margin) / 6.0
        + max(0.0, sensitivity_peak - 1.5)
        + 100.0 * (1.0 - stable_fraction)
    )


def stage_cost(report: dict[str, Any], stage: str, position_target_hz: float) -> float:
    """Return the stage-specific scalar optimized by the environment reward."""

    del position_target_hz
    if stage == "joint":
        return float(report["cost"]["total"])
    if stage == "dobc":
        return float(report["cost"]["dobc_idealized"] + report["cost"]["unsafe"])
    targets = report["targets_hz"]
    split_and_target = {
        "current": ("current_reference", float(targets["current_reference"])),
        "speed": ("speed_train", float(targets["speed_train"])),
        "position": ("position_surrogate", float(targets["position_surrogate"])),
    }
    split, target = split_and_target[stage]
    cost = _loop_stage_cost(report["splits"][split], target)
    safety = report["safety"]
    if stage in {"current", "speed"}:
        cost += max(
            0.0, 4.0 - float(safety["current_to_speed_crossover_ratio"])
        ) / 4.0
    if stage in {"speed", "position"}:
        cost += max(
            0.0, 3.0 - float(safety["speed_to_position_crossover_ratio"])
        ) / 3.0
    return float(cost)


def _time_loop_cost(summary: dict[str, float | int], target_s: float) -> float:
    """Soft time-domain cost without inventing absolute actuator limits."""

    settling_ratio = float(summary["settling_time_s_worst"]) / target_s
    overshoot_ratio = float(summary["overshoot_ratio_worst"]) / 0.10
    steady_state_ratio = float(summary["steady_state_error_worst"]) / 0.02
    iae_ratio = float(summary["iae_ratio_to_baseline_median"])
    control_peak_ratio = float(summary["control_peak_ratio_to_baseline_worst"])
    control_rms_ratio = float(summary["control_rms_ratio_to_baseline_worst"])
    control_slew_ratio = float(summary["control_slew_ratio_to_baseline_worst"])
    stable_fraction = float(summary["stable_fraction"])
    return float(
        100.0 * (1.0 - stable_fraction)
        + 0.5 * max(0.0, settling_ratio - 1.0)
        + 2.0 * max(0.0, overshoot_ratio - 1.0)
        + max(0.0, steady_state_ratio - 1.0)
        + 0.25 * iae_ratio
        + max(0.0, control_peak_ratio - 1.25)
        + 0.5 * max(0.0, control_rms_ratio - 1.25)
        + 0.5 * max(0.0, control_slew_ratio - 1.50)
    )


def time_stage_cost(report: dict[str, Any], stage: str) -> float:
    """Return the stage-specific step/disturbance cost."""

    loop_costs = {
        "current": _time_loop_cost(
            report["splits"]["current_reference"],
            TIME_TARGETS_S["current_reference"],
        ),
        "speed": _time_loop_cost(
            report["splits"]["speed_train"],
            TIME_TARGETS_S["speed_train"],
        ),
        "position": _time_loop_cost(
            report["splits"]["position_surrogate"],
            TIME_TARGETS_S["position_surrogate"],
        ),
    }
    speed = report["splits"]["speed_train"]
    dobc_cost = float(
        1.5 * float(speed["disturbance_peak_ratio_to_baseline_worst"])
        + float(speed["disturbance_iae_ratio_to_baseline_median"])
        + 0.5
        * max(
            0.0,
            float(speed["disturbance_recovery_time_s_worst"]) / 0.050 - 1.0,
        )
    )
    if stage == "dobc":
        return dobc_cost
    if stage == "joint":
        return float(sum(loop_costs.values()) + dobc_cost)
    return float(loop_costs[stage])


def combined_stage_cost(
    frequency_report: dict[str, Any],
    time_report: dict[str, Any],
    stage: str,
    position_target_hz: float,
) -> float:
    """Combined frequency/time-domain objective used by combined frequency/time-domain reward."""

    return float(
        stage_cost(frequency_report, stage, position_target_hz)
        + time_stage_cost(time_report, stage)
    )


def _time_report_safe(report: dict[str, Any]) -> bool:
    """Apply physical-limit safety when supplied, otherwise require stability."""

    stable = bool(
        report["splits"]
        and all(
            float(summary["stable_fraction"]) == 1.0
            for summary in report["splits"].values()
        )
    )
    declared_safety = report.get("safety")
    return bool(
        stable
        and (
            declared_safety is None
            or bool(declared_safety.get("safe", False))
        )
    )


class PIDTuningEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """Bounded delta-action environment with periodic full-ensemble audits."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        project_root: Path,
        *,
        stage: str = "joint",
        max_episode_steps: int = 32,
        audit_interval: int = 8,
        initial_perturbation: float = 0.05,
        base_parameters: np.ndarray | None = None,
        worker_rank: int = 0,
        render_mode: None = None,
    ) -> None:
        super().__init__()
        if stage not in STAGE_ORDER:
            raise ValueError(f"stage must be one of {STAGE_ORDER}, got {stage!r}")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if audit_interval <= 0:
            raise ValueError("audit_interval must be positive")
        if not 0.0 <= initial_perturbation <= 0.5:
            raise ValueError("initial_perturbation must be between 0 and 0.5")
        if render_mode is not None:
            raise ValueError("PIDTuningEnv does not implement rendering")
        self.project_root = Path(project_root).resolve()
        self.backend = "physics"
        self.stage = stage
        self.max_episode_steps = int(max_episode_steps)
        self.audit_interval = int(audit_interval)
        self.initial_perturbation = float(initial_perturbation)
        self.worker_rank = int(worker_rank)
        if self.worker_rank < 0:
            raise ValueError("worker_rank must be non-negative")
        self.evaluator = get_physics_controller_evaluator(self.project_root)
        self.time_evaluator = get_physics_time_domain_evaluator(self.project_root)
        self.parameter_space = self.evaluator.space
        if base_parameters is None:
            self._base_parameters = self.parameter_space.initial.copy()
        else:
            candidate_base = np.asarray(base_parameters, dtype=np.float64)
            if candidate_base.shape != (11,):
                raise ValueError("base_parameters must have shape (11,)")
            self.parameter_space.normalize(candidate_base)
            self._base_parameters = candidate_base.copy()
        self.position_target_hz = float(
            self.parameter_space.metadata["position_design"]["target_crossover_hz"]
        )
        self._action_mask = np.zeros(11, dtype=np.float32)
        self._action_mask[list(STAGE_INDICES[stage])] = 1.0
        self._stage_vector = np.zeros(len(STAGE_ORDER), dtype=np.float32)
        self._stage_vector[STAGE_ORDER.index(stage)] = 1.0

        self.action_space = spaces.Box(-1.0, 1.0, shape=(11,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "sampled_frf": spaces.Box(
                    -5.0, 5.0, shape=(96,), dtype=np.float32
                ),
                "friction_context": spaces.Box(
                    -5.0, 5.0, shape=(6,), dtype=np.float32
                ),
                "parameter_state": spaces.Box(
                    -1.0, 1.0, shape=(11,), dtype=np.float32
                ),
                "metrics": spaces.Box(
                    -5.0,
                    5.0,
                    shape=(len(METRIC_NAMES),),
                    dtype=np.float32,
                ),
                "time_metrics": spaces.Box(
                    -5.0,
                    5.0,
                    shape=(len(TIME_METRIC_NAMES),),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    0.0, 1.0, shape=(11,), dtype=np.float32
                ),
                "stage": spaces.Box(
                    0.0,
                    1.0,
                    shape=(len(STAGE_ORDER),),
                    dtype=np.float32,
                ),
            }
        )

        self._parameters: np.ndarray | None = None
        self._sampled_indices: np.ndarray | None = None
        self._sampled_frf: np.ndarray | None = None
        self._friction_context: np.ndarray | None = None
        self._report: dict[str, Any] | None = None
        self._time_report: dict[str, Any] | None = None
        self._last_audit_report: dict[str, Any] | None = None
        self._last_time_audit_report: dict[str, Any] | None = None
        self._previous_stage_cost = 0.0
        self._step_count = 0
        self._total_step_count = 0
        self._episode_count = 0

    @property
    def parameters(self) -> np.ndarray:
        if self._parameters is None:
            raise RuntimeError("environment must be reset before reading parameters")
        return self._parameters.copy()

    @property
    def base_parameters(self) -> np.ndarray:
        return self._base_parameters.copy()

    @property
    def action_mask(self) -> np.ndarray:
        return self._action_mask.copy()

    def _observation(self) -> dict[str, np.ndarray]:
        if (
            self._parameters is None
            or self._sampled_frf is None
            or self._friction_context is None
            or self._report is None
            or self._time_report is None
        ):
            raise RuntimeError("environment state is not initialized")
        observation = {
            "sampled_frf": self._sampled_frf.copy(),
            "friction_context": self._friction_context.copy(),
            "parameter_state": self.parameter_space.normalize(self._parameters).astype(
                np.float32
            ),
            "metrics": _metric_vector(self._report, self.position_target_hz),
            "time_metrics": _time_metric_vector(self._time_report),
            "action_mask": self._action_mask.copy(),
            "stage": self._stage_vector.copy(),
        }
        if not self.observation_space.contains(observation):
            raise RuntimeError("constructed observation is outside observation_space")
        return observation

    def _info(self, *, audit_performed: bool) -> dict[str, Any]:
        if (
            self._report is None
            or self._time_report is None
            or self._sampled_indices is None
            or self._parameters is None
        ):
            raise RuntimeError("environment state is not initialized")
        audit_frequency_safe = (
            None
            if self._last_audit_report is None
            else bool(self._last_audit_report["safety"]["safe"])
        )
        audit_time_safe = (
            None
            if self._last_time_audit_report is None
            else _time_report_safe(self._last_time_audit_report)
        )
        frequency_cost = stage_cost(
            self._report, self.stage, self.position_target_hz
        )
        time_cost = time_stage_cost(self._time_report, self.stage)
        fast_frequency_safe = bool(self._report["safety"]["safe"])
        fast_time_safe = _time_report_safe(self._time_report)
        return {
            "backend": self.backend,
            "stage": self.stage,
            "worker_rank": self.worker_rank,
            "step": self._step_count,
            "total_step": self._total_step_count,
            "stage_cost": float(frequency_cost + time_cost),
            "frequency_stage_cost": float(frequency_cost),
            "time_stage_cost": float(time_cost),
            "fast_safe": bool(fast_frequency_safe and fast_time_safe),
            "fast_frequency_safe": fast_frequency_safe,
            "fast_time_safe": fast_time_safe,
            "audit_performed": audit_performed,
            "audit_safe": (
                None
                if audit_frequency_safe is None or audit_time_safe is None
                else bool(audit_frequency_safe and audit_time_safe)
            ),
            "audit_frequency_safe": audit_frequency_safe,
            "audit_time_safe": audit_time_safe,
            "sampled_model_ids": self.evaluator.model_ids(self._sampled_indices),
            "parameters": self._parameters.copy(),
        }

    def _candidate_initial_parameters(
        self, perturb: bool, explicit: np.ndarray | None
    ) -> np.ndarray:
        if explicit is not None:
            candidate = np.asarray(explicit, dtype=np.float64)
            if candidate.shape != (11,):
                raise ValueError("reset option parameters must have shape (11,)")
            self.parameter_space.normalize(candidate)
            return candidate.copy()
        normalized = self.parameter_space.normalize(self._base_parameters)
        if perturb and self.initial_perturbation > 0:
            noise = self.np_random.uniform(
                -self.initial_perturbation, self.initial_perturbation, size=11
            )
            normalized = np.clip(
                normalized + noise * self._action_mask.astype(np.float64), -1.0, 1.0
            )
        return self.parameter_space.denormalize(normalized)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self._total_step_count = 0
            self._episode_count = 0
        super().reset(seed=seed)
        self._episode_count += 1
        options = {} if options is None else dict(options)
        perturb = bool(options.get("perturb", True))
        explicit_parameters = options.get("parameters")
        explicit = (
            None
            if explicit_parameters is None
            else np.asarray(explicit_parameters, dtype=np.float64)
        )
        explicit_indices = options.get("sampled_indices")
        if explicit_indices is None:
            self._sampled_indices = self.evaluator.sample_training_indices(self.np_random)
        else:
            self._sampled_indices = self.evaluator.validate_sampled_indices(
                np.asarray(explicit_indices, dtype=np.int64)
            )

        candidate = self._candidate_initial_parameters(perturb, explicit)
        report = self.evaluator.train(candidate, self._sampled_indices)
        time_report = self.time_evaluator.train(candidate, self._sampled_indices)
        audit = self.evaluator.audit(candidate)
        time_audit = self.time_evaluator.audit(candidate)
        if explicit is None and not (
            bool(report["safety"]["safe"])
            and _time_report_safe(time_report)
            and bool(audit["safety"]["safe"])
            and _time_report_safe(time_audit)
        ):
            candidate = self._base_parameters.copy()
            report = self.evaluator.train(candidate, self._sampled_indices)
            time_report = self.time_evaluator.train(candidate, self._sampled_indices)
            audit = self.evaluator.audit(candidate)
            time_audit = self.time_evaluator.audit(candidate)

        self._parameters = candidate
        self._report = report
        self._time_report = time_report
        self._last_audit_report = audit
        self._last_time_audit_report = time_audit
        sampled = self.evaluator.sampled_frf_vector(self._sampled_indices)
        self._sampled_frf = np.clip(sampled, -5.0, 5.0).astype(np.float32)
        friction = self.evaluator.friction_context_vector(self._sampled_indices)
        self._friction_context = np.clip(friction, -5.0, 5.0).astype(np.float32)
        self._previous_stage_cost = combined_stage_cost(
            report, time_report, self.stage, self.position_target_hz
        )
        self._step_count = 0
        return self._observation(), self._info(audit_performed=True)

    def export_state(self) -> dict[str, Any]:
        """Return the complete mutable state needed for exact vector-env resume."""

        if (
            self._parameters is None
            or self._sampled_indices is None
            or self._sampled_frf is None
            or self._friction_context is None
            or self._report is None
            or self._time_report is None
        ):
            raise RuntimeError("environment must be reset before exporting state")
        return {
            "schema_version": 3,
            "stage": self.stage,
            "worker_rank": self.worker_rank,
            "parameters": self._parameters.copy(),
            "sampled_indices": self._sampled_indices.copy(),
            "sampled_frf": self._sampled_frf.copy(),
            "friction_context": self._friction_context.copy(),
            "report": copy.deepcopy(self._report),
            "time_report": copy.deepcopy(self._time_report),
            "last_audit_report": copy.deepcopy(self._last_audit_report),
            "last_time_audit_report": copy.deepcopy(
                self._last_time_audit_report
            ),
            "previous_stage_cost": float(self._previous_stage_cost),
            "step_count": int(self._step_count),
            "total_step_count": int(self._total_step_count),
            "episode_count": int(self._episode_count),
            "np_random_state": copy.deepcopy(self.np_random.bit_generator.state),
            "action_space_random_state": copy.deepcopy(
                self.action_space.np_random.bit_generator.state
            ),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore a state produced by :meth:`export_state`."""

        if int(state.get("schema_version", -1)) != 3:
            raise ValueError("unsupported environment state schema")
        if state.get("stage") != self.stage:
            raise ValueError("environment state stage mismatch")
        if int(state.get("worker_rank", -1)) != self.worker_rank:
            raise ValueError("environment state worker rank mismatch")
        parameters = np.asarray(state["parameters"], dtype=np.float64)
        sampled_indices = self.evaluator.validate_sampled_indices(
            np.asarray(state["sampled_indices"], dtype=np.int64)
        )
        sampled_frf = np.asarray(state["sampled_frf"], dtype=np.float32)
        friction_context = np.asarray(state["friction_context"], dtype=np.float32)
        if (
            parameters.shape != (11,)
            or sampled_frf.shape != (96,)
            or friction_context.shape != (6,)
        ):
            raise ValueError("environment state contains invalid arrays")
        expected_friction_context = self.evaluator.friction_context_vector(
            sampled_indices
        ).astype(np.float32)
        if not np.array_equal(friction_context, expected_friction_context):
            raise ValueError("environment state friction context mismatch")
        self.parameter_space.normalize(parameters)
        self._parameters = parameters.copy()
        self._sampled_indices = sampled_indices.copy()
        self._sampled_frf = sampled_frf.copy()
        self._friction_context = friction_context.copy()
        self._report = copy.deepcopy(state["report"])
        self._time_report = copy.deepcopy(state["time_report"])
        self._last_audit_report = copy.deepcopy(state["last_audit_report"])
        self._last_time_audit_report = copy.deepcopy(
            state["last_time_audit_report"]
        )
        self._previous_stage_cost = float(state["previous_stage_cost"])
        self._step_count = int(state["step_count"])
        self._total_step_count = int(state["total_step_count"])
        self._episode_count = int(state["episode_count"])
        self._np_random = np.random.default_rng()
        self._np_random.bit_generator.state = copy.deepcopy(
            state["np_random_state"]
        )
        self.action_space.seed(0)
        self.action_space.np_random.bit_generator.state = copy.deepcopy(
            state["action_space_random_state"]
        )
        observation = self._observation()
        if not self.observation_space.contains(observation):
            raise ValueError("restored environment observation is invalid")

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._parameters is None or self._sampled_indices is None:
            raise RuntimeError("environment must be reset before step")
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (11,):
            raise ValueError(f"expected action shape (11,), got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("action contains NaN or infinite values")
        clipped = np.clip(values, -1.0, 1.0)
        masked_action = clipped * self._action_mask.astype(np.float64)
        candidate = self.parameter_space.apply_action(self._parameters, masked_action)
        report = self.evaluator.train(candidate, self._sampled_indices)
        time_report = self.time_evaluator.train(candidate, self._sampled_indices)

        self._step_count += 1
        self._total_step_count += 1
        audit_performed = self._total_step_count % self.audit_interval == 0
        audit_report = self.evaluator.audit(candidate) if audit_performed else None
        time_audit_report = (
            self.time_evaluator.audit(candidate) if audit_performed else None
        )
        fast_safe = bool(report["safety"]["safe"]) and _time_report_safe(time_report)
        audit_safe = (
            True
            if audit_report is None or time_audit_report is None
            else bool(audit_report["safety"]["safe"])
            and _time_report_safe(time_audit_report)
        )
        safe = fast_safe and audit_safe
        new_stage_cost = combined_stage_cost(
            report, time_report, self.stage, self.position_target_hz
        )
        improvement = self._previous_stage_cost - new_stage_cost
        action_penalty = 0.002 * float(np.sum(masked_action**2))
        if safe:
            reward = 10.0 * improvement - 0.02 * new_stage_cost - action_penalty
        else:
            reward = -100.0 - action_penalty

        self._parameters = candidate
        self._report = report
        self._time_report = time_report
        if audit_report is not None:
            self._last_audit_report = audit_report
        if time_audit_report is not None:
            self._last_time_audit_report = time_audit_report
        self._previous_stage_cost = new_stage_cost
        terminated = bool(not safe)
        truncated = bool(self._step_count >= self.max_episode_steps and not terminated)
        info = self._info(audit_performed=audit_performed)
        info["reward_components"] = {
            "improvement": float(improvement),
            "absolute_cost": float(new_stage_cost),
            "frequency_cost": float(
                stage_cost(report, self.stage, self.position_target_hz)
            ),
            "time_cost": float(time_stage_cost(time_report, self.stage)),
            "action_penalty": float(action_penalty),
            "unsafe_penalty": 0.0 if safe else 100.0,
        }
        return self._observation(), float(reward), terminated, truncated, info
