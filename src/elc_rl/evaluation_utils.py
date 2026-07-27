"""Shared numerical helpers for the physics-v1 controller evaluators."""

from __future__ import annotations

from typing import Any

import numpy as np

from .controller_parameters import ControllerParameterSpace


GAIN_MARGIN_CAP_DB = 120.0


def zero_crossing_locations(
    log_frequency: np.ndarray, values: np.ndarray
) -> list[tuple[float, int, float]]:
    """Locate sign changes using linear interpolation on a log-frequency grid."""

    signs = np.signbit(values)
    indices = np.flatnonzero(signs[:-1] != signs[1:])
    crossings: list[tuple[float, int, float]] = []
    for index in indices:
        y0, y1 = values[index : index + 2]
        fraction = float(-y0 / (y1 - y0))
        log_value = float(
            log_frequency[index]
            + fraction * (log_frequency[index + 1] - log_frequency[index])
        )
        crossings.append((10.0**log_value, int(index), fraction))
    return crossings


def interpolate_pair(values: np.ndarray, index: int, fraction: float) -> float:
    """Interpolate a value between two adjacent samples."""

    return float(values[index] + fraction * (values[index + 1] - values[index]))


def _finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def summarize_loop_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Summarize robust loop metrics over one fixed motor-model split."""

    if not rows:
        raise ValueError("cannot summarize an empty model split")
    crossover = _finite_values(rows, "crossover_hz")
    phase_margin = _finite_values(rows, "phase_margin_deg")
    gain_margin = _finite_values(rows, "gain_margin_db")
    bandwidth = _finite_values(rows, "bandwidth_hz")
    sensitivity_peak = _finite_values(rows, "sensitivity_peak")
    complementary_peak = _finite_values(rows, "complementary_peak")
    maximum_real_pole = _finite_values(rows, "maximum_real_pole")
    return {
        "model_count": len(rows),
        "stable_count": int(sum(bool(row["stable"]) for row in rows)),
        "stable_fraction": float(np.mean([bool(row["stable"]) for row in rows])),
        "crossover_hz_median": float(np.median(crossover)),
        "crossover_hz_min": float(np.min(crossover)),
        "crossover_hz_max": float(np.max(crossover)),
        "phase_margin_deg_median": float(np.median(phase_margin)),
        "phase_margin_deg_worst": float(np.min(phase_margin)),
        "gain_margin_db_median": float(np.median(gain_margin)),
        "gain_margin_db_worst": float(np.min(gain_margin)),
        "bandwidth_hz_median": float(np.median(bandwidth)),
        "bandwidth_hz_min": float(np.min(bandwidth)),
        "bandwidth_hz_max": float(np.max(bandwidth)),
        "sensitivity_peak_worst": float(np.max(sensitivity_peak)),
        "complementary_peak_worst": float(np.max(complementary_peak)),
        "maximum_real_pole_worst": float(np.max(maximum_real_pole)),
    }


def dobc_metrics(
    parameters: np.ndarray, space: ControllerParameterSpace
) -> dict[str, float]:
    """Compute deterministic diagnostics for the configured DOBC Q filter."""

    gain = float(parameters[6])
    time_constant_s = float(parameters[7])
    frequency_hz = np.geomspace(0.1, 10.0, 256)
    s = 1j * 2.0 * np.pi * frequency_hz
    q_filter = gain / (1.0 + time_constant_s * s)
    residual = np.abs(1.0 - q_filter)
    tau_lower = space.specs[7].lower
    return {
        "gain": gain,
        "time_constant_s": time_constant_s,
        "q_cutoff_hz": float(1.0 / (2.0 * np.pi * time_constant_s)),
        "ideal_dc_residual_ratio": float(abs(1.0 - gain)),
        "ideal_0p1_to_10hz_residual_rms": float(np.sqrt(np.mean(residual**2))),
        "aggressiveness_proxy": float(gain * tau_lower / time_constant_s),
    }
