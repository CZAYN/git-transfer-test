from __future__ import annotations

import math

import numpy as np
import pytest

from elc_rl.simulation_kernel import lugre_friction_step


DT = 2.0e-4
SIGMA_0 = 187.0
SIGMA_1 = 2.42
SIGMA_2 = 0.0237
OMEGA_S = 0.05061
TAU_C = 0.04
TAU_S = 0.07
ALPHA = 2.0


def _step(omega: float, state: float, mode: int = 1):
    return lugre_friction_step(
        mode,
        omega,
        state,
        DT,
        SIGMA_0,
        SIGMA_1,
        SIGMA_2,
        OMEGA_S,
        TAU_C,
        TAU_S,
        ALPHA,
    )


def _stribeck_torque(omega: float) -> float:
    return TAU_C + (TAU_S - TAU_C) * math.exp(
        -((abs(omega) / OMEGA_S) ** ALPHA)
    )


@pytest.mark.parametrize("omega", [0.01, 0.05, 0.25])
def test_lugre_steady_state_matches_stribeck_curve(omega: float) -> None:
    stribeck_torque = _stribeck_torque(omega)
    steady_state = stribeck_torque / SIGMA_0

    torque, next_state, rate = _step(omega, steady_state)

    assert next_state == pytest.approx(steady_state, abs=1e-15)
    assert rate == pytest.approx(0.0, abs=1e-12)
    assert torque == pytest.approx(
        stribeck_torque + SIGMA_2 * omega,
        rel=1e-12,
        abs=1e-12,
    )


@pytest.mark.parametrize("omega", [0.005, 0.05, 0.5])
def test_lugre_model_is_odd_symmetric(omega: float) -> None:
    initial_state = 1.3e-4

    positive = _step(omega, initial_state)
    negative = _step(-omega, -initial_state)

    assert negative[0] == pytest.approx(-positive[0], abs=1e-12)
    assert negative[1] == pytest.approx(-positive[1], abs=1e-15)
    assert negative[2] == pytest.approx(-positive[2], abs=1e-12)


def test_lugre_zero_speed_preserves_bristle_memory() -> None:
    initial_state = 2.5e-4

    torque, next_state, rate = _step(0.0, initial_state)

    assert next_state == initial_state
    assert rate == 0.0
    assert torque == pytest.approx(SIGMA_0 * initial_state)


def test_lugre_uses_exact_frozen_velocity_state_update() -> None:
    omega = 0.08
    initial_state = -1.0e-4
    stribeck_torque = _stribeck_torque(omega)
    decay_rate = SIGMA_0 * abs(omega) / stribeck_torque
    steady_state = stribeck_torque / SIGMA_0
    expected_state = steady_state + (
        initial_state - steady_state
    ) * math.exp(-decay_rate * DT)
    expected_rate = omega - decay_rate * expected_state

    torque, next_state, rate = _step(omega, initial_state)

    assert next_state == pytest.approx(expected_state, abs=1e-15)
    assert rate == pytest.approx(expected_rate, abs=1e-12)
    assert torque == pytest.approx(
        SIGMA_0 * expected_state + SIGMA_1 * expected_rate + SIGMA_2 * omega,
        abs=1e-12,
    )


@pytest.mark.parametrize("omega", [-2.0, -0.1, 0.0, 0.1, 2.0])
def test_viscous_mode_is_exactly_legacy_friction(omega: float) -> None:
    torque, next_state, rate = _step(omega, 123.0, mode=0)

    assert torque == SIGMA_2 * omega
    assert next_state == 0.0
    assert rate == 0.0


def test_lugre_reversal_sequence_remains_finite() -> None:
    state = 0.0
    speeds = np.concatenate(
        (
            np.linspace(0.0, 2.0, 500),
            np.linspace(2.0, -2.0, 1_000),
            np.linspace(-2.0, 0.0, 500),
        )
    )

    for omega in speeds:
        torque, state, rate = _step(float(omega), state)
        assert math.isfinite(torque)
        assert math.isfinite(state)
        assert math.isfinite(rate)

    assert abs(state) <= TAU_S / SIGMA_0 + 1e-12


@pytest.mark.parametrize(
    ("parameter_index", "invalid_value"),
    [
        (4, 0.0),
        (7, 0.0),
        (8, 0.0),
        (9, TAU_C / 2.0),
        (10, 0.0),
    ],
)
def test_lugre_rejects_nonphysical_parameters(
    parameter_index: int,
    invalid_value: float,
) -> None:
    arguments = [
        1,
        0.1,
        0.0,
        DT,
        SIGMA_0,
        SIGMA_1,
        SIGMA_2,
        OMEGA_S,
        TAU_C,
        TAU_S,
        ALPHA,
    ]
    arguments[parameter_index] = invalid_value

    with pytest.raises(ValueError):
        lugre_friction_step(*arguments)
