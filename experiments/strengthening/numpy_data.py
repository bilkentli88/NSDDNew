"""NumPy-only history generation and anchor profiles, extracted unchanged from sdde_core.

This module permits observation/profile verification without importing PyTorch.
"""
from __future__ import annotations
from typing import Iterable
import numpy as np

DT = 0.05

TAU_MIN = 0.15

TAU_MAX = 0.50

TAU_RANGE = TAU_MAX - TAU_MIN

HORIZON = 1.0

HIST_STEPS = int(round(TAU_MAX / DT))

N_STEPS = int(round(HORIZON / DT))

FULL_TIME = np.linspace(-TAU_MAX, HORIZON, HIST_STEPS + N_STEPS + 1)

FUTURE_TIME = np.arange(N_STEPS + 1, dtype=float) * DT

C5 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

D5 = np.array([2.0, -1.0, -2.0, -1.0, 2.0])

RHO_STRONG = 0.60

R_GRID = np.linspace(TAU_MIN, TAU_MAX, 351)

def tau_true_np(x: np.ndarray | float) -> np.ndarray | float:
    return 0.30 + 0.08 * np.tanh(1.30 * np.asarray(x))

def a_true_np(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x)
    return -0.75 * x + 0.12 * np.sin(2.0 * x)

def b_true_np(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x)
    return 0.60 + 0.10 * np.cos(x)

def interpolate_numpy(values: list[float] | np.ndarray, query_time: float) -> float:
    values = np.asarray(values)
    index = (query_time + TAU_MAX) / DT
    i0 = max(0, min(int(np.floor(index)), len(values) - 2))
    weight = index - i0
    return float((1.0 - weight) * values[i0] + weight * values[i0 + 1])

def simulate_ground_truth(history: np.ndarray) -> np.ndarray:
    """Simulate the unforced scalar ground-truth DDE."""
    values = list(np.asarray(history, dtype=float))
    for n in range(N_STEPS):
        t = n * DT
        current = values[HIST_STEPS + n]
        delay = float(tau_true_np(current))
        delayed = interpolate_numpy(values, t - delay)
        derivative = float(a_true_np(current) + b_true_np(current) * delayed)
        values.append(current + DT * derivative)
    return np.asarray(values)

def strong_history(
    state: float,
    c: float,
    d: float,
    rho: float = RHO_STRONG,
) -> np.ndarray:
    theta = np.linspace(-TAU_MAX, 0.0, HIST_STEPS + 1)
    return state + c * theta + rho * d * theta**2

def make_strong_dataset(
    states: Iterable[float],
    c_values: np.ndarray = C5,
    d_values: np.ndarray = D5,
    rho: float = RHO_STRONG,
) -> tuple[np.ndarray, list[dict]]:
    """Generate projectively diverse trajectories from the ground-truth DDE."""
    trajectories: list[np.ndarray] = []
    metadata: list[dict] = []
    for state in states:
        for history_index, (c, d) in enumerate(zip(c_values, d_values)):
            history = strong_history(float(state), float(c), float(d), rho)
            trajectories.append(simulate_ground_truth(history))
            metadata.append(
                {
                    "state": float(state),
                    "history_index": history_index,
                    "c": float(c),
                    "d": float(d),
                    "rho": float(rho),
                }
            )
    return np.asarray(trajectories), metadata

def make_forced_ramp_dataset(
    intercepts: Iterable[float],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Generate exact ramps of the same ground-truth delayed system.

    The known input is selected so x_c(t)=c+t is an exact solution:

        u_c(t) = 1 - a*(x_c(t))
                   - b*(x_c(t)) x_c(t-tau*(x_c(t))).

    At every matched current state, all ramps have q_s(r)=s-r, so their
    centered delayed-history signature is exactly zero.
    """
    trajectories: list[np.ndarray] = []
    inputs: list[np.ndarray] = []
    metadata: list[dict] = []
    step_times = np.arange(N_STEPS, dtype=float) * DT
    for intercept in intercepts:
        trajectory = float(intercept) + FULL_TIME
        current = float(intercept) + step_times
        delayed_true = current - tau_true_np(current)
        known_input = 1.0 - a_true_np(current) - b_true_np(current) * delayed_true
        trajectories.append(trajectory)
        inputs.append(np.asarray(known_input, dtype=float))
        metadata.append({"intercept": float(intercept)})
    return np.asarray(trajectories), np.asarray(inputs), metadata

def add_observation_noise(data: np.ndarray, noise_std: float, seed: int) -> np.ndarray:
    """Add noise only after t=0 while preserving the designed history and x(0)."""
    rng = np.random.default_rng(seed)
    noisy = np.asarray(data, dtype=float).copy()
    noisy[:, HIST_STEPS + 1 :] += rng.normal(
        0.0,
        noise_std,
        size=noisy[:, HIST_STEPS + 1 :].shape,
    )
    return noisy

def local_polynomial_derivative(
    trajectory_future: np.ndarray,
    window: int = 5,
    degree: int = 2,
) -> float:
    y = np.asarray(trajectory_future[:window], dtype=float)
    t = FUTURE_TIME[:window]
    scale = t[-1] if t[-1] > 0 else 1.0
    coefficient = np.polyfit(t / scale, y, deg=degree)
    return float(coefficient[-2] / scale)

def profile_residual(
    slopes: np.ndarray,
    state: float,
    rho: float = RHO_STRONG,
    r_grid: np.ndarray = R_GRID,
    c_values: np.ndarray = C5,
    d_values: np.ndarray = D5,
) -> np.ndarray:
    """Normalized squared distance to the candidate centered history span."""
    slopes = np.asarray(slopes, dtype=float)
    centered_slopes = slopes - slopes.mean()
    histories = (
        state
        - np.outer(r_grid, c_values)
        + rho * np.outer(r_grid**2, d_values)
    )
    centered_histories = histories - histories.mean(axis=1, keepdims=True)
    denominator = np.sum(centered_histories**2, axis=1)
    numerator = (centered_histories @ centered_slopes) ** 2
    base = float(centered_slopes @ centered_slopes)
    with np.errstate(divide="ignore", invalid="ignore"):
        residual = base - np.where(
            denominator > 1e-14,
            numerator / denominator,
            0.0,
        )
    return np.clip(residual / len(slopes), 0.0, None)

def estimate_profile_anchors(
    noisy_data: np.ndarray,
    states: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    estimates: list[float] = []
    for state_index, state in enumerate(states):
        block = noisy_data[state_index * 5 : (state_index + 1) * 5, HIST_STEPS:]
        slopes = np.asarray(
            [local_polynomial_derivative(row) for row in block],
            dtype=float,
        )
        profile = profile_residual(slopes, float(state))
        estimates.append(float(R_GRID[int(np.argmin(profile))]))
    return np.asarray(states, dtype=float), np.asarray(estimates, dtype=float)
