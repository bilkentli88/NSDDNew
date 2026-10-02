#!/usr/bin/env python3
"""Core scalar utilities for the TNNLS strengthening experiments.

The module contains the ground-truth scalar state-dependent DDE, controlled
history designs, differentiable explicit-Euler rollout, neural model classes,
profile-anchor estimation, and common evaluation helpers.  It is intentionally
small, CPU-only, and independent of manuscript-specific output files.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from numpy_data import (
    C5,
    D5,
    DT,
    FULL_TIME,
    FUTURE_TIME,
    HIST_STEPS,
    HORIZON,
    N_STEPS,
    RHO_STRONG,
    R_GRID,
    TAU_MAX,
    TAU_MIN,
    TAU_RANGE,
    a_true_np,
    add_observation_noise,
    b_true_np,
    estimate_profile_anchors,
    interpolate_numpy,
    local_polynomial_derivative,
    make_forced_ramp_dataset,
    make_strong_dataset,
    profile_residual,
    simulate_ground_truth,
    strong_history,
    tau_true_np,
)


DEVICE = torch.device("cpu")

torch.set_num_threads(1)


def set_seed(seed: int) -> None:
    """Set all random generators used by the experiments."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int, depth: int = 2):
        super().__init__()
        dims = [input_dim] + [width] * depth + [output_dim]
        layers: list[nn.Module] = []
        for index in range(len(dims) - 2):
            layers.extend([nn.Linear(dims[index], dims[index + 1]), nn.Tanh()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnedTau(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MLP(1, 1, width=20, depth=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TAU_MIN + TAU_RANGE * torch.sigmoid(self.net(x))


class RestrictedLearnedModel(nn.Module):
    """Affine delayed-state model with learned a, b, and state-dependent delay."""

    def __init__(self):
        super().__init__()
        self.a_net = MLP(1, 1, width=28, depth=2)
        self.b_net = MLP(1, 1, width=22, depth=2)
        self.tau_net = LearnedTau()

    def tau(self, x: torch.Tensor) -> torch.Tensor:
        return self.tau_net(x)

    def derivative(self, x: torch.Tensor, delayed: torch.Tensor) -> torch.Tensor:
        gain = 0.05 + torch.nn.functional.softplus(self.b_net(x))
        return self.a_net(x) + gain * delayed


class OracleRestrictedModel(nn.Module):
    """Restricted model with the generating delay fixed and a,b learned."""

    def __init__(self):
        super().__init__()
        self.a_net = MLP(1, 1, width=28, depth=2)
        self.b_net = MLP(1, 1, width=22, depth=2)

    def tau(self, x: torch.Tensor) -> torch.Tensor:
        return 0.30 + 0.08 * torch.tanh(1.30 * x)

    def derivative(self, x: torch.Tensor, delayed: torch.Tensor) -> torch.Tensor:
        gain = 0.05 + torch.nn.functional.softplus(self.b_net(x))
        return self.a_net(x) + gain * delayed


class UnrestrictedLearnedModel(nn.Module):
    """Unrestricted delayed-state vector field with a jointly learned delay."""

    def __init__(self):
        super().__init__()
        self.f_net = MLP(2, 1, width=76, depth=2)
        self.tau_net = LearnedTau()

    def tau(self, x: torch.Tensor) -> torch.Tensor:
        return self.tau_net(x)

    def derivative(self, x: torch.Tensor, delayed: torch.Tensor) -> torch.Tensor:
        return self.f_net(torch.cat([x, delayed], dim=-1))


class UnrestrictedFixedDelayModel(nn.Module):
    def __init__(self, delay: float = 0.46):
        super().__init__()
        self.delay = float(delay)
        self.f_net = MLP(2, 1, width=76, depth=2)

    def tau(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self.delay)

    def derivative(self, x: torch.Tensor, delayed: torch.Tensor) -> torch.Tensor:
        return self.f_net(torch.cat([x, delayed], dim=-1))


class RestrictedFixedDelayModel(nn.Module):
    """Same a,b architecture as RestrictedLearnedModel with a fixed delay."""

    def __init__(self, delay: float = 0.46):
        super().__init__()
        self.delay = float(delay)
        self.a_net = MLP(1, 1, width=28, depth=2)
        self.b_net = MLP(1, 1, width=22, depth=2)

    def tau(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self.delay)

    def derivative(self, x: torch.Tensor, delayed: torch.Tensor) -> torch.Tensor:
        gain = 0.05 + torch.nn.functional.softplus(self.b_net(x))
        return self.a_net(x) + gain * delayed


def initialize_tau_constant(model: nn.Module, target_delay: float) -> None:
    """Initialize a learned delay network to an exact constant function."""
    if not hasattr(model, "tau_net"):
        raise TypeError("The model does not have a learned tau_net.")
    target = float(np.clip(target_delay, TAU_MIN + 1e-6, TAU_MAX - 1e-6))
    probability = (target - TAU_MIN) / TAU_RANGE
    logit = math.log(probability / (1.0 - probability))
    final_linear = model.tau_net.net.net[-1]
    if not isinstance(final_linear, nn.Linear):
        raise TypeError("Unexpected delay-network architecture.")
    with torch.no_grad():
        # Preserve the randomly initialized hidden representation.  Zeroing
        # only the output weights gives an exact initial constant while still
        # allowing the delay to become state-dependent during optimization.
        final_linear.weight.zero_()
        final_linear.bias.fill_(logit)


def interpolate_torch(values: torch.Tensor, query_times: torch.Tensor) -> torch.Tensor:
    index = (query_times + TAU_MAX) / DT
    i0 = torch.floor(index).long().clamp(0, values.shape[1] - 2)
    weight = index - i0.to(index.dtype)
    value0 = torch.gather(values, 1, i0)
    value1 = torch.gather(values, 1, i0 + 1)
    return value0 * (1.0 - weight) + value1 * weight


def differentiable_rollout(
    model: nn.Module,
    histories: torch.Tensor,
    steps: int,
    known_inputs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Roll out a scalar SDDDE, optionally adding a known input at each step."""
    sequence = [histories[:, index] for index in range(histories.shape[1])]
    for step in range(steps):
        time = step * DT
        current = sequence[-1].unsqueeze(-1)
        delay = model.tau(current).squeeze(-1)
        values = torch.stack(sequence, dim=1)
        delayed = interpolate_torch(values, (time - delay).unsqueeze(1))
        derivative = model.derivative(current, delayed).squeeze(-1)
        if known_inputs is not None:
            derivative = derivative + known_inputs[:, step]
        sequence.append(sequence[-1] + DT * derivative)
    return torch.stack(sequence, dim=1)


@torch.no_grad()
def rollout_numpy(
    model: nn.Module,
    histories: np.ndarray,
    steps: int = N_STEPS,
    known_inputs: np.ndarray | None = None,
) -> np.ndarray:
    model.eval()
    history_tensor = torch.tensor(histories, dtype=torch.float32, device=DEVICE)
    input_tensor = (
        torch.tensor(known_inputs, dtype=torch.float32, device=DEVICE)
        if known_inputs is not None
        else None
    )
    return differentiable_rollout(model, history_tensor, steps, input_tensor).cpu().numpy()


def delay_curve(
    model: nn.Module,
    state_grid: np.ndarray,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        states = torch.tensor(state_grid, dtype=torch.float32).reshape(-1, 1)
        return model.tau(states).squeeze(-1).cpu().numpy()


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
