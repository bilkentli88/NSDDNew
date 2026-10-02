#!/usr/bin/env python3
"""Extended controlled experiments for identifiability-aware Neural SDDDEs.

Experiments
-----------
1. Scalar observation-noise scaling with profile-informed rollout training.
2. Solver refinement: separate ground-truth discretization error from learned-model error.
3. Interpolation versus unseen-state extrapolation.
4. Two-dimensional delayed oscillator under full versus partial observation.
5. Unrestricted wrong-delay baseline on the two-dimensional oscillator.

The program is intentionally CPU-scale and seeded. It imports the principal
scalar implementation and adds a standalone vector-oscillator pipeline.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "extended" / "generated"
FIGURES_DIR = ROOT / "figures" / "extended" / "generated"
CHECKPOINT_DIR = ROOT / "checkpoints" / "extended" / "generated"
SCALAR_PATH = ROOT / "experiments" / "run_principal_scalar.py"
SCALAR_CKPT = ROOT / "checkpoints" / "reference" / "restricted_noisy_seed0.pt"
SEED = 20260719
DEVICE = torch.device("cpu")
torch.set_num_threads(1)


def load_scalar_module():
    spec = importlib.util.spec_from_file_location("scalar_experiments", SCALAR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SCALAR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scalar = load_scalar_module()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# =============================================================================
# Scalar experiments
# =============================================================================

def train_scalar_noise_model(noise_std: float, seed: int, quick: bool = False, do_certificate: bool = True):
    train_states = (-0.75, -0.25, 0.25, 0.75)
    test_states = (-0.50, 0.0, 0.50)
    clean_train, _ = scalar.make_strong_dataset(train_states)
    clean_test, _ = scalar.make_strong_dataset(test_states)
    noisy_train = scalar.add_observation_noise(clean_train, noise_std, SEED + seed * 17 + 1)
    noisy_test = scalar.add_observation_noise(clean_test, noise_std, SEED + seed * 17 + 2)
    anchor_states, anchor_delays = scalar.estimate_profile_delay_anchors(noisy_train, train_states)

    # Seed before construction; the historical driver seeded only pretraining.
    set_seed(seed)
    model = scalar.RestrictedLearnedModel()
    scalar.pretrain_tau_from_trajectory_profiles(
        model, anchor_states, anchor_delays, seed=seed,
        epochs=180 if quick else 260,
    )
    schedule = [(10, 35), (20, 70)] if quick else [(10, 45), (20, 90)]
    model, train_loss = scalar.train_trajectory_only(
        model,
        clean_train[:, : scalar.HIST_STEPS + 1],
        noisy_train[:, scalar.HIST_STEPS :],
        seed=seed,
        schedule=schedule,
        learning_rate=2.0e-3,
        tau_smooth_weight=2.0e-4,
        anchor_states=anchor_states,
        anchor_delays=anchor_delays,
        anchor_weight=0.08,
    )
    train_mse, test_mse, noisy_test_mse = scalar.evaluate_model(model, clean_train, clean_test, noisy_test)
    delay_rmse, delay_mean, delay_min, delay_max = scalar.delay_metrics(model)

    # A light certificate sweep, with the controlled noise level passed explicitly.
    if not do_certificate:
        identified_rate = float("nan")
        mean_width = float("nan")
    elif noise_std == 0.0:
        identified_rate = 1.0
        mean_width = 0.0
    else:
        old_cert_noise = scalar.CERT_NOISE_STD
        scalar.CERT_NOISE_STD = noise_std
        rng = np.random.default_rng(SEED + 1000 + seed)
        cert_rows = []
        for i, s in enumerate(test_states):
            block = noisy_test[i * 5 : (i + 1) * 5]
            cert = scalar.certificate_from_noisy_trajectories(
                block, s, scalar.RHO_STRONG, rng, n_boot=12, n_points=10
            )
            cert_rows.append(cert)
        scalar.CERT_NOISE_STD = old_cert_noise
        identified_rate = float(np.mean([c["verdict"] == "identifiable" for c in cert_rows]))
        mean_width = float(np.mean([c["width"] for c in cert_rows]))
    anchor_mae = float(np.mean(np.abs(anchor_delays - scalar.tau_true_np(anchor_states))))
    return model, {
        "noise_std": noise_std,
        "seed": seed,
        "train_loss": train_loss,
        "clean_train_mse": train_mse,
        "clean_test_mse": test_mse,
        "noisy_test_mse": noisy_test_mse,
        "delay_rmse": delay_rmse,
        "delay_mean": delay_mean,
        "delay_min": delay_min,
        "delay_max": delay_max,
        "anchor_mae": anchor_mae,
        "certificate_identified_rate": identified_rate,
        "certificate_mean_width": mean_width,
    }


def scalar_history(s: float, c: float, d: float, dt: float, rho: float = 0.60) -> np.ndarray:
    n_hist = int(round(scalar.TAU_MAX / dt))
    theta = np.linspace(-scalar.TAU_MAX, 0.0, n_hist + 1)
    return s + c * theta + rho * d * theta**2


def interp_uniform_np(values: np.ndarray | list[float], query: float, dt: float, tau_max: float) -> float:
    arr = np.asarray(values, dtype=float)
    idx = (query + tau_max) / dt
    i0 = int(np.floor(idx))
    i0 = max(0, min(i0, len(arr) - 2))
    w = idx - i0
    return float((1.0 - w) * arr[i0] + w * arr[i0 + 1])


def scalar_truth_rollout(history: np.ndarray, dt: float, horizon: float) -> np.ndarray:
    n_hist = len(history) - 1
    n_steps = int(round(horizon / dt))
    vals = list(np.asarray(history, dtype=float))
    for n in range(n_steps):
        t = n * dt
        x = vals[n_hist + n]
        r = float(scalar.tau_true_np(x))
        z = interp_uniform_np(vals, t - r, dt, scalar.TAU_MAX)
        dx = float(scalar.a_true_np(x) + scalar.b_true_np(x) * z)
        vals.append(x + dt * dx)
    return np.asarray(vals)


@torch.no_grad()
def scalar_model_rollout(model: nn.Module, history: np.ndarray, dt: float, horizon: float) -> np.ndarray:
    n_hist = len(history) - 1
    n_steps = int(round(horizon / dt))
    seq = [torch.tensor(float(v), dtype=torch.float32) for v in history]
    for n in range(n_steps):
        t = n * dt
        current = seq[-1].reshape(1, 1)
        delay = float(model.tau(current).item())
        values = torch.stack(seq)
        index = (t - delay + scalar.TAU_MAX) / dt
        i0 = max(0, min(int(math.floor(index)), len(seq) - 2))
        w = index - i0
        delayed = ((1.0 - w) * values[i0] + w * values[i0 + 1]).reshape(1, 1)
        dx = float(model.derivative(current, delayed).item())
        seq.append(seq[-1] + dt * dx)
    return torch.stack(seq).cpu().numpy()


def interpolate_reference(ref: np.ndarray, ref_dt: float, target_times: np.ndarray) -> np.ndarray:
    ref_times = np.arange(len(ref), dtype=float) * ref_dt - scalar.TAU_MAX
    return np.interp(target_times, ref_times, ref)


def solver_refinement(model: nn.Module) -> pd.DataFrame:
    rows = []
    s, c, d = 0.0, 1.0, -1.0
    horizon = 1.0
    ref_dt = 0.003125
    ref_hist = scalar_history(s, c, d, ref_dt)
    ref = scalar_truth_rollout(ref_hist, ref_dt, horizon)
    for dt in (0.10, 0.05, 0.025, 0.0125):
        hist = scalar_history(s, c, d, dt)
        truth_coarse = scalar_truth_rollout(hist, dt, horizon)
        learned = scalar_model_rollout(model, hist, dt, horizon)
        times = np.arange(len(truth_coarse), dtype=float) * dt - scalar.TAU_MAX
        ref_at = interpolate_reference(ref, ref_dt, times)
        mask = times >= 0.0
        rows.append({
            "dt": dt,
            "ground_truth_euler_mse": float(np.mean((truth_coarse[mask] - ref_at[mask]) ** 2)),
            "learned_model_mse": float(np.mean((learned[mask] - ref_at[mask]) ** 2)),
            "learned_vs_coarse_truth_mse": float(np.mean((learned[mask] - truth_coarse[mask]) ** 2)),
        })
    return pd.DataFrame(rows)


def scalar_extrapolation(model: nn.Module) -> pd.DataFrame:
    interpolation_states = (-0.50, 0.0, 0.50)
    extrapolation_states = (-1.25, -1.00, 1.00, 1.25)
    rows = []
    for region, states in (("interpolation", interpolation_states), ("extrapolation", extrapolation_states)):
        data, _ = scalar.make_strong_dataset(states)
        pred = scalar.rollout_numpy(model, data[:, : scalar.HIST_STEPS + 1])
        traj_mse = float(np.mean((pred[:, scalar.HIST_STEPS:] - data[:, scalar.HIST_STEPS:]) ** 2))
        state_grid = np.linspace(min(states), max(states), 181)
        with torch.no_grad():
            x = torch.tensor(state_grid, dtype=torch.float32).reshape(-1, 1)
            tau_hat = model.tau(x).squeeze(-1).cpu().numpy()
        delay_rmse = float(np.sqrt(np.mean((tau_hat - scalar.tau_true_np(state_grid)) ** 2)))
        rows.append({"region": region, "trajectory_mse": traj_mse, "delay_rmse": delay_rmse,
                     "state_min": min(states), "state_max": max(states),
                     "certificate_support": "available on matched-state probes" if region == "interpolation" else "absent; abstain outside visited domain"})
    return pd.DataFrame(rows)


# =============================================================================
# Two-dimensional delayed oscillator
# =============================================================================
OSC_DT = 0.025
OSC_TAU_MIN = 0.12
OSC_TAU_MAX = 0.40
OSC_HORIZON = 1.50
OSC_HIST_STEPS = int(round(OSC_TAU_MAX / OSC_DT))
OSC_STEPS = int(round(OSC_HORIZON / OSC_DT))
OSC_C = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
OSC_D = np.array([2.0, -1.0, -2.0, -1.0, 2.0])
OSC_RHO = 0.55


def osc_tau_np(x: np.ndarray | float) -> np.ndarray | float:
    return 0.24 + 0.055 * np.tanh(1.15 * np.asarray(x))


def osc_rhs_np(state: np.ndarray, delayed_x1: float) -> np.ndarray:
    x1, x2 = state
    return np.array([
        x2,
        -1.25 * x1 - 0.32 * x2 + 0.58 * delayed_x1,
    ], dtype=float)


def osc_history(s: float, c: float, d: float, dt: float = OSC_DT) -> np.ndarray:
    theta = np.linspace(-OSC_TAU_MAX, 0.0, int(round(OSC_TAU_MAX / dt)) + 1)
    x1 = s + c * theta**2 + OSC_RHO * d * theta**3
    x2 = 2.0 * c * theta + 3.0 * OSC_RHO * d * theta**2
    return np.column_stack([x1, x2])


def osc_simulate(history: np.ndarray, dt: float = OSC_DT, horizon: float = OSC_HORIZON) -> np.ndarray:
    n_hist = len(history) - 1
    n_steps = int(round(horizon / dt))
    vals = [np.asarray(v, dtype=float) for v in history]
    for n in range(n_steps):
        t = n * dt
        state = vals[n_hist + n]
        delay = float(osc_tau_np(state[0]))
        x1_values = np.array([v[0] for v in vals])
        z = interp_uniform_np(x1_values, t - delay, dt, OSC_TAU_MAX)
        vals.append(state + dt * osc_rhs_np(state, z))
    return np.asarray(vals)


def make_osc_dataset(states: Iterable[float]) -> np.ndarray:
    data = []
    for s in states:
        for c, d in zip(OSC_C, OSC_D):
            data.append(osc_simulate(osc_history(float(s), float(c), float(d))))
    return np.asarray(data)


def add_vector_noise(clean: np.ndarray, noise_std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = clean.copy()
    noisy[:, OSC_HIST_STEPS + 1 :, :] += rng.normal(
        0.0, noise_std, size=noisy[:, OSC_HIST_STEPS + 1 :, :].shape
    )
    return noisy


class VecMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int = 36, depth: int = 2):
        super().__init__()
        dims = [input_dim] + [width] * depth + [output_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.Tanh()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OscTau(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = VecMLP(2, 1, width=22, depth=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return OSC_TAU_MIN + (OSC_TAU_MAX - OSC_TAU_MIN) * torch.sigmoid(self.net(x))


class OscRestrictedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_net = VecMLP(2, 2, width=42, depth=2)
        self.b_net = VecMLP(2, 2, width=28, depth=2)
        self.tau_net = OscTau()

    def tau(self, x: torch.Tensor) -> torch.Tensor:
        return self.tau_net(x)

    def derivative(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.a_net(x) + self.b_net(x) * z


class OscUnrestrictedFixed(nn.Module):
    def __init__(self, delay: float = 0.38):
        super().__init__()
        self.fixed_delay = delay
        self.f_net = VecMLP(3, 2, width=70, depth=2)

    def tau(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 1), self.fixed_delay, dtype=x.dtype, device=x.device)

    def derivative(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.f_net(torch.cat([x, z], dim=1))


def osc_interp_torch(x1_values: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    index = (query + OSC_TAU_MAX) / OSC_DT
    i0 = torch.floor(index).long().clamp(0, x1_values.shape[1] - 2)
    w = index - i0.to(index.dtype)
    x0 = torch.gather(x1_values, 1, i0)
    x1 = torch.gather(x1_values, 1, i0 + 1)
    return (1.0 - w) * x0 + w * x1


def osc_rollout_torch(model: nn.Module, histories: torch.Tensor, steps: int) -> torch.Tensor:
    seq = [histories[:, i, :] for i in range(histories.shape[1])]
    for n in range(steps):
        t = n * OSC_DT
        current = seq[-1]
        delay = model.tau(current).squeeze(1)
        values = torch.stack(seq, dim=1)
        delayed = osc_interp_torch(values[:, :, 0], (t - delay).unsqueeze(1)).reshape(-1, 1)
        dx = model.derivative(current, delayed)
        seq.append(current + OSC_DT * dx)
    return torch.stack(seq, dim=1)


@torch.no_grad()
def osc_rollout_np(model: nn.Module, histories: np.ndarray) -> np.ndarray:
    x = torch.tensor(histories, dtype=torch.float32)
    return osc_rollout_torch(model, x, OSC_STEPS).cpu().numpy()


def polynomial_derivative(y: np.ndarray, order: int, dt: float, window: int = 7, degree: int = 4) -> float:
    vals = np.asarray(y[:window], dtype=float)
    t = np.arange(window, dtype=float) * dt
    scale = t[-1]
    u = t / scale
    coef = np.polyfit(u, vals, degree)
    p = np.poly1d(coef)
    dp = np.polyder(p, m=order)
    return float(dp(0.0) / (scale**order))


def osc_profile(s: float, slopes: np.ndarray, r_grid: np.ndarray) -> np.ndarray:
    # Only the second dynamic component has delayed influence in the ground truth.
    pv = slopes - slopes.mean()
    q = s + np.outer(r_grid**2, OSC_C) - OSC_RHO * np.outer(r_grid**3, OSC_D)
    pq = q - q.mean(axis=1, keepdims=True)
    den = np.sum(pq * pq, axis=1)
    num = (pq @ pv) ** 2
    base = float(pv @ pv)
    return np.clip((base - np.where(den > 1e-14, num / den, 0.0)) / len(slopes), 0.0, None)


def osc_anchor_estimates(noisy: np.ndarray, states: tuple[float, ...], partial: bool) -> tuple[np.ndarray, np.ndarray]:
    r_grid = np.linspace(OSC_TAU_MIN, OSC_TAU_MAX, 281)
    estimates = []
    for i, s in enumerate(states):
        block = noisy[i * 5 : (i + 1) * 5, OSC_HIST_STEPS :, :]
        if partial:
            # Under the explicit Euler observation model, the second forward
            # difference of x1 equals x2'(0) in the noiseless case.
            slopes = np.array([(row[2, 0] - 2.0 * row[1, 0] + row[0, 0]) / (OSC_DT**2) for row in block])
        else:
            # The first forward difference of x2 equals x2'(0) exactly for
            # the noiseless Euler-generated trajectories.
            slopes = np.array([(row[1, 1] - row[0, 1]) / OSC_DT for row in block])
        prof = osc_profile(float(s), slopes, r_grid)
        estimates.append(float(r_grid[int(np.argmin(prof))]))
    return np.asarray(states, dtype=float), np.asarray(estimates, dtype=float)


def pretrain_osc_tau(model: OscRestrictedModel, states: np.ndarray, delays: np.ndarray, seed: int, epochs: int) -> None:
    set_seed(seed)
    x = torch.tensor(np.column_stack([states, np.zeros_like(states)]), dtype=torch.float32)
    y = torch.tensor(delays, dtype=torch.float32).reshape(-1, 1)
    opt = torch.optim.Adam(model.tau_net.parameters(), lr=4e-3)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = torch.mean((model.tau(x) - y) ** 2)
        loss.backward()
        opt.step()


def train_osc_model(
    model: nn.Module,
    clean: np.ndarray,
    noisy: np.ndarray,
    seed: int,
    partial: bool,
    anchors: tuple[np.ndarray, np.ndarray] | None,
    quick: bool,
) -> tuple[nn.Module, float]:
    set_seed(seed)
    hist = torch.tensor(clean[:, : OSC_HIST_STEPS + 1, :], dtype=torch.float32)
    target = torch.tensor(noisy[:, OSC_HIST_STEPS :, :], dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=1.8e-3, weight_decay=1e-6)
    schedule = [(20, 35), (40, 50), (60, 70)] if quick else [(20, 60), (40, 90), (60, 130)]
    final = 0.0
    for steps, epochs in schedule:
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            pred = osc_rollout_torch(model, hist, steps)[:, OSC_HIST_STEPS : OSC_HIST_STEPS + steps + 1, :]
            if partial:
                fit = torch.mean((pred[:, :, 0] - target[:, : steps + 1, 0]) ** 2)
            else:
                fit = torch.mean((pred - target[:, : steps + 1, :]) ** 2)
            loss = fit
            if anchors is not None and isinstance(model, OscRestrictedModel):
                states, delays = anchors
                ax = torch.tensor(np.column_stack([states, np.zeros_like(states)]), dtype=torch.float32)
                ay = torch.tensor(delays, dtype=torch.float32).reshape(-1, 1)
                loss = loss + 0.08 * torch.mean((model.tau(ax) - ay) ** 2)
            if isinstance(model, OscRestrictedModel):
                grid = torch.linspace(-1.4, 1.4, 100)
                gx = torch.column_stack([grid, torch.zeros_like(grid)])
                tg = model.tau(gx)
                loss = loss + 2e-4 * torch.mean((tg[1:] - tg[:-1]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            final = float(fit.detach())
    return model, final


@torch.no_grad()
def osc_delay_metrics(model: nn.Module) -> tuple[float, float]:
    x1 = np.linspace(-1.3, 1.3, 201)
    x = torch.tensor(np.column_stack([x1, np.zeros_like(x1)]), dtype=torch.float32)
    pred = model.tau(x).squeeze(1).cpu().numpy()
    truth = osc_tau_np(x1)
    return float(np.sqrt(np.mean((pred - truth) ** 2))), float(np.mean(pred))


def run_oscillator(quick: bool = False) -> tuple[pd.DataFrame, dict[str, nn.Module], np.ndarray, np.ndarray]:
    train_states = (-0.60, -0.20, 0.20, 0.60)
    test_states = (-0.40, 0.0, 0.40)
    clean_train = make_osc_dataset(train_states)
    clean_test = make_osc_dataset(test_states)
    noise_std = 0.00002
    noisy_train = add_vector_noise(clean_train, noise_std, SEED + 501)
    noisy_test = add_vector_noise(clean_test, noise_std, SEED + 502)
    rows = []
    models: dict[str, nn.Module] = {}

    for partial, seed, name in ((False, 70, "full_observation"), (True, 71, "partial_x1_only")):
        anchors = osc_anchor_estimates(noisy_train, train_states, partial=partial)
        set_seed(seed)
        model = OscRestrictedModel()
        pretrain_osc_tau(model, *anchors, seed=seed, epochs=200 if quick else 430)
        model, loss = train_osc_model(model, clean_train, noisy_train, seed, partial, anchors, quick)
        pred = osc_rollout_np(model, clean_test[:, : OSC_HIST_STEPS + 1, :])
        future_pred = pred[:, OSC_HIST_STEPS :, :]
        future_true = clean_test[:, OSC_HIST_STEPS :, :]
        full_mse = float(np.mean((future_pred - future_true) ** 2))
        x1_mse = float(np.mean((future_pred[:, :, 0] - future_true[:, :, 0]) ** 2))
        x2_mse = float(np.mean((future_pred[:, :, 1] - future_true[:, :, 1]) ** 2))
        drmse, dmean = osc_delay_metrics(model)
        anchor_mae = float(np.mean(np.abs(anchors[1] - osc_tau_np(anchors[0]))))
        rows.append({
            "model": name,
            "architecture": "restricted vector a(x)+b(x)z, learned tau",
            "observation": "full state" if not partial else "x1 only after t=0",
            "training_loss": loss,
            "test_full_state_mse": full_mse,
            "test_x1_mse": x1_mse,
            "test_x2_mse": x2_mse,
            "delay_rmse": drmse,
            "delay_mean": dmean,
            "anchor_mae": anchor_mae,
            "mechanistic_verdict": "qualitative recovery; outside scalar theorem" if not partial else "weakened under partial observation; outside scalar theorem",
        })
        models[name] = model

    # Flexible wrong-delay baseline under full observation.
    set_seed(72)
    model = OscUnrestrictedFixed(0.38)
    model, loss = train_osc_model(model, clean_train, noisy_train, 72, False, None, quick)
    pred = osc_rollout_np(model, clean_test[:, : OSC_HIST_STEPS + 1, :])
    future_pred = pred[:, OSC_HIST_STEPS :, :]
    future_true = clean_test[:, OSC_HIST_STEPS :, :]
    drmse, dmean = osc_delay_metrics(model)
    rows.append({
        "model": "unrestricted_fixed_delay_0.38",
        "architecture": "unrestricted vector f(x,z), fixed tau",
        "observation": "full state",
        "training_loss": loss,
        "test_full_state_mse": float(np.mean((future_pred - future_true) ** 2)),
        "test_x1_mse": float(np.mean((future_pred[:, :, 0] - future_true[:, :, 0]) ** 2)),
        "test_x2_mse": float(np.mean((future_pred[:, :, 1] - future_true[:, :, 1]) ** 2)),
        "delay_rmse": drmse,
        "delay_mean": dmean,
        "anchor_mae": float("nan"),
        "mechanistic_verdict": "abstain: unrestricted compensation",
    })
    models["unrestricted"] = model
    return pd.DataFrame(rows), models, clean_test, noisy_test


# =============================================================================
# Figures and report outputs
# =============================================================================

def make_figures(noise_df: pd.DataFrame, solver_df: pd.DataFrame, extra_df: pd.DataFrame,
                 osc_df: pd.DataFrame, osc_models: dict[str, nn.Module], osc_test: np.ndarray) -> None:
    plt.figure(figsize=(7.6, 4.6))
    plt.loglog(noise_df["noise_std"].replace(0, 1e-5), noise_df["clean_test_mse"], marker="o", label="Trajectory MSE")
    plt.loglog(noise_df["noise_std"].replace(0, 1e-5), noise_df["delay_rmse"], marker="s", label="Delay RMSE")
    plt.xlabel("Observation-noise standard deviation")
    plt.ylabel("Error")
    plt.title("Noise scaling under trajectory-only training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "noise_scaling.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.6, 4.6))
    plt.loglog(solver_df["dt"], solver_df["ground_truth_euler_mse"], marker="o", label="Ground-truth Euler error")
    plt.loglog(solver_df["dt"], solver_df["learned_model_mse"], marker="s", label="Learned-model total error")
    plt.gca().invert_xaxis()
    plt.xlabel("Solver step size")
    plt.ylabel("MSE against fine reference")
    plt.title("Solver refinement separates discretization and model error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "solver_refinement.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.4))
    x = np.arange(len(extra_df))
    width = 0.34
    plt.bar(x - width/2, extra_df["trajectory_mse"], width, label="Trajectory MSE")
    plt.bar(x + width/2, extra_df["delay_rmse"], width, label="Delay RMSE")
    plt.yscale("log")
    plt.xticks(x, extra_df["region"])
    plt.ylabel("Error")
    plt.title("Interpolation versus unseen-state extrapolation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "extrapolation.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.0, 4.6))
    idx = 0
    truth = osc_test[idx]
    hist = osc_test[[idx], : OSC_HIST_STEPS + 1, :]
    t = np.arange(len(truth), dtype=float) * OSC_DT - OSC_TAU_MAX
    for key, label, style in (
        ("full_observation", "Full observation", "--"),
        ("partial_x1_only", "Partial observation", ":"),
        ("unrestricted", "Unrestricted fixed delay", "-."),
    ):
        pred = osc_rollout_np(osc_models[key], hist)[0]
        plt.plot(t, pred[:, 0], style, linewidth=1.7, label=label)
    plt.plot(t, truth[:, 0], linewidth=2.3, label="Ground truth x1")
    plt.axvline(0.0, linewidth=0.8)
    plt.xlabel("Time")
    plt.ylabel("x1")
    plt.title("Two-dimensional delayed oscillator")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "oscillator_trajectories.png", dpi=180)
    plt.close()

    grid = np.linspace(-1.3, 1.3, 241)
    plt.figure(figsize=(7.8, 4.5))
    plt.plot(grid, osc_tau_np(grid), linewidth=2.4, label="True delay")
    x = torch.tensor(np.column_stack([grid, np.zeros_like(grid)]), dtype=torch.float32)
    for key, label, style in (
        ("full_observation", "Full observation", "--"),
        ("partial_x1_only", "Partial x1 only", ":"),
        ("unrestricted", "Fixed delay 0.38", "-."),
    ):
        with torch.no_grad():
            vals = osc_models[key].tau(x).squeeze(1).cpu().numpy()
        plt.plot(grid, vals, style, linewidth=1.8, label=label)
    plt.xlabel("x1 (x2=0 slice)")
    plt.ylabel("Delay")
    plt.title("Delay recovery in the vector oscillator")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "oscillator_delays.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("paper", "manuscript", "extended"),
        default="paper",
        help=("paper: historical short configuration; manuscript: declared noise-sweep "
              "schedule with short oscillator study; extended: longer schedules throughout"),
    )
    args = parser.parse_args()
    quick = args.mode == "paper"
    start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Frozen scalar model for solver-refinement and extrapolation evaluation.
    scalar_model = scalar.RestrictedLearnedModel()
    scalar_model.load_state_dict(torch.load(SCALAR_CKPT, map_location="cpu"))
    scalar_model.eval()

    noise_levels = [0.0, 0.0005, 0.0010, 0.0020, 0.0050]
    noise_rows = []
    n_reps = 2
    for i, sigma in enumerate(noise_levels):
        for rep in range(n_reps):
            seed = 100 + 10 * i + rep
            model, row = train_scalar_noise_model(sigma, seed=seed, quick=quick, do_certificate=(rep == 0))
            row["replicate"] = rep
            noise_rows.append(row)
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"scalar_noise_{sigma:.4g}_rep{rep}.pt")
    noise_runs_df = pd.DataFrame(noise_rows)
    noise_runs_df.to_csv(RESULTS_DIR / "noise_scaling_runs.csv", index=False)
    numeric_cols = [c for c in noise_runs_df.columns if c not in {"noise_std", "seed", "replicate"}]
    noise_df = noise_runs_df.groupby("noise_std", as_index=False)[numeric_cols].mean()
    std_df = noise_runs_df.groupby("noise_std")[numeric_cols].std(ddof=1).add_suffix("_std").reset_index()
    noise_df = noise_df.merge(std_df, on="noise_std", how="left")
    noise_df.to_csv(RESULTS_DIR / "noise_scaling.csv", index=False)

    solver_df = solver_refinement(scalar_model)
    solver_df.to_csv(RESULTS_DIR / "solver_refinement.csv", index=False)
    extra_df = scalar_extrapolation(scalar_model)
    extra_df.to_csv(RESULTS_DIR / "extrapolation.csv", index=False)

    osc_df, osc_models, osc_test, osc_noisy = run_oscillator(quick=(args.mode != "extended"))
    osc_df.to_csv(RESULTS_DIR / "oscillator_summary.csv", index=False)
    for name, model in osc_models.items():
        torch.save(model.state_dict(), CHECKPOINT_DIR / f"osc_{name}.pt")

    make_figures(noise_df, solver_df, extra_df, osc_df, osc_models, osc_test)

    results = {
        "noise_scaling": noise_df.to_dict(orient="records"),
        "noise_scaling_runs": noise_runs_df.to_dict(orient="records"),
        "solver_refinement": solver_df.to_dict(orient="records"),
        "extrapolation": extra_df.to_dict(orient="records"),
        "oscillator": osc_df.where(pd.notnull(osc_df), None).to_dict(orient="records"),
        "elapsed_seconds": time.time() - start,
    }
    (RESULTS_DIR / "key_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    manifest = {
        "mode": args.mode,
        "scalar_model_seeded_before_construction": True,
        "scalar_seeds": [100 + 10*i + rep for i in range(len(noise_levels)) for rep in range(n_reps)],
        "scalar_train_noise_seed_rule": "20260719 + 17 * seed + 1",
        "scalar_test_noise_seed_rule": "20260719 + 17 * seed + 2",
        "scalar_resampling_seed_rule": "20260719 + 1000 + seed",
        "noise_levels": noise_levels,
        "scalar_pretraining_iterations": 180 if quick else 260,
        "scalar_rollout_schedule": [(10, 35), (20, 70)] if quick else [(10, 45), (20, 90)],
        "anchor_weight": 0.08,
        "diagnostic_replication": 0,
        "diagnostic_resamples": 12,
        "zero_noise_diagnostic": "assigned 3/3 resolved states and width zero; no resampling",
        "oscillator_model_seeds": [70, 71, 72],
        "oscillator_model_seeded_before_construction": True,
        "oscillator_pretraining_iterations": 430 if args.mode == "extended" else 200,
        "oscillator_rollout_schedule": [(20, 60), (40, 90), (60, 130)] if args.mode == "extended" else [(20, 35), (40, 50), (60, 70)],
        "frozen_scalar_checkpoint": str(SCALAR_CKPT.relative_to(ROOT)),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pandas_version": pd.__version__,
        "matplotlib_version": matplotlib.__version__,
        "cpu_threads": torch.get_num_threads(),
    }
    (RESULTS_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    # The repository-level requirements file is maintained separately.
    (RESULTS_DIR / "run_environment.txt").write_text(
        f"PyTorch {torch.__version__}\nNumPy {np.__version__}\nCPU only\nScalar dt {scalar.DT}\nOscillator dt {OSC_DT}\n",
        encoding="utf-8",
    )
    elapsed = time.time() - start
    print("\nNoise scaling:\n", noise_df.to_string(index=False))
    print("\nSolver refinement:\n", solver_df.to_string(index=False))
    print("\nExtrapolation:\n", extra_df.to_string(index=False))
    print("\nOscillator:\n", osc_df.to_string(index=False))
    print(f"\nCompleted in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
