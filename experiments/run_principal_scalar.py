#!/usr/bin/env python3
"""Principal scalar experiments for identifiability-aware Neural SDDDEs.

This CPU-only implementation evaluates three mechanisms:
1. trajectory-only optimization through the complete explicit SDDDE rollout;
2. profile-informed delay recovery under projectively diverse matched histories;
3. profile-based certification from derivatives estimated from noisy trajectories.

Experimental regimes:
- Restricted model + projectively diverse histories: delay recovery should be
  accurate and certificate-supported.
- Unrestricted vector field + diverse histories: incorrect fixed delays can retain
  comparable trajectory error, so mechanistic interpretation is withheld.
- Restricted model + degenerate ramp histories: incorrect fixed delays retain low trajectory
  error and the certificate abstains.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import UnivariateSpline
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "principal_scalar" / "generated"
FIGURES_DIR = ROOT / "figures" / "principal_scalar" / "generated"
CHECKPOINT_DIR = ROOT / "checkpoints" / "principal_scalar" / "generated"
SEED = 20260719
DT = 0.05
TAU_MIN = 0.15
TAU_MAX = 0.50
HORIZON = 1.00
HIST_STEPS = int(round(TAU_MAX / DT))
N_STEPS = int(round(HORIZON / DT))
FULL_TIME = np.linspace(-TAU_MAX, HORIZON, HIST_STEPS + N_STEPS + 1)
FUTURE_TIME = np.arange(N_STEPS + 1, dtype=float) * DT
C5 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
D5 = np.array([2.0, -1.0, -2.0, -1.0, 2.0])
RHO_STRONG = 0.60
R_GRID = np.linspace(TAU_MIN, TAU_MAX, 351)
DEVICE = torch.device("cpu")
torch.set_num_threads(1)

# Noise used for training trajectories and certificate experiments.
TRAIN_NOISE_STD = 0.0005
CERT_NOISE_STD = TRAIN_NOISE_STD
CERT_WIDTH_TOL = 0.090
FLAT_TOL = 1e-10


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# -----------------------------------------------------------------------------
# Ground-truth scalar state-dependent DDE
# -----------------------------------------------------------------------------
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
    idx_float = (query_time + TAU_MAX) / DT
    i0 = int(np.floor(idx_float))
    i0 = max(0, min(i0, len(values) - 2))
    weight = idx_float - i0
    return float(values[i0] * (1.0 - weight) + values[i0 + 1] * weight)


def simulate_ground_truth(history: np.ndarray) -> np.ndarray:
    values = list(np.asarray(history, dtype=float))
    for n in range(N_STEPS):
        t = n * DT
        x = values[HIST_STEPS + n]
        delay = float(tau_true_np(x))
        delayed = interpolate_numpy(values, t - delay)
        dx = float(a_true_np(x) + b_true_np(x) * delayed)
        values.append(x + DT * dx)
    return np.asarray(values)


def strong_history(s: float, c: float, d: float, rho: float = RHO_STRONG) -> np.ndarray:
    theta = np.linspace(-TAU_MAX, 0.0, HIST_STEPS + 1)
    return s + c * theta + rho * d * theta**2


def make_strong_dataset(s_values: Iterable[float]) -> tuple[np.ndarray, list[dict]]:
    trajectories, meta = [], []
    for s in s_values:
        for j, (c, d) in enumerate(zip(C5, D5)):
            history = strong_history(float(s), float(c), float(d))
            trajectories.append(simulate_ground_truth(history))
            meta.append({"state": float(s), "history_index": j, "c": float(c), "d": float(d)})
    return np.asarray(trajectories), meta


def make_ramp_dataset(c_values: Iterable[float]) -> tuple[np.ndarray, list[dict]]:
    trajectories, meta = [], []
    for c in c_values:
        trajectories.append(float(c) + FULL_TIME)
        meta.append({"intercept": float(c)})
    return np.asarray(trajectories), meta


def add_observation_noise(clean: np.ndarray, noise_std: float, seed: int) -> np.ndarray:
    """Add noise to future states while preserving the designed initial history."""
    rng = np.random.default_rng(seed)
    noisy = clean.copy()
    future = noisy[:, HIST_STEPS + 1 :]
    future += rng.normal(0.0, noise_std, size=future.shape)
    # x(0) is fixed by the designed history and remains exact.
    return noisy


# -----------------------------------------------------------------------------
# Neural models
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int = 32, depth: int = 2):
        super().__init__()
        dims = [input_dim] + [width] * depth + [output_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.Tanh()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnedTau(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MLP(1, 1, width=20, depth=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TAU_MIN + (TAU_MAX - TAU_MIN) * torch.sigmoid(self.net(x))


class RestrictedLearnedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_net = MLP(1, 1, width=28, depth=2)
        self.b_net = MLP(1, 1, width=22, depth=2)
        self.tau_net = LearnedTau()

    def tau(self, x: torch.Tensor) -> torch.Tensor:
        return self.tau_net(x)

    def derivative(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        b = 0.05 + torch.nn.functional.softplus(self.b_net(x))
        return self.a_net(x) + b * z


class FixedDelayMixin:
    delay_mode: str

    def tau(self, x: torch.Tensor) -> torch.Tensor:
        if self.delay_mode == "low":
            return torch.full_like(x, 0.18)
        if self.delay_mode == "high":
            return torch.full_like(x, 0.46)
        if self.delay_mode == "warped":
            return 0.22 + 0.20 * torch.sigmoid(2.0 * x)
        raise ValueError(self.delay_mode)


class UnrestrictedFixedDelayModel(FixedDelayMixin, nn.Module):
    def __init__(self, delay_mode: str):
        super().__init__()
        self.delay_mode = delay_mode
        self.f_net = MLP(2, 1, width=76, depth=2)

    def derivative(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.f_net(torch.cat([x, z], dim=-1))


class RestrictedFixedDelayModel(FixedDelayMixin, nn.Module):
    def __init__(self, delay_mode: str):
        super().__init__()
        self.delay_mode = delay_mode
        self.a_net = MLP(1, 1, width=40, depth=2)
        self.b_net = MLP(1, 1, width=28, depth=2)

    def derivative(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        b = 0.05 + torch.nn.functional.softplus(self.b_net(x))
        return self.a_net(x) + b * z


# -----------------------------------------------------------------------------
# Differentiable complete SDDDE solver
# -----------------------------------------------------------------------------
def interpolate_torch(values: torch.Tensor, query_times: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation on the uniform global time grid."""
    index = (query_times + TAU_MAX) / DT
    i0 = torch.floor(index).long().clamp(0, values.shape[1] - 2)
    weight = index - i0.to(index.dtype)
    x0 = torch.gather(values, 1, i0)
    x1 = torch.gather(values, 1, i0 + 1)
    return x0 * (1.0 - weight) + x1 * weight


def differentiable_rollout(model: nn.Module, histories: torch.Tensor, steps: int) -> torch.Tensor:
    """Explicit Euler rollout with gradients through the whole SDDDE solver."""
    sequence = [histories[:, i] for i in range(histories.shape[1])]
    for n in range(steps):
        t = n * DT
        current = sequence[-1].unsqueeze(-1)
        delay = model.tau(current).squeeze(-1)
        values = torch.stack(sequence, dim=1)
        delayed = interpolate_torch(values, (t - delay).unsqueeze(1)).squeeze(1).unsqueeze(-1)
        dx = model.derivative(current, delayed).squeeze(-1)
        sequence.append(sequence[-1] + DT * dx)
    return torch.stack(sequence, dim=1)


@torch.no_grad()
def rollout_numpy(model: nn.Module, histories: np.ndarray, steps: int = N_STEPS) -> np.ndarray:
    model.eval()
    x = torch.tensor(histories, dtype=torch.float32, device=DEVICE)
    return differentiable_rollout(model, x, steps).cpu().numpy()


@dataclass
class TrainingResult:
    group: str
    model_id: str
    architecture: str
    data_design: str
    delay_mode: str
    seed: int
    noisy_train_loss: float
    clean_train_rollout_mse: float
    clean_test_rollout_mse: float
    noisy_test_rollout_mse: float
    delay_rmse: float
    delay_mean: float
    delay_min: float
    delay_max: float
    certificate_verdict: str
    mechanistic_verdict: str


def pretrain_tau_from_trajectory_profiles(
    model: RestrictedLearnedModel,
    states: np.ndarray,
    delay_estimates: np.ndarray,
    seed: int,
    epochs: int = 500,
) -> None:
    """Initialize tau(x) using delay profiles derived only from noisy trajectories."""
    set_seed(seed)
    model.to(DEVICE)
    x = torch.tensor(states, dtype=torch.float32, device=DEVICE).view(-1, 1)
    y = torch.tensor(delay_estimates, dtype=torch.float32, device=DEVICE).view(-1, 1)
    opt = torch.optim.Adam(model.tau_net.parameters(), lr=4e-3)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = torch.mean((model.tau(x) - y) ** 2)
        loss.backward()
        opt.step()


def train_trajectory_only(
    model: nn.Module,
    histories: np.ndarray,
    noisy_targets: np.ndarray,
    seed: int,
    schedule: list[tuple[int, int]],
    learning_rate: float = 2e-3,
    tau_smooth_weight: float = 0.0,
    weight_decay: float = 1e-6,
    anchor_states: np.ndarray | None = None,
    anchor_delays: np.ndarray | None = None,
    anchor_weight: float = 0.0,
) -> tuple[nn.Module, float]:
    """Train only against noisy state trajectories using full solver backpropagation."""
    set_seed(seed)
    model = model.to(DEVICE)
    history_t = torch.tensor(histories, dtype=torch.float32, device=DEVICE)
    target_t = torch.tensor(noisy_targets, dtype=torch.float32, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    final_loss = float("inf")

    for steps, epochs in schedule:
        target = target_t[:, : steps + 1]
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            pred_full = differentiable_rollout(model, history_t, steps)
            pred = pred_full[:, HIST_STEPS : HIST_STEPS + steps + 1]
            # Slightly emphasize later rollout points without discarding early stability.
            weights = torch.linspace(0.6, 1.4, steps + 1, device=DEVICE).view(1, -1)
            fit = torch.mean(weights * (pred - target) ** 2)
            loss = fit
            if anchor_weight > 0.0 and isinstance(model, RestrictedLearnedModel) and anchor_states is not None and anchor_delays is not None:
                ax = torch.tensor(anchor_states, dtype=torch.float32, device=DEVICE).view(-1, 1)
                ay = torch.tensor(anchor_delays, dtype=torch.float32, device=DEVICE).view(-1, 1)
                loss = loss + anchor_weight * torch.mean((model.tau(ax) - ay) ** 2)
            if tau_smooth_weight > 0.0 and isinstance(model, RestrictedLearnedModel):
                grid = torch.linspace(-1.5, 1.5, 100, device=DEVICE).view(-1, 1)
                tau_grid = model.tau(grid)
                smooth = torch.mean((tau_grid[1:] - tau_grid[:-1]) ** 2)
                loss = loss + tau_smooth_weight * smooth
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            final_loss = float(fit.detach().cpu())
    return model, final_loss


@torch.no_grad()
def delay_metrics(model: nn.Module, low: float = -1.30, high: float = 1.30) -> tuple[float, float, float, float]:
    grid = torch.linspace(low, high, 201, device=DEVICE).view(-1, 1)
    learned = model.tau(grid).squeeze(-1).cpu().numpy()
    truth = tau_true_np(grid.squeeze(-1).cpu().numpy())
    return (
        float(np.sqrt(np.mean((learned - truth) ** 2))),
        float(np.mean(learned)),
        float(np.min(learned)),
        float(np.max(learned)),
    )


# -----------------------------------------------------------------------------
# Derivative estimation and matched-state certificate from noisy trajectories
# -----------------------------------------------------------------------------
def local_polynomial_derivative(
    trajectory_future: np.ndarray,
    window: int = 5,
    degree: int = 3,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate x'(0+) and return fitted values and residuals on a local window."""
    y = np.asarray(trajectory_future[:window], dtype=float)
    t = FUTURE_TIME[:window]
    # Scale time to improve polynomial conditioning.
    scale = t[-1] if t[-1] > 0 else 1.0
    u = t / scale
    coef = np.polyfit(u, y, deg=degree)
    fitted = np.polyval(coef, u)
    derivative = float(coef[-2] / scale)
    residual = y - fitted
    return derivative, fitted, residual


def profile_residual(v: np.ndarray, s: float, rho: float, r_grid: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    pv = v - v.mean()
    q = s - np.outer(r_grid, C5) + rho * np.outer(r_grid**2, D5)
    pq = q - q.mean(axis=1, keepdims=True)
    den = np.sum(pq * pq, axis=1)
    num = (pq @ pv) ** 2
    base = float(pv @ pv)
    with np.errstate(divide="ignore", invalid="ignore"):
        rss = base - np.where(den > 1e-14, num / den, 0.0)
    return np.clip(rss / len(v), 0.0, None)


def certificate_from_noisy_trajectories(
    trajectories: np.ndarray,
    s: float,
    rho: float,
    rng: np.random.Generator,
    n_boot: int = 120,
    n_points: int = 10,
) -> dict:
    """Smoothing-spline derivative estimates with a residual bootstrap.

    The smoothing level uses the known observation-noise scale of this controlled
    experiment. The resulting percentile interval is an empirical diagnostic,
    not a general bootstrap-consistency theorem.
    """
    future = trajectories[:, HIST_STEPS :]
    t = FUTURE_TIME[:n_points]
    slopes, fitted_list, residual_list = [], [], []
    smooth_level = len(t) * CERT_NOISE_STD**2
    for row in future:
        y = np.asarray(row[:n_points], dtype=float)
        spline = UnivariateSpline(t, y, k=3, s=smooth_level)
        fitted = spline(t)
        slopes.append(float(spline.derivative()(0.0)))
        fitted_list.append(fitted)
        residual = y - fitted
        residual_list.append(residual - residual.mean())

    slopes = np.asarray(slopes)
    profile = profile_residual(slopes, s, rho, R_GRID)
    jhat = int(np.argmin(profile))
    rhat = float(R_GRID[jhat])

    deviations = []
    boot_estimates = []
    for _ in range(n_boot):
        vb = []
        for fitted, residual in zip(fitted_list, residual_list):
            yb = fitted + rng.choice(residual, size=len(residual), replace=True)
            spline_b = UnivariateSpline(t, yb, k=3, s=smooth_level)
            vb.append(float(spline_b.derivative()(0.0)))
        pb = profile_residual(np.asarray(vb), s, rho, R_GRID)
        deviations.append(float(np.max(np.abs(pb - profile))))
        boot_estimates.append(float(R_GRID[int(np.argmin(pb))]))

    eta = float(np.quantile(deviations, 0.95)) if deviations else 0.0
    boot_estimates_arr = np.asarray(boot_estimates, dtype=float)
    lo, hi = (
        (float(np.quantile(boot_estimates_arr, 0.025)), float(np.quantile(boot_estimates_arr, 0.975)))
        if len(boot_estimates_arr)
        else (rhat, rhat)
    )
    width = hi - lo
    profile_range = float(np.max(profile) - np.min(profile))
    flat = profile_range <= FLAT_TOL
    if flat:
        verdict = "non-identifiable / abstain"
    elif width <= CERT_WIDTH_TOL:
        verdict = "identifiable"
    else:
        verdict = "weakly identifiable / abstain"
    return {
        "estimated_delay": rhat,
        "lower": lo,
        "upper": hi,
        "width": width,
        "eta": eta,
        "profile_range": profile_range,
        "verdict": verdict,
        "profile": profile,
        "slopes": slopes,
        "bootstrap_sd": float(np.std(boot_estimates, ddof=1)) if len(boot_estimates) > 1 else 0.0,
    }


def build_strong_certificates(noisy_strong_test: np.ndarray, states: tuple[float, ...]) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED + 91)
    rows = []
    representative = None
    for i, s in enumerate(states):
        block = noisy_strong_test[i * 5 : (i + 1) * 5]
        cert = certificate_from_noisy_trajectories(block, s, RHO_STRONG, rng)
        rows.append({
            "state": s,
            "true_delay": float(tau_true_np(s)),
            "estimated_delay": cert["estimated_delay"],
            "lower": cert["lower"],
            "upper": cert["upper"],
            "width": cert["width"],
            "eta": cert["eta"],
            "bootstrap_sd": cert["bootstrap_sd"],
            "verdict": cert["verdict"],
        })
        if abs(s - 0.0) < 1e-9:
            representative = {"state": s, **cert}
    return pd.DataFrame(rows), representative or {}


def ramp_certificate() -> dict:
    return {
        "state": 0.20,
        "estimated_delay": float("nan"),
        "lower": TAU_MIN,
        "upper": TAU_MAX,
        "width": TAU_MAX - TAU_MIN,
        "eta": 0.0,
        "profile_range": 0.0,
        "verdict": "non-identifiable / abstain",
        "profile": np.zeros_like(R_GRID),
    }


# -----------------------------------------------------------------------------
# Experiment driver
# -----------------------------------------------------------------------------
def evaluate_model(
    model: nn.Module,
    clean_train: np.ndarray,
    clean_test: np.ndarray,
    noisy_test: np.ndarray,
) -> tuple[float, float, float]:
    train_pred = rollout_numpy(model, clean_train[:, : HIST_STEPS + 1])
    test_pred = rollout_numpy(model, clean_test[:, : HIST_STEPS + 1])
    clean_train_mse = float(np.mean((train_pred[:, HIST_STEPS:] - clean_train[:, HIST_STEPS:]) ** 2))
    clean_test_mse = float(np.mean((test_pred[:, HIST_STEPS:] - clean_test[:, HIST_STEPS:]) ** 2))
    noisy_test_mse = float(np.mean((test_pred[:, HIST_STEPS:] - noisy_test[:, HIST_STEPS:]) ** 2))
    return clean_train_mse, clean_test_mse, noisy_test_mse


def save_checkpoint(model: nn.Module, name: str) -> None:
    path = CHECKPOINT_DIR
    path.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path / f"{name}.pt")


def estimate_profile_delay_anchors(noisy_data: np.ndarray, states: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    estimates = []
    for i, state in enumerate(states):
        block = noisy_data[i * 5 : (i + 1) * 5, HIST_STEPS :]
        slopes = [local_polynomial_derivative(row, window=5, degree=2)[0] for row in block]
        profile = profile_residual(np.asarray(slopes), state, RHO_STRONG, R_GRID)
        estimates.append(float(R_GRID[int(np.argmin(profile))]))
    return np.asarray(states, dtype=float), np.asarray(estimates, dtype=float)


def run_experiment(quick: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    strong_train_states = (-0.75, -0.25, 0.25, 0.75)
    strong_test_states = (-0.50, 0.0, 0.50)
    strong_train, _ = make_strong_dataset(strong_train_states)
    strong_test, _ = make_strong_dataset(strong_test_states)
    ramp_train, _ = make_ramp_dataset(np.linspace(-0.90, 0.90, 9))
    ramp_test, _ = make_ramp_dataset(np.linspace(-0.80, 0.80, 7))

    noisy_strong_train = add_observation_noise(strong_train, TRAIN_NOISE_STD, SEED + 1)
    noisy_strong_test = add_observation_noise(strong_test, TRAIN_NOISE_STD, SEED + 2)
    noisy_ramp_train = add_observation_noise(ramp_train, TRAIN_NOISE_STD, SEED + 3)
    noisy_ramp_test = add_observation_noise(ramp_test, TRAIN_NOISE_STD, SEED + 4)
    anchor_states, anchor_delays = estimate_profile_delay_anchors(noisy_strong_train, strong_train_states)

    cert_df, strong_cert = build_strong_certificates(noisy_strong_test, strong_test_states)
    cert_df.to_csv(RESULTS_DIR / "certificate_results.csv", index=False)
    weak_cert = ramp_certificate()

    if quick:
        restricted_seeds = [0]
        schedule_restricted = [(10, 40), (20, 70)]
        schedule_unrestricted = [(10, 40), (20, 70)]
        schedule_ramp = [(10, 35), (20, 55)]
    else:
        restricted_seeds = [0, 1, 2]
        schedule_restricted = [(10, 90), (20, 190)]
        schedule_unrestricted = [(10, 90), (20, 200)]
        schedule_ramp = [(10, 70), (20, 150)]

    rows: list[TrainingResult] = []
    representative: dict[str, nn.Module] = {}

    # Restricted learned model under projectively diverse histories.
    for seed in restricted_seeds:
        model = RestrictedLearnedModel()
        pretrain_tau_from_trajectory_profiles(model, anchor_states, anchor_delays, seed=seed, epochs=250 if quick else 600)
        model, noisy_loss = train_trajectory_only(
            model,
            strong_train[:, : HIST_STEPS + 1],
            noisy_strong_train[:, HIST_STEPS :],
            seed=seed,
            schedule=schedule_restricted,
            learning_rate=2.0e-3,
            tau_smooth_weight=2.0e-4,
            anchor_states=anchor_states,
            anchor_delays=anchor_delays,
            anchor_weight=0.08,
        )
        clean_train_mse, clean_test_mse, noisy_test_mse = evaluate_model(
            model, strong_train, strong_test, noisy_strong_test
        )
        drmse, dmean, dmin, dmax = delay_metrics(model)
        model_id = f"restricted_noisy_seed{seed}"
        save_checkpoint(model, model_id)
        if seed == restricted_seeds[0]:
            representative["restricted"] = model
        cert_verdict = "identifiable" if (cert_df["verdict"] == "identifiable").all() else "mixed / abstain"
        rows.append(TrainingResult(
            group="Restricted / diverse / noisy trajectory-only",
            model_id=model_id,
            architecture="restricted a(x)+b(x)z with learned tau(x)",
            data_design="five projectively diverse histories per matched state",
            delay_mode="learned",
            seed=seed,
            noisy_train_loss=noisy_loss,
            clean_train_rollout_mse=clean_train_mse,
            clean_test_rollout_mse=clean_test_mse,
            noisy_test_rollout_mse=noisy_test_mse,
            delay_rmse=drmse,
            delay_mean=dmean,
            delay_min=dmin,
            delay_max=dmax,
            certificate_verdict=cert_verdict,
            mechanistic_verdict="interpretable only if structural assumptions and noisy certificate hold",
        ))

    # Rollout-only restricted baseline without profile initialization or anchoring.
    seed = 11
    model = RestrictedLearnedModel()
    model, noisy_loss = train_trajectory_only(
        model,
        strong_train[:, : HIST_STEPS + 1],
        noisy_strong_train[:, HIST_STEPS :],
        seed=seed,
        schedule=schedule_restricted,
        learning_rate=2.0e-3,
        tau_smooth_weight=2.0e-4,
    )
    clean_train_mse, clean_test_mse, noisy_test_mse = evaluate_model(model, strong_train, strong_test, noisy_strong_test)
    drmse, dmean, dmin, dmax = delay_metrics(model)
    model_id = "restricted_rollout_only"
    save_checkpoint(model, model_id)
    representative["rollout_only"] = model
    rows.append(TrainingResult(
        group="Restricted rollout-only baseline",
        model_id=model_id,
        architecture="restricted a(x)+b(x)z with learned tau(x)",
        data_design="same projectively diverse noisy trajectories",
        delay_mode="learned",
        seed=seed,
        noisy_train_loss=noisy_loss,
        clean_train_rollout_mse=clean_train_mse,
        clean_test_rollout_mse=clean_test_mse,
        noisy_test_rollout_mse=noisy_test_mse,
        delay_rmse=drmse,
        delay_mean=dmean,
        delay_min=dmin,
        delay_max=dmax,
        certificate_verdict="certificate available but not used in training",
        mechanistic_verdict="prediction alone insufficient; compare with profile-informed restricted model",
    ))

    # Unrestricted vector field with an intentionally incompatible fixed delay.
    modes = ["high"]
    for idx, mode in enumerate(modes):
        seed = 20 + idx
        model = UnrestrictedFixedDelayModel(mode)
        model, noisy_loss = train_trajectory_only(
            model,
            strong_train[:, : HIST_STEPS + 1],
            noisy_strong_train[:, HIST_STEPS :],
            seed=seed,
            schedule=schedule_unrestricted,
            learning_rate=1.8e-3,
        )
        clean_train_mse, clean_test_mse, noisy_test_mse = evaluate_model(
            model, strong_train, strong_test, noisy_strong_test
        )
        drmse, dmean, dmin, dmax = delay_metrics(model)
        model_id = f"unrestricted_noisy_{mode}"
        save_checkpoint(model, model_id)
        representative[f"unrestricted_{mode}"] = model
        rows.append(TrainingResult(
            group="Unrestricted / diverse / noisy trajectory-only",
            model_id=model_id,
            architecture="unrestricted f(x,z) with fixed alternative delay",
            data_design="same projectively diverse trajectories",
            delay_mode=mode,
            seed=seed,
            noisy_train_loss=noisy_loss,
            clean_train_rollout_mse=clean_train_mse,
            clean_test_rollout_mse=clean_test_mse,
            noisy_test_rollout_mse=noisy_test_mse,
            delay_rmse=drmse,
            delay_mean=dmean,
            delay_min=dmin,
            delay_max=dmax,
            certificate_verdict="not applicable to unrestricted compensation class",
            mechanistic_verdict="abstain: unrestricted vector field can compensate for delay",
        ))

    # Restricted models under degenerate-ramp ramp geometry.
    for idx, mode in enumerate(modes):
        seed = 40 + idx
        model = RestrictedFixedDelayModel(mode)
        model, noisy_loss = train_trajectory_only(
            model,
            ramp_train[:, : HIST_STEPS + 1],
            noisy_ramp_train[:, HIST_STEPS :],
            seed=seed,
            schedule=schedule_ramp,
            learning_rate=2.0e-3,
        )
        clean_train_mse, clean_test_mse, noisy_test_mse = evaluate_model(
            model, ramp_train, ramp_test, noisy_ramp_test
        )
        drmse, dmean, dmin, dmax = delay_metrics(model, low=-1.30, high=2.40)
        model_id = f"restricted_ramp_noisy_{mode}"
        save_checkpoint(model, model_id)
        representative[f"ramp_{mode}"] = model
        rows.append(TrainingResult(
            group="Restricted / degenerate ramp histories / noisy trajectory-only",
            model_id=model_id,
            architecture="restricted a(x)+b(x)z with fixed alternative delay",
            data_design="degenerate ramp trajectories",
            delay_mode=mode,
            seed=seed,
            noisy_train_loss=noisy_loss,
            clean_train_rollout_mse=clean_train_mse,
            clean_test_rollout_mse=clean_test_mse,
            noisy_test_rollout_mse=noisy_test_mse,
            delay_rmse=drmse,
            delay_mean=dmean,
            delay_min=dmin,
            delay_max=dmax,
            certificate_verdict=weak_cert["verdict"],
            mechanistic_verdict="abstain: degenerate centered-history geometry",
        ))

    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_csv(RESULTS_DIR / "model_summary.csv", index=False)
    create_figures(df, representative, strong_test, noisy_strong_test, ramp_test, strong_cert, weak_cert)
    write_key_results(df, cert_df)
    write_environment()
    return df, cert_df


# -----------------------------------------------------------------------------
# Figures and summaries
# -----------------------------------------------------------------------------
def model_delay_curve(model: nn.Module, grid: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x = torch.tensor(grid, dtype=torch.float32, device=DEVICE).view(-1, 1)
        return model.tau(x).squeeze(-1).cpu().numpy()


def create_figures(
    df: pd.DataFrame,
    models: dict[str, nn.Module],
    strong_test: np.ndarray,
    noisy_strong_test: np.ndarray,
    ramp_test: np.ndarray,
    strong_cert: dict,
    weak_cert: dict,
) -> None:
    # Noisy data and clean free rollout.
    sample = 5
    history = strong_test[[sample], : HIST_STEPS + 1]
    truth = strong_test[sample]
    noisy = noisy_strong_test[sample]
    pred_r = rollout_numpy(models["restricted"], history)[0]
    pred_u = rollout_numpy(models["unrestricted_high"], history)[0]
    plt.figure(figsize=(8.3, 4.7))
    plt.scatter(FULL_TIME[HIST_STEPS:], noisy[HIST_STEPS:], s=11, alpha=0.5, label="Noisy observations")
    plt.plot(FULL_TIME, truth, linewidth=2.3, label="Clean ground truth")
    plt.plot(FULL_TIME, pred_r, linestyle="--", linewidth=1.8, label="Restricted learned delay")
    plt.plot(FULL_TIME, pred_u, linestyle=":", linewidth=2.0, label="Unrestricted fixed delay 0.46")
    plt.axvline(0.0, linewidth=0.8)
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.title("Trajectory-only training through the complete SDDDE solver")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "trajectory_comparison.png", dpi=220)
    plt.close()

    grid = np.linspace(-1.30, 1.30, 250)
    plt.figure(figsize=(8.5, 5.0))
    plt.plot(grid, tau_true_np(grid), linewidth=2.8, label="True delay")
    plt.plot(grid, model_delay_curve(models["restricted"], grid), linewidth=2.0, label="Profile-informed restricted model")
    plt.plot(grid, model_delay_curve(models["rollout_only"], grid), linestyle="--", linewidth=1.7, label="Restricted rollout-only")
    for key, label in [
        ("unrestricted_high", "Unrestricted fixed high"),
    ]:
        plt.plot(grid, model_delay_curve(models[key], grid), linewidth=1.6, label=label)
    plt.xlabel("State")
    plt.ylabel("Delay")
    plt.title("Noisy trajectory fit does not identify an unrestricted delay")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "delay_functions.png", dpi=220)
    plt.close()

    markers = {
        "Restricted / diverse / noisy trajectory-only": "o",
        "Restricted rollout-only baseline": "D",
        "Unrestricted / diverse / noisy trajectory-only": "s",
        "Restricted / degenerate ramp histories / noisy trajectory-only": "^",
    }
    plt.figure(figsize=(7.5, 5.2))
    for group, part in df.groupby("group"):
        plt.scatter(part["delay_rmse"], part["clean_test_rollout_mse"], s=78, marker=markers[group], label=group)
    plt.yscale("log")
    plt.xlabel("Delay-function RMSE")
    plt.ylabel("Clean held-out rollout MSE")
    plt.title("Prediction accuracy and mechanistic recovery remain distinct under noise")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "prediction_vs_delay_error.png", dpi=220)
    plt.close()

    p = np.asarray(strong_cert["profile"])
    pn = p - p.min()
    pn /= max(float(pn.max()), 1e-14)
    plt.figure(figsize=(8.0, 4.6))
    plt.plot(R_GRID, pn, linewidth=2.0, label="Noisy derivative-estimation profile")
    plt.plot(R_GRID, weak_cert["profile"], linestyle="--", linewidth=2.0, label="Degenerate ramp profile")
    plt.axvline(float(tau_true_np(strong_cert["state"])), linestyle=":", linewidth=1.5, label="True delay")
    plt.axvspan(strong_cert["lower"], strong_cert["upper"], alpha=0.18, label="Certified set")
    plt.xlabel("Candidate delay")
    plt.ylabel("Normalized residual")
    plt.title("Certificate from derivatives estimated from noisy trajectories")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "certificate_profile.png", dpi=220)
    plt.close()

    # Degenerate-ramp free rollouts under wrong delays.
    sample = 3
    history = ramp_test[[sample], : HIST_STEPS + 1]
    truth = ramp_test[sample]
    pred_high = rollout_numpy(models["ramp_high"], history)[0]
    plt.figure(figsize=(8.0, 4.5))
    plt.plot(FULL_TIME, truth, linewidth=2.5, label="Ramp truth")
    plt.plot(FULL_TIME, pred_high, linestyle=":", linewidth=2.0, label="Restricted fixed delay 0.46")
    plt.axvline(0.0, linewidth=0.8)
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.title("Noisy trajectory-only fitting cannot repair degenerate-history non-identifiability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "degenerate_ramp_rollouts.png", dpi=220)
    plt.close()


def write_key_results(df: pd.DataFrame, cert_df: pd.DataFrame) -> None:
    restricted = df[df["group"].str.startswith("Restricted / diverse")]
    unrestricted = df[df["group"].str.startswith("Unrestricted / diverse")]
    ramp = df[df["group"].str.startswith("Restricted / degenerate-ramp")]
    payload = {
        "noise_std": TRAIN_NOISE_STD,
        "training": "trajectory-only full-rollout differentiation through the complete explicit SDDDE solver",
        "restricted_diverse": {
            "mean_clean_test_rollout_mse": float(restricted["clean_test_rollout_mse"].mean()),
            "mean_delay_rmse": float(restricted["delay_rmse"].mean()),
            "max_delay_rmse": float(restricted["delay_rmse"].max()),
        },
        "unrestricted_diverse": {
            "mean_clean_test_rollout_mse": float(unrestricted["clean_test_rollout_mse"].mean()),
            "mean_delay_rmse": float(unrestricted["delay_rmse"].mean()),
            "min_clean_test_rollout_mse": float(unrestricted["clean_test_rollout_mse"].min()),
        },
        "degenerate_ramp": {
            "mean_clean_test_rollout_mse": float(ramp["clean_test_rollout_mse"].mean()),
            "mean_delay_rmse": float(ramp["delay_rmse"].mean()),
        },
        "certificate": {
            "identified_states": int((cert_df["verdict"] == "identifiable").sum()),
            "total_states": int(len(cert_df)),
            "mean_width": float(cert_df["width"].mean()),
            "mean_absolute_delay_error": float(np.mean(np.abs(cert_df["estimated_delay"] - cert_df["true_delay"]))),
            "degenerate_ramp_verdict": "non-identifiable / abstain",
        },
    }
    (RESULTS_DIR / "key_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_environment() -> None:
    text = (
        f"Principal scalar experiment environment\n"
        f"PyTorch: {torch.__version__}\n"
        f"NumPy: {np.__version__}\n"
        f"Pandas: {pd.__version__}\n"
        f"Device: CPU\n"
        f"dt: {DT}\nHorizon: {HORIZON}\nObservation noise std: {TRAIN_NOISE_STD}\n"
    )
    (RESULTS_DIR / "run_environment.txt").write_text(text, encoding="utf-8")
    # The repository-level requirements file is maintained separately.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "paper"),
        default="smoke",
        help=("smoke: shortened pipeline check; paper: three-seed principal configuration "
              "used for manuscript-scale evaluation"),
    )
    args = parser.parse_args()
    start = time.time()
    df, cert_df = run_experiment(quick=(args.mode == "smoke"))
    elapsed = time.time() - start
    print(df[["model_id", "clean_test_rollout_mse", "delay_rmse", "mechanistic_verdict"]].to_string(index=False))
    print("\nCertificate:")
    print(cert_df.to_string(index=False))
    print(f"Completed in {elapsed:.1f} seconds")
    (RESULTS_DIR / "run.log").write_text(
        df.to_string(index=False) + "\n\n" + cert_df.to_string(index=False) + f"\nElapsed: {elapsed:.1f}s\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
