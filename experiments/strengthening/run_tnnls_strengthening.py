#!/usr/bin/env python3
"""Run the integrated Step-2/Step-3 TNNLS experiments on one CPU thread.

This script implements:

* the predeclared 2x2 profile-initialization/anchor-loss ablation;
* an oracle-delay restricted control;
* an unrestricted Neural SDDDE with a jointly learned delay;
* an unrestricted fixed-wrong-delay constructive control;
* a corrected degenerate-ramp study with known inputs, so the ramps are exact
  solutions of the same generating delayed system;
* a standardized 21-state residual-resampling certificate;
* state-domain coverage, abstention, and false-confidence evaluation for every
  trained delay candidate.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
from torch import nn

import certificate as cert
import sdde_core as core

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
OUTPUT_DIR = ROOT / "results" / "strengthening" / "generated"
FIGURE_DIR = ROOT / "figures" / "strengthening" / "generated"
CHECKPOINT_DIR = ROOT / "checkpoints" / "strengthening" / "generated"

PUBLICATION_SEEDS = [20260719, 20260720, 20260721, 20260722, 20260723]
TAU_INITIALIZATIONS = [0.18, 0.25, 0.325, 0.40, 0.47]
TRAIN_STATES = (-0.75, -0.25, 0.25, 0.75)
TEST_STATES = (-0.50, 0.0, 0.50)
CERTIFICATE_STATES = tuple(np.linspace(-0.70, 0.70, 21))
NOISE_STD = 5.0e-4
MECHANISTIC_TOLERANCE = 0.10 * core.TAU_RANGE
PRINCIPAL_DATA_SEED = 20260801
DEGENERATE_DATA_SEED = 20260802
CERTIFICATE_DATA_SEED = 20260803
CERTIFICATE_BOOTSTRAP_SEED = 20260804
PUBLICATION_CERTIFICATE_RESAMPLES = 120
QUICK_CERTIFICATE_RESAMPLES = 12


@dataclass(frozen=True)
class Configuration:
    config_id: str
    label: str
    model_class: str
    use_profile_initialization: bool
    use_anchor_loss: bool
    learned_delay: bool
    fixed_delay: float | None = None


PRINCIPAL_CONFIGURATIONS = (
    Configuration(
        "P0",
        "Oracle restricted",
        "oracle_restricted",
        False,
        False,
        False,
    ),
    Configuration(
        "P1",
        "Full identifiability-aware",
        "restricted_learned",
        True,
        True,
        True,
    ),
    Configuration(
        "P2",
        "Profile initialization only",
        "restricted_learned",
        True,
        False,
        True,
    ),
    Configuration(
        "P3",
        "Anchor loss only",
        "restricted_learned",
        False,
        True,
        True,
    ),
    Configuration(
        "P4",
        "Restricted rollout-only",
        "restricted_learned",
        False,
        False,
        True,
    ),
    Configuration(
        "P5",
        "Unrestricted jointly learned delay",
        "unrestricted_learned",
        False,
        False,
        True,
    ),
)

FIXED_CONTROL = Configuration(
    "P6",
    "Unrestricted fixed wrong delay",
    "unrestricted_fixed",
    False,
    False,
    False,
    fixed_delay=0.46,
)


def make_model(
    configuration: Configuration,
    seed: int,
    tau_initialization: float | None,
) -> nn.Module:
    core.set_seed(seed)
    if configuration.model_class == "oracle_restricted":
        return core.OracleRestrictedModel()
    if configuration.model_class == "restricted_learned":
        model: nn.Module = core.RestrictedLearnedModel()
    elif configuration.model_class == "unrestricted_learned":
        model = core.UnrestrictedLearnedModel()
    elif configuration.model_class == "unrestricted_fixed":
        return core.UnrestrictedFixedDelayModel(configuration.fixed_delay or 0.46)
    elif configuration.model_class == "restricted_fixed":
        return core.RestrictedFixedDelayModel(configuration.fixed_delay or 0.46)
    else:
        raise ValueError(f"Unknown model class: {configuration.model_class}")
    if tau_initialization is not None:
        core.initialize_tau_constant(model, tau_initialization)
    return model


def pretrain_tau(
    model: nn.Module,
    states: np.ndarray,
    delays: np.ndarray,
    seed: int,
    epochs: int,
) -> None:
    if not hasattr(model, "tau_net"):
        raise TypeError("Profile initialization requires a learned tau_net.")
    core.set_seed(seed)
    model.to(core.DEVICE)
    state_tensor = torch.tensor(states, dtype=torch.float32).reshape(-1, 1)
    delay_tensor = torch.tensor(delays, dtype=torch.float32).reshape(-1, 1)
    optimizer = torch.optim.Adam(model.tau_net.parameters(), lr=4.0e-3)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model.tau(state_tensor) - delay_tensor) ** 2)
        loss.backward()
        optimizer.step()


def train_model(
    model: nn.Module,
    histories: np.ndarray,
    noisy_targets: np.ndarray,
    seed: int,
    schedule: list[tuple[int, int]],
    known_inputs: np.ndarray | None = None,
    anchor_states: np.ndarray | None = None,
    anchor_delays: np.ndarray | None = None,
    anchor_weight: float = 0.0,
    smoothness_weight: float = 2.0e-4,
) -> tuple[nn.Module, float]:
    """Train all configurations with a common rollout objective and budget."""
    core.set_seed(seed)
    model = model.to(core.DEVICE)
    history_tensor = torch.tensor(histories, dtype=torch.float32)
    target_tensor = torch.tensor(noisy_targets, dtype=torch.float32)
    input_tensor = (
        torch.tensor(known_inputs, dtype=torch.float32)
        if known_inputs is not None
        else None
    )
    anchor_state_tensor = (
        torch.tensor(anchor_states, dtype=torch.float32).reshape(-1, 1)
        if anchor_states is not None
        else None
    )
    anchor_delay_tensor = (
        torch.tensor(anchor_delays, dtype=torch.float32).reshape(-1, 1)
        if anchor_delays is not None
        else None
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=2.0e-3,
        weight_decay=1.0e-6,
    )
    final_fit = float("inf")
    for steps, epochs in schedule:
        target = target_tensor[:, : steps + 1]
        partial_inputs = input_tensor[:, :steps] if input_tensor is not None else None
        weights = torch.linspace(0.6, 1.4, steps + 1).reshape(1, -1)
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            prediction_full = core.differentiable_rollout(
                model,
                history_tensor,
                steps,
                partial_inputs,
            )
            prediction = prediction_full[
                :, core.HIST_STEPS : core.HIST_STEPS + steps + 1
            ]
            fit = torch.mean(weights * (prediction - target) ** 2)
            loss = fit
            if (
                anchor_weight > 0.0
                and anchor_state_tensor is not None
                and anchor_delay_tensor is not None
                and hasattr(model, "tau_net")
            ):
                loss = loss + anchor_weight * torch.mean(
                    (model.tau(anchor_state_tensor) - anchor_delay_tensor) ** 2
                )
            if smoothness_weight > 0.0 and hasattr(model, "tau_net"):
                grid = torch.linspace(-1.5, 1.5, 100).reshape(-1, 1)
                tau_grid = model.tau(grid)
                loss = loss + smoothness_weight * torch.mean(
                    (tau_grid[1:] - tau_grid[:-1]) ** 2
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            final_fit = float(fit.detach())
    return model, final_fit


def trajectory_metrics(
    model: nn.Module,
    clean_train: np.ndarray,
    clean_test: np.ndarray,
    noisy_test: np.ndarray,
    train_inputs: np.ndarray | None = None,
    test_inputs: np.ndarray | None = None,
) -> tuple[float, float, float]:
    train_prediction = core.rollout_numpy(
        model,
        clean_train[:, : core.HIST_STEPS + 1],
        known_inputs=train_inputs,
    )
    test_prediction = core.rollout_numpy(
        model,
        clean_test[:, : core.HIST_STEPS + 1],
        known_inputs=test_inputs,
    )
    train_mse = float(
        np.mean(
            (
                train_prediction[:, core.HIST_STEPS :]
                - clean_train[:, core.HIST_STEPS :]
            )
            ** 2
        )
    )
    test_mse = float(
        np.mean(
            (
                test_prediction[:, core.HIST_STEPS :]
                - clean_test[:, core.HIST_STEPS :]
            )
            ** 2
        )
    )
    noisy_test_mse = float(
        np.mean(
            (
                test_prediction[:, core.HIST_STEPS :]
                - noisy_test[:, core.HIST_STEPS :]
            )
            ** 2
        )
    )
    return train_mse, test_mse, noisy_test_mse


def delay_metrics(
    model: nn.Module,
    state_grid: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    learned = core.delay_curve(model, state_grid)
    truth = np.asarray(core.tau_true_np(state_grid), dtype=float)
    error = learned - truth
    rmse = float(np.sqrt(np.mean(error**2)))
    return (
        {
            "delay_rmse": rmse,
            "delay_nrmse": rmse / core.TAU_RANGE,
            "delay_mae": float(np.mean(np.abs(error))),
            "delay_max_abs_error": float(np.max(np.abs(error))),
            "delay_mean": float(np.mean(learned)),
            "delay_min": float(np.min(learned)),
            "delay_max": float(np.max(learned)),
        },
        learned,
    )


def run_principal(
    quick: bool,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, list[np.ndarray]],
    list[dict[str, object]],
]:
    clean_train, _ = core.make_strong_dataset(TRAIN_STATES)
    clean_test, _ = core.make_strong_dataset(TEST_STATES)
    noisy_train = core.add_observation_noise(
        clean_train,
        NOISE_STD,
        PRINCIPAL_DATA_SEED,
    )
    noisy_test = core.add_observation_noise(
        clean_test,
        NOISE_STD,
        PRINCIPAL_DATA_SEED + 1,
    )
    anchor_states, anchor_delays = core.estimate_profile_anchors(
        noisy_train,
        TRAIN_STATES,
    )
    state_grid = np.linspace(-0.75, 0.75, 201)
    seeds = PUBLICATION_SEEDS[:1] if quick else PUBLICATION_SEEDS
    schedule = [(10, 12), (20, 20)] if quick else [(10, 90), (20, 190)]
    pretrain_epochs = 40 if quick else 600

    rows: list[dict] = []
    curve_rows: list[dict] = []
    curves_by_config: dict[str, list[np.ndarray]] = {}
    model_records: list[dict[str, object]] = []

    configurations = list(PRINCIPAL_CONFIGURATIONS)
    fixed_seeds = seeds[:1] if quick else seeds[:3]
    runs = [
        (configuration, seed_index, seed)
        for configuration in configurations
        for seed_index, seed in enumerate(seeds)
    ]
    runs.extend(
        (FIXED_CONTROL, seed_index, seed)
        for seed_index, seed in enumerate(fixed_seeds)
    )

    for configuration, seed_index, seed in runs:
        tau_initialization = (
            TAU_INITIALIZATIONS[seed_index]
            if configuration.config_id == "P5"
            else None
        )
        model = make_model(configuration, seed, tau_initialization)
        if configuration.use_profile_initialization:
            pretrain_tau(
                model,
                anchor_states,
                anchor_delays,
                seed + 303,
                pretrain_epochs,
            )
        model, final_fit = train_model(
            model,
            clean_train[:, : core.HIST_STEPS + 1],
            noisy_train[:, core.HIST_STEPS :],
            seed + 404,
            schedule,
            anchor_states=anchor_states if configuration.use_anchor_loss else None,
            anchor_delays=anchor_delays if configuration.use_anchor_loss else None,
            anchor_weight=0.08 if configuration.use_anchor_loss else 0.0,
        )
        train_mse, test_mse, noisy_test_mse = trajectory_metrics(
            model,
            clean_train,
            clean_test,
            noisy_test,
        )
        delay_result, learned_curve = delay_metrics(model, state_grid)
        rows.append(
            {
                **asdict(configuration),
                "seed": seed,
                "run_index": seed_index,
                "tau_initialization": tau_initialization,
                "noise_std": NOISE_STD,
                "data_seed": PRINCIPAL_DATA_SEED,
                "parameter_count": int(
                    sum(parameter.numel() for parameter in model.parameters())
                ),
                "anchor_mae": float(
                    np.mean(
                        np.abs(
                            anchor_delays
                            - core.tau_true_np(anchor_states)
                        )
                    )
                ),
                "final_weighted_fit": final_fit,
                "clean_train_mse": train_mse,
                "clean_test_mse": test_mse,
                "noisy_test_mse": noisy_test_mse,
                **delay_result,
            }
        )
        curves_by_config.setdefault(configuration.config_id, []).append(learned_curve)
        curve_rows.extend(
            {
                "config_id": configuration.config_id,
                "label": configuration.label,
                "seed": seed,
                "state": float(state),
                "learned_delay": float(delay),
                "true_delay": float(core.tau_true_np(state)),
            }
            for state, delay in zip(state_grid, learned_curve)
        )
        core.save_checkpoint(
            model,
            CHECKPOINT_DIR / f"{configuration.config_id}_seed{seed}.pt",
        )
        model_records.append(
            {
                "configuration": configuration,
                "seed": seed,
                "model": model,
            }
        )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(curve_rows),
        curves_by_config,
        model_records,
    )


def run_degenerate(
    quick: bool,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, list[np.ndarray]],
    list[dict[str, object]],
]:
    clean_train, train_inputs, _ = core.make_forced_ramp_dataset(
        np.linspace(-0.90, 0.90, 9)
    )
    clean_test, test_inputs, _ = core.make_forced_ramp_dataset(
        np.linspace(-0.80, 0.80, 7)
    )
    noisy_train = core.add_observation_noise(
        clean_train,
        NOISE_STD,
        DEGENERATE_DATA_SEED,
    )
    noisy_test = core.add_observation_noise(
        clean_test,
        NOISE_STD,
        DEGENERATE_DATA_SEED + 1,
    )
    seeds = PUBLICATION_SEEDS[:1] if quick else PUBLICATION_SEEDS
    schedule = [(10, 12), (20, 20)] if quick else [(10, 70), (20, 150)]
    state_grid = np.linspace(-0.75, 0.75, 201)
    rows: list[dict] = []
    curve_rows: list[dict] = []
    curves_by_config: dict[str, list[np.ndarray]] = {}
    model_records: list[dict[str, object]] = []

    configurations = [
        Configuration(
            "D1",
            "Restricted learned delay / forced degenerate ramps",
            "restricted_learned",
            False,
            False,
            True,
        ),
        Configuration(
            "D2",
            "Restricted fixed wrong delay / forced degenerate ramps",
            "restricted_fixed",
            False,
            False,
            False,
            fixed_delay=0.46,
        ),
    ]
    for configuration in configurations:
        configuration_seeds = seeds if configuration.config_id == "D1" else seeds[:1]
        for seed_index, seed in enumerate(configuration_seeds):
            tau_initialization = (
                TAU_INITIALIZATIONS[seed_index]
                if configuration.learned_delay
                else None
            )
            model = make_model(configuration, seed, tau_initialization)
            model, final_fit = train_model(
                model,
                clean_train[:, : core.HIST_STEPS + 1],
                noisy_train[:, core.HIST_STEPS :],
                seed + 707,
                schedule,
                known_inputs=train_inputs,
                anchor_weight=0.0,
            )
            train_mse, test_mse, noisy_test_mse = trajectory_metrics(
                model,
                clean_train,
                clean_test,
                noisy_test,
                train_inputs=train_inputs,
                test_inputs=test_inputs,
            )
            delay_result, learned_curve = delay_metrics(model, state_grid)
            rows.append(
                {
                    **asdict(configuration),
                    "seed": seed,
                    "run_index": seed_index,
                    "tau_initialization": tau_initialization,
                    "noise_std": NOISE_STD,
                    "data_seed": DEGENERATE_DATA_SEED,
                    "parameter_count": int(
                        sum(parameter.numel() for parameter in model.parameters())
                    ),
                    "profile_range": 0.0,
                    "final_weighted_fit": final_fit,
                    "clean_train_mse": train_mse,
                    "clean_test_mse": test_mse,
                    "noisy_test_mse": noisy_test_mse,
                    **delay_result,
                }
            )
            curves_by_config.setdefault(configuration.config_id, []).append(
                learned_curve
            )
            curve_rows.extend(
                {
                    "config_id": configuration.config_id,
                    "label": configuration.label,
                    "seed": seed,
                    "state": float(state),
                    "learned_delay": float(delay),
                    "true_delay": float(core.tau_true_np(state)),
                }
                for state, delay in zip(state_grid, learned_curve)
            )
            core.save_checkpoint(
                model,
                CHECKPOINT_DIR / f"{configuration.config_id}_seed{seed}.pt",
            )
            model_records.append(
                {
                    "configuration": configuration,
                    "seed": seed,
                    "model": model,
                }
            )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(curve_rows),
        curves_by_config,
        model_records,
    )


def candidate_table(
    model_records: list[dict[str, object]],
    states: tuple[float, ...],
) -> pd.DataFrame:
    """Evaluate every trained delay function on the certificate state grid."""
    rows: list[dict] = []
    state_array = np.asarray(states, dtype=float)
    for record in model_records:
        configuration = record["configuration"]
        if not isinstance(configuration, Configuration):
            raise TypeError("Invalid configuration in model record.")
        model = record["model"]
        if not isinstance(model, nn.Module):
            raise TypeError("Invalid model in model record.")
        learned_delays = core.delay_curve(model, state_array)
        rows.extend(
            {
                "config_id": configuration.config_id,
                "label": configuration.label,
                "model_class": configuration.model_class,
                "seed": int(record["seed"]),
                "state": float(state),
                "learned_delay": float(delay),
            }
            for state, delay in zip(state_array, learned_delays)
        )
    return pd.DataFrame(rows)


def run_standardized_certificate(
    quick: bool,
    principal_models: list[dict[str, object]],
    degenerate_models: list[dict[str, object]],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    """Run informative and degenerate certificates on one fixed 21-state grid."""
    clean_certificate, _ = core.make_strong_dataset(CERTIFICATE_STATES)
    noisy_certificate = core.add_observation_noise(
        clean_certificate,
        NOISE_STD,
        CERTIFICATE_DATA_SEED,
    )
    n_resamples = (
        QUICK_CERTIFICATE_RESAMPLES
        if quick
        else PUBLICATION_CERTIFICATE_RESAMPLES
    )
    informative_geometry, informative_profiles = (
        cert.build_informative_certificates(
            noisy_certificate,
            CERTIFICATE_STATES,
            NOISE_STD,
            CERTIFICATE_BOOTSTRAP_SEED,
            n_resamples,
        )
    )
    degenerate_geometry, degenerate_profiles = (
        cert.build_degenerate_certificates(
            CERTIFICATE_STATES,
            NOISE_STD,
            n_resamples,
        )
    )

    principal_candidates = candidate_table(
        principal_models,
        CERTIFICATE_STATES,
    )
    degenerate_candidates = candidate_table(
        degenerate_models,
        CERTIFICATE_STATES,
    )
    principal_candidate_results = cert.evaluate_candidates(
        informative_geometry,
        informative_profiles,
        principal_candidates,
        MECHANISTIC_TOLERANCE,
    )
    degenerate_candidate_results = cert.evaluate_candidates(
        degenerate_geometry,
        degenerate_profiles,
        degenerate_candidates,
        MECHANISTIC_TOLERANCE,
    )
    candidate_summary = cert.summarize_candidates(
        pd.concat(
            [principal_candidate_results, degenerate_candidate_results],
            ignore_index=True,
        )
    )
    geometry_summary = {
        "informative": cert.summarize_geometry(informative_geometry),
        "degenerate": cert.summarize_geometry(degenerate_geometry),
    }
    return (
        informative_geometry,
        informative_profiles,
        degenerate_geometry,
        degenerate_profiles,
        principal_candidate_results,
        degenerate_candidate_results,
        candidate_summary,
        geometry_summary,
    )


def pairwise_curve_dispersion(
    curves_by_config: dict[str, list[np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict] = []
    for config_id, curves in curves_by_config.items():
        pairwise: list[float] = []
        for first in range(len(curves)):
            for second in range(first + 1, len(curves)):
                pairwise.append(
                    float(np.sqrt(np.mean((curves[first] - curves[second]) ** 2)))
                )
        rows.append(
            {
                "config_id": config_id,
                "n_runs": len(curves),
                "n_pairs": len(pairwise),
                "mean_pairwise_delay_rmse": (
                    float(np.mean(pairwise)) if pairwise else float("nan")
                ),
                "median_pairwise_delay_rmse": (
                    float(np.median(pairwise)) if pairwise else float("nan")
                ),
                "max_pairwise_delay_rmse": (
                    float(np.max(pairwise)) if pairwise else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_results(per_run: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "clean_train_mse",
        "clean_test_mse",
        "noisy_test_mse",
        "delay_rmse",
        "delay_nrmse",
        "delay_mae",
        "delay_max_abs_error",
        "delay_mean",
        "delay_min",
        "delay_max",
    ]
    rows: list[dict] = []
    for (config_id, label), group in per_run.groupby(["config_id", "label"]):
        row: dict[str, float | int | str] = {
            "config_id": config_id,
            "label": label,
            "n_runs": len(group),
        }
        for column in numeric:
            values = group[column].to_numpy(dtype=float)
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_sample_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            )
            row[f"{column}_median"] = float(np.median(values))
            row[f"{column}_q25"] = float(np.quantile(values, 0.25))
            row[f"{column}_q75"] = float(np.quantile(values, 0.75))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("config_id")


def evaluate_predeclared_criteria(
    principal: pd.DataFrame,
    degenerate: pd.DataFrame,
    dispersion: pd.DataFrame,
    certificate_summary: pd.DataFrame,
    geometry_summary: dict,
) -> dict:
    medians = principal.groupby("config_id").median(numeric_only=True)
    criteria: dict[str, object] = {
        "mechanistic_tolerance": MECHANISTIC_TOLERANCE,
        "status": "integrated learning and certification evaluation",
    }
    if {"P1", "P4"}.issubset(medians.index):
        paired = principal[
            principal["config_id"].isin(["P1", "P4"])
        ].pivot(index="seed", columns="config_id", values="delay_rmse")
        criteria["P1_vs_P4"] = {
            "median_delay_rmse_ratio": float(
                medians.loc["P1", "delay_rmse"] / medians.loc["P4", "delay_rmse"]
            ),
            "paired_improvements": int(np.sum(paired["P1"] < paired["P4"])),
            "n_paired_runs": int(len(paired)),
            "median_trajectory_mse_ratio": float(
                medians.loc["P1", "clean_test_mse"]
                / medians.loc["P4", "clean_test_mse"]
            ),
            "passes_predeclared_effect_criteria": bool(
                medians.loc["P1", "delay_rmse"]
                <= 0.5 * medians.loc["P4", "delay_rmse"]
                and np.sum(paired["P1"] < paired["P4"]) >= min(4, len(paired))
                and medians.loc["P1", "clean_test_mse"]
                <= 2.0 * medians.loc["P4", "clean_test_mse"]
            ),
        }
    if {"P1", "P5"}.issubset(medians.index):
        p5_dispersion = dispersion.loc[
            dispersion["config_id"] == "P5",
            "median_pairwise_delay_rmse",
        ]
        dispersion_value = (
            float(p5_dispersion.iloc[0]) if len(p5_dispersion) else float("nan")
        )
        criteria["P5_unrestricted_ambiguity"] = {
            "median_trajectory_mse_ratio_to_P1": float(
                medians.loc["P5", "clean_test_mse"]
                / medians.loc["P1", "clean_test_mse"]
            ),
            "median_delay_rmse_ratio_to_P1": float(
                medians.loc["P5", "delay_rmse"]
                / medians.loc["P1", "delay_rmse"]
            ),
            "median_pairwise_delay_rmse": dispersion_value,
            "passes_predeclared_effect_criteria": bool(
                medians.loc["P5", "clean_test_mse"]
                <= 3.0 * medians.loc["P1", "clean_test_mse"]
                and (
                    medians.loc["P5", "delay_rmse"]
                    >= 3.0 * medians.loc["P1", "delay_rmse"]
                    or dispersion_value >= MECHANISTIC_TOLERANCE
                )
            ),
        }
    if "D1" in set(degenerate["config_id"]):
        d1 = degenerate[degenerate["config_id"] == "D1"]
        d1_dispersion = dispersion.loc[
            dispersion["config_id"] == "D1",
            "median_pairwise_delay_rmse",
        ]
        criteria["D1_degenerate_geometry"] = {
            "profile_range": float(d1["profile_range"].max()),
            "median_clean_test_mse": float(d1["clean_test_mse"].median()),
            "median_delay_rmse": float(d1["delay_rmse"].median()),
            "median_pairwise_delay_rmse": (
                float(d1_dispersion.iloc[0])
                if len(d1_dispersion)
                else float("nan")
            ),
        }
    certificate_by_id = certificate_summary.set_index("config_id")
    informative = geometry_summary["informative"]
    degenerate_geometry = geometry_summary["degenerate"]
    p1_false_confidence = (
        float(
            certificate_by_id.loc[
                "P1",
                "conditional_false_confidence_rate",
            ]
        )
        if "P1" in certificate_by_id.index
        else float("nan")
    )
    p1_acceptance = (
        float(certificate_by_id.loc["P1", "pooled_acceptance_rate"])
        if "P1" in certificate_by_id.index
        else float("nan")
    )
    unrestricted_acceptance_rates = [
        float(certificate_by_id.loc[config_id, "pooled_acceptance_rate"])
        for config_id in ("P5", "P6")
        if config_id in certificate_by_id.index
    ]
    unrestricted_scope_abstention = (
        1.0 - float(np.mean(unrestricted_acceptance_rates))
        if unrestricted_acceptance_rates
        else float("nan")
    )
    criteria["standardized_certificate"] = {
        "n_states": int(informative["n_states"]),
        "informative_geometry_acceptance_rate": float(
            informative["geometry_acceptance_rate"]
        ),
        "true_delay_interval_coverage": float(
            informative["true_delay_interval_coverage"]
        ),
        "true_delay_empirical_set_coverage": float(
            informative["true_delay_empirical_set_coverage"]
        ),
        "P1_candidate_acceptance_rate": p1_acceptance,
        "P1_conditional_false_confidence_rate": p1_false_confidence,
        "unrestricted_model_scope_abstention_rate": (
            unrestricted_scope_abstention
        ),
        "degenerate_abstention_rate": float(
            1.0 - degenerate_geometry["geometry_acceptance_rate"]
        ),
        "passes_predeclared_effect_criteria": bool(
            informative["geometry_acceptance_rate"] >= 0.80
            and informative["true_delay_interval_coverage"] >= 0.80
            and p1_acceptance >= 0.70
            and p1_false_confidence <= 0.05
            and unrestricted_scope_abstention == 1.0
            and degenerate_geometry["geometry_acceptance_rate"] == 0.0
        ),
        "note": (
            "The residual-resampling intervals are empirical sensitivity "
            "diagnostics, not calibrated confidence intervals."
        ),
    }
    return criteria


def create_figures(
    principal: pd.DataFrame,
    principal_curves: pd.DataFrame,
    degenerate_curves: pd.DataFrame,
    informative_geometry: pd.DataFrame,
    certificate_summary: pd.DataFrame,
) -> None:
    plt.figure(figsize=(7.2, 4.8))
    for config_id, group in principal.groupby("config_id"):
        plt.scatter(
            group["clean_test_mse"],
            group["delay_rmse"],
            label=config_id,
            s=48,
            alpha=0.85,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Clean held-out trajectory MSE")
    plt.ylabel("Delay RMSE")
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "step3_prediction_vs_delay.png", dpi=220)
    plt.close()

    ablation_order = ["P1", "P2", "P3", "P4"]
    plt.figure(figsize=(7.2, 4.8))
    for index, config_id in enumerate(ablation_order):
        values = principal.loc[
            principal["config_id"] == config_id,
            "delay_rmse",
        ].to_numpy()
        plt.scatter(
            np.full_like(values, index, dtype=float),
            values,
            s=52,
            alpha=0.85,
        )
    plt.xticks(range(len(ablation_order)), ablation_order)
    plt.yscale("log")
    plt.xlabel("Ablation configuration")
    plt.ylabel("Delay RMSE")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "step3_ablation_delay_rmse.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    p5 = principal_curves[principal_curves["config_id"] == "P5"]
    for seed, group in p5.groupby("seed"):
        plt.plot(group["state"], group["learned_delay"], label=str(seed))
    if len(p5):
        truth = p5.groupby("state", as_index=False)["true_delay"].first()
        plt.plot(
            truth["state"],
            truth["true_delay"],
            color="black",
            linewidth=2.2,
            label="Truth",
        )
    plt.xlabel("State")
    plt.ylabel("Delay")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "step3_unrestricted_delay_curves.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    d1 = degenerate_curves[degenerate_curves["config_id"] == "D1"]
    for seed, group in d1.groupby("seed"):
        plt.plot(group["state"], group["learned_delay"], label=str(seed))
    if len(d1):
        truth = d1.groupby("state", as_index=False)["true_delay"].first()
        plt.plot(
            truth["state"],
            truth["true_delay"],
            color="black",
            linewidth=2.2,
            label="Truth",
        )
    plt.xlabel("State")
    plt.ylabel("Delay")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "step3_degenerate_delay_curves.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    accepted = informative_geometry["geometry_accepted"].to_numpy(dtype=bool)
    colors = np.where(accepted, "#1b7f3a", "#b5472f")
    plt.vlines(
        informative_geometry["state"],
        informative_geometry["interval_lower"],
        informative_geometry["interval_upper"],
        color=colors,
        linewidth=2.0,
        alpha=0.85,
    )
    plt.scatter(
        informative_geometry["state"],
        informative_geometry["estimated_delay"],
        color=colors,
        s=30,
        label="Profile minimizer",
        zorder=3,
    )
    plt.plot(
        informative_geometry["state"],
        informative_geometry["true_delay"],
        color="black",
        linewidth=1.8,
        label="True delay",
    )
    plt.xlabel("Matched state")
    plt.ylabel("Delay")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "step3_certificate_intervals.png", dpi=220)
    plt.close()

    order = list(certificate_summary["config_id"])
    x = np.arange(len(order))
    plt.figure(figsize=(7.8, 4.8))
    plt.bar(
        x - 0.18,
        certificate_summary["pooled_acceptance_rate"],
        width=0.36,
        label="Candidate acceptance",
        color="#315b8a",
    )
    plt.bar(
        x + 0.18,
        certificate_summary["conditional_false_confidence_rate"],
        width=0.36,
        label="Accepted large error / accepted",
        color="#b5472f",
    )
    plt.xticks(x, order)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Rate (denominators differ; see legend)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "step3_acceptance_false_confidence.png",
        dpi=220,
    )
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all five predeclared seeds and the publication training schedule.",
    )
    args = parser.parse_args()
    quick = not args.full
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    (
        principal,
        principal_curves,
        principal_curve_arrays,
        principal_models,
    ) = run_principal(quick)
    (
        degenerate,
        degenerate_curves,
        degenerate_curve_arrays,
        degenerate_models,
    ) = run_degenerate(quick)
    (
        informative_geometry,
        informative_profiles,
        degenerate_geometry,
        degenerate_profiles,
        principal_candidate_results,
        degenerate_candidate_results,
        certificate_summary,
        geometry_summary,
    ) = run_standardized_certificate(
        quick,
        principal_models,
        degenerate_models,
    )
    combined_curves = {**principal_curve_arrays, **degenerate_curve_arrays}
    dispersion = pairwise_curve_dispersion(combined_curves)
    principal_summary = aggregate_results(principal)
    degenerate_summary = aggregate_results(degenerate)
    criteria = evaluate_predeclared_criteria(
        principal,
        degenerate,
        dispersion,
        certificate_summary,
        geometry_summary,
    )

    principal.to_csv(OUTPUT_DIR / "step3_principal_runs.csv", index=False)
    principal_summary.to_csv(
        OUTPUT_DIR / "step3_principal_summary.csv",
        index=False,
    )
    principal_curves.to_csv(
        OUTPUT_DIR / "step3_principal_delay_curves.csv",
        index=False,
    )
    degenerate.to_csv(OUTPUT_DIR / "step3_degenerate_runs.csv", index=False)
    degenerate_summary.to_csv(
        OUTPUT_DIR / "step3_degenerate_summary.csv",
        index=False,
    )
    degenerate_curves.to_csv(
        OUTPUT_DIR / "step3_degenerate_delay_curves.csv",
        index=False,
    )
    dispersion.to_csv(OUTPUT_DIR / "step3_delay_dispersion.csv", index=False)
    informative_geometry.to_csv(
        OUTPUT_DIR / "step3_informative_certificate_statewise.csv",
        index=False,
    )
    informative_profiles.to_csv(
        OUTPUT_DIR / "step3_informative_profiles.csv",
        index=False,
    )
    degenerate_geometry.to_csv(
        OUTPUT_DIR / "step3_degenerate_certificate_statewise.csv",
        index=False,
    )
    degenerate_profiles.to_csv(
        OUTPUT_DIR / "step3_degenerate_profiles.csv",
        index=False,
    )
    principal_candidate_results.to_csv(
        OUTPUT_DIR / "step3_principal_candidate_certificates.csv",
        index=False,
    )
    degenerate_candidate_results.to_csv(
        OUTPUT_DIR / "step3_degenerate_candidate_certificates.csv",
        index=False,
    )
    certificate_summary.to_csv(
        OUTPUT_DIR / "step3_certificate_summary.csv",
        index=False,
    )
    (OUTPUT_DIR / "step3_geometry_summary.json").write_text(
        json.dumps(geometry_summary, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "step3_success_criteria.json").write_text(
        json.dumps(criteria, indent=2),
        encoding="utf-8",
    )
    create_figures(
        principal,
        principal_curves,
        degenerate_curves,
        informative_geometry,
        certificate_summary,
    )

    elapsed = time.time() - start
    run_manifest = {
        "mode": "full" if args.full else "quick smoke test",
        "publication_seeds": PUBLICATION_SEEDS[:1] if quick else PUBLICATION_SEEDS,
        "tau_initializations": TAU_INITIALIZATIONS[:1] if quick else TAU_INITIALIZATIONS,
        "training_schedule": [(10, 12), (20, 20)] if quick else [(10, 90), (20, 190)],
        "degenerate_training_schedule": [(10, 12), (20, 20)] if quick else [(10, 70), (20, 150)],
        "profile_pretraining_epochs": 40 if quick else 600,
        "anchor_weight": 0.08,
        "anchor_state_weights": "uniform; all four training anchors retained",
        "noise_std": NOISE_STD,
        "principal_data_seed": PRINCIPAL_DATA_SEED,
        "degenerate_data_seed": DEGENERATE_DATA_SEED,
        "certificate_data_seed": CERTIFICATE_DATA_SEED,
        "certificate_bootstrap_seed": CERTIFICATE_BOOTSTRAP_SEED,
        "certificate_states": [float(state) for state in CERTIFICATE_STATES],
        "certificate_resamples": (
            QUICK_CERTIFICATE_RESAMPLES
            if quick
            else PUBLICATION_CERTIFICATE_RESAMPLES
        ),
        "certificate_width_tolerance": cert.CERTIFICATE_WIDTH_TOLERANCE,
        "flat_profile_tolerance": cert.FLAT_PROFILE_TOLERANCE,
        "mechanistic_tolerance": MECHANISTIC_TOLERANCE,
        "principal_configurations": [
            asdict(configuration)
            for configuration in (*PRINCIPAL_CONFIGURATIONS, FIXED_CONTROL)
        ],
        "elapsed_seconds": elapsed,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "scipy_version": scipy.__version__,
        "pandas_version": pd.__version__,
        "matplotlib_version": matplotlib.__version__,
        "cpu_threads": torch.get_num_threads(),
    }
    (OUTPUT_DIR / "step3_run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2),
        encoding="utf-8",
    )

    print("\nPrincipal experiment")
    print(
        principal_summary[
            [
                "config_id",
                "label",
                "n_runs",
                "clean_test_mse_median",
                "delay_rmse_median",
            ]
        ].to_string(index=False)
    )
    print("\nDegenerate experiment")
    print(
        degenerate_summary[
            [
                "config_id",
                "label",
                "n_runs",
                "clean_test_mse_median",
                "delay_rmse_median",
            ]
        ].to_string(index=False)
    )
    print("\nStandardized certificate")
    print(
        certificate_summary[
            [
                "config_id",
                "n_runs",
                "eligible_candidate_states",
                "pooled_acceptance_rate",
                "conditional_false_confidence_rate",
            ]
        ].to_string(index=False)
    )
    print("\nInformative geometry")
    print(json.dumps(geometry_summary["informative"], indent=2))
    print(f"\nCompleted in {elapsed:.1f} seconds")
    if quick:
        print(
            "This was a smoke test with 12 certificate resamples per state. "
            "Use --full for the 120-resample publication run."
        )


if __name__ == "__main__":
    main()
