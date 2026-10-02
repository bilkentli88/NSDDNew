#!/usr/bin/env python3
"""Standardized empirical certificate utilities for the scalar experiments.

The percentile intervals produced here are residual-resampling sensitivity
diagnostics.  They are not claimed to be calibrated frequentist confidence
intervals.  Candidate acceptance additionally requires membership in the
empirical profile sublevel set, keeping the neural interpretation check
separate from profile geometry.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

DT = 0.05
TAU_MIN = 0.15
TAU_MAX = 0.50
HIST_STEPS = int(round(TAU_MAX / DT))
FUTURE_TIME = np.arange(21, dtype=float) * DT
C5 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
D5 = np.array([2.0, -1.0, -2.0, -1.0, 2.0])
RHO_STRONG = 0.60
R_GRID = np.linspace(TAU_MIN, TAU_MAX, 351)

CERTIFICATE_WIDTH_TOLERANCE = 0.090
FLAT_PROFILE_TOLERANCE = 1.0e-10
LOWER_PERCENTILE = 0.025
UPPER_PERCENTILE = 0.975
ELIGIBLE_MODEL_CLASSES = {
    "oracle_restricted",
    "restricted_learned",
    "restricted_fixed",
}


def tau_true_np(x: np.ndarray | float) -> np.ndarray | float:
    """Generating delay used only for synthetic evaluation."""
    return 0.30 + 0.08 * np.tanh(1.30 * np.asarray(x))


def profile_residual(
    slopes: np.ndarray,
    state: float,
    r_grid: np.ndarray = R_GRID,
) -> np.ndarray:
    """Normalized squared profiled residual for the diverse quadratic design."""
    slopes = np.asarray(slopes, dtype=float)
    centered_slopes = slopes - slopes.mean()
    histories = (
        state
        - np.outer(r_grid, C5)
        + RHO_STRONG * np.outer(r_grid**2, D5)
    )
    centered_histories = histories - histories.mean(axis=1, keepdims=True)
    denominator = np.sum(centered_histories**2, axis=1)
    numerator = (centered_histories @ centered_slopes) ** 2
    base = float(centered_slopes @ centered_slopes)
    with np.errstate(divide="ignore", invalid="ignore"):
        residual = base - np.where(
            denominator > 1.0e-14,
            numerator / denominator,
            0.0,
        )
    return np.clip(residual / len(slopes), 0.0, None)


def _count_components(mask: np.ndarray) -> int:
    """Count connected True components on an ordered one-dimensional grid."""
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return 0
    return int(mask[0]) + int(np.sum(mask[1:] & ~mask[:-1]))


def certificate_from_noisy_trajectories(
    trajectories: np.ndarray,
    state: float,
    noise_std: float,
    rng: np.random.Generator,
    n_resamples: int = 120,
    n_points: int = 10,
) -> tuple[dict, pd.DataFrame]:
    """Construct one statewise profile and residual-resampling certificate."""
    if trajectories.shape[0] != len(C5):
        raise ValueError("Each matched state must have exactly five histories.")
    if n_points < 4:
        raise ValueError("Cubic smoothing splines require at least four points.")
    if n_resamples < 1:
        raise ValueError("At least one residual resample is required.")

    future = np.asarray(trajectories[:, HIST_STEPS:], dtype=float)
    t = FUTURE_TIME[:n_points]
    smoothing_level = len(t) * float(noise_std) ** 2
    slopes: list[float] = []
    fitted_list: list[np.ndarray] = []
    residual_list: list[np.ndarray] = []

    for row in future:
        y = np.asarray(row[:n_points], dtype=float)
        spline = UnivariateSpline(t, y, k=3, s=smoothing_level)
        fitted = np.asarray(spline(t), dtype=float)
        residual = y - fitted
        slopes.append(float(spline.derivative()(0.0)))
        fitted_list.append(fitted)
        residual_list.append(residual - residual.mean())

    profile = profile_residual(np.asarray(slopes), float(state))
    minimum_index = int(np.argmin(profile))
    estimated_delay = float(R_GRID[minimum_index])

    deviations: list[float] = []
    bootstrap_minimizers: list[float] = []
    for _ in range(n_resamples):
        resampled_slopes: list[float] = []
        for fitted, residual in zip(fitted_list, residual_list):
            resampled_y = fitted + rng.choice(
                residual,
                size=len(residual),
                replace=True,
            )
            spline = UnivariateSpline(
                t,
                resampled_y,
                k=3,
                s=smoothing_level,
            )
            resampled_slopes.append(float(spline.derivative()(0.0)))
        resampled_profile = profile_residual(
            np.asarray(resampled_slopes),
            float(state),
        )
        deviations.append(float(np.max(np.abs(resampled_profile - profile))))
        bootstrap_minimizers.append(
            float(R_GRID[int(np.argmin(resampled_profile))])
        )

    bootstrap_array = np.asarray(bootstrap_minimizers, dtype=float)
    lower = float(np.quantile(bootstrap_array, LOWER_PERCENTILE))
    upper = float(np.quantile(bootstrap_array, UPPER_PERCENTILE))
    interval_width = upper - lower
    eta = float(np.quantile(np.asarray(deviations), 0.95))
    profile_minimum = float(np.min(profile))
    profile_range = float(np.max(profile) - profile_minimum)

    certified_mask = profile <= profile_minimum + 2.0 * eta
    component_count = _count_components(certified_mask)
    certified_grid = R_GRID[certified_mask]
    certified_lower = float(certified_grid[0])
    certified_upper = float(certified_grid[-1])
    certified_diameter = certified_upper - certified_lower

    true_delay = float(tau_true_np(float(state)))
    true_in_interval = bool(lower <= true_delay <= upper)
    true_profile_value = float(np.interp(true_delay, R_GRID, profile))
    true_in_empirical_set = bool(true_profile_value <= profile_minimum + 2.0 * eta)

    if profile_range <= FLAT_PROFILE_TOLERANCE:
        verdict = "non-identifiable / abstain"
        reason = "structurally flat profile"
    elif component_count != 1:
        verdict = "non-identifiable / abstain"
        reason = "disconnected empirical profile set"
    elif interval_width > CERTIFICATE_WIDTH_TOLERANCE:
        verdict = "weakly identifiable / abstain"
        reason = "resampling interval exceeds resolution threshold"
    else:
        verdict = "identifiable"
        reason = "resolved connected profile"

    result = {
        "state": float(state),
        "true_delay": true_delay,
        "estimated_delay": estimated_delay,
        "interval_lower": lower,
        "interval_upper": upper,
        "interval_width": interval_width,
        "bootstrap_sd": float(np.std(bootstrap_array, ddof=1)),
        "eta_95": eta,
        "profile_minimum": profile_minimum,
        "profile_range": profile_range,
        "empirical_set_lower": certified_lower,
        "empirical_set_upper": certified_upper,
        "empirical_set_diameter": certified_diameter,
        "empirical_set_components": component_count,
        "true_in_interval": true_in_interval,
        "true_in_empirical_set": true_in_empirical_set,
        "verdict": verdict,
        "verdict_reason": reason,
        "geometry_accepted": verdict == "identifiable",
        "n_resamples": int(n_resamples),
        "noise_std": float(noise_std),
    }
    profile_table = pd.DataFrame(
        {
            "state": float(state),
            "candidate_delay": R_GRID,
            "profile_value": profile,
            "empirical_set_member": certified_mask,
        }
    )
    return result, profile_table


def build_informative_certificates(
    noisy_trajectories: np.ndarray,
    states: Iterable[float],
    noise_std: float,
    seed: int,
    n_resamples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the same certificate at every predeclared matched state."""
    states = tuple(float(state) for state in states)
    expected_rows = len(states) * len(C5)
    if noisy_trajectories.shape[0] != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} trajectories, got "
            f"{noisy_trajectories.shape[0]}."
        )

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    profiles: list[pd.DataFrame] = []
    for state_index, state in enumerate(states):
        block = noisy_trajectories[
            state_index * len(C5) : (state_index + 1) * len(C5)
        ]
        result, profile = certificate_from_noisy_trajectories(
            block,
            state,
            noise_std,
            rng,
            n_resamples=n_resamples,
        )
        rows.append(result)
        profiles.append(profile)
    return pd.DataFrame(rows), pd.concat(profiles, ignore_index=True)


def build_degenerate_certificates(
    states: Iterable[float],
    noise_std: float,
    n_resamples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Declare the analytically flat ramp profile on the same state grid."""
    rows: list[dict] = []
    profiles: list[pd.DataFrame] = []
    for state in states:
        state = float(state)
        rows.append(
            {
                "state": state,
                "true_delay": float(tau_true_np(state)),
                "estimated_delay": float("nan"),
                "interval_lower": TAU_MIN,
                "interval_upper": TAU_MAX,
                "interval_width": TAU_MAX - TAU_MIN,
                "bootstrap_sd": float("nan"),
                "eta_95": 0.0,
                "profile_minimum": 0.0,
                "profile_range": 0.0,
                "empirical_set_lower": TAU_MIN,
                "empirical_set_upper": TAU_MAX,
                "empirical_set_diameter": TAU_MAX - TAU_MIN,
                "empirical_set_components": 1,
                "true_in_interval": True,
                "true_in_empirical_set": True,
                "verdict": "non-identifiable / abstain",
                "verdict_reason": "structurally flat profile",
                "geometry_accepted": False,
                "n_resamples": int(n_resamples),
                "noise_std": float(noise_std),
            }
        )
        profiles.append(
            pd.DataFrame(
                {
                    "state": state,
                    "candidate_delay": R_GRID,
                    "profile_value": np.zeros_like(R_GRID),
                    "empirical_set_member": np.ones_like(R_GRID, dtype=bool),
                }
            )
        )
    return pd.DataFrame(rows), pd.concat(profiles, ignore_index=True)


def evaluate_candidates(
    geometry: pd.DataFrame,
    profiles: pd.DataFrame,
    candidates: pd.DataFrame,
    mechanistic_tolerance: float,
) -> pd.DataFrame:
    """Apply geometry and profile-membership checks to trained delay candidates."""
    geometry_index = geometry.set_index("state")
    profile_groups = {
        float(state): group.sort_values("candidate_delay")
        for state, group in profiles.groupby("state")
    }
    rows: list[dict] = []
    for candidate in candidates.to_dict(orient="records"):
        state = float(candidate["state"])
        certificate = geometry_index.loc[state]
        profile = profile_groups[state]
        learned_delay = float(candidate["learned_delay"])
        profile_value = float(
            np.interp(
                learned_delay,
                profile["candidate_delay"].to_numpy(dtype=float),
                profile["profile_value"].to_numpy(dtype=float),
            )
        )
        empirical_threshold = (
            float(certificate["profile_minimum"])
            + 2.0 * float(certificate["eta_95"])
        )
        candidate_in_empirical_set = bool(profile_value <= empirical_threshold)
        candidate_in_interval = bool(
            float(certificate["interval_lower"])
            <= learned_delay
            <= float(certificate["interval_upper"])
        )
        certificate_eligible = bool(
            candidate["model_class"] in ELIGIBLE_MODEL_CLASSES
        )
        accepted = bool(
            certificate_eligible
            and certificate["geometry_accepted"]
            and candidate_in_interval
            and candidate_in_empirical_set
        )
        absolute_error = abs(learned_delay - float(certificate["true_delay"]))
        false_confidence = bool(
            accepted and absolute_error > mechanistic_tolerance
        )

        if not certificate_eligible:
            reason = "model class outside affine-certificate scope"
        elif accepted:
            reason = "accepted"
        elif not bool(certificate["geometry_accepted"]):
            reason = str(certificate["verdict_reason"])
        elif not candidate_in_interval:
            reason = "neural delay outside resampling minimizer interval"
        else:
            reason = "neural delay outside empirical profile set"

        rows.append(
            {
                **candidate,
                "true_delay": float(certificate["true_delay"]),
                "profile_minimizer": float(certificate["estimated_delay"]),
                "interval_lower": float(certificate["interval_lower"]),
                "interval_upper": float(certificate["interval_upper"]),
                "interval_width": float(certificate["interval_width"]),
                "geometry_verdict": str(certificate["verdict"]),
                "candidate_profile_value": profile_value,
                "empirical_profile_threshold": empirical_threshold,
                "candidate_in_interval": candidate_in_interval,
                "candidate_in_empirical_set": candidate_in_empirical_set,
                "certificate_eligible": certificate_eligible,
                "candidate_accepted": accepted,
                "candidate_abstained": not accepted,
                "candidate_verdict_reason": reason,
                "absolute_delay_error": absolute_error,
                "mechanistic_tolerance": float(mechanistic_tolerance),
                "false_confidence_event": false_confidence,
            }
        )
    return pd.DataFrame(rows)


def summarize_geometry(geometry: pd.DataFrame) -> dict[str, float | int]:
    """Aggregate statewise profile behavior without reference to a neural model."""
    return {
        "n_states": int(len(geometry)),
        "accepted_states": int(geometry["geometry_accepted"].sum()),
        "abstained_states": int((~geometry["geometry_accepted"]).sum()),
        "geometry_acceptance_rate": float(geometry["geometry_accepted"].mean()),
        "true_delay_interval_coverage": float(geometry["true_in_interval"].mean()),
        "true_delay_empirical_set_coverage": float(
            geometry["true_in_empirical_set"].mean()
        ),
        "mean_interval_width": float(geometry["interval_width"].mean()),
        "median_interval_width": float(geometry["interval_width"].median()),
        "max_interval_width": float(geometry["interval_width"].max()),
    }


def summarize_candidates(candidate_results: pd.DataFrame) -> pd.DataFrame:
    """Produce pooled and run-aware coverage/false-confidence summaries."""
    rows: list[dict] = []
    for (config_id, label), group in candidate_results.groupby(
        ["config_id", "label"]
    ):
        run_rates = group.groupby("seed").agg(
            acceptance_rate=("candidate_accepted", "mean"),
            false_confidence_rate=("false_confidence_event", "mean"),
        )
        accepted = int(group["candidate_accepted"].sum())
        false_events = int(group["false_confidence_event"].sum())
        eligible = int(group["certificate_eligible"].sum())
        rows.append(
            {
                "config_id": config_id,
                "label": label,
                "n_runs": int(group["seed"].nunique()),
                "n_candidate_states": int(len(group)),
                "eligible_candidate_states": eligible,
                "ineligible_candidate_states": int(len(group) - eligible),
                "accepted_candidate_states": accepted,
                "abstained_candidate_states": int(len(group) - accepted),
                "pooled_acceptance_rate": float(
                    group["candidate_accepted"].mean()
                ),
                "mean_run_acceptance_rate": float(
                    run_rates["acceptance_rate"].mean()
                ),
                "eligible_acceptance_rate": (
                    float(
                        group.loc[
                            group["certificate_eligible"],
                            "candidate_accepted",
                        ].mean()
                    )
                    if eligible
                    else float("nan")
                ),
                "candidate_interval_inclusion_rate": float(
                    group["candidate_in_interval"].mean()
                ),
                "candidate_empirical_set_inclusion_rate": float(
                    group["candidate_in_empirical_set"].mean()
                ),
                "false_confidence_events": false_events,
                "conditional_false_confidence_rate": (
                    float(false_events / accepted) if accepted else 0.0
                ),
                "unconditional_false_confidence_rate": float(
                    group["false_confidence_event"].mean()
                ),
                "mean_run_false_confidence_rate": float(
                    run_rates["false_confidence_rate"].mean()
                ),
                "mean_absolute_delay_error": float(
                    group["absolute_delay_error"].mean()
                ),
                "mean_accepted_absolute_delay_error": (
                    float(
                        group.loc[
                            group["candidate_accepted"],
                            "absolute_delay_error",
                        ].mean()
                    )
                    if accepted
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("config_id")
