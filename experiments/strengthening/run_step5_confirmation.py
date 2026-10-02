#!/usr/bin/env python3
"""Run the predeclared Step-5 fresh-noise scalar confirmation on one CPU thread.

The Step-2/3 ablation showed that the profile-anchor loss, rather than profile
pretraining, was the effective identifiability-aware intervention.  This script
tests that observation on five independent noisy training datasets and three
paired neural initializations per dataset.  It does not add post-hoc seeds to
the original fixed noisy dataset.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch

import certificate as cert
import run_tnnls_strengthening as base
import sdde_core as core

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
OUTPUT_DIR = ROOT / "results" / "confirmation" / "generated"
FIGURE_DIR = ROOT / "figures" / "confirmation" / "generated"
CHECKPOINT_DIR = ROOT / "checkpoints" / "confirmation" / "generated"

CONFIRMATION_DATA_SEEDS = [
    20260901,
    20260902,
    20260903,
    20260904,
    20260905,
]
CONFIRMATION_OPTIMIZATION_SEEDS = [20261001, 20261002, 20261003]
CONFIRMATION_CONFIG_IDS = ("P1", "P2", "P3", "P4")
PRIMARY_CONFIG_ID = "P3"
REFERENCE_CONFIG_ID = "P4"

PRIMARY_MAX_DELAY_RATIO = 0.50
PRIMARY_MIN_PAIRED_IMPROVEMENTS = 12
PRIMARY_MAX_TRAJECTORY_RATIO = 2.0
PRIMARY_MIN_ACCEPTANCE_RATE = 0.70
PRIMARY_MAX_FALSE_CONFIDENCE_RATE = 0.05


def selected_configurations() -> tuple[base.Configuration, ...]:
    configurations = {
        configuration.config_id: configuration
        for configuration in base.PRINCIPAL_CONFIGURATIONS
    }
    return tuple(configurations[config_id] for config_id in CONFIRMATION_CONFIG_IDS)


def fixed_certificate_geometry(
    quick: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    clean_certificate, _ = core.make_strong_dataset(base.CERTIFICATE_STATES)
    noisy_certificate = core.add_observation_noise(
        clean_certificate,
        base.NOISE_STD,
        base.CERTIFICATE_DATA_SEED,
    )
    n_resamples = (
        base.QUICK_CERTIFICATE_RESAMPLES
        if quick
        else base.PUBLICATION_CERTIFICATE_RESAMPLES
    )
    geometry, profiles = cert.build_informative_certificates(
        noisy_certificate,
        base.CERTIFICATE_STATES,
        base.NOISE_STD,
        base.CERTIFICATE_BOOTSTRAP_SEED,
        n_resamples,
    )
    return geometry, profiles, n_resamples


def run_confirmation(
    quick: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_train, _ = core.make_strong_dataset(base.TRAIN_STATES)
    clean_test, _ = core.make_strong_dataset(base.TEST_STATES)
    state_grid = np.linspace(-0.75, 0.75, 201)
    certificate_states = np.asarray(base.CERTIFICATE_STATES, dtype=float)
    configurations = selected_configurations()
    data_seeds = CONFIRMATION_DATA_SEEDS[:1] if quick else CONFIRMATION_DATA_SEEDS
    optimization_seeds = (
        CONFIRMATION_OPTIMIZATION_SEEDS[:1]
        if quick
        else CONFIRMATION_OPTIMIZATION_SEEDS
    )
    schedule = [(10, 12), (20, 20)] if quick else [(10, 90), (20, 190)]
    pretrain_epochs = 40 if quick else 600
    total_runs = len(data_seeds) * len(optimization_seeds) * len(configurations)

    run_rows: list[dict] = []
    candidate_rows: list[dict] = []
    run_number = 0
    for data_index, data_seed in enumerate(data_seeds):
        noisy_train = core.add_observation_noise(
            clean_train,
            base.NOISE_STD,
            data_seed,
        )
        noisy_test = core.add_observation_noise(
            clean_test,
            base.NOISE_STD,
            data_seed + 1000,
        )
        anchor_states, anchor_delays = core.estimate_profile_anchors(
            noisy_train,
            base.TRAIN_STATES,
        )
        anchor_mae = float(
            np.mean(
                np.abs(
                    anchor_delays
                    - np.asarray(core.tau_true_np(anchor_states), dtype=float)
                )
            )
        )

        for optimization_seed in optimization_seeds:
            model_seed = optimization_seed + 100 * data_index
            for configuration in configurations:
                run_number += 1
                print(
                    f"[{run_number:02d}/{total_runs:02d}] "
                    f"data={data_seed} opt={optimization_seed} "
                    f"{configuration.config_id}",
                    flush=True,
                )
                model = base.make_model(configuration, model_seed, None)
                if configuration.use_profile_initialization:
                    base.pretrain_tau(
                        model,
                        anchor_states,
                        anchor_delays,
                        model_seed + 303,
                        pretrain_epochs,
                    )
                model, final_fit = base.train_model(
                    model,
                    clean_train[:, : core.HIST_STEPS + 1],
                    noisy_train[:, core.HIST_STEPS :],
                    model_seed + 404,
                    schedule,
                    anchor_states=(
                        anchor_states
                        if configuration.use_anchor_loss
                        else None
                    ),
                    anchor_delays=(
                        anchor_delays
                        if configuration.use_anchor_loss
                        else None
                    ),
                    anchor_weight=(
                        0.08 if configuration.use_anchor_loss else 0.0
                    ),
                )
                train_mse, test_mse, noisy_test_mse = base.trajectory_metrics(
                    model,
                    clean_train,
                    clean_test,
                    noisy_test,
                )
                delay_result, _ = base.delay_metrics(model, state_grid)
                run_rows.append(
                    {
                        **asdict(configuration),
                        "data_seed": data_seed,
                        "data_replication": data_index,
                        "optimization_seed": optimization_seed,
                        "model_seed": model_seed,
                        "noise_std": base.NOISE_STD,
                        "anchor_mae": anchor_mae,
                        "final_weighted_fit": final_fit,
                        "clean_train_mse": train_mse,
                        "clean_test_mse": test_mse,
                        "noisy_test_mse": noisy_test_mse,
                        **delay_result,
                    }
                )
                certificate_delays = core.delay_curve(
                    model,
                    certificate_states,
                )
                candidate_rows.extend(
                    {
                        "config_id": configuration.config_id,
                        "label": configuration.label,
                        "model_class": configuration.model_class,
                        "seed": model_seed,
                        "data_seed": data_seed,
                        "optimization_seed": optimization_seed,
                        "state": float(state),
                        "learned_delay": float(delay),
                    }
                    for state, delay in zip(
                        certificate_states,
                        certificate_delays,
                    )
                )
                checkpoint_name = (
                    f"{configuration.config_id}_data{data_seed}"
                    f"_opt{optimization_seed}.pt"
                )
                core.save_checkpoint(
                    model,
                    CHECKPOINT_DIR / checkpoint_name,
                )
    return pd.DataFrame(run_rows), pd.DataFrame(candidate_rows)


def aggregate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "clean_train_mse",
        "clean_test_mse",
        "noisy_test_mse",
        "delay_rmse",
        "delay_nrmse",
        "delay_mae",
        "delay_max_abs_error",
        "anchor_mae",
    ]
    rows: list[dict] = []
    for (config_id, label), group in runs.groupby(["config_id", "label"]):
        row: dict[str, float | int | str] = {
            "config_id": config_id,
            "label": label,
            "n_runs": int(len(group)),
            "n_data_replications": int(group["data_seed"].nunique()),
            "n_optimization_seeds": int(group["optimization_seed"].nunique()),
        }
        for column in columns:
            values = group[column].to_numpy(dtype=float)
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_sample_sd"] = float(np.std(values, ddof=1))
            row[f"{column}_median"] = float(np.median(values))
            row[f"{column}_q25"] = float(np.quantile(values, 0.25))
            row[f"{column}_q75"] = float(np.quantile(values, 0.75))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("config_id")


def paired_comparison(
    runs: pd.DataFrame,
    first_id: str,
    second_id: str,
) -> tuple[pd.DataFrame, dict]:
    selected = runs[runs["config_id"].isin([first_id, second_id])]
    wide_delay = selected.pivot(
        index=["data_seed", "optimization_seed"],
        columns="config_id",
        values="delay_rmse",
    )
    wide_mse = selected.pivot(
        index=["data_seed", "optimization_seed"],
        columns="config_id",
        values="clean_test_mse",
    )
    pairs = pd.DataFrame(
        {
            "data_seed": wide_delay.index.get_level_values("data_seed"),
            "optimization_seed": wide_delay.index.get_level_values(
                "optimization_seed"
            ),
            f"{first_id}_delay_rmse": wide_delay[first_id].to_numpy(),
            f"{second_id}_delay_rmse": wide_delay[second_id].to_numpy(),
            "delay_rmse_ratio": (
                wide_delay[first_id] / wide_delay[second_id]
            ).to_numpy(),
            "delay_rmse_difference": (
                wide_delay[first_id] - wide_delay[second_id]
            ).to_numpy(),
            f"{first_id}_clean_test_mse": wide_mse[first_id].to_numpy(),
            f"{second_id}_clean_test_mse": wide_mse[second_id].to_numpy(),
            "clean_test_mse_ratio": (
                wide_mse[first_id] / wide_mse[second_id]
            ).to_numpy(),
        }
    )
    first = runs[runs["config_id"] == first_id]
    second = runs[runs["config_id"] == second_id]
    summary = {
        "first_config": first_id,
        "second_config": second_id,
        "n_pairs": int(len(pairs)),
        "paired_delay_improvements": int(
            (pairs["delay_rmse_ratio"] < 1.0).sum()
        ),
        "delay_ratio_of_medians": float(
            first["delay_rmse"].median() / second["delay_rmse"].median()
        ),
        "median_paired_delay_ratio": float(
            pairs["delay_rmse_ratio"].median()
        ),
        "clean_test_mse_ratio_of_medians": float(
            first["clean_test_mse"].median()
            / second["clean_test_mse"].median()
        ),
        "median_paired_clean_test_mse_ratio": float(
            pairs["clean_test_mse_ratio"].median()
        ),
    }
    return pairs, summary


def evaluate_confirmation_criteria(
    runs: pd.DataFrame,
    certificate_summary: pd.DataFrame,
    primary_comparison: dict,
    quick: bool,
) -> dict:
    certificate_by_id = certificate_summary.set_index("config_id")
    p3_acceptance = float(
        certificate_by_id.loc[PRIMARY_CONFIG_ID, "pooled_acceptance_rate"]
    )
    p3_false_confidence = float(
        certificate_by_id.loc[
            PRIMARY_CONFIG_ID,
            "conditional_false_confidence_rate",
        ]
    )
    required_improvements = (
        1 if quick else PRIMARY_MIN_PAIRED_IMPROVEMENTS
    )
    return {
        "mode": "quick smoke test" if quick else "full confirmation",
        "primary_comparison": f"{PRIMARY_CONFIG_ID}_vs_{REFERENCE_CONFIG_ID}",
        "predeclared_thresholds": {
            "maximum_delay_ratio_of_medians": PRIMARY_MAX_DELAY_RATIO,
            "minimum_paired_improvements": required_improvements,
            "maximum_trajectory_mse_ratio_of_medians": (
                PRIMARY_MAX_TRAJECTORY_RATIO
            ),
            "minimum_candidate_acceptance_rate": (
                PRIMARY_MIN_ACCEPTANCE_RATE
            ),
            "maximum_conditional_false_confidence_rate": (
                PRIMARY_MAX_FALSE_CONFIDENCE_RATE
            ),
        },
        "observed": {
            **primary_comparison,
            "P3_candidate_acceptance_rate": p3_acceptance,
            "P3_conditional_false_confidence_rate": p3_false_confidence,
        },
        "passes_predeclared_confirmation": bool(
            primary_comparison["delay_ratio_of_medians"]
            <= PRIMARY_MAX_DELAY_RATIO
            and primary_comparison["paired_delay_improvements"]
            >= required_improvements
            and primary_comparison["clean_test_mse_ratio_of_medians"]
            <= PRIMARY_MAX_TRAJECTORY_RATIO
            and p3_acceptance >= PRIMARY_MIN_ACCEPTANCE_RATE
            and p3_false_confidence
            <= PRIMARY_MAX_FALSE_CONFIDENCE_RATE
        ),
        "interpretation": (
            "P3 is the profile-anchor objective without profile pretraining. "
            "P4 is the architecture-matched rollout-only reference."
        ),
    }


def create_figures(
    runs: pd.DataFrame,
    primary_pairs: pd.DataFrame,
    certificate_summary: pd.DataFrame,
) -> None:
    order = list(CONFIRMATION_CONFIG_IDS)
    plt.figure(figsize=(7.2, 4.8))
    for index, config_id in enumerate(order):
        values = runs.loc[
            runs["config_id"] == config_id,
            "delay_rmse",
        ].to_numpy()
        jitter = np.linspace(-0.12, 0.12, len(values))
        plt.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=36,
            alpha=0.75,
        )
    plt.xticks(range(len(order)), order)
    plt.yscale("log")
    plt.xlabel("Confirmation configuration")
    plt.ylabel("Delay RMSE")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "step5_confirmation_delay_rmse.png",
        dpi=220,
    )
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    ratios = primary_pairs["delay_rmse_ratio"].to_numpy(dtype=float)
    plt.axhline(1.0, color="black", linewidth=1.5, linestyle="--")
    plt.axhline(
        PRIMARY_MAX_DELAY_RATIO,
        color="#b5472f",
        linewidth=1.5,
        linestyle=":",
        label="Reference ratio 0.50",
    )
    plt.scatter(
        np.arange(1, len(ratios) + 1),
        ratios,
        color="#315b8a",
        s=42,
    )
    plt.yscale("log")
    plt.xlabel("Paired data/optimization replication")
    plt.ylabel("P3/P4 delay-RMSE ratio")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "step5_paired_delay_ratios.png",
        dpi=220,
    )
    plt.close()

    certificate = certificate_summary.set_index("config_id").loc[order]
    x = np.arange(len(order))
    plt.figure(figsize=(7.2, 4.8))
    plt.bar(
        x - 0.18,
        certificate["pooled_acceptance_rate"],
        width=0.36,
        color="#315b8a",
        label="Candidate acceptance",
    )
    plt.bar(
        x + 0.18,
        certificate["conditional_false_confidence_rate"],
        width=0.36,
        color="#b5472f",
        label="Accepted large error / accepted",
    )
    plt.xticks(x, order)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Rate (denominators differ; see legend)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "step5_confirmation_certificate.png",
        dpi=220,
    )
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run five independent noise datasets and three paired seeds.",
    )
    args = parser.parse_args()
    quick = not args.full
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    geometry, profiles, n_resamples = fixed_certificate_geometry(quick)
    runs, candidates = run_confirmation(quick)
    candidate_results = cert.evaluate_candidates(
        geometry,
        profiles,
        candidates,
        base.MECHANISTIC_TOLERANCE,
    )
    run_summary = aggregate_runs(runs)
    certificate_summary = cert.summarize_candidates(candidate_results)

    primary_pairs, primary_comparison = paired_comparison(
        runs,
        PRIMARY_CONFIG_ID,
        REFERENCE_CONFIG_ID,
    )
    p1_p3_pairs, p1_p3_comparison = paired_comparison(runs, "P1", "P3")
    p2_p4_pairs, p2_p4_comparison = paired_comparison(runs, "P2", "P4")
    criteria = evaluate_confirmation_criteria(
        runs,
        certificate_summary,
        primary_comparison,
        quick,
    )
    comparisons = {
        "P3_vs_P4_primary": primary_comparison,
        "P1_vs_P3_pretraining_effect": p1_p3_comparison,
        "P2_vs_P4_pretraining_only_effect": p2_p4_comparison,
    }

    runs.to_csv(OUTPUT_DIR / "step5_confirmation_runs.csv", index=False)
    run_summary.to_csv(
        OUTPUT_DIR / "step5_confirmation_summary.csv",
        index=False,
    )
    candidate_results.to_csv(
        OUTPUT_DIR / "step5_confirmation_candidate_certificates.csv",
        index=False,
    )
    certificate_summary.to_csv(
        OUTPUT_DIR / "step5_confirmation_certificate_summary.csv",
        index=False,
    )
    geometry.to_csv(
        OUTPUT_DIR / "step5_confirmation_geometry.csv",
        index=False,
    )
    profiles.to_csv(OUTPUT_DIR / "step5_confirmation_profiles.csv", index=False)
    primary_pairs.to_csv(
        OUTPUT_DIR / "step5_P3_vs_P4_pairs.csv",
        index=False,
    )
    p1_p3_pairs.to_csv(
        OUTPUT_DIR / "step5_P1_vs_P3_pairs.csv",
        index=False,
    )
    p2_p4_pairs.to_csv(
        OUTPUT_DIR / "step5_P2_vs_P4_pairs.csv",
        index=False,
    )
    (OUTPUT_DIR / "step5_confirmation_comparisons.json").write_text(
        json.dumps(comparisons, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "step5_confirmation_criteria.json").write_text(
        json.dumps(criteria, indent=2),
        encoding="utf-8",
    )
    create_figures(
        runs,
        primary_pairs,
        certificate_summary,
    )

    elapsed = time.time() - start
    manifest = {
        "mode": "full" if args.full else "quick smoke test",
        "data_seeds": CONFIRMATION_DATA_SEEDS[:1] if quick else CONFIRMATION_DATA_SEEDS,
        "optimization_seeds": CONFIRMATION_OPTIMIZATION_SEEDS[:1] if quick else CONFIRMATION_OPTIMIZATION_SEEDS,
        "model_seeds": sorted(int(seed) for seed in runs["model_seed"].unique()),
        "model_seed_rule": "optimization_seed + 100 * data_replication_index",
        "training_schedule": [(10, 12), (20, 20)] if quick else [(10, 90), (20, 190)],
        "profile_pretraining_epochs": 40 if quick else 600,
        "anchor_weight": 0.08,
        "anchor_state_weights": "uniform; all four training anchors retained",
        "certificate_data_seed": base.CERTIFICATE_DATA_SEED,
        "certificate_bootstrap_seed": base.CERTIFICATE_BOOTSTRAP_SEED,
        "configurations": [
            asdict(configuration)
            for configuration in selected_configurations()
        ],
        "noise_std": base.NOISE_STD,
        "certificate_states": [
            float(state) for state in base.CERTIFICATE_STATES
        ],
        "certificate_resamples": n_resamples,
        "elapsed_seconds": elapsed,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "scipy_version": scipy.__version__,
        "pandas_version": pd.__version__,
        "matplotlib_version": matplotlib.__version__,
        "cpu_threads": torch.get_num_threads(),
    }
    (OUTPUT_DIR / "step5_confirmation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nStep-5 confirmation summary")
    print(
        run_summary[
            [
                "config_id",
                "n_runs",
                "clean_test_mse_median",
                "delay_rmse_median",
            ]
        ].to_string(index=False)
    )
    print("\nCertificate summary")
    print(
        certificate_summary[
            [
                "config_id",
                "pooled_acceptance_rate",
                "conditional_false_confidence_rate",
            ]
        ].to_string(index=False)
    )
    print("\nPrimary P3-versus-P4 comparison")
    print(json.dumps(primary_comparison, indent=2))
    print(
        "\nPredeclared confirmation pass:",
        criteria["passes_predeclared_confirmation"],
    )
    print(f"Completed in {elapsed:.1f} seconds")
    if quick:
        print(
            "This was a smoke test. Use --full for five data replications "
            "and three paired optimization seeds."
        )


if __name__ == "__main__":
    main()
