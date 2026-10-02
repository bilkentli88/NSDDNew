"""Verify recorded statistics and optionally reconstruct the profile pipeline.

This checks records and profile computations, not neural training or coverage.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIRMATION = ROOT / "results/confirmation/reference"
EXTENDED = ROOT / "results/extended/reference"


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)
    print(f"[PASS] {message}")


def close(actual, expected):
    return np.allclose(actual, expected, rtol=1e-10, atol=1e-13, equal_nan=True)


def check_confirmation():
    runs = pd.read_csv(CONFIRMATION / "step5_confirmation_runs.csv")
    candidates = pd.read_csv(CONFIRMATION / "step5_confirmation_candidate_certificates.csv")
    summary = pd.read_csv(CONFIRMATION / "step5_confirmation_summary.csv").set_index("config_id")
    cert_summary = pd.read_csv(CONFIRMATION / "step5_confirmation_certificate_summary.csv").set_index("config_id")
    criteria = json.loads((CONFIRMATION / "step5_confirmation_criteria.json").read_text())
    require(len(runs) == 60 and len(candidates) == 1260, "60 run records and 1260 candidate-state records")
    keys = ["config_id", "data_seed", "optimization_seed"]
    expected = {(config, data, opt) for config in ["P1", "P2", "P3", "P4"]
                for data in range(20260901, 20260906) for opt in range(20261001, 20261004)}
    require(not runs.duplicated(keys).any() and set(runs[keys].itertuples(index=False, name=None)) == expected,
            "five datasets and three paired seeds for every configuration")
    require((runs.model_seed == runs.optimization_seed + 100 * runs.data_replication).all(), "model seed offsets")
    metrics = ["clean_train_mse", "clean_test_mse", "noisy_test_mse", "delay_rmse", "delay_nrmse",
               "delay_mae", "delay_max_abs_error", "anchor_mae"]
    for cid, group in runs.groupby("config_id"):
        for metric in metrics:
            v = group[metric].to_numpy()
            actual = [v.mean(), v.std(ddof=1), np.median(v), np.quantile(v, .25), np.quantile(v, .75)]
            stored = [summary.loc[cid, metric + suffix] for suffix in ["_mean", "_sample_sd", "_median", "_q25", "_q75"]]
            require(close(actual, stored), f"{cid}: {metric} summary")
    require(not candidates.duplicated(keys+["state"]).any() and (candidates.groupby(keys).size() == 21).all(),
            "21 unique states for every candidate")
    decision = (candidates.certificate_eligible & candidates.geometry_verdict.eq("identifiable")
                & candidates.candidate_in_interval & candidates.candidate_in_empirical_set)
    error = (candidates.learned_delay - candidates.true_delay).abs()
    event = decision & (error > candidates.mechanistic_tolerance)
    require(decision.equals(candidates.candidate_accepted), "all recorded acceptance gates")
    require(close(error, candidates.absolute_delay_error) and event.equals(candidates.false_confidence_event), "all large-error events")
    counts = {"P1": (247, 0), "P2": (162, 16), "P3": (244, 0), "P4": (165, 10)}
    for cid, group in candidates.groupby("config_id"):
        accepted, errors = int(group.candidate_accepted.sum()), int(group.false_confidence_event.sum())
        require((accepted, errors) == counts[cid], f"{cid}: accepted and accepted-large-error counts")
        stored = cert_summary.loc[cid]
        require(int(stored.accepted_candidate_states) == accepted and int(stored.false_confidence_events) == errors
                and close(stored.pooled_acceptance_rate, accepted / len(group))
                and close(stored.conditional_false_confidence_rate, errors / accepted), f"{cid}: candidate summary")
    paired = runs.pivot(index=["data_seed", "optimization_seed"], columns="config_id", values="delay_rmse")
    for first, second in [("P3", "P4"), ("P1", "P3"), ("P2", "P4")]:
        stored = pd.read_csv(CONFIRMATION / f"step5_{first}_vs_{second}_pairs.csv").set_index(["data_seed", "optimization_seed"])
        require(close(paired[first] / paired[second], stored.delay_rmse_ratio), f"{first}/{second} paired ratios")
    p3, p4 = runs[runs.config_id.eq("P3")], runs[runs.config_id.eq("P4")]
    ratio = p3.delay_rmse.median() / p4.delay_rmse.median()
    mse_ratio = p3.clean_test_mse.median() / p4.clean_test_mse.median()
    observed = criteria["observed"]
    require(close(ratio, observed["delay_ratio_of_medians"]) and close(np.median(paired.P3 / paired.P4), observed["median_paired_delay_ratio"]),
            "distinct marginal-median and paired-median ratios")
    t = criteria["predeclared_thresholds"]
    p3c = candidates[candidates.config_id.eq("P3")]
    passes = (ratio <= t["maximum_delay_ratio_of_medians"]
              and (paired.P3 < paired.P4).sum() >= t["minimum_paired_improvements"]
              and mse_ratio <= t["maximum_trajectory_mse_ratio_of_medians"]
              and p3c.candidate_accepted.mean() >= t["minimum_candidate_acceptance_rate"]
              and p3c.false_confidence_event.sum()/p3c.candidate_accepted.sum() <= t["maximum_conditional_false_confidence_rate"])
    require(bool(passes) == criteria["passes_predeclared_confirmation"], "confirmation decision from individual records")
    require(round(100*(1-ratio), 1) == 71.7 and (paired.P3 < paired.P4).sum() == 15, "71.7% reduction and 15/15 improvements")
    return candidates


def check_extended():
    runs = pd.read_csv(EXTENDED / "noise_scaling_runs.csv")
    summary = pd.read_csv(EXTENDED / "noise_scaling.csv").set_index("noise_std")
    require(len(runs) == 10 and set(runs.noise_std) == {0., .0005, .001, .002, .005}, "five noise levels, two records per level")
    for sigma, group in runs.groupby("noise_std"):
        require(len(group) == 2, f"noise {sigma:g}: two runs")
        for metric in ["clean_test_mse", "delay_rmse"]:
            require(close([group[metric].mean(), group[metric].std(ddof=1)], [summary.loc[sigma, metric], summary.loc[sigma, metric+"_std"]]),
                    f"noise {sigma:g}: {metric} mean and sample SD")
        require(group.certificate_identified_rate.notna().sum() == 1, f"noise {sigma:g}: diagnostic recorded only for first run")
    require(close(summary.certificate_identified_rate, [1, 1, 2/3, 1/3, 0]), "resolved-state counts")
    require(close(pd.read_csv(EXTENDED / "solver_refinement.csv").dt, [.1, .05, .025, .0125]), "four frozen-model Euler resolutions")
    require(set(pd.read_csv(EXTENDED / "extrapolation.csv").region) == {"interpolation", "extrapolation"}, "both state-domain evaluations")
    require(len(pd.read_csv(EXTENDED / "oscillator_summary.csv")) == 3, "three oscillator regimes")
    require((ROOT / "checkpoints/reference/restricted_noisy_seed0.pt").is_file(), "extended reference checkpoint")


def rebuild_geometry(candidates, output=None):
    from experiments.strengthening import certificate as cert
    from experiments.strengthening import numpy_data as data
    states = tuple(np.linspace(-.7, .7, 21))
    clean, _ = data.make_strong_dataset(states)
    noisy = data.add_observation_noise(clean, .0005, 20260803)
    geometry, profiles = cert.build_informative_certificates(noisy, states, .0005, 20260804, 120)
    archived = pd.read_csv(CONFIRMATION / "step5_confirmation_geometry.csv")
    for col in ["state", "true_delay", "estimated_delay", "interval_lower", "interval_upper", "interval_width",
                "eta_95", "profile_minimum", "profile_range", "empirical_set_diameter"]:
        require(close(geometry[col], archived[col]), f"rebuilt geometry: {col}")
    cols = ["config_id", "label", "model_class", "seed", "data_seed", "optimization_seed", "state", "learned_delay"]
    candidate_input = candidates[cols].copy()
    candidate_input["state"] = [states[int(np.argmin(abs(np.asarray(states)-s)))] for s in candidate_input.state]
    actual = cert.evaluate_candidates(geometry, profiles, candidate_input, .035)
    for col in ["candidate_accepted", "candidate_in_interval", "candidate_in_empirical_set", "false_confidence_event"]:
        require(actual[col].equals(candidates[col]), f"rebuilt {col} flags")
    require(close(actual.candidate_profile_value, candidates.candidate_profile_value), "rebuilt complete candidate profile values")
    from verify_strengthening import check_strengthening_geometry
    check_strengthening_geometry(require, close, ROOT, geometry, profiles)
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        geometry.to_csv(output/"rebuilt_geometry.csv", index=False)
        profiles.to_csv(output/"rebuilt_profiles.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", action="store_true")
    parser.add_argument("--save-rebuilt", type=Path)
    args = parser.parse_args()
    candidates = check_confirmation()
    from verify_strengthening import check_strengthening
    check_strengthening(require, close, ROOT)
    check_extended()
    if args.geometry:
        rebuild_geometry(candidates, args.save_rebuilt)
    print("\nArchived-record verification passed. Neural training was not rerun.")


if __name__ == "__main__":
    main()
