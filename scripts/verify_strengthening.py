"""Verification of the recovered full fixed-noise study, without PyTorch."""
from __future__ import annotations
import itertools
import json
import numpy as np
import pandas as pd


def check_strengthening(require, close, root):
    from experiments.strengthening import certificate as cert
    base = root / "results/strengthening/reference"
    principal = pd.read_csv(base/"step3_principal_runs.csv")
    degenerate = pd.read_csv(base/"step3_degenerate_runs.csv")
    runs = pd.concat([principal, degenerate], ignore_index=True)
    require(len(principal) == 33 and len(degenerate) == 6, "fixed-noise: 33 principal and six ramp runs")
    manifest = json.loads((base/"step3_run_manifest.json").read_text())
    require(manifest["mode"] == "full" and manifest["publication_seeds"] == list(range(20260719,20260724))
            and manifest["certificate_resamples"] == 120, "fixed-noise: full five-seed execution manifest")
    expected_counts = {**{f"P{i}":5 for i in range(6)}, "P6":3, "D1":5, "D2":1}
    require(runs.groupby("config_id").size().to_dict() == expected_counts
            and not runs.duplicated(["config_id","seed"]).any(), "fixed-noise: per-configuration run counts and keys")
    summaries = pd.concat([pd.read_csv(base/"step3_principal_summary.csv"),
                           pd.read_csv(base/"step3_degenerate_summary.csv")]).set_index("config_id")
    metrics = [c[:-5] for c in summaries.columns if c.endswith("_mean")]
    for cid, group in runs.groupby("config_id"):
        require(len(group) == summaries.loc[cid,"n_runs"], f"fixed-noise {cid}: summary run count")
        for metric in metrics:
            v = group[metric]
            actual = [v.mean(), v.std(ddof=1), v.median(), v.quantile(.25), v.quantile(.75)]
            stored = [summaries.loc[cid,metric+s] for s in ["_mean","_sample_sd","_median","_q25","_q75"]]
            require(close(actual, stored), f"fixed-noise {cid}: {metric} summaries")
    # Rounded values transcribed from Table S2 of the supplied supplement.
    published = {
        "P0":("5.525e-05","1.469e-08"), "P1":("8.624e-05","1.593e-02"),
        "P2":("1.108e-04","3.335e-02"), "P3":("6.789e-05","1.313e-02"),
        "P4":("9.737e-05","3.041e-02"), "P5":("1.059e-05","3.194e-02"),
        "P6":("5.453e-05","1.646e-01"), "D1":("9.589e-05","1.193e-01"),
        "D2":("1.157e-04","1.646e-01"),
    }
    for cid, group in runs.groupby("config_id"):
        actual = (f"{group.clean_test_mse.median():.3e}",f"{group.delay_rmse.median():.3e}")
        require(actual == published[cid], f"fixed-noise {cid}: published Table S2 medians")
    require(all((root/f"checkpoints/strengthening/reference/{r.config_id}_seed{r.seed}.pt").is_file()
                for r in runs.itertuples()), "fixed-noise: all 39 checkpoints present")
    curve_frames = [pd.read_csv(base/f"step3_{kind}_delay_curves.csv") for kind in ["principal","degenerate"]]
    curves = pd.concat(curve_frames, ignore_index=True)
    require(len(curves) == 39*201 and not curves.duplicated(["config_id","seed","state"]).any(),
            "fixed-noise: 201 saved delay values for every run")
    indexed = runs.set_index(["config_id","seed"])
    for (cid,seed), group in curves.groupby(["config_id","seed"]):
        error = group.learned_delay.to_numpy()-group.true_delay.to_numpy()
        actual = [np.sqrt(np.mean(error**2)), np.sqrt(np.mean(error**2))/.35,
                  np.mean(abs(error)), max(abs(error)), np.mean(group.learned_delay.to_numpy(dtype=np.float32)),
                  group.learned_delay.min(), group.learned_delay.max()]
        stored = indexed.loc[(cid,seed),["delay_rmse","delay_nrmse","delay_mae","delay_max_abs_error",
                                        "delay_mean","delay_min","delay_max"]].to_numpy(dtype=float)
        require(close(actual,stored), f"fixed-noise {cid}/{seed}: delay metrics from saved curves")
    dispersion = pd.read_csv(base/"step3_delay_dispersion.csv").set_index("config_id")
    for cid, group in curves.groupby("config_id"):
        # The original neural curve arrays and pairwise arithmetic used float32.
        values = group.pivot(index="seed",columns="state",values="learned_delay").to_numpy(dtype=np.float32)
        pairs = [float(np.sqrt(np.mean((a-b)**2))) for a,b in itertools.combinations(values,2)]
        stored = dispersion.loc[cid]
        require(stored.n_pairs == len(pairs), f"fixed-noise {cid}: dispersion pair count")
        actual = [np.mean(pairs),np.median(pairs),max(pairs)] if pairs else [np.nan]*3
        require(close(actual,stored[["mean_pairwise_delay_rmse","median_pairwise_delay_rmse",
                                      "max_pairwise_delay_rmse"]].to_numpy(dtype=float)),
                f"fixed-noise {cid}: between-run delay dispersion")
    principal_candidates = pd.read_csv(base/"step3_principal_candidate_certificates.csv")
    degenerate_candidates = pd.read_csv(base/"step3_degenerate_candidate_certificates.csv")
    candidates = pd.concat([principal_candidates,degenerate_candidates],ignore_index=True)
    require(len(candidates) == 819 and (candidates.groupby(["config_id","seed"]).size()==21).all(),
            "fixed-noise: 819 candidate-state records")
    decision = (candidates.certificate_eligible & candidates.geometry_verdict.eq("identifiable")
                & candidates.candidate_in_interval & candidates.candidate_in_empirical_set)
    error = abs(candidates.learned_delay-candidates.true_delay)
    require(decision.equals(candidates.candidate_accepted)
            and (decision & (error > candidates.mechanistic_tolerance)).equals(candidates.false_confidence_event),
            "fixed-noise: acceptance gates and accepted-large-error flags")
    actual = cert.summarize_candidates(candidates).set_index("config_id")
    stored = pd.read_csv(base/"step3_certificate_summary.csv").set_index("config_id")
    numeric = actual.select_dtypes(include="number").columns
    require(close(actual.loc[stored.index,numeric],stored[numeric]), "fixed-noise: all candidate summary values")
    p3 = candidates[candidates.config_id.eq("P3")]
    require(p3.candidate_accepted.sum() == 75 and not p3.false_confidence_event.any(),
            "fixed-noise: P3 accepted 75/105 with zero observed accepted-large-error events")
    paired = principal.pivot(index="seed",columns="config_id",values="delay_rmse")
    require((paired.P3<paired.P4).sum()==4, "fixed-noise: P3 improves four of five paired runs")
    criteria = json.loads((base/"step3_success_criteria.json").read_text())["P1_vs_P4"]
    p1, p4 = principal[principal.config_id.eq("P1")], principal[principal.config_id.eq("P4")]
    require(close(p1.delay_rmse.median()/p4.delay_rmse.median(),criteria["median_delay_rmse_ratio"])
            and (paired.P1<paired.P4).sum()==criteria["paired_improvements"]
            and close(p1.clean_test_mse.median()/p4.clean_test_mse.median(),criteria["median_trajectory_mse_ratio"]),
            "fixed-noise: archived P1/P4 effect statistics")
    require(not criteria["passes_predeclared_effect_criteria"], "fixed-noise: initial P1 comparison did not pass its effect criterion")
    return principal_candidates, degenerate_candidates


def check_strengthening_geometry(require, close, root, geometry, profiles):
    from experiments.strengthening import certificate as cert
    base = root / "results/strengthening/reference"
    for kind, actual_geometry, actual_profiles in [
        ("informative", geometry, profiles),
        ("degenerate", *cert.build_degenerate_certificates(tuple(np.linspace(-.7,.7,21)),.0005,120))
    ]:
        stored = pd.read_csv(base/f"step3_{kind}_certificate_statewise.csv")
        numeric = actual_geometry.select_dtypes(include="number").columns
        require(close(actual_geometry[numeric],stored[numeric]), f"fixed-noise rebuilt {kind}: all geometry values")
        require((actual_geometry.geometry_accepted == stored.geometry_accepted).all(),
                f"fixed-noise rebuilt {kind}: geometry verdicts")
        archived_profiles = pd.read_csv(base/f"step3_{kind}_profiles.csv")
        require(close(actual_profiles[["state","candidate_delay","profile_value"]],
                      archived_profiles[["state","candidate_delay","profile_value"]])
                and actual_profiles.empirical_set_member.equals(archived_profiles.empirical_set_member),
                f"fixed-noise rebuilt {kind}: all 7371 profile grid values and memberships")
        label = "principal" if kind=="informative" else "degenerate"
        archived = pd.read_csv(base/f"step3_{label}_candidate_certificates.csv")
        source = archived[["config_id","label","model_class","seed","state","learned_delay"]].copy()
        states = actual_geometry.state.to_numpy()
        source["state"] = [states[int(np.argmin(abs(states-s)))] for s in source.state]
        actual = cert.evaluate_candidates(actual_geometry,actual_profiles,source,.035)
        flags = ["candidate_accepted","candidate_in_interval","candidate_in_empirical_set",
                 "certificate_eligible","false_confidence_event"]
        require(actual[flags].equals(archived[flags]), f"fixed-noise rebuilt {kind}: all candidate decision flags")
        require(close(actual.candidate_profile_value,archived.candidate_profile_value),
                f"fixed-noise rebuilt {kind}: all candidate profile values")
