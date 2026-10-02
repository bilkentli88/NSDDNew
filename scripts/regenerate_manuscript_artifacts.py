#!/usr/bin/env python3
"""Regenerate manuscript tables and confirmation plots from verified CSV records.

Both Figure 2 panels, Figure S1 and Tables S2-S6 are built from records.
No PyTorch dependency is needed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def table(path, headers, rows):
    # Booktabs is used by the accompanying manuscript.
    lines = [r"\begin{tabular}{" + "l"*len(headers) + "}", r"\toprule",
             " & ".join(h.replace("_", r"\_") for h in headers)+r" \\", r"\midrule"]
    lines += [" & ".join(map(str, row))+r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines)+"\n")


def sci(value):
    if value == 0:
        return "$0$"
    mantissa, exponent = f"{float(value):.3e}".split("e")
    return rf"\({mantissa}\times10^{{{int(exponent)}}}\)"


def mean_sd(mean, sd):
    exponent = int(np.floor(np.log10(abs(mean))))
    scale = 10.**exponent
    return rf"\(({mean/scale:.3f}\pm{sd/scale:.3f})\times10^{{{exponent}}}\)"



def fixed_noise_artifacts(out, base):
    runs = pd.concat([pd.read_csv(base/"step3_principal_runs.csv"),
                      pd.read_csv(base/"step3_degenerate_runs.csv")], ignore_index=True)
    groups = {cid:g for cid,g in runs.groupby("config_id")}
    labels = {"P0":"Oracle-delay restricted", "P1":"Profile initialization and anchor",
              "P2":"Profile initialization only", "P3":"Anchor only",
              "P4":"Restricted rollout only", "P5":"Unrestricted, learned delay",
              "P6":"Unrestricted, fixed delay 0.46", "D1":"Restricted, learned delay, ramps",
              "D2":"Restricted, fixed delay, ramps"}
    table(out/"table_S2_fixed_noise.tex", ["ID","Configuration","Runs","Clean trajectory MSE","Delay RMSE"],
          [[cid,labels[cid],len(groups[cid]),sci(groups[cid].clean_test_mse.median()),
            sci(groups[cid].delay_rmse.median())] for cid in labels])
    pairs = runs[runs.config_id.isin(["P3","P4"])].pivot(index="seed",columns="config_id",values="delay_rmse")
    pairs["P3_P4_ratio"] = pairs.P3/pairs.P4
    pairs["P3_improves"] = pairs.P3 < pairs.P4
    pairs.to_csv(out/"fixed_noise_P3_vs_P4_pairs.csv")
    comparison = {"paired_runs":len(pairs), "paired_improvements":int(pairs.P3_improves.sum()),
                  "ratio_of_marginal_medians":float(pairs.P3.median()/pairs.P4.median()),
                  "median_paired_ratio":float(pairs.P3_P4_ratio.median()),
                  "scope":"initial fixed-noise study; distinct from fresh-noise confirmation"}
    (out/"fixed_noise_comparison.json").write_text(json.dumps(comparison,indent=2)+"\n")
    fig = plt.figure(figsize=(221.1337585/72,196.804306/72))
    ax = fig.add_axes([.17,.20,.81,.71])
    markers = {"P1":"o","P2":"s","P3":"*","P4":"D","P5":"^","P6":"v","D1":"P"}
    offsets = {"P1":(5,3),"P2":(5,5),"P3":(5,-8),"P4":(5,-9),
               "P5":(5,5),"P6":(-21,6),"D1":(4,5)}
    for cid in markers:
        g = groups[cid]
        x = g.clean_test_mse.median(); y = g.delay_rmse.median()
        xlo,xhi = g.clean_test_mse.quantile([.25,.75])
        ylo,yhi = g.delay_rmse.quantile([.25,.75])
        color = "#315b8a" if cid.startswith("P") and int(cid[1:]) <= 4 else "#c55a11" if cid.startswith("P") else "#777777"
        ax.errorbar(x,y,xerr=[[x-xlo],[xhi-x]],yerr=[[y-ylo],[yhi-y]],fmt=markers[cid],
                    color=color,ecolor=color,elinewidth=.7,capsize=2,markersize=7 if cid=="P3" else 5)
        ax.annotate(cid,(x,y),xytext=offsets[cid],textcoords="offset points",
                    fontsize=7,fontweight="bold" if cid=="P3" else "normal")
    ax.set(xscale="log",yscale="log",xlim=(6e-6,1.8e-4),ylim=(.0068,.27))
    ax.set_xlabel("Clean held-out trajectory MSE",fontsize=6.4)
    ax.set_ylabel("Delay RMSE",fontsize=6.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(alpha=.25)
    ax.set_title("(a) Fixed-noise mechanism study",fontsize=8,fontweight="bold",loc="left")
    panel = out/"figure_2_panel_a.pdf"
    fig.savefig(panel)
    plt.close(fig)
    return panel


def generate(confirmation, out, strengthening, extended):
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(confirmation/"step5_confirmation_runs.csv")
    cert = pd.read_csv(confirmation/"step5_confirmation_certificate_summary.csv").set_index("config_id")
    summary = runs.groupby("config_id").median(numeric_only=True)
    configs = ["P1", "P2", "P3", "P4"]
    rows = []
    for cid in configs:
        c = cert.loc[cid]
        rows.append([cid, sci(summary.loc[cid,"clean_test_mse"]), sci(summary.loc[cid,"delay_rmse"]),
                     f"{int(c.accepted_candidate_states)}/{int(c.eligible_candidate_states)}",
                     f"{int(c.false_confidence_events)}/{int(c.accepted_candidate_states)}"])
    table(out/"table_S3_confirmation.tex", ["ID", "Clean trajectory MSE", "Delay RMSE", "Accepted/eligible", "Large error/accepted"], rows)
    solver = pd.read_csv(extended/"solver_refinement.csv")
    table(out/"table_S4_solver.tex", solver.columns.tolist(), [[f"{v:.4f}" if j == 0 else sci(v) for j,v in enumerate(row)] for row in solver.itertuples(index=False,name=None)])
    extra = pd.read_csv(extended/"extrapolation.csv")
    table(out/"table_S5_domain.tex", extra.columns.tolist(), [[sci(v) if isinstance(v,(float,np.floating)) else str(v).replace("_",r"\_") for v in row] for row in extra.itertuples(index=False,name=None)])
    noise = pd.read_csv(extended/"noise_scaling.csv")
    table(out/"table_S6_noise.tex", [r"\(\sigma\)", "Clean trajectory MSE", "Delay RMSE", "Resolved states", "Mean interval width"],
          [[sci(r.noise_std), mean_sd(r.clean_test_mse,r.clean_test_mse_std), mean_sd(r.delay_rmse,r.delay_rmse_std),
            f"{int(round(3*r.certificate_identified_rate))}/3", f"{r.certificate_mean_width:.4f}"] for r in noise.itertuples()])

    plt.rcParams.update({"font.family":"DejaVu Serif", "font.size":9, "axes.titlesize":10, "pdf.fonttype":42})
    left_panel = fixed_noise_artifacts(out, strengthening)
    fig, axes = plt.subplots(1,2,figsize=(10,3.7))
    values = [runs.loc[runs.config_id.eq(cid),"delay_rmse"].to_numpy() for cid in configs]
    axes[0].boxplot(values, tick_labels=configs)
    for i,v in enumerate(values,1):
        axes[0].scatter(i+np.linspace(-.12,.12,len(v)),v,s=14,color="#315b8a",zorder=3)
    axes[0].set(yscale="log", ylabel="Delay RMSE", xlabel="Configuration")
    axes[0].set_title(f"{len(values[0])} runs per configuration")
    x = np.arange(4)
    axes[1].bar(x-.18,cert.loc[configs,"pooled_acceptance_rate"],.36,label="Candidate acceptance",color="#4b83c4")
    axes[1].bar(x+.18,cert.loc[configs,"conditional_false_confidence_rate"],.36,label="Accepted large error / accepted",color="#c46d33")
    for i,cid in enumerate(configs):
        c=cert.loc[cid]
        for xpos,rate,label in [(i-.18,c.pooled_acceptance_rate,f"{int(c.accepted_candidate_states)}/{int(c.eligible_candidate_states)}"),
                                (i+.18,c.conditional_false_confidence_rate,f"{int(c.false_confidence_events)}/{int(c.accepted_candidate_states)}")]:
            axes[1].text(xpos,rate+.025,label,ha="center",fontsize=7)
    axes[1].set(xticks=x,xticklabels=configs,ylim=(0,1),ylabel="Rate",xlabel="Configuration")
    axes[1].legend(fontsize=7,loc="upper center",bbox_to_anchor=(.5,1.18))
    fig.tight_layout()
    for ext in ["pdf","png"]: fig.savefig(out/f"figure_S1_confirmation.{ext}",dpi=220,bbox_inches="tight")
    plt.close(fig)

    pair = runs.pivot(index=["data_seed","optimization_seed"],columns="config_id",values="delay_rmse")
    ratios = pair.P3/pair.P4
    fig=plt.figure(figsize=(220/72,196.8043/72))
    ax=fig.add_axes([.23,.19,.75,.72])
    med=np.median(ratios)
    for index,data_seed in enumerate(sorted(runs.data_seed.unique()),1):
        y=ratios.xs(data_seed).to_numpy()
        ax.scatter(index+np.linspace(-.18,.18,len(y)),y,s=13,color="#1f4e79")
    ax.axhline(1,color="black",ls="--",lw=.8,label="Equal delay RMSE")
    ax.axhline(.5,color="#c55a11",ls=":",lw=1,label="Reference ratio 0.50")
    ax.axhline(med,color="#5b8dd6",lw=.8)
    ax.text(5.35,med,f"paired median={med:.3f}",ha="right",va="bottom",fontsize=6,color="#1f4e79")
    ax.set(yscale="log",ylim=(.06,1.2),xlim=(.6,5.4),xticks=range(1,6),xticklabels=[f"D{i}" for i in range(1,6)])
    ax.set_xlabel("Independent noisy dataset (three paired seeds)",fontsize=6.4)
    ax.set_ylabel("Anchor-only / rollout-only delay-RMSE ratio",fontsize=6.4)
    ax.tick_params(labelsize=6.5)
    ax.set_title("(b) Fresh-noise confirmation",fontsize=8,fontweight="bold",loc="left")
    ax.legend(loc="lower left",fontsize=5.5)
    panel=out/"figure_2_panel_b.pdf";fig.savefig(panel);plt.close(fig)
    # Both panels are rebuilt from recovered run records.
    original=fitz.open(left_panel)
    right=fitz.open(panel)
    combined=fitz.open();page=combined.new_page(width=441.1337585,height=196.8043060)
    page.show_pdf_page(fitz.Rect(0,0,221.1337585,196.804306),original,0,clip=fitz.Rect(0,0,221.1337585,196.804306))
    page.show_pdf_page(fitz.Rect(221.1337585,0,441.1337585,196.804306),right,0)
    combined.save(out/"empirical_confirmation.pdf",garbage=4,deflate=True)
    provenance={"confirmation_source":str(confirmation.relative_to(ROOT)) if confirmation.is_relative_to(ROOT) else str(confirmation),
                "fixed_noise_source":str(strengthening.relative_to(ROOT)),
                "extended_source":str(extended.relative_to(ROOT)),
                "figure_2_panel_a":"medians and interquartile ranges recomputed from fixed-noise CSV records",
                "figure_2_panel_b":"recomputed from paired confirmation CSVs",
                "noise_zero_row":"diagnostic values assigned in historical implementation",
                "tables":"S2-S6 generated from included CSV records; zero-noise note belongs in table caption"}
    (out/"artifact_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(f"Generated Figure 2, Figure S1 and Tables S2-S6 in {out}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-source",choices=("reference","generated"),default="reference")
    parser.add_argument("--strengthening-source",choices=("reference","generated"),default="reference")
    parser.add_argument("--extended-source",choices=("reference","generated"),default="reference")
    parser.add_argument("--output",type=Path,default=ROOT/"figures/manuscript/generated")
    args=parser.parse_args()
    generate(ROOT/"results/confirmation"/args.confirmation_source,args.output.resolve(),
             ROOT/"results/strengthening"/args.strengthening_source,
             ROOT/"results/extended"/args.extended_source)


if __name__ == "__main__":
    main()
