# Matched-State Identifiability and Certification of Learned State-Dependent Delays in Neural Delay Differential Equations

This repository contains the scalar Neural SDDDE experiments, recovered
fresh-noise confirmation records, checkpoints, and reproducibility tools for
the accompanying manuscript and supplement.

The work reports trajectory prediction, learned delay accuracy, and empirical
profile acceptance separately. The positive recovery theorem concerns the
scalar affine delayed-state model and a fixed matched-slope inverse problem.
Residual resampling supplies sensitivity diagnostics; it does not establish
the uniform error envelope needed for deterministic certification or calibrated
statistical confidence.

## What this revision includes

- Fixed-noise and five-dataset confirmation runners with their recovered records and checkpoints.
- The full confirmation records: 60 trained runs, 1,260 candidate-state records,
  paired comparisons, and all 60 model checkpoints.
- Restored extended-experiment records, checkpoints, and figures.
- Checks that recompute summaries and rebuild the 21-state empirical profiles
  and candidate decisions without retraining.
- Regenerated confirmation figures and LaTeX table fragments.
- Dependency files, execution commands, experiment mapping, and provenance notes.

The original fixed-noise study has now been recovered: 39 runs, all 39
checkpoints, complete profiles and candidate decisions. Table S2 and both
Figure 2 panels can be rebuilt from the included records. Verification found
one manuscript correction: P3 improves over P4 in 4/5 initial fixed-noise
pairs; the fresh-noise result remains 15/15. See
[manuscript correction](docs/manuscript_corrections.md).
The original WP9 execution note confirms the archived noise-sweep schedule:
180 delay-pretraining iterations and 35/70 rollout epochs. See
[provenance and limits](docs/provenance_and_limits.md).

## Install

Use Python 3.12 or newer in a virtual environment. Cached-result verification
and figure generation need only the analysis dependencies:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements-analysis.txt
```

For neural training, additionally install PyTorch:

```bash
python -m pip install -r requirements.txt
```

The recovered confirmation manifest records NumPy 2.4.2 and PyTorch
2.10.0+cpu. `requirements-confirmation.txt` pins their public package versions;
other historical dependency versions were not recorded. It is not a complete
lockfile or a guarantee of identical results on every machine.

Alternatively, use `conda env create -f environment.yml`, then
`conda activate neural-sddde-identifiability`. Training runs on one CPU thread.

## Verify the recovered results

From the repository root:

```bash
python scripts/reproduce_all.py --mode verify
python -m unittest discover -s tests -v
```

The first command validates cached statistics, reconstructs the empirical
geometry and decisions, and regenerates manuscript artifacts. Verification
does not load neural checkpoints or rerun optimization. Neural-model tests
are explicitly skipped when PyTorch is unavailable.

Individual commands are also available:

```bash
python scripts/validate_precomputed_results.py --geometry
python scripts/regenerate_manuscript_artifacts.py
```

The preparation checks and their limits are recorded in
[verification report](docs/verification_report.md).

## Run experiments

A reduced workflow check:

```bash
python scripts/reproduce_all.py --mode quick
```

Full strengthening and confirmation schedules, plus the archived article's
noise-sweep schedule:

```bash
python scripts/reproduce_all.py --mode paper
```

Inspect commands without executing training, or select one stage:

```bash
python scripts/reproduce_all.py --mode paper --dry-run
python scripts/reproduce_all.py --mode paper --stages confirmation
```

Equivalent individual commands:

```bash
python experiments/strengthening/run_tnnls_strengthening.py --full
python experiments/strengthening/run_step5_confirmation.py --full
python experiments/run_extended_experiments.py --mode paper
```

The extended driver's `--mode paper` matches the configuration identified by
the original WP9 README: 180 pretraining iterations and 35/70 rollout epochs.
The same archived configuration uses the short oscillator study. The legacy
`--mode manuscript` option is retained for compatibility and uses a longer
noise sweep (260 pretraining iterations and 45/90 rollout epochs); it is not
the configuration used for the archived article results. `--mode extended`
also lengthens the oscillator study. The original execution note is included
in [docs/archive/WP9_README.txt](docs/archive/WP9_README.txt).

New runs seed models before construction and record their own execution
settings, so they must be reported as new results rather than assumed to
duplicate the archived metrics exactly.

The legacy `experiments/run_principal_scalar.py` remains available because
the extended driver imports its model and solver utilities. Its standalone
three-seed experiment is distinct from the current confirmation.

## Files and outputs

| Location | Contents |
|---|---|
| `experiments/strengthening/` | Strengthening and confirmation runners, scalar core, NumPy data utilities, empirical profile procedure |
| `results/strengthening/reference/` | Original fixed-noise records and complete profiles |
| `checkpoints/strengthening/reference/` | 39 original fixed-noise model states |
| `results/confirmation/reference/` | Recovered confirmation CSVs and JSON records |
| `checkpoints/confirmation/reference/` | 60 recovered model states |
| `results/extended/reference/` | Noise sweep, solver refinement, state-domain and oscillator records |
| `checkpoints/reference/` | Frozen scalar model used by solver and domain evaluations |
| `figures/manuscript/reference/` | Conceptual artwork and archived Figure 2 |
| `figures/manuscript/generated/` | Rebuilt Figure 2, Figure S1, and Tables S2-S6 |
| `docs/` | Mapping, methods, provenance, and verification |

Training writes new files to `generated/` directories and preserves
`reference/` records. Full runs also save their profiles and execution
metadata. Generated directories are ignored by Git so reruns do not
accidentally replace the archived evidence.

The artifact generator rebuilds Figure 2(a) medians and interquartile ranges
from the original fixed-noise runs and Figure 2(b) from all 15 fresh-noise
paired ratios. The 0.50 plot line is labelled as a
reference ratio. The confirmation criterion applies to the ratio of marginal
medians (0.283); the median paired ratio is a different statistic (0.332).
Figure S1 uses accepted large-error counts divided by accepted candidates.
For plots from a new confirmation run:

```bash
python scripts/regenerate_manuscript_artifacts.py --confirmation-source generated
# To use newly generated records for all three experiment families:
python scripts/regenerate_manuscript_artifacts.py --confirmation-source generated --strengthening-source generated --extended-source generated
```

The table fragments require LaTeX `booktabs`. Their captions should retain
the experimental qualifications, including the assigned zero-noise diagnostic.

## Interpretation and replication

The confirmation uses five independently generated noisy datasets, with three
optimization repeats within each. Its 15 pairs are nested repeats, not 15
independent datasets. Four equally weighted delay anchors re-express the same
training observations used for rollout fitting. Certificate-gated training
anchors are an optional extension, not the implemented experiment.

The noise sweep uses two runs per level and a separate, first-run-only
12-resample diagnostic. These are resolved probe-state counts, not trained
candidate acceptance counts. The zero-noise diagnostic was assigned directly
in the historical code.

See [experiment mapping](docs/experiment_mapping.md) and
[methods and scope](docs/methods_and_scope.md) for the full distinctions.
The oscillator is a qualitative stress test outside the scalar theorem.
Solver refinement evaluates a frozen model; it does not establish
solver-independent training or delay recovery.

## Citation and license

Bibliographic metadata are in [CITATION.cff](CITATION.cff). Cite the final
published manuscript when available. Code and supplied checkpoints are
distributed under the [MIT License](LICENSE).
