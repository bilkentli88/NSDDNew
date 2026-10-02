# Matched-State Identifiability and Certification of Learned State-Dependent Delays in Neural Delay Differential Equations

This repository provides experiment code, reference results, model checkpoints,
and reproducibility tools for the accompanying article and supplementary document.
The experiments cover scalar neural state-dependent delay differential equations
(SDDDEs) and a two-dimensional delayed-oscillator stress test.

The study reports trajectory prediction, learned delay accuracy, and empirical
profile acceptance separately. The positive recovery theorem concerns the scalar
affine delayed-state model and a fixed matched-slope inverse problem. Residual
resampling supplies sensitivity diagnostics; it does not establish the uniform
profile-error envelope required for deterministic certification or calibrated
statistical confidence.

## Repository contents

- Fixed-noise ablation: 39 runs, model checkpoints, complete empirical profiles,
  and candidate-state decisions.
- Fresh-noise confirmation: five independently generated noisy datasets, three
  paired optimization seeds per dataset, 60 trained models across P1–P4, and
  1,260 candidate-state evaluations.
- Observation-noise sweep, Euler solver refinement, state-domain evaluation,
  and delayed-oscillator experiments.
- Scripts for verifying reference statistics, reconstructing empirical profiles,
  and generating Figure 2, Figure S1, and Tables S2–S6.
- Dependency specifications, experiment mapping, methods, and verification notes.

## Installation

Use Python 3.12 or newer. Create a virtual environment from the repository root:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

For reference-result verification and figure generation:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-analysis.txt
```

For neural training, also install the training dependencies:

```bash
python -m pip install -r requirements.txt
```

The confirmation execution record specifies NumPy 2.4.2 and PyTorch 2.10.0+cpu.
`requirements-confirmation.txt` pins the corresponding public package versions.
Other dependency versions were not recorded, so this file is not a complete
environment lockfile and does not guarantee identical results across machines.

Alternatively:

```bash
conda env create -f environment.yml
conda activate neural-sddde-identifiability
```

The training scripts use one CPU thread.

## Verify reference results

Run from the repository root:

```bash
python scripts/reproduce_all.py --mode verify
python -m unittest discover -s tests -v
```

The first command checks reference statistics, reconstructs empirical profile
geometry and candidate decisions, and generates the manuscript figures and table
fragments. It does not load neural checkpoints or rerun optimization.
Neural-model tests are skipped when PyTorch is unavailable.

The verification and artifact-generation steps can also be run separately:

```bash
python scripts/validate_precomputed_results.py --geometry
python scripts/regenerate_manuscript_artifacts.py
```

See [verification report](docs/verification_report.md) for the completed checks
and their scope.

## Run experiments

To run the fixed-noise ablation, fresh-noise confirmation, and extended
experiments with the article's training schedules:

```bash
python scripts/reproduce_all.py --mode paper
```

To inspect the commands without starting training:

```bash
python scripts/reproduce_all.py --mode paper --dry-run
```

To run only the fresh-noise confirmation:

```bash
python scripts/reproduce_all.py --mode paper --stages confirmation
```

For a reduced training-budget check of the ablation and confirmation workflows:

```bash
python scripts/reproduce_all.py --mode quick --stages strengthening confirmation
```

The individual article experiment commands are:

```bash
python experiments/strengthening/run_tnnls_strengthening.py --full
python experiments/strengthening/run_step5_confirmation.py --full
python experiments/run_extended_experiments.py --mode paper
```

For the extended driver, `--mode paper` uses 180 delay-pretraining iterations
followed by 35 and 70 rollout-training epochs at the 10- and 20-step horizons,
respectively. It also selects the article's oscillator training schedule.
The alternative `--mode manuscript` uses a longer noise-sweep schedule
(260 pretraining iterations and 45/90 rollout epochs), while `--mode extended`
also lengthens oscillator training.

Reruns record their seeds, schedules, and execution environments. Their metrics
may differ from the supplied reference results. In particular, the extended
driver uses explicit seeds before model construction; the reference experiments
did not record those initialization seeds. See
[provenance and reproducibility limits](docs/provenance_and_limits.md).

`experiments/run_principal_scalar.py` supplies model and solver utilities used
by the extended driver. Its standalone three-seed experiment is separate from
the five-dataset confirmation.

## Files and outputs

| Location | Contents |
|---|---|
| `experiments/strengthening/` | Ablation and confirmation runners, scalar core, NumPy data utilities, and empirical profile procedure |
| `results/strengthening/reference/` | Fixed-noise run records and complete profiles |
| `checkpoints/strengthening/reference/` | 39 fixed-noise model checkpoints |
| `results/confirmation/reference/` | Confirmation CSV and JSON records |
| `checkpoints/confirmation/reference/` | 60 confirmation model checkpoints |
| `results/extended/reference/` | Noise-sweep, solver-refinement, state-domain, and oscillator results |
| `checkpoints/reference/` | Frozen scalar model used for solver and state-domain evaluations |
| `figures/manuscript/reference/` | Conceptual figure and reference Figure 2 |
| `figures/manuscript/generated/` | Figure 2, Figure S1, and Tables S2–S6 generated from numerical records |
| `docs/` | Experiment mapping, methods, provenance, and verification notes |

Training writes its outputs to `generated/` directories and preserves the
supplied `reference/` records. Full ablation and confirmation runs also save
empirical profiles and execution metadata.

The artifact generator computes Figure 2(a) medians and interquartile ranges
from the fixed-noise runs and Figure 2(b) from the 15 fresh-noise paired ratios.
The plot's 0.50 line is a reference ratio. The confirmation criterion uses the
ratio of marginal median delay RMSEs (0.283); the median paired ratio is a
separate statistic (0.332). Figure S1 reports accepted large-error events
divided by accepted candidates.

To generate artifacts from a new confirmation run:

```bash
python scripts/regenerate_manuscript_artifacts.py --confirmation-source generated
```

To use generated records for all three experiment families:

```bash
python scripts/regenerate_manuscript_artifacts.py --confirmation-source generated --strengthening-source generated --extended-source generated
```

The LaTeX table fragments require `booktabs`. When incorporating them into a
document, retain the experiment-specific qualifications described below.

## Interpretation and replication

The confirmation has five independent noisy datasets and three optimization
repeats within each. Its 15 paired runs are not 15 independent datasets.
Four equally weighted delay anchors use the same training observations as the
rollout loss. The implemented experiments retain all four anchors; empirical
acceptance checks are applied during post-training evaluation.

The noise sweep uses two training runs per noise level. At positive noise
levels, the first run alone supplies the three-state diagnostic with 12 residual
resamples per state. These counts describe resolved probe states rather than
accepted trained candidates. At zero noise, the implementation assigns three
resolved states and zero interval width without resampling.

Solver refinement evaluates one frozen model and history. It measures rollout
sensitivity to the Euler step and does not establish solver-independent training
or delay recovery. In the state-domain experiment, the wider delay-evaluation
grid includes interior and exterior states; its aggregate delay RMSE does not
isolate outside-domain error. Exterior delay predictions remain outside the
assessed mechanistic support.

The oscillator experiment is a qualitative stress test outside the scalar
identifiability theorem. See [experiment mapping](docs/experiment_mapping.md)
and [methods and scope](docs/methods_and_scope.md) for further details.

## Citation and license

Bibliographic metadata are provided in [CITATION.cff](CITATION.cff). Cite the
published article when available. Code and supplied checkpoints are distributed
under the [MIT License](LICENSE).
