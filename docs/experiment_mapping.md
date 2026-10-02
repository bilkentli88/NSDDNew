# Experiment-to-manuscript mapping

| Manuscript material | Program or source | Cached evidence | Reproduction status |
|---|---|---|---|
| Fixed-noise P0-P6 and D1/D2 study, Table S2 | `experiments/strengthening/run_tnnls_strengthening.py --full` | Original run/profile CSVs and 39 checkpoints | Full records recovered; summaries, curves, profiles and decisions verified |
| Five-dataset P1-P4 confirmation, Table S3 | `experiments/strengthening/run_step5_confirmation.py --full` | `results/confirmation/reference/`, 60 checkpoints | All summaries, paired comparisons and decisions recalculated from records |
| 21-state empirical profile assessment | `certificate.py`, `numpy_data.py`; `scripts/validate_precomputed_results.py --geometry` | `step5_confirmation_geometry.csv` and fixed-noise informative/ramp profiles | Confirmation 1260 and fixed-noise 819 decisions reconstructed without retraining |
| Observation-noise sweep, Table S6 | `experiments/run_extended_experiments.py --mode paper` | `results/extended/reference/noise_scaling*.csv` | Archived means/SD verified; original WP9 README confirms 180 pretraining iterations and 35/70 rollout epochs |
| Frozen-model solver refinement, Table S4 | Same extended driver | `solver_refinement.csv`, supplied reference checkpoint | Four step sizes; one frozen model and one history, not a retraining comparison |
| Interpolation/extrapolation, Table S5 | Same extended driver | `extrapolation.csv`, reference checkpoint | Profile assessment remains restricted to the designed state domain |
| Full/partial oscillator stress test | Same extended driver | `oscillator_summary.csv`, oscillator checkpoints | One run per regime; outside the scalar theorem |
| Figure 1 | `figures/manuscript/reference/conceptual_framework.pdf` | Archived conceptual vector artwork | Included figure asset, not generated from numerical data |
| Figure 2 | `scripts/regenerate_manuscript_artifacts.py` | Fixed-noise run records and fresh-noise paired records | Both panels rebuilt; medians and interquartile ranges recomputed |
| Figure S1 | Same artifact script | Confirmation run and candidate-summary CSVs | Rebuilt distributions and conditional accepted-large-error frequencies |

The standalone legacy `run_principal_scalar.py` is retained because the extended
driver imports its solver, model and observation utilities. Its three-seed
training experiment is not the five-dataset confirmation or the fixed-noise
strengthening study.

Full experiment runs write to `results/*/generated`, `figures/*/generated` and
`checkpoints/*/generated`. Archived `reference` files are never overwritten by
those commands. Historical filenames beginning `tnnls` or `step3/step5` are
retained to make the recovered results traceable.
