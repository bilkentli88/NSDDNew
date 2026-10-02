WP9 - Manuscript-Scale Stress Tests

Run the complete CPU experiment:
    python wp9_manuscript_experiments.py --quick

The --quick flag is the validated configuration used for the included results. It trains two scalar replicates at each of five noise levels and one model per vector-observation regime. Typical runtime is about 25 seconds on one CPU thread.

Main outputs:
- wp9_manuscript_experiment_note.pdf: research note
- wp9_noise_scaling.csv: aggregated scalar noise results
- wp9_noise_scaling_runs.csv: per-initialization scalar results
- wp9_solver_refinement.csv: solver study
- wp9_extrapolation.csv: visited versus unvisited state study
- wp9_oscillator_summary.csv: vector oscillator study
- wp9_key_results.json: machine-readable results
- checkpoints/: trained model state dictionaries

The script is self-contained within this folder. wp8_dependency.py and reference_checkpoints/ contain the frozen scalar implementation and checkpoint used by the refinement and extrapolation studies.
