# Verification report

## Checks completed during repository preparation

- Python syntax compilation passed for all experiment, utility and test files.
- `python scripts/reproduce_all.py --mode verify` passed 280 cached-record
  and reconstruction checks, then regenerated the manuscript artifacts.
- Four numerical tests passed. The legacy neural-model test module was skipped
  because PyTorch was unavailable.
- The full-training command routing was checked with `--mode paper --dry-run`.
  The extended stage selects `--mode paper`, matching the archived 180/35/70
  noise-sweep schedule confirmed by the original WP9 README.
- All 12 NumPy-only function definitions match the recovered scalar core by
  abstract-syntax-tree comparison.
- All 74 copied confirmation files (CSV/JSON, checkpoints and diagnostic PNGs)
  are byte-identical to the supplied confirmation output ZIP.
- All 20 restored extended numerical records and checkpoint files are
  byte-identical to the recovered complete package.
- All 62 recovered fixed-noise record/checkpoint/plot files are byte-identical
  to their files in the new ZIP.
- Both Figure 2 panels and Figure S1 were rendered and visually inspected.

## Fixed-noise evidence recovered

All nine configurations in Table S2 reproduce their published rounded medians.
The 39 saved delay curves reproduce their delay metrics and between-run
curve dispersion. Both complete 21-by-351 profile grids and all 819
fixed-noise candidate-state decisions reconstruct from the empirical procedure.
P3 acceptance is 75/105 with no observed accepted-large-error event.

The original fixed-noise P3/P4 paired count is **4/5**, not the 5/5 claim in
Sections 10.1 and S.VIII-A of the supplied article. This correction is recorded
in manuscript_corrections.md. The original P1/P4 effect criterion failed;
this fact is retained in the archived record. The later confirmation criterion
and 15/15 confirmation improvements remain unchanged.

## Numerical evidence reproduced

The checks cover 60 P1-P4 confirmation runs, five training-noise datasets,
three optimization repeats within each, all reported configuration summaries,
three paired-comparison tables, and all 1,260 candidate-state records.

The independently reconstructed 21-state probe geometry reproduces the stored
interval endpoints, widths, empirical squared-profile values and sublevel-set
diameters. Reapplying the candidate procedure reproduces every acceptance,
interval-membership, sublevel-membership and accepted-large-error flag.

The primary P3/P4 ratio of marginal median delay RMSEs is approximately 0.283,
the median paired ratio is approximately 0.332, and P3 improves all 15 pairs.
P3 has 244/315 accepted candidate-state evaluations and no observed accepted
large-error event. P4 has 165/315 accepted evaluations and 10 such events.

The archived noise sweep, solver refinement, state-domain evaluation and
oscillator summaries were checked against their included records. The
noise-sweep means and sample standard deviations were recomputed from its
two runs at each level; the diagnostic comes from the designated first run.

## Preparation environment

| Component | Version |
|---|---|
| Python | 3.12.14 |
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| SciPy | 1.17.0 |
| Matplotlib | 3.10.8 |
| PyMuPDF | 1.26.6 |
| PyTorch | Not installed |

This environment differs from the recovered training manifest. Agreement of
the rebuilt profiles and decisions checks those computations; it does not
establish that neural training will reproduce the same metrics here.

## Limits

No neural training, checkpoint inference or training-specific test was
executed. The updated training runners were syntax-checked and their commands
inspected. New initialization seeding in the extended runner is documented and
can change newly trained results.

The missing fixed-noise record gap is closed by the newly recovered archive.
The original WP9 README identifies its validated `--quick` configuration as
the source of the included results. The two archived noise-sweep CSVs match
that package byte for byte. Article reproduction now selects the matching
180-pretraining, 35/70-rollout-epoch schedule through the extended driver's
`--mode paper` option.

Generated LaTeX files are table fragments requiring `booktabs`; the complete
manuscript was not compiled as part of this repository task. Successful
empirical reconstruction does not validate a deterministic uniform error
envelope or statistical coverage.
