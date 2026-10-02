# Methods and interpretation

## Theory and experimental diagnostics

The positive theorem concerns the scalar affine delayed-state model and a fixed
matched-slope inverse problem. It is not a characterization of the entire
trajectory observation operator. Deterministic certified-set guarantees require
a valid uniform profile-error envelope. An empirical resampling statistic does
not establish that envelope or calibrated frequentist coverage.

The numerical code profiles the normalized squared residual `Q`, whereas the
theorem uses the residual norm `P = sqrt(M * Q)`. The stored fields `eta` and
`eta_95` are historical names for the 0.95 quantile of resampled squared-profile
deviations over the finite grid. They are not the theorem's raw-norm envelope.
Finite-grid set diameters and 2.5/97.5 percentile minimizer intervals are
empirical sensitivity summaries. Off-grid candidate values are checked by
linear interpolation of `Q`.

The 21-state geometry uses five probe trajectories per state, independent probe
noise with seed 20260803, 120 resamples and resampling seed 20260804. The same
geometry is used for all candidates. Candidate decisions are evaluated after
training. A profile can be constructed in advance because it does not use the
trained neural parameters.

## Training anchors

Four anchor states, -0.75, -0.25, 0.25 and 0.75, use quadratic fits to five
samples including t=0. All four anchors are retained with equal weights; the
implementation uses their mean squared delay discrepancy. Certificate-gated
training-anchor selection is an optional generalization and was not tested.
Probe response estimates instead use cubic smoothing splines over ten samples
including t=0. Resampling includes the initial fitting sample; it does not keep
that sample constrained to its known exact value.

## Confirmation

Five independent training-noise realizations have data seeds 20260901-20260905.
Each has three optimization seed labels 20261001-20261003. The actual model
seed is `optimization_seed + 100 * data_replication_index`. Optimizer repeats
within a dataset are nested; 15 pairs are not 15 independent datasets. The
success criterion uses the ratio of marginal median delay RMSEs (0.283), not
the median paired ratio (0.332). The plot's 0.50 line is a reference ratio.

`false_confidence_event` is a historical column name for an accepted candidate
whose absolute error exceeds 0.035. Its frequency is descriptive, not a
confidence or coverage guarantee. Figure S1 displays errors conditional on
acceptance; this denominator differs from an unconditional rate over all states.

## Noise sweep and numerical resolution

Two training runs per noise level also have different noise seeds. At positive
noise levels the first run alone supplies the three-state, 12-resample profile
diagnostic. Its counts are resolved probe states, not accepted trained candidates.
At zero noise the historical implementation assigns 3/3 resolved states and zero
interval width without resampling. The sweep is descriptive and does not isolate
noise magnitude using a common random realization.

Solver refinement uses one frozen model/history. Learned delays do not change
with the rollout step in that check. Numerical effects on learned delay recovery
or the anchor advantage require retraining at multiple resolutions.

## Original fixed-noise study

The original five-seed study is now included. P3 has lower delay RMSE than
P4 in four of five paired runs; its ratio of marginal medians is 0.431672.
The median paired ratio is 0.515114 and should not be substituted for the
marginal-median ratio. Table S2 medians and the 75/105 P3 candidate acceptance
count reproduce from the recovered records. This initial single-noise-dataset
study is distinct from the later five-dataset confirmation.
