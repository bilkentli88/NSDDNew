# Recovered sources and verification limits

This revision consolidates the supplied GitHub ZIP with the recovered
`TNNLS_Strengthening_Experiments.zip`, `run_step5_confirmation.py`,
`tnnls_step5_confirmation_outputs(2).zip`, the earlier complete repository,
WP8/WP9 reference assets, the corrected manuscript Figure 2, and the newly
recovered NewIEEEPro.zip fixed-noise study.

Archived numerical records and checkpoint bytes are preserved. Numerical-only
functions were moved without changing their source bodies into `numpy_data.py`
so that profile reconstruction can run without PyTorch. Original source digests
are recorded in `recovered_source_hashes.json`.

The full original fixed-noise study was recovered from
NewIEEEPro/tnnls_step3_outputs/: 33 principal runs, six ramp runs, 39
checkpoints, 819 candidate-state evaluations, and complete informative and
ramp profiles. All four recovered program hashes agree with the earlier
algorithm sources. Table S2 medians and Figure 2(a) error bars are now
recomputed from those records. The older one-seed Step 2 and unrelated
prototype experiments in the new ZIP are not used for the manuscript study.

The records reveal one manuscript count error: P3 improves over P4 in four
of the five initial fixed-noise pairs, rather than five. Its marginal median
ratio remains 0.432. Fresh-noise confirmation still improves all 15 pairs.
Exact manuscript replacements are given in manuscript_corrections.md.

The legacy extended outputs lack a full execution-mode/environment manifest.
Their individual noise-sweep records reproduce the means and sample SD in the
revised Table S6. Exact training schedules cannot be inferred from those records.
The original extended `paper` option runs shorter schedules (180 pretraining
iterations; 35/70 rollout epochs). This revision retains that historical option
and adds `manuscript` for the declared noise-sweep schedule (260; 45/90), with
the short oscillator configuration. `extended` uses longer schedules throughout.

The revised extended driver seeds scalar and oscillator models before construction. The
original initialized these models before setting their training seeds. This fix makes
new initializations explicit and can change rerun metrics. Archived results
remain unchanged. New generated files are not bitwise replicas of the archives
and must be identified by their own execution manifests.

The recovered full-confirmation manifest records NumPy 2.4.2, PyTorch 2.10.0+cpu,
one CPU thread, five datasets and three optimization seed labels. It does not
record exact SciPy, pandas or Matplotlib versions; these are not invented here.
New training manifests record actual used seeds, schedules, versions and scope.

Repository preparation verified cached statistics, rebuilt geometry and all
candidate decisions, checked numerical invariants, and regenerated figures.
Neural training was not executed because PyTorch was absent from the preparation
environment. Training-specific tests are explicitly skipped in that environment.
