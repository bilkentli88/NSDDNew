# Checkpoints

- `reference/restricted_noisy_seed0.pt` is the frozen scalar checkpoint used by the solver-refinement and state-domain experiments.
- `extended/reference/` contains the 10 noise-sweep and three oscillator state dictionaries recovered from the extended package.
- `confirmation/reference/` contains all 60 state dictionaries from the five-dataset, three-optimization-seed P1-P4 confirmation.
- `strengthening/reference/` contains the 39 recovered original fixed-noise model states.
- `strengthening/generated/` and `confirmation/generated/` are populated by new strengthening/confirmation runs.
- `principal_scalar/generated/` is populated when the principal experiment is rerun.

Checkpoint bytes are preserved from the recovered packages. They were not loaded
or reevaluated during this repository revision because PyTorch was unavailable.
