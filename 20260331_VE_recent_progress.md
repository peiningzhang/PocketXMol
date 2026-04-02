# VE Recent Progress

This note records the latest evaluation progress for the VE distillation line.

## Fixed Sampling Setup

The current best-performing sampler configuration remains:

- `coordinate_update = snr_renoise`
- `discrete_update = categorical_transition`

This is the default comparison baseline for the recent checkpoint sweeps.

## Formal Checkpoint Comparison

We ran the same evaluation protocol on:

- `step_50000.pt`
- `step_70000.pt`
- `step_80000.pt`
- `step_100000.pt`

Protocol:

- `sample_steps = 1 2 4 8`
- `num_mols = 1000`
- `standard_eval = true`
- standard SDF metrics were appended into the consistency summary

Key takeaway from the first sweep:

- `4-step` favored `step_50000.pt`
- `8-step` favored `step_100000.pt`
- `connectivity` was already high at `4/8-step`, so the remaining gap was mostly in recoverable molecule count, not just graph connectivity

## Extended Step Sweep

We then reran the two most relevant checkpoints with longer sampling:

- `step_50000.pt`
- `step_100000.pt`

Protocol:

- `sample_steps = 1 2 4 8 16 32`
- `num_mols = 1000`
- `standard_eval = true`

### Results: `step_50000.pt`

- `8-step`: `recon_success = 0.381`, `complete = 0.364`
- `16-step`: `recon_success = 0.483`, `complete = 0.473`
- `32-step`: `recon_success = 0.518`, `complete = 0.500`

Standard metrics at `32-step`:

- `standard_complete = 0.503`
- `standard_recon_success = 0.523`
- `standard_atom_type_js = 0.0987`

### Results: `step_100000.pt`

- `8-step`: `recon_success = 0.392`, `complete = 0.360`
- `16-step`: `recon_success = 0.479`, `complete = 0.456`
- `32-step`: `recon_success = 0.543`, `complete = 0.518`

Standard metrics at `32-step`:

- `standard_complete = 0.527`
- `standard_recon_success = 0.552`
- `standard_atom_type_js = 0.0913`

## Interpretation

The extended sweep changed the conclusion slightly:

- `step_50000.pt` is still slightly better for the cheaper `16-step` regime.
- `step_100000.pt` is better for the stronger `32-step` regime.
- Both checkpoints still improve when going from `8 -> 16 -> 32` steps, so the sampler has not fully saturated.
- The dominant bottleneck is still the fraction of samples that fail reconstruction entirely, not connectivity once reconstruction succeeds.

## Practical Conclusion

Current recommended operating points:

- `16-step` deployment: `step_50000.pt`
- `32-step` deployment: `step_100000.pt`

If future work continues, the most useful next direction is not more sampler variants, but reducing `bad` samples through better training balance or stronger discrete supervision.
