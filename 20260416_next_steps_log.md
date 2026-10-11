# 2026-04-16 Next Steps Log

## Current Result

Latest checkpoint:

- `outputs_distill_ve_2/consistency_distill_20260404_005352/checkpoints/step_100000.pt`

Latest Euler evaluation:

- `outputs_consistency_online_eval_euler/consistency_eval_20260405_204628`

Main result:

- `recon_success = 1.0` across `1/2/4/8/16/32/100-step`
- `complete` stays at or near `0`
- `standard_no_clashes` stays very low, roughly `0.005` to `0.010`
- `standard_atom_type_js` stays in the `0.34` to `0.44` range

Interpretation:

- Switching from `snr_renoise` to `euler` improves throughput.
- The dominant problem is still structural fidelity rather than sampling speed.
- The current model can reconstruct an output object, but most outputs are still incomplete or structurally poor.

## Comparison Summary

Compared with the earlier `snr_renoise` sweep:

- Euler is faster, especially at low step counts.
- Euler does not fix the completeness problem.
- Neither sweep reaches parity with the baseline.

## Next Step Plan

Priority order:

1. Debug why `complete` collapses to near zero even when `recon_success` is `1.0`.
2. Inspect generated SDFs and `gen_info.csv` for a small fixed subset of samples across `1-step`, `8-step`, and `100-step`.
3. Determine whether the dominant failure is:
   - disconnected graphs
   - malformed atom typing / valence
   - coordinate quality causing reconstruction artifacts
4. Add a lightweight diagnostic script that summarizes:
   - fraction of disconnected molecules
   - atom-count distribution
   - ring-count distribution
   - parse / reconstruction failure modes
5. Only after the failure mode is localized, change training or sampling again.

## What Not To Do Next

- Do not spend more time sweeping sampling step counts alone.
- Do not treat Euler as a parity fix.
- Do not optimize docking-style metrics before fixing completeness.

## Immediate Actionable Task

Implement a diagnostic pass on generated outputs that explains why `recon_success = 1.0` coexists with `complete ~= 0`.

## 2026-04-16 Diagnostic Update

Added a lightweight diagnostic script:

- `scripts/diagnose_consistency_outputs.py`

Ran it on:

- `outputs_consistency_online_eval_euler/consistency_eval_20260405_204628` with `1/8/100-step`
- `outputs_consistency_online_eval/consistency_eval_20260404_213705` with `1/8-step`

Generated summaries:

- `outputs_consistency_online_eval_euler/consistency_eval_20260405_204628/diagnostic_summary.csv`
- `outputs_consistency_online_eval_euler/consistency_eval_20260405_204628/diagnostic_summary.json`
- `outputs_consistency_online_eval/consistency_eval_20260404_213705/diagnostic_summary.csv`
- `outputs_consistency_online_eval/consistency_eval_20260404_213705/diagnostic_summary.json`

### Main Finding

`complete ~= 0` is not caused by widespread reconstruction exceptions.

- `bad = 0` for the checked Euler steps.
- Almost every sample is reconstructed successfully but is tagged as `incomp`.
- In `scripts/eval_consistency.py`, `incomp` means the reconstructed `smiles` contains `.`.
- So the current failure mode is overwhelmingly: reconstructed molecules are disconnected multi-fragment outputs.

### Euler Diagnostic Summary

`1-step`

- `complete = 0.005` (`5 / 1000`)
- `incomp = 995 / 1000`
- mean atom count `22.64`
- mean largest-fragment atom count only `3.32`
- most common fragment species: `C`, `CC`, `CCC`
- modal disconnected fragment count: `15`

`8-step`

- `complete = 0.0` (`0 / 1000`)
- `incomp = 1000 / 1000`
- mean atom count `22.22`
- mean largest-fragment atom count only `2.57`
- most common fragment species: `C`, `P`, `CC`, `S`
- modal disconnected fragment count: `15`

`100-step`

- `complete = 0.0` (`0 / 1000`)
- `incomp = 1000 / 1000`
- mean atom count `22.26`
- mean largest-fragment atom count only `2.42`
- most common fragment species: `C`, `P`, `CC`, `S`
- modal disconnected fragment count: `20`

### Comparison Against Earlier `snr_renoise`

The same disconnected-fragment failure already existed before Euler.

- `snr_renoise 1-step`: `995 / 1000` are `incomp`, mean largest fragment `3.32`
- `snr_renoise 8-step`: `1000 / 1000` are `incomp`, mean largest fragment `1.07`

This narrows the issue:

- Euler did not create the completeness failure.
- The failure is upstream of the update rule choice.
- Increasing step count does not merge fragments; it often makes the dominant fragment even smaller.

### Confidence Signal

The disconnected outputs are not low-confidence edge cases.

- Euler `1-step`: complete samples have mean `cfd_edge = 1.60`, while incomplete samples have mean `cfd_edge = 7.33`
- Euler `8-step`: incomplete samples have mean `cfd_edge = 17.72`
- Euler `100-step`: incomplete samples have mean `cfd_edge = 19.15`

So the model is confidently producing disconnected graphs. This points more toward a systematic mismatch in edge / bond generation than toward sampling noise or occasional reconstruction instability.

### Bond Collapse Signal

The stronger signal is that bond generation itself is collapsing as step count increases.

Euler:

- `1-step`: mean atom count `22.64`, mean bond count `7.85`, mean atom degree `0.74`, isolated-atom fraction `0.38`
- `8-step`: mean atom count `22.22`, mean bond count `4.71`, mean atom degree `0.44`, isolated-atom fraction `0.59`
- `100-step`: mean atom count `22.26`, mean bond count `3.89`, mean atom degree `0.36`, isolated-atom fraction `0.67`

Earlier `snr_renoise`:

- `1-step`: same coarse pattern as Euler `1-step`
- `8-step`: mean bond count only `0.07`, mean atom degree `0.008`, isolated-atom fraction `0.992`

Interpretation:

- Atom count does not collapse; molecules still contain roughly `22` atoms.
- What collapses is connectivity.
- Higher step counts are currently eroding bonds rather than refining them.
- This makes the edge-generation path the primary suspect, ahead of coordinate noise or occasional valence failure.

## Revised Next Step Plan

Priority order is now:

1. Inspect the predicted edge-type distribution before reconstruction for the same `1/8/100-step` outputs.
2. Check whether the model is assigning mostly `no-bond` edges or only very local singleton bonds.
3. Compare generated bond-count / degree statistics against the baseline evaluator outputs.
4. Only after confirming the edge failure mode, decide whether to change:
   - training loss weighting
   - discrete edge transition setup
   - sampler update rule

## 2026-04-16 Training-Side Findings

Checked:

- [`outputs_distill_ve_2/consistency_distill_20260404_005352/log.txt`](/shared/healthinfolab/phz24002/PocketXMol/outputs_distill_ve_2/consistency_distill_20260404_005352/log.txt)
- [`outputs_distill_ve_2/consistency_distill_20260404_005352/distill_pxm.yaml`](/shared/healthinfolab/phz24002/PocketXMol/outputs_distill_ve_2/consistency_distill_20260404_005352/distill_pxm.yaml)
- [`models/consistency_loss.py`](/shared/healthinfolab/phz24002/PocketXMol/models/consistency_loss.py)
- [`scripts/train_consistency.py`](/shared/healthinfolab/phz24002/PocketXMol/scripts/train_consistency.py)
- [`data/trained_models/pxm/train_config/train.yml`](/shared/healthinfolab/phz24002/PocketXMol/data/trained_models/pxm/train_config/train.yml)

### Loss Weighting

Current distillation loss weights are:

- `pos = 1.0`
- `node = 1.0`
- `halfedge = 1.0`

Teacher training used a larger edge weight:

- base training config has `edge = 1.5`

So the distilled run is already relatively softer on edges than the original teacher training.

### Loss Magnitude Drift

Parsed `2000` logged training points from the distilled run.

Early training (`step <= 10k`):

- mean `pos loss = 0.01085`
- mean `node loss = 0.00983`
- mean `edge loss = 0.00288`
- `edge / pos = 0.266`
- `edge / total = 0.122`

Mid training (`10k < step <= 50k`):

- mean `pos loss = 0.01274`
- mean `node loss = 0.00811`
- mean `edge loss = 0.00238`
- `edge / pos = 0.187`
- `edge / total = 0.103`

Late training (`step > 50k`):

- mean `pos loss = 0.01099`
- mean `node loss = 0.00174`
- mean `edge loss = 0.00071`
- `edge / pos = 0.064`
- `edge / total = 0.053`

At the end of training, representative points look like:

- `step=99900`: `pos=0.009124`, `node=0.000988`, `edge=0.000731`
- `step=100000`: `pos=0.006743`, `node=0.001143`, `edge=0.000415`

Interpretation:

- `edge loss` is smaller than `pos` from the start.
- The imbalance becomes more severe late in training.
- By the end, edge contributes only around `5%` of total loss on average.

### Distillation Objective Risk

The current consistency loss is:

- MSE on positions
- KL on node logits
- KL on edge logits
- all with plain scalar weights

There is no explicit class reweighting for the heavy `no-bond` imbalance.

Also, the consistency target is EMA-student prediction, not direct ground-truth bond labels. So if edge predictions drift toward trivial `no-bond` behavior, the EMA target can reinforce that collapse instead of correcting it.

### Noise / Prior Interaction

The distilled run inherits the teacher task noise config for `sbdd`.

For edges, the teacher config uses:

- `prior_type: tomask_half`

And `scripts/train_consistency.py` passes that edge prior into `build_transitions(...)`.

This matters because:

- the model is already under-weighting edges in the loss
- while the discrete edge corruption prior is relatively aggressive
- so the training setup is biased toward recovering from heavily masked edges without a correspondingly strong edge objective

## Current Working Hypothesis

The most likely failure is not a single bug, but a training imbalance:

1. edge supervision is weaker than pos/node in the distilled objective
2. edge corruption is strong (`tomask_half`)
3. the consistency target can self-reinforce a collapsed edge distribution

This is consistent with the observed outcome:

- atom counts remain reasonable
- connectivity collapses
- higher sampling steps erase bonds rather than refine them

## Immediate Next Checks

Before changing training, verify one more thing:

1. inspect generated / predicted edge-type distributions directly, especially the fraction of `edge_type == 0`
2. compare that against reference edge statistics from the test data if available
3. if the collapse is confirmed, run a targeted retrain on `gpu33` with:
   - larger `halfedge` loss weight
   - unchanged sampler first
   - no simultaneous algorithm changes

## 2026-04-16 Direct Edge-Type Diagnostic

Added a direct edge inspection script:

- [scripts/inspect_edge_collapse.py](/shared/healthinfolab/phz24002/PocketXMol/scripts/inspect_edge_collapse.py)

Ran on `gpu35` with the latest checkpoint:

- checkpoint:
  `outputs_distill_ve_2/consistency_distill_20260404_005352/checkpoints/step_100000.pt`
- output:
  [`outputs_edge_diagnostics/euler_step100k_20260416_bs4/edge_diagnostic.csv`](/shared/healthinfolab/phz24002/PocketXMol/outputs_edge_diagnostics/euler_step100k_20260416_bs4/edge_diagnostic.csv)
- raw json:
  [`outputs_edge_diagnostics/euler_step100k_20260416_bs4/edge_diagnostic.json`](/shared/healthinfolab/phz24002/PocketXMol/outputs_edge_diagnostics/euler_step100k_20260416_bs4/edge_diagnostic.json)

Setup:

- `coordinate_update = euler`
- `discrete_update = categorical_transition`
- `sample_steps = 1 / 8 / 100`
- `num_graphs = 32` per step

### Result

The collapse is directly confirmed in the model's predicted halfedge classes.

`1-step`

- predicted positive-bond fraction: `0.0284`
- predicted `edge_type = 0` fraction: `0.9716`
- only nonzero predicted bond class observed: `edge_type = 1`

`8-step`

- predicted positive-bond fraction: `0.0166`
- predicted `edge_type = 0` fraction: `0.9834`
- only nonzero predicted bond class observed: `edge_type = 1`

`100-step`

- predicted positive-bond fraction: `0.0130`
- predicted `edge_type = 0` fraction: `0.9870`
- only nonzero predicted bond class observed: `edge_type = 1`

The final sampled `edge_state` distribution is identical to the argmax class distribution from `pred_halfedge` in this diagnostic run, so the problem is already present in the model output itself, before any downstream chemistry reconstruction.

### Interpretation

This is the strongest evidence so far.

- The model is not merely producing the wrong bond orders.
- It is predicting almost all candidate edges as `no-bond`.
- As step count increases, the positive-bond fraction shrinks even further:
  `2.84% -> 1.66% -> 1.30%`
- The model is also not meaningfully using higher bond classes in this sample; it effectively degenerates to a near-binary `{no-bond, single-bond}` output, dominated by `no-bond`.

This lines up with all earlier observations:

- atom counts stay normal
- largest fragment size stays tiny
- isolated atom fraction grows with step count
- reconstructed molecules become bags of small disconnected fragments

## Updated Working Conclusion

The primary bottleneck is now clearly the edge generator, not the coordinate sampler.

Most plausible causes, in descending order:

1. the distilled edge objective is too weak relative to `pos`
2. the `tomask_half` edge corruption prior is too aggressive for the current consistency objective
3. EMA-based consistency is allowing a collapsed edge distribution to self-reinforce

## Recommended First Retrain

When moving to `gpu33`, the first retrain should be minimal and controlled.

Change only one axis first:

1. increase `loss_weights.halfedge`
2. keep the current sampler and evaluation protocol unchanged
3. keep `coordinate_update=euler` fixed for now
4. do not mix in extra algorithm changes until we see whether bond fraction recovers

Suggested first experiment:

- raise `halfedge` loss weight from `1.0` to `3.0` or `5.0`
- leave `pos=1.0`, `node=1.0`
- keep the same checkpoint initialization and same training data
- evaluate the same `1/8/100-step` edge diagnostic before doing a full parity sweep

## 2026-04-19 Edge-Weighted Retrain Started

Started the first controlled retrain for the edge-collapse hypothesis.

Config:

- [configs/distill/distill_pxm_euler_edge5.yaml](/shared/healthinfolab/phz24002/PocketXMol/configs/distill/distill_pxm_euler_edge5.yaml)

Only intended training-axis change:

- `loss_weights.halfedge: 1.0 -> 5.0`

Kept fixed:

- `loss_weights.pos = 1.0`
- `loss_weights.node = 1.0`
- `coordinate_update = euler`
- `discrete_update = categorical_transition`
- `schedule = uniform`
- teacher initialization
- training data override

Disabled for this controlled run:

- `wandb`
- online evaluation

Launch command:

```bash
ssh gpu35 'export LD_LIBRARY_PATH=/shared/healthinfolab/phz24002/anaconda3/envs/PocketXMol/lib:$LD_LIBRARY_PATH; export CUDA_VISIBLE_DEVICES=1; cd /shared/healthinfolab/phz24002/PocketXMol; mkdir -p outputs_distill_edge5_launch; nohup /shared/healthinfolab/phz24002/anaconda3/envs/PocketXMol/bin/python scripts/train_consistency.py --config configs/distill/distill_pxm_euler_edge5.yaml --device cuda:0 --outdir outputs_distill_edge5 > outputs_distill_edge5_launch/train_edge5_gpu35_gpu1_20260419.out 2>&1 & echo $!'
```

Runtime:

- host: `gpu35`
- physical GPU: `GPU1`
- PID: `18750`
- output dir: [outputs_distill_edge5/consistency_distill_20260419_031626](/shared/healthinfolab/phz24002/PocketXMol/outputs_distill_edge5/consistency_distill_20260419_031626)
- launch stdout: [outputs_distill_edge5_launch/train_edge5_gpu35_gpu1_20260419.out](/shared/healthinfolab/phz24002/PocketXMol/outputs_distill_edge5_launch/train_edge5_gpu35_gpu1_20260419.out)

Initial health check:

- training started successfully
- `nvidia-smi -L` maps `GPU1` to `GPU-49670f54-7688-0dd5-248d-e1a37f53f938`
- PID `18750` is running on that UUID
- first logged point:
  - `step=50`
  - `total=0.062145`
  - `pos=0.020859`
  - `node=0.018370`
  - `edge=0.004583`

Note:

- `edge` in the log is the raw unweighted edge loss from `MixedStateConsistencyLoss`.
- `total` includes the configured `halfedge=5.0` multiplier.

Next check:

- wait for the first checkpoint at `step=1000`
- run [scripts/inspect_edge_collapse.py](/shared/healthinfolab/phz24002/PocketXMol/scripts/inspect_edge_collapse.py) on that checkpoint before any full parity sweep

## 2026-04-19 Edge-Weighted Retrain Result

The `halfedge=5.0` retrain completed normally.

Final checkpoint:

- [outputs_distill_edge5/consistency_distill_20260419_031626/checkpoints/step_100000.pt](/shared/healthinfolab/phz24002/PocketXMol/outputs_distill_edge5/consistency_distill_20260419_031626/checkpoints/step_100000.pt)

Final training log:

- [outputs_distill_edge5/consistency_distill_20260419_031626/log.txt](/shared/healthinfolab/phz24002/PocketXMol/outputs_distill_edge5/consistency_distill_20260419_031626/log.txt)

Completion:

- `step=100000`
- saved final checkpoint
- `Consistency distillation completed`

### Training Loss Effect

The loss weight change did take effect.

Parsed `2000` logged points.

Early training (`step <= 10k`):

- mean `total = 0.03522`
- mean raw `pos = 0.01108`
- mean raw `node = 0.00985`
- mean raw `edge = 0.00286`
- weighted edge contribution to total: about `40.5%`

Mid training (`10k < step <= 50k`):

- mean `total = 0.03320`
- mean raw `pos = 0.01324`
- mean raw `node = 0.00826`
- mean raw `edge = 0.00234`
- weighted edge contribution to total: about `35.2%`

Late training (`step > 50k`):

- mean `total = 0.01681`
- mean raw `pos = 0.01129`
- mean raw `node = 0.00209`
- mean raw `edge = 0.00069`
- weighted edge contribution to total: about `20.4%`

Compared with the previous `halfedge=1.0` run, late-stage edge contribution increased from roughly `5%` to roughly `20%`.

So this run was a valid test of the simple loss-weight hypothesis.

### Final Edge Diagnostic

Ran:

```bash
ssh gpu35 'export LD_LIBRARY_PATH=/shared/healthinfolab/phz24002/anaconda3/envs/PocketXMol/lib:$LD_LIBRARY_PATH; export CUDA_VISIBLE_DEVICES=1; cd /shared/healthinfolab/phz24002/PocketXMol && /shared/healthinfolab/phz24002/anaconda3/envs/PocketXMol/bin/python scripts/inspect_edge_collapse.py --config_distill configs/distill/distill_pxm_euler_edge5.yaml --consistency_ckpt outputs_distill_edge5/consistency_distill_20260419_031626/checkpoints/step_100000.pt --sample_steps 1 8 100 --num_mols 32 --batch_size 4 --num_workers 1 --device cuda:0 --outdir outputs_edge_diagnostics/euler_edge5_step100k_20260419_bs4'
```

Output:

- [outputs_edge_diagnostics/euler_edge5_step100k_20260419_bs4/edge_diagnostic.csv](/shared/healthinfolab/phz24002/PocketXMol/outputs_edge_diagnostics/euler_edge5_step100k_20260419_bs4/edge_diagnostic.csv)
- [outputs_edge_diagnostics/euler_edge5_step100k_20260419_bs4/edge_diagnostic.json](/shared/healthinfolab/phz24002/PocketXMol/outputs_edge_diagnostics/euler_edge5_step100k_20260419_bs4/edge_diagnostic.json)

Result:

| sample_steps | pred positive-bond frac | pred `edge_type=0` frac |
|---:|---:|---:|
| 1 | 0.0237 | 0.9763 |
| 8 | 0.0175 | 0.9825 |
| 100 | 0.0183 | 0.9817 |

Compared with the previous checkpoint:

| sample_steps | old positive-bond frac | edge5 positive-bond frac |
|---:|---:|---:|
| 1 | 0.0284 | 0.0237 |
| 8 | 0.0166 | 0.0175 |
| 100 | 0.0130 | 0.0183 |

Interpretation:

- `halfedge=5.0` increased the training loss pressure on edges.
- It did not materially recover positive-bond generation.
- The model still predicts `no-bond` for roughly `97.6%` to `98.3%` of candidate halfedges.
- The output still only uses `edge_type=0` and `edge_type=1` in this diagnostic sample.

### Updated Conclusion

The simple loss-weight-only fix is insufficient.

The edge collapse is likely not just because `halfedge` loss had scalar weight `1.0`. More likely contributors now are:

1. severe class imbalance inside edge KL, where `no-bond` dominates candidate halfedges
2. EMA consistency target reinforcing the collapsed `no-bond` distribution
3. the `tomask_half` edge prior / categorical transition making edge recovery too easy to satisfy with `no-bond`

Next useful direction:

- add class-aware edge loss or positive-edge reweighting, not just scalar halfedge weight
- inspect target edge distribution from EMA/teacher during training
- consider direct supervised CE on teacher or ground-truth edge labels for a warmup / auxiliary term

## 2026-04-19 Positive-Edge Reweighting Retrain Started

After `halfedge=5.0` failed to recover positive bond generation, started a class-aware edge-loss run.

Code changes:

- [models/consistency_loss.py](/shared/healthinfolab/phz24002/PocketXMol/models/consistency_loss.py)
- [scripts/train_consistency.py](/shared/healthinfolab/phz24002/PocketXMol/scripts/train_consistency.py)

Implemented:

- optional `loss_weights.edge_positive`
- edge KL is weighted by ground-truth `batch["halfedge_type"] > 0`
- positive edges get weight `edge_positive`
- no-bond edges keep weight `1.0`
- `halfedge` scalar remains separate

Config:

- [configs/distill/distill_pxm_euler_edgepos20.yaml](/shared/healthinfolab/phz24002/PocketXMol/configs/distill/distill_pxm_euler_edgepos20.yaml)

Loss setup:

- `pos = 1.0`
- `node = 1.0`
- `halfedge = 1.0`
- `edge_positive = 20.0`

Rationale:

- This follows the updated hypothesis that scalar edge weight alone is too blunt.
- It raises pressure specifically on true positive bonds.
- It keeps the overall halfedge scalar smaller than the failed `halfedge=5.0` run.

Launch command:

```bash
ssh gpu35 'export LD_LIBRARY_PATH=/shared/healthinfolab/phz24002/anaconda3/envs/PocketXMol/lib:$LD_LIBRARY_PATH; export CUDA_VISIBLE_DEVICES=1; cd /shared/healthinfolab/phz24002/PocketXMol; mkdir -p outputs_distill_edgepos20_launch; nohup /shared/healthinfolab/phz24002/anaconda3/envs/PocketXMol/bin/python scripts/train_consistency.py --config configs/distill/distill_pxm_euler_edgepos20.yaml --device cuda:0 --outdir outputs_distill_edgepos20 > outputs_distill_edgepos20_launch/train_edgepos20_gpu35_gpu1_20260419.out 2>&1 & echo $!'
```

Runtime:

- host: `gpu35`
- physical GPU: `GPU1`
- PID: `169306`
- output dir: [outputs_distill_edgepos20/consistency_distill_20260419_172400](/shared/healthinfolab/phz24002/PocketXMol/outputs_distill_edgepos20/consistency_distill_20260419_172400)
- launch stdout: [outputs_distill_edgepos20_launch/train_edgepos20_gpu35_gpu1_20260419.out](/shared/healthinfolab/phz24002/PocketXMol/outputs_distill_edgepos20_launch/train_edgepos20_gpu35_gpu1_20260419.out)

Initial health check:

- process is running on `GPU-49670f54-7688-0dd5-248d-e1a37f53f938`, which maps to physical `GPU1`
- first logged points:
  - `step=50 total=0.058282 pos=0.020690 node=0.018394 edge=0.019198`
  - `step=100 total=0.008625 pos=0.004120 node=0.001829 edge=0.002675`

Note:

- in this run, logged `edge` is the weighted edge mean, not the old raw unweighted edge KL.
- the larger initial `edge` term indicates positive-edge weighting is active.

Next check:

- after `step_1000.pt`, run the direct edge diagnostic and compare positive-bond fraction against:
  - old baseline distilled: `1-step 0.0284`, `8-step 0.0166`, `100-step 0.0130`
  - scalar `halfedge=5`: `1-step 0.0237`, `8-step 0.0175`, `100-step 0.0183`

## 2026-04-19 Positive-Edge Reweighting Early Result

The `edge_positive=20` run is still training.

Runtime status:

- PID `169306`
- still running on `gpu35 GPU1`
- latest checked training step: `4400`

Available checkpoints:

- `step_1000.pt`
- `step_2000.pt`
- `step_3000.pt`
- `step_4000.pt`

Ran direct edge diagnostic on `step_4000.pt`.

Command:

```bash
ssh gpu35 'export LD_LIBRARY_PATH=/shared/healthinfolab/phz24002/anaconda3/envs/PocketXMol/lib:$LD_LIBRARY_PATH; export CUDA_VISIBLE_DEVICES=1; cd /shared/healthinfolab/phz24002/PocketXMol && /shared/healthinfolab/phz24002/anaconda3/envs/PocketXMol/bin/python scripts/inspect_edge_collapse.py --config_distill configs/distill/distill_pxm_euler_edgepos20.yaml --consistency_ckpt outputs_distill_edgepos20/consistency_distill_20260419_172400/checkpoints/step_4000.pt --sample_steps 1 8 100 --num_mols 32 --batch_size 4 --num_workers 1 --device cuda:0 --outdir outputs_edge_diagnostics/euler_edgepos20_step4000_20260419_bs4'
```

Output:

- [outputs_edge_diagnostics/euler_edgepos20_step4000_20260419_bs4/edge_diagnostic.csv](/shared/healthinfolab/phz24002/PocketXMol/outputs_edge_diagnostics/euler_edgepos20_step4000_20260419_bs4/edge_diagnostic.csv)
- [outputs_edge_diagnostics/euler_edgepos20_step4000_20260419_bs4/edge_diagnostic.json](/shared/healthinfolab/phz24002/PocketXMol/outputs_edge_diagnostics/euler_edgepos20_step4000_20260419_bs4/edge_diagnostic.json)

Early diagnostic:

| sample_steps | pred positive-bond frac | pred `edge_type=0` frac |
|---:|---:|---:|
| 1 | 0.0020 | 0.9980 |
| 8 | 0.0777 | 0.9223 |
| 100 | 0.0805 | 0.9195 |

Comparison:

| sample_steps | old distilled | scalar `halfedge=5` | `edge_positive=20` step 4000 |
|---:|---:|---:|---:|
| 1 | 0.0284 | 0.0237 | 0.0020 |
| 8 | 0.0166 | 0.0175 | 0.0777 |
| 100 | 0.0130 | 0.0183 | 0.0805 |

Interpretation:

- `edge_positive=20` is the first change that clearly increases positive-bond prediction for multi-step sampling.
- `8-step` and `100-step` move from roughly `1-2%` positive bonds to roughly `8%`.
- The model also starts using higher bond classes (`edge_type=2/3/4`) in the diagnostic sample.
- `1-step` gets worse at this early checkpoint, so the behavior is not uniformly improved.

This is an encouraging early signal, but not yet a final result.

Next:

- let the run continue
- repeat edge diagnostic at later checkpoints, e.g. `step_10000`, `step_50000`, and final `step_100000`
- only run full SDF/parity evaluation if the positive-bond fraction remains elevated and does not collapse later

## 2026-04-19 Positive-Edge Reweighting Mid-Run Check

The `edge_positive=20` run is still training.

Runtime status:

- PID `169306`
- still running on `gpu35 GPU1`
- latest checked training step: `24150`
- latest checked checkpoint: `step_24000.pt`

Ran direct edge diagnostic on `step_24000.pt`.

Output:

- [outputs_edge_diagnostics/euler_edgepos20_step24000_20260419_bs4/edge_diagnostic.csv](/shared/healthinfolab/phz24002/PocketXMol/outputs_edge_diagnostics/euler_edgepos20_step24000_20260419_bs4/edge_diagnostic.csv)
- [outputs_edge_diagnostics/euler_edgepos20_step24000_20260419_bs4/edge_diagnostic.json](/shared/healthinfolab/phz24002/PocketXMol/outputs_edge_diagnostics/euler_edgepos20_step24000_20260419_bs4/edge_diagnostic.json)

Diagnostic:

| sample_steps | pred positive-bond frac | pred `edge_type=0` frac |
|---:|---:|---:|
| 1 | 0.0183 | 0.9817 |
| 8 | 0.0685 | 0.9315 |
| 100 | 0.0701 | 0.9299 |

Comparison against earlier checkpoints:

| checkpoint | 1-step | 8-step | 100-step |
|---|---:|---:|---:|
| old distilled | 0.0284 | 0.0166 | 0.0130 |
| scalar `halfedge=5` final | 0.0237 | 0.0175 | 0.0183 |
| `edge_positive=20` step 4000 | 0.0020 | 0.0777 | 0.0805 |
| `edge_positive=20` step 24000 | 0.0183 | 0.0685 | 0.0701 |

Interpretation:

- The `8-step` and `100-step` positive-bond improvement is holding so far.
- It is slightly lower than `step_4000`, but still far above both previous runs.
- `1-step` recovered from the very low `step_4000` value, but remains below old distilled.
- Higher bond-class usage did not persist strongly at `step_24000`; `8-step` and `100-step` are again mostly `edge_type=1`.

Current read:

- positive-edge reweighting is a real improvement for multi-step edge connectivity
- it is not yet a complete chemistry/bond-order solution
- continue training and re-check at `step_50000` and final `step_100000`

## 2026-04-19 Step-24000 Small Evaluation

Ran a small SDF/evaluation pass on `edge_positive=20` checkpoint `step_24000.pt`.

Command:

```bash
ssh gpu35 'export LD_LIBRARY_PATH=/shared/healthinfolab/phz24002/anaconda3/envs/PocketXMol/lib:$LD_LIBRARY_PATH; export CUDA_VISIBLE_DEVICES=2; cd /shared/healthinfolab/phz24002/PocketXMol && /shared/healthinfolab/phz24002/anaconda3/envs/PocketXMol/bin/python scripts/eval_consistency.py --config_distill configs/distill/distill_pxm_euler_edgepos20.yaml --consistency_ckpt outputs_distill_edgepos20/consistency_distill_20260419_172400/checkpoints/step_24000.pt --config_task configs/sample/test/sbdd_csd/simple.yml --sample_steps 1 8 100 --num_mols 100 --batch_size 8 --num_workers 1 --device cuda:0 --outdir outputs_consistency_eval_edgepos20_step24000 --standard_eval --standard_docking_mode none --standard_n_workers 1 --standard_max_mols 100'
```

Output:

- [outputs_consistency_eval_edgepos20_step24000/consistency_eval_20260419_202451/summary.csv](/shared/healthinfolab/phz24002/PocketXMol/outputs_consistency_eval_edgepos20_step24000/consistency_eval_20260419_202451/summary.csv)

Summary:

| sample_steps | recon_success | complete | validity | connectivity | standard_complete | standard_no_clashes | standard_atom_type_js | QED | SA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.57 | 0.00 | 0.57 | 0.00 | 0.00 | 0.03 | 0.2511 | 0.4145 | 0.5609 |
| 8 | 0.99 | 0.05 | 0.99 | 0.0505 | 0.05 | 0.01 | 0.3310 | 0.4130 | 0.8176 |
| 100 | 1.00 | 0.02 | 1.00 | 0.02 | 0.02 | 0.00 | 0.3281 | 0.4060 | 0.7829 |

Additional standard metrics:

| sample_steps | JSD_CC_2A | JSD_All_12A | stereo |
|---:|---:|---:|---:|
| 1 | 0.6446 | 0.2192 | 0.2807 |
| 8 | 0.6483 | 0.3125 | 0.1717 |
| 100 | 0.6570 | 0.3271 | 0.1800 |

Interpretation:

- The positive-edge fix is now visible beyond the edge diagnostic.
- `8-step` and `100-step` reconstruct successfully and have nonzero completeness/connectivity.
- `8-step` currently looks better than `100-step` on this small sample.
- However, structural quality is still poor:
  - `standard_no_clashes` is only `0.01` at `8-step`
  - `standard_no_clashes` is `0.00` at `100-step`
  - bond-distance JSD remains very poor where available
- So this is not parity. It is a partial recovery from disconnected/no-bond collapse.

Current implication:

- positive-edge reweighting fixes part of the edge-collapse failure
- the next bottleneck is geometry / clash quality and realistic bond-distance distribution
- continue the run to later checkpoints before deciding whether to add geometry-aware or bond-distance-aware terms

## 2026-04-20 Positive-Edge Reweighting Final Result

The `edge_positive=20` run completed normally.

Final checkpoint:

- [outputs_distill_edgepos20/consistency_distill_20260419_172400/checkpoints/step_100000.pt](/shared/healthinfolab/phz24002/PocketXMol/outputs_distill_edgepos20/consistency_distill_20260419_172400/checkpoints/step_100000.pt)

Training log:

- [outputs_distill_edgepos20/consistency_distill_20260419_172400/log.txt](/shared/healthinfolab/phz24002/PocketXMol/outputs_distill_edgepos20/consistency_distill_20260419_172400/log.txt)

Completion:

- `step=100000`
- saved final checkpoint
- `Consistency distillation completed`

### Final Edge Diagnostic

Ran direct edge diagnostic on `step_100000.pt`.

Output:

- [outputs_edge_diagnostics/euler_edgepos20_step100000_20260420_bs4/edge_diagnostic.csv](/shared/healthinfolab/phz24002/PocketXMol/outputs_edge_diagnostics/euler_edgepos20_step100000_20260420_bs4/edge_diagnostic.csv)
- [outputs_edge_diagnostics/euler_edgepos20_step100000_20260420_bs4/edge_diagnostic.json](/shared/healthinfolab/phz24002/PocketXMol/outputs_edge_diagnostics/euler_edgepos20_step100000_20260420_bs4/edge_diagnostic.json)

| sample_steps | pred positive-bond frac | pred `edge_type=0` frac |
|---:|---:|---:|
| 1 | 0.1045 | 0.8955 |
| 8 | 0.0486 | 0.9514 |
| 100 | 0.0438 | 0.9562 |

Comparison:

| checkpoint | 1-step | 8-step | 100-step |
|---|---:|---:|---:|
| old distilled | 0.0284 | 0.0166 | 0.0130 |
| scalar `halfedge=5` final | 0.0237 | 0.0175 | 0.0183 |
| `edge_positive=20` step 4000 | 0.0020 | 0.0777 | 0.0805 |
| `edge_positive=20` step 24000 | 0.0183 | 0.0685 | 0.0701 |
| `edge_positive=20` step 100000 | 0.1045 | 0.0486 | 0.0438 |

Interpretation:

- final `1-step` positive-bond fraction is much higher than previous runs
- final `8-step` and `100-step` retain improvement over old distilled, but are worse than the mid-run checkpoints
- positive-edge reweighting does not monotonically improve with training time; the best multi-step edge fraction appeared earlier around `step_4000` to `step_24000`

### Final Small Evaluation

Ran small `100`-molecule evaluation on `step_100000.pt` with `1/8/100-step`, standard SDF eval, no docking.

Output:

- [outputs_consistency_eval_edgepos20_step100000/consistency_eval_20260420_211536/summary.csv](/shared/healthinfolab/phz24002/PocketXMol/outputs_consistency_eval_edgepos20_step100000/consistency_eval_20260420_211536/summary.csv)

| sample_steps | recon_success | complete | validity | connectivity | standard_complete | standard_no_clashes | standard_atom_type_js | QED | SA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.70 | 0.29 | 0.70 | 0.4143 | 0.29 | 0.00 | 0.3479 | 0.4640 | 0.5487 |
| 8 | 1.00 | 0.02 | 1.00 | 0.0200 | 0.02 | 0.02 | 0.5099 | 0.2822 | 0.4512 |
| 100 | 1.00 | 0.00 | 1.00 | 0.0000 | 0.00 | 0.02 | 0.4894 | 0.2802 | 0.4533 |

Additional standard metrics:

| sample_steps | JSD_CC_2A | JSD_All_12A | stereo |
|---:|---:|---:|---:|
| 1 | 0.6414 | 0.2220 | 0.0429 |
| 8 | 0.7879 | 0.3339 | 0.1600 |
| 100 | 0.7489 | 0.3124 | 0.1800 |

Comparison with `step_24000` small eval:

| checkpoint | sample_steps | complete | connectivity | standard_no_clashes | QED | SA |
|---|---:|---:|---:|---:|---:|---:|
| step 24000 | 1 | 0.00 | 0.0000 | 0.03 | 0.4145 | 0.5609 |
| step 24000 | 8 | 0.05 | 0.0505 | 0.01 | 0.4130 | 0.8176 |
| step 24000 | 100 | 0.02 | 0.0200 | 0.00 | 0.4060 | 0.7829 |
| step 100000 | 1 | 0.29 | 0.4143 | 0.00 | 0.4640 | 0.5487 |
| step 100000 | 8 | 0.02 | 0.0200 | 0.02 | 0.2822 | 0.4512 |
| step 100000 | 100 | 0.00 | 0.0000 | 0.02 | 0.2802 | 0.4533 |

### Current Conclusion

`edge_positive=20` fixed part of the no-bond collapse, but it overcorrects / destabilizes chemistry.

Clear improvements:

- final `1-step` connectivity and completeness are much better than previous runs
- positive-bond fraction is no longer stuck near `1-2%`

Still failing:

- `standard_no_clashes` remains essentially zero
- `8/100-step` degrade by the end of training
- atom-type JS and pair-distance JS are poor
- many SDF warnings show invalid valence / over-bonded carbon

Most likely new failure mode:

- positive-edge weighting is producing too many or geometrically inconsistent bonds in some regimes
- this helps connectivity but creates severe clashes / invalid valence / unrealistic bond-distance distributions

Recommended next direction:

- do not just increase edge-positive weight further
- checkpoint selection matters: evaluate earlier checkpoints, especially `step_24000`, more seriously
- next training change should combine edge-positive weighting with a geometry or valence/bond-distance regularizer, or cap positive-edge pressure

## 2026-04-23 Teacher Input Diagnostic

Tested whether restoring SNR-scaled coordinate input helps the frozen teacher's edge prediction directly.

Added diagnostic:

- [scripts/inspect_teacher_edge_inputs.py](/shared/healthinfolab/phz24002/PocketXMol/scripts/inspect_teacher_edge_inputs.py)

Ran on:

- teacher checkpoint:
  `data/trained_models/pxm/checkpoints/pocketxmol.ckpt`
- config:
  [configs/distill/distill_pxm_euler_edgepos20.yaml](/shared/healthinfolab/phz24002/PocketXMol/configs/distill/distill_pxm_euler_edgepos20.yaml)
- output:
  [outputs_teacher_edge_input_diagnostics/teacher_raw_vs_snr_20260423_gpu35_gpu1/teacher_edge_input_diagnostic.csv](/shared/healthinfolab/phz24002/PocketXMol/outputs_teacher_edge_input_diagnostics/teacher_raw_vs_snr_20260423_gpu35_gpu1/teacher_edge_input_diagnostic.csv)

Compared two teacher inputs on the same noised batches:

- `position_input_mode = raw`
- `position_input_mode = snr_scaled`

Main result:

- SNR-scaled input does **not** clearly rescue teacher edge prediction.
- At low / mid noise, `raw` and `snr_scaled` are close.
- At the noisiest tested step (`t=49`), `snr_scaled` is actually worse.

Key numbers:

| timestep | mode | pred positive frac | gt positive recall | gt positive precision |
|---:|---|---:|---:|---:|
| 25 | raw | 0.0752 | 0.7939 | 0.6550 |
| 25 | snr_scaled | 0.0760 | 0.8136 | 0.6550 |
| 49 | raw | 0.0256 | 0.1681 | 0.3989 |
| 49 | snr_scaled | 0.0182 | 0.1128 | 0.4000 |

Interpretation:

- The earlier hypothesis "`pos_in` should be SNR-scaled because that will directly fix teacher edge targets" is not supported.
- Any benefit from `snr_scaled` is more likely coming from student / EMA consistency dynamics, not from making the frozen teacher itself much better at edge prediction.

## 2026-04-23 SNR-Input Controlled Retrain Started

Started a controlled `4k`-step retrain to test whether `snr_scaled` position input helps multi-step sampling, even though it did not directly fix teacher edge prediction.

Config:

- [configs/distill/distill_pxm_euler_snrinput_edgepos20_4k.yaml](/shared/healthinfolab/phz24002/PocketXMol/configs/distill/distill_pxm_euler_snrinput_edgepos20_4k.yaml)

Changes relative to the previous `edge_positive=20` setting:

- `position_input_mode: raw -> snr_scaled`
- kept:
  - `pos_preconditioning = cosine`
  - `edge_positive = 20`
  - `coordinate_update = euler`
  - `discrete_update = categorical_transition`

Runtime:

- host: `gpu35`
- physical GPU: `GPU1`
- output dir:
  [outputs_distill_snrinput_edgepos20_4k/consistency_distill_20260423_233516](/shared/healthinfolab/phz24002/PocketXMol/outputs_distill_snrinput_edgepos20_4k/consistency_distill_20260423_233516)

## 2026-04-24 SNR-Input + Cosine 4k Result

The `snr_scaled + cosine + edge_positive=20` controlled run completed normally.

Final checkpoint:

- [outputs_distill_snrinput_edgepos20_4k/consistency_distill_20260423_233516/checkpoints/step_4000.pt](/shared/healthinfolab/phz24002/PocketXMol/outputs_distill_snrinput_edgepos20_4k/consistency_distill_20260423_233516/checkpoints/step_4000.pt)

Edge diagnostic:

- [outputs_edge_diagnostics/euler_snrinput_edgepos20_step4000_20260424_bs4/edge_diagnostic.csv](/shared/healthinfolab/phz24002/PocketXMol/outputs_edge_diagnostics/euler_snrinput_edgepos20_step4000_20260424_bs4/edge_diagnostic.csv)

Small eval:

- [outputs_consistency_eval_snrinput_edgepos20_step4000/consistency_eval_20260424_015313/summary.csv](/shared/healthinfolab/phz24002/PocketXMol/outputs_consistency_eval_snrinput_edgepos20_step4000/consistency_eval_20260424_015313/summary.csv)

Edge fractions:

| sample_steps | pred positive-bond frac |
|---:|---:|
| 1 | 0.0010 |
| 8 | 0.0824 |
| 100 | 0.0825 |

Eval summary:

| sample_steps | recon_success | complete | validity | connectivity |
|---:|---:|---:|---:|---:|
| 1 | 1.00 | 0.00 | 1.00 | 0.00 |
| 8 | 0.62 | 0.20 | 0.62 | 0.3226 |
| 100 | 0.89 | 0.26 | 0.89 | 0.2921 |

Interpretation:

- `snr_scaled` does not help `1-step`.
- But it clearly improves multi-step behavior relative to the old `raw` edge-positive runs.
- This suggests the main gain is in multi-step refinement, not in one-shot denoising.

## 2026-04-24 No-Preconditioning Controlled Retrain Started

Based on the poor `1-step` behavior and the observation that `snr_scaled` mainly helps multi-step refinement, started a second controlled `4k` run with cosine position preconditioning removed.

Config:

- [configs/distill/distill_pxm_euler_snrinput_noprecond_edgepos20_4k.yaml](/shared/healthinfolab/phz24002/PocketXMol/configs/distill/distill_pxm_euler_snrinput_noprecond_edgepos20_4k.yaml)

Changes relative to the previous run:

- keep `position_input_mode = snr_scaled`
- change `pos_preconditioning: cosine -> none`
- keep `edge_positive = 20`

Runtime:

- host: `gpu35`
- physical GPU: `GPU1`
- output dir:
  [outputs_distill_snrinput_noprecond_edgepos20_4k/consistency_distill_20260424_105301](/shared/healthinfolab/phz24002/PocketXMol/outputs_distill_snrinput_noprecond_edgepos20_4k/consistency_distill_20260424_105301)

## 2026-04-24 No-Preconditioning + EdgePos20 4k Result

The `snr_scaled + no_preconditioning + edge_positive=20` run completed normally.

Final checkpoint:

- [outputs_distill_snrinput_noprecond_edgepos20_4k/consistency_distill_20260424_105301/checkpoints/step_4000.pt](/shared/healthinfolab/phz24002/PocketXMol/outputs_distill_snrinput_noprecond_edgepos20_4k/consistency_distill_20260424_105301/checkpoints/step_4000.pt)

Edge diagnostic:

- [outputs_edge_diagnostics/euler_snrinput_noprecond_edgepos20_step4000_20260424_bs4/edge_diagnostic.csv](/shared/healthinfolab/phz24002/PocketXMol/outputs_edge_diagnostics/euler_snrinput_noprecond_edgepos20_step4000_20260424_bs4/edge_diagnostic.csv)

Small eval:

- [outputs_consistency_eval_snrinput_noprecond_edgepos20_step4000/consistency_eval_20260424_142131/summary.csv](/shared/healthinfolab/phz24002/PocketXMol/outputs_consistency_eval_snrinput_noprecond_edgepos20_step4000/consistency_eval_20260424_142131/summary.csv)

Edge fractions:

| sample_steps | pred positive-bond frac |
|---:|---:|
| 8 | 0.1020 |
| 100 | 0.1017 |

Eval summary:

| sample_steps | recon_success | complete | validity | connectivity |
|---:|---:|---:|---:|---:|
| 8 | 0.31 | 0.30 | 0.31 | 0.9677 |
| 100 | 0.47 | 0.47 | 0.47 | 1.0000 |

Interpretation:

- Removing cosine preconditioning strongly increases connectivity and completeness.
- But it also collapses chemical validity / reconstruction quality.
- The dominant failure mode is no longer "too few bonds / disconnected fragments".
- The new failure mode is closer to "strongly connected but chemically invalid / unreconstructable outputs".

## 2026-04-24 EdgePos10 Controlled Retrain Started

To see whether the no-preconditioning setup was simply overdriving positive-edge pressure, started a third controlled `4k` run with reduced positive-edge weight.

Config:

- [configs/distill/distill_pxm_euler_snrinput_noprecond_edgepos10_4k.yaml](/shared/healthinfolab/phz24002/PocketXMol/configs/distill/distill_pxm_euler_snrinput_noprecond_edgepos10_4k.yaml)

Changes relative to the previous run:

- keep:
  - `position_input_mode = snr_scaled`
  - `pos_preconditioning = none`
- change:
  - `edge_positive: 20 -> 10`

Runtime:

- host: `gpu35`
- physical GPU: `GPU1`
- output dir:
  [outputs_distill_snrinput_noprecond_edgepos10_4k/consistency_distill_20260424_144656](/shared/healthinfolab/phz24002/PocketXMol/outputs_distill_snrinput_noprecond_edgepos10_4k/consistency_distill_20260424_144656)

## 2026-04-25 No-Preconditioning + EdgePos10 4k Result

The `snr_scaled + no_preconditioning + edge_positive=10` run completed normally.

Final checkpoint:

- [outputs_distill_snrinput_noprecond_edgepos10_4k/consistency_distill_20260424_144656/checkpoints/step_4000.pt](/shared/healthinfolab/phz24002/PocketXMol/outputs_distill_snrinput_noprecond_edgepos10_4k/consistency_distill_20260424_144656/checkpoints/step_4000.pt)

Edge diagnostic:

- [outputs_edge_diagnostics/euler_snrinput_noprecond_edgepos10_step4000_20260425_bs4/edge_diagnostic.csv](/shared/healthinfolab/phz24002/PocketXMol/outputs_edge_diagnostics/euler_snrinput_noprecond_edgepos10_step4000_20260425_bs4/edge_diagnostic.csv)

Small eval:

- [outputs_consistency_eval_snrinput_noprecond_edgepos10_step4000/consistency_eval_20260425_001354/summary.csv](/shared/healthinfolab/phz24002/PocketXMol/outputs_consistency_eval_snrinput_noprecond_edgepos10_step4000/consistency_eval_20260425_001354/summary.csv)

Edge fractions:

| sample_steps | pred positive-bond frac |
|---:|---:|
| 8 | 0.1010 |
| 100 | 0.1010 |

Eval summary:

| sample_steps | recon_success | complete | validity | connectivity |
|---:|---:|---:|---:|---:|
| 8 | 0.34 | 0.33 | 0.34 | 0.9706 |
| 100 | 0.51 | 0.50 | 0.51 | 0.9804 |

Comparison against `edge_positive=20`:

| config | sample_steps | complete | validity | connectivity |
|---|---:|---:|---:|---:|
| no-precond, edgepos20 | 8 | 0.30 | 0.31 | 0.9677 |
| no-precond, edgepos10 | 8 | 0.33 | 0.34 | 0.9706 |
| no-precond, edgepos20 | 100 | 0.47 | 0.47 | 1.0000 |
| no-precond, edgepos10 | 100 | 0.50 | 0.51 | 0.9804 |

Current conclusion:

- `edge_positive=10` is slightly better than `20` in the same `snr_scaled + no_preconditioning` regime.
- However, the core bottleneck has clearly shifted:
  - disconnected-fragment collapse is no longer the main issue
  - the main issue is now chemical validity / reconstruction failure in highly connected samples
- Further scalar tuning of `edge_positive` alone is unlikely to solve the remaining gap.

Recommended next step:

- stop treating connectivity as the dominant bottleneck
- run a failure-mode diagnostic on invalid / unreconstructable `8-step` samples from the `edgepos10` run
- separate whether the remaining failures are dominated by:
  - invalid valence
  - wrong bond order
  - atom-type mismatch
  - geometry causing reconstruction failure

## 2026-04-30 8-Step Failure-Mode Diagnostic

Ran a targeted failure analysis on the best current short-run checkpoint:

- config:
  `snr_scaled + no_preconditioning + edge_positive=10`
- eval root:
  [outputs_consistency_eval_snrinput_noprecond_edgepos10_step4000/consistency_eval_20260425_001354](/shared/healthinfolab/phz24002/PocketXMol/outputs_consistency_eval_snrinput_noprecond_edgepos10_step4000/consistency_eval_20260425_001354)

Diagnostic outputs:

- sample-level summary:
  [diagnostic_summary.csv](/shared/healthinfolab/phz24002/PocketXMol/outputs_consistency_eval_snrinput_noprecond_edgepos10_step4000/consistency_eval_20260425_001354/diagnostic_summary.csv)
- invalid-SDF summary:
  [invalid_sdf_summary.csv](/shared/healthinfolab/phz24002/PocketXMol/outputs_consistency_eval_snrinput_noprecond_edgepos10_step4000/consistency_eval_20260425_001354/invalid_sdf_summary.csv)
- invalid-SDF per-sample table:
  [invalid_sdf_samples.csv](/shared/healthinfolab/phz24002/PocketXMol/outputs_consistency_eval_snrinput_noprecond_edgepos10_step4000/consistency_eval_20260425_001354/invalid_sdf_samples.csv)

### Outcome split

For `8-step`:

- `count_complete = 33`
- `count_incomp = 1`
- `count_bad = 66`

So the current failure mode is **not** disconnected fragments anymore. There is only `1` incomplete sample; almost every failure is a `bad` molecule that cannot be sanitized / reconstructed.

### Invalid-SDF error families

For the `66` bad `8-step` samples:

- `55` are `explicit_valence`
- `10` are `kekulize`
- `1` is `sanitize_ok` but still tagged upstream as bad

For comparison, `100-step` has the same pattern:

- `42` `explicit_valence`
- `7` `kekulize`

So the dominant remaining bottleneck is now:

- over-bonded / over-ordered local chemistry
- aromatic / bond-order assignments that cannot be kekulized

### Which atoms are failing valence

Among the `8-step` explicit-valence failures, the offending elements are:

- `C: 28`
- `O: 20`
- `F: 5`
- `N: 1`
- `P: 1`

Observed invalid valence values are dominated by:

- valence `5`: `23`
- valence `3`: `19`
- valence `2`: `5`
- valence `6`: `5`

Representative recurring failures:

- carbon with valence `5` or `6` (and one valence `7`)
- oxygen with valence `3`
- fluorine with valence `2`

This is consistent with a regime where the model has largely fixed connectivity but is now assigning too many bonds / overly aggressive bond orders.

### Sample characteristics

For bad `8-step` samples:

- `bad_num_atoms_mean = 24.88`
- `bad_num_bonds_mean = 29.98`
- `bad_cfd_node_mean = 5.62`
- `bad_cfd_edge_mean = 12.38`

Mean bad-sample confidence by family:

| error_family | cfd_node | cfd_edge | num_atoms | num_bonds |
|---|---:|---:|---:|---:|
| explicit_valence | 5.679 | 12.296 | 24.836 | 30.127 |
| kekulize | 5.463 | 12.902 | 25.200 | 29.600 |

So these invalid samples are not obviously low-confidence outliers; the model is making fairly confident chemically illegal decisions.

### Current interpretation

The project has now clearly moved out of the original "missing edges / disconnected fragments" failure regime.

The main remaining failure mode is:

- high-connectivity molecules with illegal local chemistry
- especially carbon/oxygen valence overflow
- plus a smaller aromatic / kekulization problem

This means further tuning of `edge_positive` alone is unlikely to be enough. The next corrective experiments should target chemical legality more directly, e.g.:

- add a more direct supervised edge CE term instead of only EMA edge KL
- add valence-aware constraints / penalties / post-processing
- inspect bond-order predictions on bad samples, not just edge presence

## 2026-05-08 Edge Supervised CE Controlled Run

Based on the April 30 failure-mode diagnostic, started a controlled run that adds a small direct supervised edge cross-entropy term. The goal is to test whether a ground-truth edge anchor can reduce the over-bonding / invalid-valence regime without losing the 8-step connectivity gains.

Code changes:

- [models/consistency_loss.py](/shared/healthinfolab/phz24002/PocketXMol/models/consistency_loss.py)
  - added `edge_supervised_weight`
  - when `edge_target_type` is provided, computes:
    `CE(student_pred_halfedge_logits_x0, batch["halfedge_type"])`
  - adds `edge_supervised_weight * edge_supervised_loss` to total loss
- [scripts/train_consistency.py](/shared/healthinfolab/phz24002/PocketXMol/scripts/train_consistency.py)
  - reads `distill.loss_weights.edge_supervised`
  - logs `train/loss_edge_supervised`

Config:

- [configs/distill/distill_pxm_euler_snrinput_noprecond_edgepos10_edgece01_4k.yaml](/shared/healthinfolab/phz24002/PocketXMol/configs/distill/distill_pxm_euler_snrinput_noprecond_edgepos10_edgece01_4k.yaml)

Only intended change relative to the previous best short-run config:

- keep:
  - `position_input_mode = snr_scaled`
  - `pos_preconditioning = none`
  - `edge_positive = 10.0`
  - `max_steps = 4000`
- add:
  - `edge_supervised = 0.1`

Runtime:

- host: `gpu37`
- physical GPU: `GPU1`
- output root:
  [outputs_distill_snrinput_noprecond_edgepos10_edgece01_4k](/shared/healthinfolab/phz24002/PocketXMol/outputs_distill_snrinput_noprecond_edgepos10_edgece01_4k)
- current run:
  [outputs_distill_snrinput_noprecond_edgepos10_edgece01_4k/consistency_distill_20260508_031946](/shared/healthinfolab/phz24002/PocketXMol/outputs_distill_snrinput_noprecond_edgepos10_edgece01_4k/consistency_distill_20260508_031946)
- launch log:
  [outputs_distill_snrinput_noprecond_edgepos10_edgece01_4k_launch/train_gpu37_gpu1_20260507.out](/shared/healthinfolab/phz24002/PocketXMol/outputs_distill_snrinput_noprecond_edgepos10_edgece01_4k_launch/train_gpu37_gpu1_20260507.out)

Current status at note update:

- training is running normally
- checkpoint saved through `step_2000.pt`
- latest observed log line: `step=2500`
- example logged supervised CE values:
  - `step=2000`: `edge_ce = 0.110090`
  - `step=2400`: `edge_ce = 0.009671`
  - `step=2500`: `edge_ce = 0.054677`

Evaluation plan after `step_4000.pt`:

- run edge diagnostic on `8-step` and `100-step`
- run small SBDD eval focused on `8-step`
- rerun invalid-SDF diagnostic
- compare against the previous `edge_positive=10` baseline:
  - `8-step complete = 0.33`
  - `8-step validity = 0.34`
  - `8-step connectivity = 0.9706`
  - `8-step bad = 66`
  - `8-step explicit_valence = 55`

Primary success criterion:

- reduce `8-step` explicit-valence failures without collapsing connectivity.
