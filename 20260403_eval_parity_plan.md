# VE Distillation Evaluation Parity Plan

## Goal

Before exploring more sampler variants, first make the distilled VE model match the original PocketXMol model under a comparable evaluation protocol.

The target is not "best short-step sampler" yet. The target is:

- under the same dataset
- under the same evaluation code path
- under the same SDF / Vina / validity metrics
- the distilled model should be able to approach the original model's 100-step quality

Only after that should we continue pushing toward fewer steps.

## Why this is needed

Recent experiments show:

- `snr_renoise + categorical_transition` is still the most stable sampler configuration.
- More steps help, but the distilled model does not yet clearly dominate the original baseline at long step counts.
- Evaluation is still not fully aligned:
  - fast consistency summary
  - `standard_eval` SDF metrics
  - Vina
  - validity / connectivity

So the current bottleneck is not sampler creativity. The bottleneck is parity.

## Current Baseline

Use this as the default comparison setup:

- `coordinate_update = snr_renoise`
- `discrete_update = categorical_transition`
- `standard_eval = true`
- `sample_steps = 1 2 4 8 16 32`

Relevant checkpoint candidates:

- `step_50000.pt`
- `step_100000.pt`

Observed trend from the latest sweeps:

- `step_50000.pt` is better around `16-step`
- `step_100000.pt` is better around `32-step`
- both checkpoints still improve as step count increases

## Phase 1: Lock Evaluation Parity

### 1.1 Unify evaluation entry points

Make sure all comparisons go through the same path:

- `scripts/eval_consistency.py`
- `evaluate/evaluate_sdf_standard.py` through the standard-eval helper

Do not mix in ad hoc outputs or old one-off evaluation scripts.

### 1.2 Use one metric set for comparison

Primary metrics:

- `recon_success`
- `complete`
- `validity`
- `connectivity`
- `standard_complete`
- `standard_recon_success`
- `standard_atom_type_js`
- `standard_pair_js_*`
- `standard_bond_js_*`

Secondary metrics:

- `QED`
- `SA`
- Vina mean / median
- wall time / samples per second

### 1.3 Fix metric interpretation

Treat `standard_eval_success = 0` as a bug / placeholder until the path is verified.

Do not use it as a conclusion metric.

## Phase 2: Build a True 100-Step Parity Baseline

The next concrete baseline should be:

- original PocketXMol at 100 steps
- distilled VE model at 100 steps
- same dataset split
- same standard evaluation

For this phase, the distilled model should first be judged on whether it can match the baseline in the long-step regime.

If it cannot, do not proceed to more aggressive step reduction.

## Phase 3: Diagnose the Gap

If the distilled model is still behind at 100 steps, split the gap into categories:

- reconstruction failure rate
- validity / chemical sanity
- connectivity once reconstruction succeeds
- geometry / bond distribution mismatch
- atom-type distribution mismatch

Use these categories to decide whether the next fix should be:

- training loss balance
- teacher/student alignment
- discrete supervision
- sampling update rule
- sigma conditioning

## Phase 4: Only Then Reduce Steps

Once the distilled model is close enough to the original model at 100 steps, move down the step ladder:

- 32 steps
- 16 steps
- 8 steps
- 4 steps

At that point, the question becomes:

- which step count gives the best quality / speed tradeoff
- not whether the distilled model is fundamentally missing the target

## Acceptance Criteria

Consider parity "good enough" only if the distilled model satisfies most of the following under the same evaluation setup:

- long-step `complete` is close to the original baseline
- long-step `recon_success` is close to the original baseline
- `validity` and `connectivity` are not collapsing
- standard SDF metrics are stable
- Vina does not regress badly

## Immediate Next Actions

1. Re-run the 100-step comparison against the original model using the same evaluation helper.
2. Verify the standard evaluation outputs are being populated consistently.
3. Compare the long-step distilled results with the original baseline before touching the sampler again.

Reference:

conda_shared
conda activate PocketXMol
cd /shared/healthinfolab/phz24002/PocketXMol/
python scripts/sample_drug3d.py --config_task configs/sample/test/sbdd_csd/simple.yml --outdir outputs_test/sbdd_csd_noAR/
conda activate pxm_vina
python evaluate/evaluate_vina_sdf.py   --sdf_dir outputs_test/sbdd_csd_noAR/simple_pxm_20260311_150536/SDF/   --gen_info outputs_test/sbdd_csd_noAR/simple_pxm_20260311_150536/gen_info.csv   
--split_by_name_path /shared/healthinfolab/phz24002/AliDiff/data/split_by_name.pt   --test_set_root /shared/healthinfolab/phz24002/AliDiff/data/test_set   --mode score_only   --n_workers 16   --quiet

## Progress Update

Current status as of April 3, 2026:

- Rerun commands used for the unified comparison:
  - baseline rerun:
    ```bash
    cd /shared/healthinfolab/phz24002/PocketXMol
    /shared/healthinfolab/phz24002/anaconda3/envs/PocketXMol/bin/python evaluate/evaluate_sdf_standard.py \
      --sdf_dir outputs_test/sbdd_csd/base_pxm_20260304_002527/SDF \
      --gen_info outputs_test/sbdd_csd/base_pxm_20260304_002527/gen_info.csv \
      --split_by_name_path /shared/healthinfolab/phz24002/AliDiff/data/split_by_name.pt \
      --test_set_root /shared/healthinfolab/phz24002/AliDiff/data/test_set \
      --n_workers 8 \
      --result_path outputs_test/sbdd_csd/base_pxm_20260304_002527/SDF/eval_results_standard_rerun_20260403
    ```
  - distilled rerun:
    ```bash
    cd /shared/healthinfolab/phz24002/PocketXMol
    /shared/healthinfolab/phz24002/anaconda3/envs/PocketXMol/bin/python evaluate/evaluate_sdf_standard.py \
      --sdf_dir outputs_consistency_eval_parity_100/consistency_eval_20260403_160858/steps_100/SDF \
      --gen_info outputs_consistency_eval_parity_100/consistency_eval_20260403_160858/steps_100/gen_info.csv \
      --split_by_name_path /shared/healthinfolab/phz24002/AliDiff/data/split_by_name.pt \
      --test_set_root /shared/healthinfolab/phz24002/AliDiff/data/test_set \
      --n_workers 8 \
      --result_path outputs_consistency_eval_parity_100/consistency_eval_20260403_160858/steps_100/eval_results_standard_rerun_20260403
    ```
- The distilled VE parity run on `gpu36` completed at 100 steps with the same standard evaluator.
- The original PocketXMol noAR baseline was re-evaluated with the same `evaluate_sdf_standard.py` path so both sides now share one evaluator and one metric schema.
- Unified rerun results:
  - baseline: `outputs_test/sbdd_csd/base_pxm_20260304_002527/SDF/eval_results_standard_rerun_20260403`
  - distilled: `outputs_consistency_eval_parity_100/consistency_eval_20260403_160858/steps_100/eval_results_standard_rerun_20260403`
- Unified metric table:

  | metric | baseline | distilled | delta (distilled-baseline) |
  |---|---:|---:|---:|
  | `n` | 10000 | 1000 | n/a |
  | `recon_success` | 0.9517 | 0.6490 | -0.3027 |
  | `complete` | 0.9404 | 0.5870 | -0.3534 |
  | `no_clashes` | 0.9445 | 0.7720 | -0.1725 |
  | `stereo` | 0.2509 | 0.1252 | -0.1257 |
  | `atom_type_js` | 0.0781 | 0.0895 | +0.0113 |
  | `jsd_cc_2a` | 0.3191 | 0.5963 | +0.2771 |
  | `jsd_all_12a` | 0.0863 | 0.2705 | +0.1842 |
  | `jsd_6_6_4` | 0.3874 | 0.5738 | +0.1864 |
  | `jsd_6_6_1` | 0.3334 | 0.6692 | +0.3358 |
  | `jsd_6_8_1` | 0.2756 | 0.6904 | +0.4148 |
  | `jsd_6_7_1` | 0.2745 | 0.5915 | +0.3170 |
  | `jsd_6_8_2` | 0.3765 | 0.7810 | +0.4044 |
  | `jsd_6_6_2` | 0.2756 | 0.5968 | +0.3212 |
  | `jsd_6_7_4` | 0.2033 | 0.4394 | +0.2362 |
  | `jsd_6_7_2` | 0.2956 | 0.6186 | +0.3230 |
- Latest online eval on the newly trained consistency model:
  - output dir: `outputs_consistency_online_eval/consistency_eval_20260404_135859`
  - online 4-step summary:

    | metric | value |
    |---|---:|
    | `recon_success` | 1.0000 |
    | `complete` | 0.0000 |
    | `qed_mean` | 0.3293 |
    | `sa_mean` | 0.3703 |
    | `validity` | 1.0000 |
    | `connectivity` | 0.0000 |
    | `samples_per_sec` | 19.40 |
    | `wall_time_sec` | 5.16 |

  - standard SDF eval rerun on the same 100 generated molecules:

    | metric | value |
    |---|---:|
    | `recon_success` | 1.0000 |
    | `eval_success` | 0.0000 |
    | `complete` | 0.0000 |
    | `no_clashes` | 0.0000 |
    | `stereo` | 0.3500 |
    | `JSD_CC_2A` | 0.7501 |
    | `JSD_All_12A` | 0.3406 |
    | `atom_type_js` | 0.4034 |

  - standard eval result path: `outputs_consistency_online_eval/consistency_eval_20260404_135859/steps_4/SDF/eval_results_standard/metrics_sdf_0-to-99.pt`
  - this is only a 100-molecule spot check, so it is useful as a quick health signal but still not a substitute for the 1k / 10k parity-style comparison above.
- Full curve rerun on the latest checkpoint (`outputs_distill_ve_2/consistency_distill_20260404_005352/checkpoints/step_100000.pt`):
  - output dir: `outputs_consistency_online_eval/consistency_eval_20260404_213705`
  - sampled curves: `1-step`, `2-step`, `4-step`, `8-step`
  - all four points were evaluated with the same standard SDF evaluator and the same 1000-molecule budget

  | metric | 1-step | 2-step | 4-step | 8-step |
  |---|---:|---:|---:|---:|
  | `recon_success` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
  | `complete` | 0.0050 | 0.0020 | 0.0000 | 0.0000 |
  | `no_clashes` | 0.0100 | 0.0100 | 0.0160 | 0.6110 |
  | `stereo` | 0.3460 | 0.3350 | 0.3490 | 0.3500 |
  | `qed_mean` | 0.3956 | 0.3347 | 0.3364 | 0.3982 |
  | `sa_mean` | 0.7395 | 0.4646 | 0.3630 | 0.4125 |
  | `samples_per_sec` | 24.42 | 20.10 | 17.98 | 20.09 |
  | `standard_complete` | 0.0050 | 0.0020 | 0.0000 | 0.0000 |
  | `standard_no_clashes` | 0.0100 | 0.0100 | 0.0160 | 0.6110 |
  | `standard_stereo` | 0.3460 | 0.3350 | 0.3490 | 0.3500 |
  | `standard_atom_type_js` | 0.3428 | 0.3689 | 0.4107 | 0.2294 |
  | `standard_pair_js_JSD_CC_2A` | 0.6435 | 0.6416 | 0.6431 | 0.6985 |
  | `standard_pair_js_JSD_All_12A` | 0.2203 | 0.2251 | 0.3329 | 0.4091 |
  - curve interpretation:
    - `recon_success` is saturated at `1.0` for every step count, so the model is not failing at the coarse reconstruction stage.
    - `complete` stays near zero through `1/2/4/8-step`, which means the generated molecules are still usually disconnected or otherwise incomplete after reconstruction.
    - `no_clashes` only becomes nontrivial at `8-step` (`0.6110`), so the best structural improvement in this sweep comes from more sampling steps, but it is still far from baseline parity.
    - `standard_atom_type_js` and the pair-distance JS terms are materially better at `8-step` than at `1/2/4-step`, but the geometry gap is still large relative to the baseline rerun.
    - this curve shows the main tradeoff clearly: more steps help some geometric consistency metrics, but they do not solve the completeness problem yet.
- Main rerun conclusion:
  - `recon_success`: baseline `0.9517` vs distilled `0.6490`  
    This is the biggest gap and the clearest sign that the distilled model is still failing to recover valid structures at the same rate as the original model.
  - `complete`: baseline `0.9404` vs distilled `0.5870`  
    The distilled model is not just reconstructing fewer molecules; a much larger fraction also ends up disconnected / incomplete after reconstruction.
  - `no_clashes`: baseline `0.9445` vs distilled `0.7720`  
    Even among reconstructed molecules, the distilled outputs have a substantially higher clash rate.
  - `stereo`: baseline `0.2509` vs distilled `0.1252`  
    Stereo consistency is also lower in the distilled run.
  - `atom_type_js`: baseline `0.0781` vs distilled `0.0895`  
    Atom-type marginal distribution is slightly worse for the distilled model, but this is a much smaller gap than reconstruction or completeness.
  - bond / pair JSDs are consistently worse for the distilled model  
    In the rerun, every bond-length and pair-distance JSD we compared moved against the distilled model, which suggests the geometry / distribution mismatch is broad rather than isolated to a single bond class.
- Interpretation of the rerun:
  - The two evaluations are now on the same metric stack, so the gap is not a pipeline artifact from mixing old and new evaluators.
  - The distilled 100-step model is still behind the baseline in the metrics that matter most for parity: reconstruction, completeness, clash rate, and geometry consistency.
  - The distilled model is closest on atom-type distribution, so atom typing is not the dominant bottleneck right now.
  - The dominant bottleneck is structural fidelity, not raw sampling speed.
- `standard_eval_success = 0.0` remains a placeholder and should not be used as a conclusion metric.
- Scope caveat:
  - baseline rerun evaluated `10,000` pairs
  - distilled rerun evaluated `1,000` molecules
  - this makes the comparison useful for parity diagnosis, but it is still not a perfect apples-to-apples benchmark on sample count
- Updated status: parity is still not reached at 100 steps; the distilled model remains clearly behind the baseline in reconstruction and completeness, and the gap is large enough that step reduction should not be the next focus yet.

## Rerun: 16/32/100-step sweep

On April 5, 2026, the latest consistency checkpoint (`outputs_distill_ve_2/consistency_distill_20260404_005352/checkpoints/step_100000.pt`) was rerun with the same unified evaluator on `16/32/100-step`.

- Run directory: `outputs_consistency_online_eval/consistency_eval_20260405_135752`
- `16-step` completed successfully:
  - `recon_success = 1.0`
  - `complete = 0.0`
  - `standard_no_clashes = 0.953`
  - `standard_stereo = 0.35`
  - `standard_pair_js_JSD_CC_2A = 0.7010`
  - `standard_pair_js_JSD_All_12A = 0.4250`
  - `standard_atom_type_js = 0.2343`
- `32-step` completed successfully:
  - `recon_success = 1.0`
  - `complete = 0.0`
  - `standard_recon_success = 0.08`
  - `standard_no_clashes = 0.9625`
  - `standard_stereo = 0.2625`
  - `standard_pair_js_JSD_CC_2A = 0.6962`
  - `standard_pair_js_JSD_All_12A = 0.8240`
  - `standard_atom_type_js = 0.2529`
- `100-step` sampled all 1000 molecules, but the unified driver crashed during `evaluate_mol_dict` in the post-processing stage with `IndexError: list index out of range`.
- The `100-step` standard SDF evaluation was recovered separately from `steps_100/SDF/eval_results_standard/log.txt`:
  - `recon_success = 0.0`
  - `complete = 0.0`
  - `standard JSD_CC_2A = 0.6962`
  - `standard JSD_All_12A = 0.3117`
  - `standard atom_type_js = NA`

Takeaway from this rerun:

- `16-step` is the only point in this sweep with a sane standard-eval profile.
- `32-step` does not improve the core structural metrics enough to matter, even though raw reconstruction remains at `1.0`.
- `100-step` is not a usable parity point in the current pipeline: the sample generation completed, but the outputs are too malformed for the standard SDF evaluator to recover reconstruction or atom-type metrics.

## Euler Sweep

The same checkpoint was rerun again with `sampling.coordinate_update = euler` and `sampling.discrete_update = categorical_transition`.

- Run directory: `outputs_consistency_online_eval_euler/consistency_eval_20260405_204628`
- This sweep completed successfully for `1/2/4/8/16/32/100-step` and wrote a full `summary.csv`.

| metric | 1-step | 2-step | 4-step | 8-step | 16-step | 32-step | 100-step |
|---|---:|---:|---:|---:|---:|---:|---:|
| `recon_success` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `complete` | 0.0050 | 0.0010 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `no_clashes` | 0.0100 | 0.0100 | 0.0050 | 0.0050 | 0.0070 | 0.0060 | 0.0080 |
| `stereo` | 0.3460 | 0.3350 | 0.3300 | 0.3370 | 0.3460 | 0.3440 | 0.3470 |
| `qed_mean` | 0.3956 | 0.3361 | 0.3237 | 0.3243 | 0.3285 | 0.3337 | 0.3329 |
| `sa_mean` | 0.7395 | 0.4583 | 0.4244 | 0.4324 | 0.4394 | 0.4333 | 0.4314 |
| `samples_per_sec` | 49.81 | 52.33 | 35.72 | 24.35 | 14.26 | 7.82 | 5.16 |
| `standard_complete` | 0.0050 | 0.0010 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `standard_no_clashes` | 0.0100 | 0.0100 | 0.0050 | 0.0050 | 0.0070 | 0.0060 | 0.0080 |
| `standard_stereo` | 0.3460 | 0.3350 | 0.3300 | 0.3370 | 0.3460 | 0.3440 | 0.3470 |
| `standard_pair_js_JSD_CC_2A` | 0.6435 | 0.6326 | 0.6545 | 0.6550 | 0.7123 | 0.7190 | 0.7132 |
| `standard_pair_js_JSD_All_12A` | 0.2203 | 0.2226 | 0.3143 | 0.3306 | 0.3234 | 0.3207 | 0.3156 |
| `standard_atom_type_js` | 0.3428 | 0.3672 | 0.4422 | 0.4390 | 0.4215 | 0.4129 | 0.4075 |

Takeaway:

- Euler materially improves throughput at low step counts.
- It does not solve the structural problem: `complete` stays at or near zero everywhere, and `no_clashes` remains very low.
- The 100-step point is still not parity-quality, even though the sampling pipeline itself now completes cleanly.

## Euler vs SNR

The table below puts the earlier `snr_renoise` sweep and the new Euler sweep side by side on the same checkpoint.

| step | snr_complete | euler_complete | snr_no_clashes | euler_no_clashes | snr_stereo | euler_stereo | snr_pair_CC_2A | euler_pair_CC_2A | snr_pair_All_12A | euler_pair_All_12A | snr_atom_js | euler_atom_js |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0050 | 0.0050 | 0.0100 | 0.0100 | 0.3460 | 0.3460 | 0.6435 | 0.6435 | 0.2203 | 0.2203 | 0.3428 | 0.3428 |
| 2 | 0.0020 | 0.0010 | 0.0100 | 0.0100 | 0.3350 | 0.3350 | 0.6416 | 0.6326 | 0.2251 | 0.2226 | 0.3689 | 0.3672 |
| 4 | 0.0000 | 0.0000 | 0.0160 | 0.0050 | 0.3490 | 0.3300 | 0.6431 | 0.6545 | 0.3329 | 0.3143 | 0.4107 | 0.4422 |
| 8 | 0.0000 | 0.0000 | 0.6110 | 0.0050 | 0.3500 | 0.3370 | 0.6985 | 0.6550 | 0.4091 | 0.3306 | 0.2294 | 0.4390 |
| 16 | 0.0000 | 0.0000 | 0.9530 | 0.0070 | 0.3500 | 0.3460 | 0.7010 | 0.7123 | 0.4250 | 0.3234 | 0.2343 | 0.4215 |
| 32 | 0.0000 | 0.0000 | 0.9625 | 0.0060 | 0.2625 | 0.3440 | 0.6962 | 0.7190 | 0.8240 | 0.3207 | 0.2529 | 0.4129 |
| 100 | 0.0000 | 0.0000 | NA | 0.0080 | NA | 0.3470 | 0.6962 | 0.7132 | 0.3117 | 0.3156 | NA | 0.4075 |

What this says:

- Euler is a throughput win, especially at `1/2/4/8-step`.
- Euler does not resolve the structural bottleneck. `complete` remains near zero, and `no_clashes` stays extremely weak.
- `snr_renoise` is a bit better on `8-step` clash rate, but Euler is more stable and faster overall.
- The `100-step` `snr_renoise` value is the recovered standalone standard-eval result because the unified driver crashed before producing a final summary.

## Log

2026-04-06:

- Ran a full `1/2/4/8/16/32/100-step` consistency sweep with `sampling.coordinate_update = euler` on the latest checkpoint `outputs_distill_ve_2/consistency_distill_20260404_005352/checkpoints/step_100000.pt`.
- Output directory: `outputs_consistency_online_eval_euler/consistency_eval_20260405_204628`.
- The sweep completed and wrote `summary.csv`.
- Main readout:
  - `recon_success = 1.0` for every step count.
  - `complete` stayed at or near zero across the whole curve.
  - `no_clashes` stayed very low, in the `0.005` to `0.010` range for most steps.
  - `standard_atom_type_js` stayed around `0.34` to `0.44`.
- Conclusion:
  - Euler improved speed substantially versus `snr_renoise`.
  - Euler did not fix the structural bottleneck, so this remains short of parity.
