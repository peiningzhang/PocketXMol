# Frozen PocketXMol test-time scaling

Branch: `codex/test-time-scaling`. Original teacher weights; no training.
Existing uncommitted distillation changes are preserved, not included in this experiment.

## Questions and metrics fixed before the pilot

Validity is a guardrail, not the main objective. The historical no-AR teacher has
validity around 0.90 and the existing refined teacher around 0.95. They use different
sampling paths: the historical `base.yml` uses repeated partial refinement, so its
“100 steps” is not 100 total model calls.

Primary endpoints:

1. Geometry pass: PoseBusters `mol` checks for bond lengths, bond angles, internal
   clashes, aromatic-ring flatness and double-bond flatness. Report each component
   separately. This fast subset excludes the 50-conformer energy check and is not
   called full PB-validity.
2. Qualified yield: sanitizable AND single fragment AND geometry pass AND no
   ligand–protein clash, divided by all attempted outputs. Pocket clash uses the
   repository's PoseBusters-derived heavy-atom check with cutoff 0.75.
3. As-generated Vina score (no coordinate minimization/redocking): mean/median and
   score coverage; physically qualified fraction with Vina <= -7. Ligand efficiency
   and heavy-atom count accompany Vina to expose molecular-size effects. The -7
   threshold is an exploratory fixed threshold, not an experimentally validated
   binding criterion.
4. Quality versus measured model calls and search time, including ranking
   verifiers; validity/complete, QED, SMILES/scaffold diversity are guardrails.

Independent energy diagnostic: official PoseBusters `mol` energy ratio using its
50-conformer ensemble and threshold 100, applied after selection to all pilot
methods and the historical baseline / geometry Best-of-4. Report continuous median
energy ratio, pass rate, coverage, and qualified+energy pass yield. Energy is never
used by these selectors.

Never optimize reference-ligand identity/stereochemistry for de-novo SBDD. The old
`stereo` score compares to a different reference molecule and is not an intrinsic
stereochemical quality score. Missing scores fail pass-rate metrics and their
coverage is reported; conditional means never replace unconditional yield.

## First methods

- Original sampler: 100, 200, 400 steps, no AR.
- Nested Best-of-N: first N=1/2/4 independent 100-step candidates per pocket.
  Confidence-only uses the repository trajectory confidence. Geometry+confidence
  first prefers qualified molecules, then number of geometry checks passed,
  connectivity and trajectory confidence. This is an explicit heuristic baseline.
- Vina-oracle Best-of-N: qualified candidates first, then lowest Vina. This
  optimizes its reported score and is an optimistic search reference, not
  independent evidence of affinity improvement.
- Native confidence-guided partial refinement: existing `base.yml`, original
  `maskfill` sampler and `ar_config.strategy=refine`, capped at 3 extra rounds.
  Actual forward calls are measured, including batch members already fixed.

This starts with existing code paths and simple search baselines. It is not a
claim to reproduce IPR or FK steering. IPR re-noises selected regions while
conditioning on the rest; FK steering resamples particles at intermediate times.
Their dedicated molecular adaptations can follow after rewards and budgets are
validated.

References: [IPR](https://arxiv.org/abs/2605.19317),
[FK steering](https://proceedings.mlr.press/v267/singhal25b.html),
[PoseBusters](https://posebusters.readthedocs.io/en/latest/).

## Execution

- GPU host: gpu35, existing Slurm job 31228047, allocated physical GPU 0 (A100).
- SSH uses `-F /dev/null` because the host system SSH config has a permissions error.
- GPU sampling uses PocketXMol; geometry uses the installed PoseBusters in
  reed_molecule; Vina uses pxm_vina. No environments are modified.
- Pilot: first 20 pockets, seed 2026, batch size 4. Candidate repeats have
  independent seeds 2026+r, reused across methods. Identical pockets do not imply
  identical noisy trajectories, especially between AR and no-AR transforms.
- Pilot root: `outputs_test_time_scaling/pilot_20261011`.
- Historical no-AR diagnostic: 100 pockets x first 4 candidates, output
  `outputs_test_time_scaling/historical_noar_100x4`; historical seed differs from
  the new pilot. This identifies headroom rather than a paired method comparison.

```bash
ssh -F /dev/null gpu35
cd /shared/healthinfolab/phz24002/PocketXMol
srun --jobid=31228047 --overlap -n 1 bash scripts/run_test_time_scaling.sh \
  outputs_test_time_scaling/pilot_20261011 20
```

Every sampled run saves SDFs, gen_info.csv, effective config and run.json with
actual NFE/timings. Geometry and search summaries are in tts_eval/. The pipeline
writes comparison.csv, report.md and COMPLETED only after all methods finish.
Scripts fail on existing sampling output directories to prevent overwriting.

## Historical 100-pocket results

Re-evaluated the first four candidates for each of 100 pockets from the historical
no-AR run. For the first candidate: validity 0.93, complete 0.91, geometry pass 0.88,
qualified yield 0.86. Across all 400 candidates, qualified yield is 0.7975. Thus
qualified yield has more room than validity, and conditional geometry metrics
should also be inspected to isolate chemistry from geometry.

| Selection | Qualified yield | Qualified and Vina <= -7 | Mean Vina | Score coverage |
|---|---:|---:|---:|---:|
| First candidate | 0.86 | 0.30 | -5.86 | 0.89 |
| Confidence Best-of-2 | 0.87 | 0.39 | -6.36 | 0.92 |
| Geometry+confidence Best-of-2 | 0.97 | 0.41 | -6.29 | 0.98 |
| Confidence Best-of-4 | 0.88 | 0.40 | -5.62 | 0.88 |
| Geometry+confidence Best-of-4 | 0.98 | 0.41 | -6.20 | 0.95 |
| Qualified-first Vina Best-of-4 | 0.98 | 0.51 | -7.48 | 0.99 |

The geometry selector never sees Vina, but its higher qualified yield also partly
reflects filtering invalid molecules. The Vina selector optimizes its reported
score. Confidence is not monotonic with independent quality as N increases.
Vina selection also increases mean heavy-atom count (21.43 first candidate vs
24.97 Vina Best-of-4); size-normalized ligand efficiency is in the CSV.
No historical runtime/NFE budget is claimed because it was not instrumented.

The held-out energy diagnostic is especially informative: the first candidate's
qualified+energy yield is **0.77**, compared with **0.83** after geometry Best-of-4.
Energy pass rates over all attempts are 0.82 and 0.85; energy coverage is 0.93 and
0.99. The median energy ratio decreases from 5.77 to 4.86. Thus 0.98 qualified yield
after geometric selection is not 0.98 full physical plausibility. For subsequent
methods, prioritize qualified+energy yield and qualified+energy+Vina threshold
yield, with independent component metrics. This energy diagnostic was added after
the first geometric pilot result, and its thresholds were taken from the installed
PoseBusters configuration rather than tuned to these outputs.

For the historical first-candidate / geometry Best-of-4 comparison, the joint
qualified+energy+Vina<=-7 yield is **0.25 / 0.35** over all 100 pockets. This is the
more demanding endpoint for subsequent test-time scaling experiments.

## Completed new 20-pocket pilot

All methods finished on gpu35. Total: 140 newly generated candidates, 28000
graph-forward evaluations. Native refinement used exactly 400 calls per candidate
(100 initial + 3 x 100 refinement), so it can be compared to the 400-call baselines.

| Method | NFE per output | Qualified + energy | Qualified + energy + Vina <= -7 | Approx. search seconds for 20 outputs |
|---|---:|---:|---:|---:|
| First 100-step candidate | 100 | 0.80 | 0.25 | 25.5 |
| 200-step candidate | 200 | 0.60 | 0.20 | 50.5 |
| Geometry+confidence Best-of-2 | 200 | 0.90 | 0.35 | 54.8 |
| 400-step candidate | 400 | 0.65 | 0.10 | 99.8 |
| Native partial refinement, 3 rounds | 400 | 0.70 | 0.10 | 103.4 |
| Confidence Best-of-4 | 400 | 0.65 | 0.30 | 101.8 |
| Geometry+confidence Best-of-4 | 400 | 0.90 | 0.40 | 109.0 |
| Qualified-first Vina Best-of-4 | 400 | 0.95 | 0.55 | 184.0 |

Energy is a held-out measurement for all these selectors. Vina is held out for
confidence/geometry selection, but not for the Vina selector. Search seconds
amortize shared generation costs and include selection verifier costs, excluding
offline held-out evaluation and environment/model startup. These are exploratory
quality/budget measurements, not latency benchmarks or statistically conclusive
improvements. The 100-pocket historical diagnostic provides broader headroom
evidence but uses a different seed and lacks instrumented generation timing.

Results: `outputs_test_time_scaling/pilot_20261011/comparison.csv` and `report.md`.
Protocol checks passed: Python compilation, shell syntax, exact forward counting
in a 2-pocket/10-step smoke run, and nested-selection/denominator checks. No
environment dependencies were installed; Markdown reporting uses a small native
formatter because tabulate is absent in the evaluation environment.

## Interpretation rules

- 20-pocket pilot is for detecting large effects and failures, not significance.
- Compare Best-of-2 against 200 steps and Best-of-4 against 400 steps for the same
  nominal NFE; inspect actual refinement NFE and verifier time before comparison.
- Improvements in a metric used for ranking are expected selection effects.
  Check Vina, size and diversity that were not used by the geometry selector.
- Only move to internal looping/adaptive allocation once at least one measured
  metric has clear headroom and the simple baselines establish a useful frontier.
