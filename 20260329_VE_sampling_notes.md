# VE Distillation Sampling Notes

This note describes the current sampling logic used by the distilled VE consistency model.

## High-level picture

The sampler is not a standard diffusion sampler such as Euler / Heun / DDIM / ancestral sampling.
It is a **consistency-style predict-`x0`-then-re-noise loop**:

1. Start from a noisy prior state for positions and categorical priors for node / edge types.
2. Pick a small set of decreasing sampling timesteps, e.g. `1`, `2`, `4`, `8`.
3. At each step, feed the current noisy state to the model.
4. The model predicts `x0`-style outputs:
   - `pred_pos`
   - `pred_node`
   - `pred_halfedge`
5. If this is not the final step, re-noise the predicted `x0` to the next timestep.
6. At the last step, decode the predicted `x0` directly.

The core sampling loop is in [`scripts/eval_consistency.py`](./scripts/eval_consistency.py:108).

## What the sampler feeds into the model

At each timestep `t_cur`, the sampler prepares:

- `node_in = node_state`
- `halfedge_in = edge_state`
- `pos_in = pos_state * scale(t_cur)`

The scaling factor is:

`scale(t) = 1 / sqrt(1 + sigma(t)^2)`

This is implemented in [`utils/consistency.py`](./utils/consistency.py:142) and used in [`scripts/eval_consistency.py`](./scripts/eval_consistency.py:122).

This is important:

- The **VE state** itself is still `x = x0 + sigma * eps`.
- The model input is **scaled** to match the magnitude expected by the pretrained `x0`-prediction teacher.
- This is a compatibility transform, not a sampler update rule.

## Coordinate update

### Short answer

No, coordinate sampling is **not Euler**.

### What actually happens

For positions, the sampler uses a VE forward noising rule:

`x_t = x0 + sigma_t * eps`

At the first step, the sampler initializes:

`pos_state ~ N(0, sigma_max^2 I)`

from [`utils/consistency.py`](./utils/consistency.py:178).

At intermediate steps, after the model predicts `pred_pos_x0`, the code re-noises it to the next timestep:

`pos_next = pred_pos_x0 + sigma_next * eps`

This is done by [`renoise_from_pred_x0`](./utils/consistency.py:191) and the underlying VE transition in [`models/transition.py`](./models/transition.py:19).

### Interpretation

This is a **forward re-noising step**, not an ODE step.
There is:

- no Euler drift term
- no learned velocity field integration
- no explicit score-based ODE solver

### `euler` vs `snr_renoise`

| Mode | Input to update | Update rule | Noise injected? | Interpretation |
| --- | --- | --- | --- | --- |
| `snr_renoise` | predicted `x0` | `x_next = x0_pred + sigma_next * eps` | Yes, fresh Gaussian noise at every step | VE-style re-noising / consistency loop |
| `euler` | current `x_t` and predicted `x0` | `eps_hat = (x_t - x0_pred) / sigma_t`, then `x_next = x_t + (sigma_next - sigma_t) * eps_hat` | No fresh noise in the coordinate update itself | Deterministic Euler step on the VE path |

Practical difference:

- `snr_renoise` re-centers every step around `x0_pred` and keeps stochasticity.
- `euler` propagates the current noisy state forward deterministically once `x0_pred` is known.

So if you want to classify it, it is closer to:

- consistency distillation with re-noising
- not classical VE SDE sampling
- not Euler / Heun / ancestral update

## Discrete feature update

### Short answer

Discrete node and edge features are **not** updated by simple proportional replacement.
They are sampled from a **discrete diffusion / finite-state Markov chain** transition.

### What the code does

The model predicts logits for:

- node types: `pred_node`
- halfedge types: `pred_halfedge`

Then, for non-final steps, the sampler calls:

- `transitions["node"].q_vt_sample(...)`
- `transitions["edge"].q_vt_sample(...)`

from [`utils/consistency.py`](./utils/consistency.py:202) and [`utils/consistency.py`](./utils/consistency.py:205).

These methods sample from the forward marginal `q(v_t | v0)` using the transition matrices in [`models/transition.py`](./models/transition.py:161).

### Is this CTMC?

No.

This is a **discrete-time Markov chain** with a schedule of transition matrices built from `beta_t`.
The code does not implement a continuous-time Markov chain sampler.

The transition structure is defined by:

- `q(v_t | v_{t-1})`
- `q(v_t | v0)`
- `q(v_{t-1} | v_t, v0)`

in [`models/transition.py`](./models/transition.py:87) and [`models/transition.py`](./models/transition.py:118).

### Practical meaning

For each step:

- the model predicts an `x0`-style categorical distribution
- the sampler draws a new categorical state from the diffusion transition

So the update is stochastic and transition-based, not a deterministic “replace X% of tokens” rule.

## Is there noise injection during sampling?

Yes.

### Position noise

Positions get Gaussian noise at:

- initialization: `randn * sigma_max`
- every intermediate re-noising step: `x0 + sigma_next * eps`

### Discrete noise

Discrete node / edge states are re-sampled stochastically through the transition distribution.

### Final step

On the final step, there is no extra re-noising:

- node = `argmax(pred_node_logits_x0)`
- edge = `argmax(pred_halfedge_logits_x0)`
- pos = `pred_pos_x0`

This is in [`scripts/eval_consistency.py`](./scripts/eval_consistency.py:129).

## Exact step schedule

Sampling timesteps are built by:

`np.linspace(total_steps - 1, 0, sample_steps)`

with rounding and deduplication.

This is implemented in [`utils/consistency.py`](./utils/consistency.py:163).

So `sample_steps = 4` means the sampler uses a small descending subset of the full noise ladder, not four Euler steps on a continuous trajectory.

## Important code paths

- [`scripts/eval_consistency.py`](./scripts/eval_consistency.py:108)
- [`utils/consistency.py`](./utils/consistency.py:142)
- [`utils/consistency.py`](./utils/consistency.py:178)
- [`utils/consistency.py`](./utils/consistency.py:191)
- [`models/transition.py`](./models/transition.py:9)
- [`models/transition.py`](./models/transition.py:55)

## Configurable update modes

The current default modes are:

- `coordinate_update: snr_renoise`
- `coordinate_update: euler` means use the current noisy coordinate `x_t` and the predicted `x0` to take a direct Euler step to the next sigma, without re-noising from `x0`.
- `discrete_update: categorical_transition`
- `discrete_update: campbell_dfm` means sample a clean categorical proposal from the predicted logits, then replace a subset of current tokens with that proposal using a Campbell-style stochastic mask/unmask rule.

These are now exposed in the distillation sampling config so evaluation and generation can read them directly from the saved config snapshot.

## What are the `betas` here?

The `betas` used for node and edge features are **not** continuous-time Markov rates.

They are derived from the VE `sigma` schedule by:

1. mapping `sigma` to an implied VP-style cumulative signal level
   `alpha_bar = 1 / (1 + sigma^2)`
2. converting consecutive `alpha_bar` values into per-step `alpha_t`
3. defining `beta_t = 1 - alpha_t`

This happens in [`utils/consistency.py`](./utils/consistency.py:94).

So the role of `beta_t` is:

- a **discrete-time transition strength** for the categorical chain
- not a CTMC generator / rate
- not the continuous VE noise level itself

In other words, `sigma` drives the position VE process, and the same `sigma` schedule is reused to fabricate a compatible discrete diffusion schedule for node / edge transitions.

## Bottom line

- Coordinates: **VE Gaussian re-noising**, not Euler.
- Node / edge: **discrete diffusion transitions**, not simple ratio replacement.
- Sampling noise: **yes**, both continuous and discrete.
- Overall sampler: **consistency distillation with predict-`x0` and re-noise**, not a standard diffusion solver.
