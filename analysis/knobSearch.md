# Stable Audio 3: Search for Audio Manipulation Affordances ("Knob Search")

This document surveys **every point in the local MLX pipeline where generated audio can be
steered**, records which of them we currently expose as controls, and tabulates the ones we
are not yet using so we can revisit them deliberately rather than rediscovering them.

The motivating question was whether Stable Audio 3 has an intermediate **codebook**
representation — as EnCodec, DAC or SoundStream do — that we could manipulate directly.
Section 1 answers that. Sections 2–6 map the surfaces that do exist.

---

## 1. There Is No Codebook

A search of the entire repository for `codebook`, `vector_quant`, `residual_vq`, `rvq`,
`fsq`, `lookup_free`, and the EnCodec/DAC/SoundStream lineage returns **zero matches**.

SAME (Semantic-Acoustic Music Encoder) is a **continuous** VAE. Its entire bottleneck is an
affine transform followed by a division by a running standard deviation
([`stable_audio_3/models/bottleneck.py:4`](../stable_audio_3/models/bottleneck.py)):

```python
x = x * self.scaling_factor + self.bias
x = x / self.running_std
```

The MLX ports carry the same three tensors (`scaling_factor`, `bias`, `running_std`) at the
tail of both encoders.

### What this rules out

Because there are no discrete tokens, none of the codec-style manipulations are available,
and none can be retrofitted without retraining the autoencoder:

* swapping or substituting codebook entries
* token-level splicing or transposition
* dropping RVQ stages to trade quality for bitrate
* discrete token "style transfer" or token-sequence language modelling

### What this enables instead

Continuous latents support operations that discrete tokens cannot do smoothly:
arithmetic, interpolation along a path, and filtering. We trade token surgery for calculus.
Every affordance below follows from that.

---

## 2. Latent Geometry (why the numbers matter)

| Property | Value |
| :--- | :--- |
| Sample rate | 44 100 Hz |
| Samples per latent frame | 4 096 |
| **Latent frame rate** | **10.767 Hz** (92.9 ms per frame) |
| Latent dimensionality | 256 channels |
| 1 s clip | 11 frames × 256 |
| 5 s clip | **54 frames** × 256 |
| 10 s clip | 108 frames × 256 |

A five-second clip is **54 vectors**. This is small enough to manipulate directly, to
visualise, and even to hand-edit. It also fixes the *control rate* for every per-frame
affordance below at roughly 11 Hz — slow enough to draw envelopes over, fast enough to feel
gestural in performance.

---

## 3. What We Currently Expose

The live app and the studio between them drive only the outermost layer of the pipeline.

| Control | Where it acts | Code |
| :--- | :--- | :--- |
| Prompt / negative prompt | T5Gemma cross-attention | `engine.py:283` |
| Init audio + `init_noise_level` | start point of the sampling trajectory | [`engine.py:399`](../optimized/mlx/webapp/backend/engine.py) |
| `steps` | number of ping-pong sampler steps | `sa3_pipeline.py:104` |
| `cfg` | classifier-free guidance scale | `engine.py:441` |
| Inpaint range (binary) | `local_add_cond` mask | [`engine.py:411`](../optimized/mlx/webapp/backend/engine.py) |
| Variant ladder (fixed +0.02) | repeated runs at stepped noise | `live.py` |

Everything in sections 4–6 is **currently unused**.

---

## 4. Conditioning Surfaces

The DiT accepts conditioning at three structurally different injection points
([`dit_mlx.py:232`](../optimized/mlx/models/defs/dit_mlx.py)). We use a fraction of each.

### A. Cross-attention context — `[B, 257, 768]`

The T5Gemma prompt embedding concatenated with the seconds embedding, cross-attended at
**every** transformer block.

* **Prompt strength decoupled from CFG.** Blend the prompt embedding toward the learned
  `padding_embedding` (`apply_prompt_padding`, [`sa3_pipeline.py:59`](../optimized/mlx/models/defs/sa3_pipeline.py)).
  CFG amplifies the *difference* between the conditional and unconditional predictions;
  blending toward padding weakens the *conditioning signal itself*. These are orthogonal
  axes with different character.
* **Per-step prompt scheduling.** Use prompt A during early high-noise steps, which settle
  structure, and prompt B during late steps, which settle timbre. The sampler already calls
  an arbitrary `model_fn` per step, so this costs no extra forward passes.
* **Token-level masking** within the prompt sequence.

### B. Global conditioning — `[B, 768]`, adaLN modulation

The highest-leverage hidden surface. It is projected to `6 × EMBED_DIM` and split into
`scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff`, modulating **every
block's** attention and feed-forward ([`dit_mlx.py:139`](../optimized/mlx/models/defs/dit_mlx.py)).

Today it carries only `seconds_total` (`engine.py:302`). Which means:

> **We can lie about the duration.** Telling the model it is producing 380 seconds while
> generating 5 changes how it paces and structures the material — a "structural scale" knob
> that is already fully plumbed.

**Caveat.** `global_embed = global_pre + t_embed` ([`dit_mlx.py:252`](../optimized/mlx/models/defs/dit_mlx.py)):
the seconds embedding is *summed with the diffusion timestep* before projection. Lying
perturbs the same vector the model uses to know how noisy it currently is. Mild lies should
be musical; extreme ones may destabilise sampling. Worth sweeping carefully.

### C. Local additive conditioning — `[B, T_lat, 257]`

`concat(mask, masked_latents)`, embedded per-layer via `to_local_embed`
([`dit_mlx.py:136`](../optimized/mlx/models/defs/dit_mlx.py)). We use it only for binary
inpainting: `1 = keep`, `0 = regenerate` ([`engine.py:411`](../optimized/mlx/webapp/backend/engine.py)).

**Nothing requires the mask to be binary.** A fractional value means "partially adhere to
the source", and the mask is **per latent frame**. Instead of one global noise scalar we
could draw an *envelope* at ~11 Hz: hold the intro, dissolve through the middle, fully
regenerate the end.

This is the single richest unexploited affordance in the codebase, because it converts our
most important control from a scalar into a gesture.

---

## 5. Sampling-Time Surfaces

| Surface | Current state | Opportunity |
| :--- | :--- | :--- |
| `apg` (Adaptive Projected Guidance) | real engine parameter, **hardcoded to 1.0** in [`live.py:258`](../optimized/mlx/webapp/backend/live.py) | expose it — zero implementation cost |
| `anchor_logsnr = -6.2`, `logsnr_end = 2.0` | hardcoded in [`sa3_pipeline.py:75`](../optimized/mlx/models/defs/sa3_pipeline.py) | decides *where steps land* in noise space; reallocates effort between structure and detail at **fixed compute** |
| Ping-pong churn | fixed by the schedule: `x = (1-t_next)·denoised + t_next·noise` ([`sa3_pipeline.py:132`](../optimized/mlx/models/defs/sa3_pipeline.py)) | scale the injected noise independently — a texture/variation control nothing else provides |
| CFG trajectory | constant across steps | ramp it (high early, low late) for a markedly different character |
| Unconditional branch | `mx.zeros_like(cross_attn)` when no negative prompt ([`engine.py:316`](../optimized/mlx/webapp/backend/engine.py)) | the padding embedding, not zeros, is what makes an empty prompt behave; worth an A/B since the studio defaults to `cfg = 4.0` |

---

## 6. Latent-Space Surfaces

Operating directly on `z: [B, 256, T_lat]` between encode and sample.

| Operation | Effect | Confidence |
| :--- | :--- | :--- |
| Time reverse | plays the semantic content backwards | high |
| Time-stretch (resample along `T`) | changes pace without pitch shift, since latents are semantic | high |
| Freeze / hold a frame | sustains a moment | high |
| Loop a sub-segment | rhythmic repetition at ~93 ms granularity | high |
| Low-pass along time | smears transients → more ambient | medium |
| High-pass along time | emphasises onsets and change | medium |
| Slerp / linear blend between two clips | morphing (see `audio_blending_analysis.md`) | high |
| Latent arithmetic (A − B + C) | analogy-style transfer | low, untested |
| **Pitch probe direction** | SAME is trained with a pitch regressor (§ Part II.1) — a linear probe should find it | **medium–high** |
| **Stereo-image probe direction** | likewise explicitly regressed during training | **medium–high** |
| **Per-channel gains** | unknown; 256 dims are unlabelled | **low — see below** |
| **Global norm scaling** | fights the softnorm's expected scale | **low — see below** |

---

## 7. Ranked Shortlist

Ordered by expected musical payoff per unit of implementation cost.

| # | Affordance | Cost | Rationale |
| :---: | :--- | :---: | :--- |
| 1 | Soft, time-varying inpaint mask | small | scalar → 11 Hz envelope; the biggest expressive jump available |
| 2 | Expose `apg` | trivial | already a parameter, merely hardcoded |
| 3 | `seconds_total` lie | trivial | structural pacing, already plumbed |
| 4 | Prompt ↔ padding blend | small | prompt strength orthogonal to CFG |
| 5 | Ping-pong churn scale | small | texture knob nothing else gives |
| 6 | `anchor_logsnr` | small | reallocates steps at fixed compute |
| 7 | Per-step prompt schedule | medium | splits structure from timbre |
| 8 | Latent time ops (reverse / stretch / freeze / loop) | medium | 54 frames is very tractable |

---

## 8. Where To Be Sceptical

Two commonly-suggested manipulations are **not** recommended without groundwork:

* **Per-channel latent gains.** The 256 dimensions are unlabelled. Without knowing which
  (if any) correspond to perceptual attributes, scaling them individually is a random walk.
* **Global latent norm scaling.** The softnorm bottleneck means the decoder expects a
  specific scale. Pushing on it likely leaves the learned manifold and produces glitch
  rather than anything musical.

### Prerequisite research

Both become tractable after one analysis pass, which is cheap precisely because the tensors
are so small: **encode a corpus, run PCA over the latents, and test whether any directions
track brightness, density, or onset rate.** If stable semantic directions exist, per-channel
control becomes a real instrument rather than a lottery. If they do not, we drop both ideas
with evidence instead of intuition.

---

## 9. Summary

There is no codebook to manipulate, and there never will be without retraining the
autoencoder. What we have instead is a continuous 256-dimensional latent at ~11 Hz, three
distinct conditioning injection points, and a sampler whose schedule constants are currently
hardcoded. **We are presently steering only the outermost layer of that stack.**

The nearest large win is making the inpaint mask soft and time-varying, which turns our most
important control from a number into a drawn gesture at the latent frame rate.

---
---

# Part II — Steering Generated Audio Toward a Target Audio File

Part I asked *what can we turn*. Part II asks a narrower question: **can the live output,
which is driven by the input audio stream, be pulled toward the characteristics of a chosen
target audio file?** In other words, is there arithmetic — in the audio domain or in any of
the representations above — that transfers a reference track's character onto what we
generate?

## II.1 The Constraint and the Opening

### The constraint: no audio-reference conditioning exists

The Diffusion Transformer accepts exactly three conditions
([`docs/guides/model-overview.md`](../docs/guides/model-overview.md)):

1. **Text** — via T5Gemma
2. **Duration** — sinusoidal embedding of total length
3. **Inpainting** — a SAME-encoded clip with start/end times

The conditioner module confirms this: the only classes are `NumberConditioner` and
`T5GemmaConditioner` ([`stable_audio_3/models/conditioners.py`](../stable_audio_3/models/conditioners.py)).

> **There is no trained pathway that accepts an audio clip as a *style reference*.**
> Every technique below is off-label. The model will not have been optimised for it, and
> results may be unstable. Expectations should be set accordingly.

### The opening: the latent space is cross-modally aligned

SAME was **not** trained for reconstruction alone. Two of its five training losses matter
enormously here:

* **Contrastive Latent** — "a text/audio contrastive critic that encourages the latents to
  encode rich, cross-modal meaning"
* **Semantic Regression** — "small regression models for **pitch** and **stereo image**"

So the latent space is deliberately structured to carry semantic, text-aligned meaning, and
at least two perceptual attributes are *known to be regressable from it*. This is far better
ground for latent arithmetic than a plain reconstruction VAE. The DiT additionally received a
**CLAP loss** during adversarial post-training, pushing its outputs toward text alignment.

The practical consequence: **semantic structure in the latent space is a design goal of this
model, not an accident we are hoping for.**

---

## II.2 Candidate Mechanisms

### A. AdaIN / moment matching in latent space — *run this first*

Take per-channel statistics over time and transplant them:

$$z_{\text{out}} = \sigma_B \odot \frac{z_A - \mu_A}{\sigma_A} + \mu_B$$

Normalising strips A's channel statistics while preserving its **temporal dynamics**;
re-scaling imposes B's. Interpolating rather than replacing gives a continuous control:

$$\mu = (1-\lambda)\mu_A + \lambda\mu_B, \qquad \sigma = (1-\lambda)\sigma_A + \lambda\sigma_B$$

**Why this is the first experiment.** It can be tested with **no diffusion at all**: encode
A, apply AdaIN toward B, decode, listen. That is milliseconds of compute and a few dozen
lines. It decisively answers whether this latent space supports style transfer, and gates
every other idea here.

**The upgrade.** Feed the AdaIN'd latent in as `init_audio` at moderate noise. The diffusion
model then acts as a **projector back onto the data manifold**, repairing whatever went
off-distribution while retaining the imposed statistics. This requires no new plumbing — it
is exactly the existing audio-to-audio path.

### B. Gradient guidance toward the target — *most controllable*

Nudge each sampling step toward B:

$$v' = v + \lambda \nabla_x \mathcal{L}\big(x - t\cdot v,\; B\big)$$

where $\mathcal{L}$ is a statistical distance computed **in latent space**, so neither a
decode nor a backprop through the DiT is required. MLX provides `mx.grad` and
`mx.value_and_grad`, both verified working in the project venv.

Unlike (A) this assumes nothing about linear disentanglement — it is optimisation, applied
gradually across steps, with λ as a true continuous dial that self-corrects each iteration.
It fits the existing `model_fn` seam in `sample_flow_pingpong` with no sampler rewrite.
Expected cost is roughly +30–60 % per step.

### C. Textual inversion — *likely best sounding, most work*

Optimise a pseudo-token embedding $e^*$ such that the model reproduces B, then blend it with
the live prompt at runtime:

$$e = (1-\lambda)\,e_{\text{prompt}} + \lambda\,e^*$$

This speaks the model's **native, trained control language**, so output should stay musically
coherent rather than drifting into artifacts. The cost is an offline optimisation per
reference track — which is free in practice, because performance targets are known in
advance.

### D. Caption the target and use it as the prompt — *the baseline*

Run a music tagger on B, or simply describe it. Native pathway, zero engineering. **Any
fancier method must be shown to beat this**, and it may well prove sufficient.

### E. `local_add_cond` as a reference channel — *low confidence*

The only trained audio-input path, but it was trained for **temporal** inpainting: real
latents in the keep region, fill the gap. Supplying B as "context" while the trajectory comes
from A is a train/test mismatch the model never saw. Cheap to try, since the plumbing exists;
not worth betting on.

### F. CLAP bridge — *the "proper" solution, out of scope*

Embed B with CLAP and project into the T5Gemma conditioning space. Since SA3 used a CLAP loss
in post-training, such a bridge is conceptually well-founded — but the repo ships neither a
CLAP encoder nor a projection, so this is a training project rather than a knob. Recorded
here as what a purpose-built solution would look like.

---

## II.3 The Answer Depends on Which Characteristics

| Target attribute | Best mechanism | Confidence |
| :--- | :--- | :---: |
| Timbre, texture, production | AdaIN latent statistics (A) | medium–high |
| Genre, mood, overall "vibe" | textual inversion (C) or prompt (D) | high |
| **Pitch / key** | linear probe direction — explicitly regressed in training | medium–high |
| **Stereo image / width** | linear probe direction — likewise | medium–high |
| **Rhythm, groove, tempo** | **none of the above** | **low** |

The final row is the important honesty. Every statistics-based method **time-averages**, and
therefore carries *no rhythmic information whatsoever*. "Make my guitar carry this track's
groove" is not reachable by A, B or C; it would require beat-aware alignment that does not
exist in this codebase.

The pitch and stereo-image rows are the inverse — a gift from the training recipe. Because
Stability explicitly trained regressors for those attributes off the latents, linear probes
are likely to find them, and they could become well-founded knobs rather than guesses.

---

## II.4 The Live-Performance Pattern

For live use the target is **known in advance**, which changes the economics completely:

> **Precompute a target descriptor offline; expose a single λ knob at runtime.**

Whether the descriptor is a pair of statistics vectors (A), a loss target (B), or an
optimised embedding (C), all the expensive work happens before the performance. At runtime
only a cheap blend remains, well inside the live loop's budget.

---

## II.5 Recommended Order of Work

1. **The AdaIN decode test.** No diffusion, no UI. Answers "does this latent space support
   style transfer at all" in an afternoon, and gates everything else.
2. If it passes, wire λ into the existing `init_audio` path — the diffusion step doubles as
   the manifold projector, for free.
3. **In parallel, run (D)** as the baseline, so we know what any fancier method must beat.
4. Only then consider (B) or (C), which are genuine projects rather than knobs.
