# Stable Audio 3: Source Audio Blending Analysis

This document explores different methods to blend two audio sources (Source Audio 1 and Source Audio 2) inside the Stable Audio 3 local MLX diffusion pipeline. 

In Stable Audio 3, the Rectified Flow formulation maps a path between pure Gaussian noise ($\sigma = 1.0$) and clean latent representations ($\sigma = 0.0$). By introducing two audio sources, we can blend them at different stages of this mapping.

Below is an analysis of five distinct blending methodologies, their mathematical formulations, implementation viability in MLX, and qualitative trade-offs.

---

## 1. Blending Methodologies

### A. Pre-Encoder Waveform Blending (Source Mixing)
This method mixes the raw audio waveforms in the time-domain before they are passed into the SAME (Semantic-Acoustic Music Encoder) VAE.

#### Mathematical Formulation
Given waveforms $x_1(t)$ and $x_2(t)$, and a blending weight $w \in [0, 1]$:
$$x_{\text{blend}}(t) = (1 - w) \cdot x_1(t) + w \cdot x_2(t)$$

The blended latent code is then obtained by encoding:
$$z_{\text{blend}} = \text{Encoder}(x_{\text{blend}})$$

Noising is then applied normally during sampling setup:
$$z_{\text{start}} = z_{\text{blend}} \cdot (1 - \sigma_{\text{init}}) + \epsilon \cdot \sigma_{\text{init}}$$

#### Pros & Cons
*   **Pros**: 
    *   Trivial implementation. Zero backend modifications needed inside the sampling loops.
    *   No overhead; requires only one encoder forward pass.
*   **Cons**:
    *   Linear waveform mixing causes phase cancellation (comb filtering) and acoustic clutter (e.g. vocals overlapping vocals in a messy way) before the model can try to resolve them.
    *   The model treats the mixed input as a single static source rather than intelligently co-synthesizing features.

---

### B. Post-Encoder Latent Linear Blending (Feature Crossfade)
This method encodes both audios separately and interpolates their clean latents before noising them.

#### Mathematical Formulation
Encodes both inputs separately:
$$z_1 = \text{Encoder}(x_1), \quad z_2 = \text{Encoder}(x_2)$$

Linearly interpolates the latent states:
$$z_{\text{blend}} = (1 - w) \cdot z_1 + w \cdot z_2$$

Applies target noise level:
$$z_{\text{start}} = z_{\text{blend}} \cdot (1 - \sigma_{\text{init}}) + \epsilon \cdot \sigma_{\text{init}}$$

#### Pros & Cons
*   **Pros**:
    *   Very fast; requires two encoder passes (only once at the start of generation) but maintains standard sampling speed.
    *   Avoids time-domain phase cancellation.
*   **Cons**:
    *   Linear interpolation between points in high-dimensional latent space can result in out-of-distribution latents that sit in low-probability regions, causing decoded audio to sound washed-out or glitchy.

---

### C. Spherical Linear Interpolation (Slerp) in Latent Space
Instead of linear interpolation, Slerp (Spherical Linear Interpolation) is performed on the latent representations. This preserves the geometric structure (norms and angles) of high-dimensional vectors.

#### Mathematical Formulation
Given latents $z_1$ and $z_2$, we normalize and interpolate along the spherical arc:
$$\theta = \arccos\left(\frac{z_1 \cdot z_2}{\|z_1\| \|z_2\|}\right)$$
$$z_{\text{blend}} = \frac{\sin((1 - w)\theta)}{\sin(\theta)} z_1 + \frac{\sin(w\theta)}{\sin(\theta)} z_2$$

#### Pros & Cons
*   **Pros**:
    *   Preserves the energy distribution and spherical variance of the latent spaces.
    *   Results in significantly sharper, less muddy blends than linear latent mixing.
*   **Cons**:
    *   Slightly higher mathematical complexity but negligible computational overhead.

---

### D. Multi-Source Noising Stage Blending (Asymmetric Denoising)
Instead of blending clean states, we noise both inputs separately using potentially different noise schedules, and interpolate their noisy representations.

#### Mathematical Formulation
$$z_{\text{noisy}, 1} = z_1 \cdot (1 - \sigma_{\text{init}}) + \epsilon_1 \cdot \sigma_{\text{init}}$$
$$z_{\text{noisy}, 2} = z_2 \cdot (1 - \sigma_{\text{init}}) + \epsilon_2 \cdot \sigma_{\text{init}}$$

We can then blend the noisy starting states:
$$z_{\text{start}} = (1 - w) \cdot z_{\text{noisy}, 1} + w \cdot z_{\text{noisy}, 2}$$
Or we can share the same noise source ($\epsilon_1 = \epsilon_2 = \epsilon$) to prevent phase/variance issues:
$$z_{\text{start}} = ((1 - w)z_1 + w z_2) \cdot (1 - \sigma_{\text{init}}) + \epsilon \cdot \sigma_{\text{init}}$$

#### Pros & Cons
*   **Pros**:
    *   Allows us to use different noise profiles or noise levels for each source (e.g. keeping source 1 mostly intact with low noise while using source 2 as a highly-noised variation).
*   **Cons**:
    *   If using independent noise vectors ($\epsilon_1 \neq \epsilon_2$), blending them reduces the variance of the starting noise by a factor of $\sqrt{(1-w)^2 + w^2}$, violating the unit-variance assumption of Gaussian noise.

---

### E. Sampling-Step Vector Field Interpolation (Dynamic Path Negotiation)
This is the most advanced method. During each step of the diffusion sampling loop, we run the DiT transformer forward pass twice (once conditioned on Source 1, once on Source 2) and interpolate the predicted velocity/vector fields before taking the next integration step.

#### Mathematical Formulation
At step $t$ with latent state $x_t$:
1.  Predict velocity field for Source 1:
    $$v_1 = \text{DiT}(x_t, t, \text{Conditioning}_1)$$
2.  Predict velocity field for Source 2:
    $$v_2 = \text{DiT}(x_t, t, \text{Conditioning}_2)$$
3.  Blend the velocity vectors:
    $$v_{\text{blend}} = (1 - w) \cdot v_1 + w \cdot v_2$$
4.  Update the latent state:
    $$x_{t-1} = x_t - \Delta t \cdot v_{\text{blend}}$$

#### Pros & Cons
*   **Pros**:
    *   **Highest Quality**: The model dynamically co-synthesizes the features, finding a path that satisfies acoustic elements of both sources.
    *   Allows blending different prompts, or blending audio-to-audio conditioning dynamically at each step.
*   **Cons**:
    *   **Double Computation**: Requires 2 forward passes of the DiT model per step. Since the DiT model is the most expensive part of the pipeline, generation time will double.

---

## 2. Evaluation & Ranking Matrix

Here is the evaluation of the five methods, ranked by **Ease of Implementation** (how easy it is to write the code) and **Semantic Blending Quality** (how organically the model fuses structural features like rhythm, melody, and timbre, rather than just layering them).

| Methodology | Ease of Implementation | Semantic Blending Quality | Computational Cost | MLX Code Footprint |
| :--- | :---: | :---: | :---: | :--- |
| **A. Waveform Blending** | **1 (Easiest)** | **5 (Worst)** | Negligible (1x Encode) | Client-side JS or single line in backend upload |
| **B. Latent Linear Blending** | **2** | **4** | Negligible (2x Encode) | Minimal changes to `engine.py` |
| **C. Latent Slerp Blending** | **3** | **2** | Negligible (2x Encode) | Minimal changes to `engine.py` + `slerp` utility |
| **D. Asymmetric Noising** | **3.5** | **3** | Negligible (2x Encode) | Moderate modifications to noising logic |
| **E. Vector Field Interpolation** | **5 (Hardest)** | **1 (Best)** | Double (2x DiT steps) | Modifies main sampler loop in `sa3_pipeline.py` |

---

### Detailed Ranking & Rationale

#### Ease of Implementation
1.  **Waveform Blending (Rank 1)**: Can be implemented entirely on the frontend using standard Web Audio API before transmission, or in three lines of Python before calling the generator.
2.  **Latent Linear Blending (Rank 2)**: Requires sending two audio paths to the backend, loading the SAME encoder model, and running a simple linear interpolation: `z_blend = (1 - w) * z1 + w * z2`.
3.  **Latent Slerp Blending (Rank 3)**: Same integration footprint as Linear Blending, but requires writing a custom geometric slerp function in MLX (handling unit normalization and cosine angle clamps).
4.  **Asymmetric Noising (Rank 4)**: Requires passing multiple noise values through the API and modifying the initial noisy latent calculation to balance variances.
5.  **Vector Field Interpolation (Rank 5)**: Requires modifying the internal pingpong flow sampler loop (`sample_flow_pingpong` in `models/defs/sa3_pipeline.py`). We must change the loop to perform two forward passes (or batched passes) of the DiT model per step and average the vector velocities.

#### Semantic Blending Quality
1.  **Vector Field Interpolation (Rank 1)**: The absolute best for creative experiments. Because the paths are merged at each step, the neural network resolves structural compromises (e.g. keeping the tempo of Source 1 but using the instrument timbres of Source 2).
2.  **Latent Slerp Blending (Rank 2)**: The best value/performance compromise. Unlike linear blending, Slerp traverses the latent hypersphere along a constant-energy arc, keeping audio elements sharp and clear without volume loss.
3.  **Asymmetric Noising (Rank 3)**: Useful when you want to use one audio clip as a structural "mold" (low noise) and let another clip's features fill in the gaps (high noise).
4.  **Latent Linear Blending (Rank 4)**: Often leads to muddy results because linear averages of features sit in low-probability regions of the latents, confusing the decoder.
5.  **Waveform Blending (Rank 5)**: Has no semantic understanding. It is identical to playing two audio files at the same time on two separate audio tracks.

---

## 3. Recommendation & Implementation Strategy

For the local Stable Audio Workstation, we recommend offering **two blending modes** to the user:

1.  **Slerp Latent Blending (Default)**: Extremely fast, zero sampling overhead, and delivers clean morphs.
2.  **Vector Field Interpolation (High Quality)**: Doubled generation time, but results in high-quality hybrid syntheses where structures (like rhythms or melodies) merge intelligently.

### UI Interaction Flow
*   A toggle button above the Source Audio box: **[Single Source Mode]** vs **[Dual Source Blend]**.
*   In **[Dual Source Blend]** mode:
    *   The Left column displays two waveform boxes: **Source Audio 1** and **Source Audio 2**.
    *   A new slider is exposed: **Source Blend Ratio (A vs B)** from `100% Source 1` to `100% Source 2`.
    *   A selector dropdown for **Blend Mode**: `Fast (Slerp Latent)` vs `High-Quality (Vector Interpolation)`.
