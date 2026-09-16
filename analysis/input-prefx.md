# Input pre-FX

Processing that runs on an input box **before the ring buffer**, so Stable Audio
conditions on the processed signal rather than the raw feed. This is the
distinction from the output filters (LPF / HPF / Comb), which shape only what
you hear and never reach the model.

Code lives in `optimized/mlx/webapp/frontend/src/live/inputfx/`.

---

## 1. The goal, stated as signal processing

The brief was: *spread every chunk of the input across the spectrogram, keep
pitch and timbre, lose the attack entirely.* Written as a constraint on the
signal that is:

> **preserve the magnitude spectrum, destroy the temporal envelope.**

That framing sorts the options along one sharp line.

**Anything LTI** — any convolution, including all of Smear and Disperse — takes
the input spectrum and multiplies it by `H(f)`. If `|H(f)| = 1` at every
frequency the magnitude spectrum is preserved *exactly* and only the phase
changes; and phase is precisely where time structure lives. The attack is still
mathematically present after such a filter — an inverse filter would recover it
— it simply no longer *sounds* like an attack, and no longer looks like one on
a spectrogram.

**Spectral magnitude smoothing** (Paulstretch, phase vocoders) is not LTI. It
flattens each frequency bin's envelope across frames, so the attack is genuinely
deleted rather than hidden.

For our purposes — what a listener hears, and what a diffusion model conditions
on — a few hundred milliseconds of LTI smear is indistinguishable from the
non-LTI version, at a fraction of the code. That is why the shipped methods are
LTI (or nearly so) and the FFT approach stayed on the shelf.

### Why symmetry matters

A **decaying** impulse response is reverb: the onset survives intact and a tail
follows it. You still hear the hit.

Removing an attack means spreading energy **backwards as well as forwards** —
and backwards is lookahead, which is latency. This is not an implementation
detail that can be optimised away; it is the trade. Every method's latency is
reported in the rack next to the knob, in amber, and the model's input window is
that much older as a result.

---

## 2. Shipped methods

The dropdown is the method; the knob is the amount. Amount `0` is a true bypass
in every method — implementations tear their guts down rather than run them at
zero mix, since an idle convolver still costs CPU on every box.

### Smear — convolution with an enveloped noise burst

`ConvolverNode` against a self-generated impulse response: white noise shaped by
a **Gaussian centred on the impulse**, with ±3σ spanning the buffer so it does
not end on a step. Each input sample is splattered across T seconds. Noise is
spectrally flat on average, so pitch and timbre survive; a drum hit becomes a
noise-shaped swell.

| | |
|---|---|
| Knob | T, exponential, 20 ms → 1.2 s |
| Latency | T / 2 (600 ms at the top) |
| Mix | equal-power dry/wet across the whole throw |
| Cost | one `ConvolverNode` (partitioned FFT convolution, native) |

The impulse response is normalised to unit **energy**, not unit peak. Energy is
the right invariant: sustained material comes out at the level it went in, while
transients correctly lose their peak — that loss is the effect working. A unit
*peak* normalisation would make every smear setting quieter than the last.

Two channels of independent noise, so the result is a wide diffuse field rather
than a point source.

Rebuilding a second-long impulse response is not something to do per mouse move,
so the **length follows the knob on a 150 ms debounce** while the dry/wet mix
tracks it instantly. Moving the knob therefore feels immediate and the texture
settles a moment later. A length change under 5% is ignored outright.

This is the most thorough smear of the three. An impulse comes out as the impulse
response itself, whose unit-energy samples peak at roughly 3/√N for an N-sample
Gaussian, a few percent of the input's peak at 200 ms. That figure is estimated
from the construction, not measured: Smear runs on the native `ConvolverNode`,
which has no Node test harness, unlike Scramble.

### Disperse — cascade of Schroeder allpass sections

Eight allpass sections, `y = -g·v + v[n-d]`, `v = x + g·v[n-d]`, built from a
`DelayNode` and two gains apiece.

An allpass filter has unity magnitude at every frequency, which makes this the
*mathematically exact* version of the brief: the magnitude spectrum is preserved
to the last decimal and only phase changes. It also adds **no bulk latency** —
group delay varies with frequency, but the first sample leaves as soon as it
arrives.

| | |
|---|---|
| Knob | `g` 0 → 0.85, and a 1× → 3× stretch of the delays |
| Latency | 0 |
| Cost | ~40 nodes per box |

Base delays are 7.3, 11.1, 17.9, 23.3, 31.7, 41.3, 53.9, 67.1 ms — spread across
an octave and a half and sharing no simple ratios, so the echoes of one section
land between the echoes of the next rather than reinforcing them.

The honest caveat: **dispersion is not diffusion**. Eight sections turn a click
into a metallic "boing" rather than a wash. It reads as a blur, not a pad. Where
it wins is that it is exactly spectrum-preserving and free of latency, which
makes it the method to reach for when you want the model's window to stay
aligned with the performance.

### Scramble — slices cut on the hits, reordered within the bar

Where Smear and Disperse spread each moment across time, Scramble **rearranges**
time and averages nothing. The input is cut into rhythmic slices, and each slice
either plays in place or is swapped for an earlier slice from the last bar. Nothing
is resampled, so every slice keeps its pitch, timbre and its own attack. The
rhythm changes because the order does.

| | |
|---|---|
| Knob | probability that any one slice is swapped, 0 → 100% |
| Slice | a 1/16 note at the app's BPM (125 ms until a tempo is known) |
| Source range | the last bar, 16 slices |
| Latency | 52 ms, independent of tempo |
| Cost | one `ScriptProcessor` (2048 frames); per-sample detector, plus a chroma and an alignment search at each cut |

It never blends with the dry signal. The engine runs 52 ms behind its input, so a
dry/wet mix would sound as a flam, and the probability already *is* the amount: at
20% most of the original is there, untouched.

**Cuts follow the playing.** A detected hit takes the cut the moment it is heard,
5 ms ahead of the onset so the seam crossfade sits in the quiet before it. The next
grid line is then measured from that hit. With no hit, a cut waits 40 ms past its
grid line and then falls on the grid. A hit arriving later still overrides that
guess. That override is what lets a performance offset from the engine's grid by
half a slice lock on at all: with a symmetric snap window no hit ever falls inside
it. Only these grid cuts constrain latency. A hit that has just been detected is
always behind the output already.

**The onset detector** compares two bands against *themselves 15 ms earlier*:
plain energy (any pitch, bass included) and first-difference energy (quiet bright
hits over a loud low end). A band that rises 6 dB is a hit. The first design, high
frequency energy against its running mean, found **1 hit in 128**: first-difference
energy scales with frequency squared, so a low note after a high one never cleared
the average. A second failure showed up in testing. A hard note-off, such as a gated
synth or chopped sample, is a first-difference spike too, and it stole the cut from
the real note behind it. The high band now only counts while plain energy isn't
falling. A real hit never comes with falling energy; a note ending always does.

**Three rules make a swap sound right**, each added because a measurement failed
without it:

- **Harmony gate.** A bar-long pool doesn't protect harmony on its own. Slice *k*
  of a bar draws on average half its candidates from the previous bar, and across a
  chord-per-bar progression only **47%** of slices carried the right chord. Each
  slice now gets a chroma fingerprint (Goertzel, C4–B6). The last few slices are
  matched against the 24 major and minor triads, and a candidate is only eligible if
  ≥70% of its chroma sits in the chord sounding now. That took it to **90%**, with
  98% of slices still swapping. The gate stays off when the context doesn't look like
  a triad, which covers drums, noise, and a lone melody note (0.58 against its best
  triad). So percussion swaps 100% and a melody over one chord still scrambles to
  near chance.
- **No double hits.** A swapped slice can be shorter than the slot it fills. If it
  kept reading past its end, the next hit in the source would play late inside the
  slot. A swapped slice therefore stops where its source did whenever a hit came
  next. Without this, level ran +0.78 dB hot at full probability; with it, −0.08 dB.
- **No flutter.** On sustained tones a displacement of whole slices lands at an
  arbitrary phase. At 125 ms slices and 220 Hz that's 27.5 cycles, so every other
  seam is anti-phase and cancels. It measured as a sine reading 217.5 Hz, and on a
  pad you'd hear an 8 Hz flutter. Grid-cut swaps are nudged by up to ±10 ms to the
  offset that best continues the waveform across the seam (the alignment step of
  WSOLA). The sine now reads 220.0 Hz and the deepest seam dip is −0.6 dB. Hit cuts
  are never nudged: a hit masks its own seam, and nudging it would move the hit.

Measured by `node scripts/test-scramble.mjs`, all passing:

| claim | result |
|---|---|
| hits get their own cut (±20 ms humanised) | 128/128, mean error −0.8 ms |
| locks onto a performance 60 ms off-grid | 128/128 |
| probability 0 | bit-exact copy, delayed 52 ms |
| order at 0 / 25 / 50 / 100% | 100 / 81 / 54 / 2% of steps in place |
| every slice a single clean pitch | 99% purity at every setting |
| level at every setting | within 0.1 dB |
| attack sharpness, scrambled vs input | 7.1 vs 6.7 dB median jump |
| sustained sine | 220.0 Hz, no step or dip at seams |
| right chord across a progression | 90% (47% before the gate) |
| untonal hits swapped | 100% |

Two things to know. The attack test uses notes that ring into each other on
purpose. With notes that stop dead before the next hit, the input's "before" is
silence and the measurement reads the gap, not the attack. Reordering slices of
different lengths legitimately changes how long a note rings before the next hit.
And the harmony gate's thresholds were tuned on synthetic triads with overtones.
Dense jazz voicings or heavy distortion may gate more or less than intended; the
constants are named and commented in `scrambleEngine.ts`.

---

## 3. Considered and not shipped

**Grains, replaced by Scramble.** 50 ms Hann grains read at randomly scattered
offsets and overlap-added, which is Paulstretch's smearing without the FFT. It
shipped first, and two findings from it are worth keeping. **Grain density decides
whether it smears or merely doubles.** At 4× overlap an impulse came out as four
separate clicks, and whether the attack survived was down to which offsets came up.
16× was the knee (~120 ms of spread, peak down to 41%). **Its level compensation
has a closed form.** Scattered grains are decorrelated, so they sum in *power*, not
amplitude, and the Hann overlap-add constant doesn't apply: `1/√(OVERLAP · 3/8)` =
0.408, against 0.413 measured. It was replaced because what was wanted was rhythm
*rearranged*, and a 16-copy average dissolves rhythm instead. The two are close to
opposites: overlap, window and read offsets all point the other way.

**STFT magnitude smoothing (Paulstretch proper).** Frame the audio, keep
magnitudes, smooth each bin across frames with a long time constant, randomise
phase, overlap-add. The textbook answer and the only approach that truly
*removes* attack information rather than hiding it. Needs an FFT and an
`AudioWorklet` — call it 200 lines and a real debugging session. Worth building
if Smear at 1.2 s still leaves more transient than wanted.

**Noise-excited channel vocoder.** A filter bank, slow envelope followers, and
band-filtered noise as the carrier. Preserves timbre and flattens envelopes
beautifully, but destroys pitch unless the bands are very narrow — which fails
the brief.

**Heavy compression / limiting.** Reduces attack but does not spread anything in
time. Does not meet the brief at all.

---

## 4. Adding a method

1. Write `src/live/inputfx/<name>.ts` exporting an `FxMethod`:

```ts
export const myFx: FxMethod = {
  id: 'myfx',
  label: 'My FX',              // dropdown text
  blurb: 'One line...',        // dropdown tooltip
  format: (v) => `${v}`,       // readout under the knob
  latencyAt: (v) => 0,         // pure: the rack shows this without
                               // waiting for the node to catch up
  create: (ctx) => new MyFx(ctx),
};
```

2. The `FxImpl` it creates exposes stable `input` / `output` nodes, plus
   `setAmount`, `latency` and `dispose`. Amount `0` must tear the guts down.
   Implement the optional `setTempo(bpm | null)` if the method works in musical
   units; `FxChain` forwards the app's BPM to it, before any amount, so guts built
   on the first non-zero amount are built at the right tempo.
3. Add it to `FX_METHODS` in `index.ts`. Nothing else changes.

`FxChain` keeps the surrounding graph — the box's fader on one side, the track
mixes and the monitor on the other — connected to fixed endpoints, so switching
method mid-performance rebuilds only the middle and cannot drop a box out of its
patch.

**If the method does sample-level DSP, put the arithmetic in a plain class with
no Web Audio and no imports**, the way `scrambleEngine.ts` is separated from
`scramble.ts`. Node 22.6+ strips TypeScript types natively, so a test script can
`import` the engine's `.ts` file directly, with no build step. The Scramble figures
above came out of `node scripts/test-scramble.mjs`, and the Grains figures came out
of its predecessor. It earns its keep: every failure in that script (a blind
detector, false hits on note-offs, double hits, seam flutter, a harmony claim that
held only 47% of the time) looked plausible on paper and was caught only by running
the code.

---

## 5. Where it sits in the graph

```
mic / file / Strudel
        ↓
     box fader            (level)
        ↓
   FxChain.input
        ↓
   [ method guts ]        ← swapped when the dropdown changes
        ↓
   FxChain.output ────┬──→ monitor gain        (what you hear)
                      └──→ track mixes → ScriptProcessor → ring buffer
                                                              ↓
                                                        backend snapshot
```

Per box, not global: each input box carries its own `fxMethod` / `fxAmount`, so
a mic can be washed out while a Live Code pattern stays sharp. The rack's Input
FX group edits whichever box is selected — click a box to select it, the same
gesture that picks a track on the output side. The selected box's name is shown
under the group caption.
