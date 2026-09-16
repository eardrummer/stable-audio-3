import { expMap, mixGains, type FxImpl, type FxMethod } from './types';

/**
 * SMEAR — convolution with an enveloped noise burst.
 *
 * Each input sample is splattered across T seconds of decorrelated noise. Noise
 * is spectrally flat on average, so the magnitude spectrum — pitch and timbre —
 * survives while the temporal envelope is destroyed: a drum hit becomes a
 * noise-shaped swell of length T.
 *
 * The envelope is a GAUSSIAN CENTRED ON THE IMPULSE, not a decay. That is the
 * difference between this and reverb: a decaying impulse response leaves the
 * onset intact and puts a tail behind it, so you still hear the hit. Spreading
 * energy backwards as well as forwards is what actually dissolves the attack,
 * and it costs T/2 of latency because "backwards" means lookahead.
 */

export const T_MIN = 0.02, T_MAX = 1.2;
export const smearTime = expMap(T_MIN, T_MAX);

/**
 * Gaussian-enveloped stereo noise, normalised to unit ENERGY rather than unit
 * peak. Energy is the right invariant here: it keeps sustained material at the
 * level it came in at, while transients correctly lose their peak — that loss
 * is the effect working.
 */
export function makeSmearIR(ctx: BaseAudioContext, seconds: number): AudioBuffer {
  const n = Math.max(64, Math.round(seconds * ctx.sampleRate));
  const buf = ctx.createBuffer(2, n, ctx.sampleRate);
  const centre = (n - 1) / 2;
  // ±3 sigma spans the buffer, so the envelope is down to ~1% at the edges and
  // the impulse response does not end on a step.
  const sigma = n / 6;
  for (let ch = 0; ch < 2; ch++) {
    const d = buf.getChannelData(ch);
    let energy = 0;
    for (let i = 0; i < n; i++) {
      const z = (i - centre) / sigma;
      d[i] = (Math.random() * 2 - 1) * Math.exp(-0.5 * z * z);
      energy += d[i] * d[i];
    }
    // Independent noise per channel: the smear comes out as a wide diffuse
    // field rather than a point source.
    const k = 1 / Math.sqrt(energy || 1);
    for (let i = 0; i < n; i++) d[i] *= k;
  }
  return buf;
}

/** Rebuilding a second-long impulse response is not something to do per mouse
 *  move, so the length follows the knob once it settles. */
const REBUILD_DEBOUNCE_MS = 150;

class Smear implements FxImpl {
  readonly input: GainNode;
  readonly output: GainNode;
  private dry: GainNode;
  private wet: GainNode;
  private conv: ConvolverNode | null = null;
  private amount = 0;
  private timer: number | null = null;
  private built = 0;      // seconds of impulse response currently loaded
  private ctx: AudioContext;

  constructor(ctx: AudioContext) {
    this.ctx = ctx;
    this.input = ctx.createGain();
    this.output = ctx.createGain();
    this.dry = ctx.createGain();
    this.wet = ctx.createGain();
    this.dry.gain.value = 1;
    this.wet.gain.value = 0;
    this.input.connect(this.dry);
    this.dry.connect(this.output);
  }

  setAmount(v: number) {
    this.amount = v;
    const { dry, wet } = mixGains(v);
    const t = this.ctx.currentTime;
    this.dry.gain.setTargetAtTime(dry, t, 0.02);
    this.wet.gain.setTargetAtTime(wet, t, 0.02);
    if (v <= 0) { this.teardown(); return; }
    if (this.timer !== null) window.clearTimeout(this.timer);
    this.timer = window.setTimeout(() => this.reload(), REBUILD_DEBOUNCE_MS);
  }

  private reload() {
    this.timer = null;
    if (this.amount <= 0) return;
    const seconds = smearTime(this.amount);
    // A few percent of drift is not worth a reallocation and a re-partition.
    if (this.conv && Math.abs(seconds - this.built) / this.built < 0.05) return;
    if (!this.conv) {
      this.conv = this.ctx.createConvolver();
      this.conv.normalize = false;   // the impulse response is already scaled
      this.input.connect(this.wet);
      this.wet.connect(this.conv);
      this.conv.connect(this.output);
    }
    this.conv.buffer = makeSmearIR(this.ctx, seconds);
    this.built = seconds;
  }

  private teardown() {
    if (this.timer !== null) { window.clearTimeout(this.timer); this.timer = null; }
    if (!this.conv) return;
    try { this.input.disconnect(this.wet); } catch { /* already */ }
    this.wet.disconnect();
    this.conv.disconnect();
    this.conv = null;
    this.built = 0;
  }

  latency() { return smear.latencyAt(this.amount); }

  dispose() {
    this.teardown();
    this.input.disconnect();
    this.dry.disconnect();
    this.output.disconnect();
  }
}

export const smear: FxMethod = {
  id: 'smear',
  label: 'Smear',
  blurb: 'Convolves with a Gaussian noise burst: keeps the spectrum, dissolves the attack. Adds T/2 latency.',
  format: (v) => (v <= 0 ? 'off' : `${(smearTime(v) * 1000).toFixed(0)}ms`),
  latencyAt: (v) => (v > 0 ? smearTime(v) / 2 : 0),
  create: (ctx) => new Smear(ctx),
};
