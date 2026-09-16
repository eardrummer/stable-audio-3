import { type FxImpl, type FxMethod } from './types';

/**
 * DISPERSE — a cascade of Schroeder allpass sections.
 *
 * An allpass filter has unity magnitude at every frequency, so this is the
 * mathematically exact version of "spread it in time, leave pitch and timbre
 * alone": the magnitude spectrum is preserved to the last decimal and only the
 * phase — the time structure — changes. It also adds no bulk latency.
 *
 * The price is that dispersion is not the same thing as diffusion. A handful of
 * sections turns a click into a metallic "boing" rather than a wash; smoothness
 * comes from stacking enough sections at delays with no common factor, which is
 * why the base delays below are spread across an octave and a half and share no
 * simple ratios. Compared with Smear it keeps more of the onset, so it reads as
 * a blur rather than a pad.
 */

/** One section: y = -g·v + v[n-d], v = x + g·v[n-d]. */
function allpass(ctx: AudioContext, delaySec: number) {
  const sum = ctx.createGain();          // v
  const delay = ctx.createDelay(1);
  const fb = ctx.createGain();           // +g around the delay
  const ff = ctx.createGain();           // -g feedforward
  const out = ctx.createGain();
  delay.delayTime.value = delaySec;
  sum.connect(delay);
  delay.connect(fb);
  fb.connect(sum);
  sum.connect(ff);
  ff.connect(out);
  delay.connect(out);
  return { input: sum, output: out, delay, fb, ff };
}

/** Milliseconds, deliberately sharing no simple ratios so the echoes of one
 *  section land between the echoes of the next instead of reinforcing them. */
const BASE_MS = [7.3, 11.1, 17.9, 23.3, 31.7, 41.3, 53.9, 67.1];
const G_MAX = 0.85;      // past ~0.9 the ring stops decaying within a snapshot
const SPAN_MAX = 3;      // how far the delays stretch at full amount

class Disperse implements FxImpl {
  readonly input: GainNode;
  readonly output: GainNode;
  private ctx: AudioContext;
  private sections: ReturnType<typeof allpass>[] = [];
  private dry: GainNode;

  constructor(ctx: AudioContext) {
    this.ctx = ctx;
    this.input = ctx.createGain();
    this.output = ctx.createGain();
    this.dry = ctx.createGain();
    this.dry.gain.value = 1;
    this.input.connect(this.dry);
    this.dry.connect(this.output);
  }

  setAmount(v: number) {
    const t = this.ctx.currentTime;
    if (v <= 0) {
      this.dry.gain.setTargetAtTime(1, t, 0.02);
      this.teardown();
      return;
    }
    if (!this.sections.length) this.build();
    // An allpass cascade is already unity gain, so there is nothing to
    // crossfade against: the dry path is simply switched out of circuit.
    this.dry.gain.setTargetAtTime(0, t, 0.02);
    const g = G_MAX * v;
    const span = 1 + (SPAN_MAX - 1) * v;
    this.sections.forEach((s, i) => {
      s.fb.gain.setTargetAtTime(g, t, 0.03);
      s.ff.gain.setTargetAtTime(-g, t, 0.03);
      s.delay.delayTime.setTargetAtTime((BASE_MS[i] / 1000) * span, t, 0.05);
    });
  }

  private build() {
    let node: AudioNode = this.input;
    for (const ms of BASE_MS) {
      const s = allpass(this.ctx, ms / 1000);
      node.connect(s.input);
      this.sections.push(s);
      node = s.output;
    }
    node.connect(this.output);
  }

  private teardown() {
    if (!this.sections.length) return;
    try { this.input.disconnect(this.sections[0].input); } catch { /* already */ }
    for (const s of this.sections) {
      s.input.disconnect(); s.delay.disconnect();
      s.fb.disconnect(); s.ff.disconnect(); s.output.disconnect();
    }
    this.sections = [];
  }

  /** Group delay varies with frequency, but there is no bulk delay: the first
   *  sample leaves as soon as it arrives. */
  latency() { return 0; }

  dispose() {
    this.teardown();
    this.input.disconnect();
    this.dry.disconnect();
    this.output.disconnect();
  }
}

export const disperse: FxMethod = {
  id: 'disperse',
  label: 'Disperse',
  blurb: 'Eight allpass sections. Magnitude spectrum preserved exactly, no latency, blurs rather than dissolves.',
  format: (v) => (v <= 0 ? 'off' : `g ${(G_MAX * v).toFixed(2)}`),
  latencyAt: () => 0,
  create: (ctx) => new Disperse(ctx),
};
