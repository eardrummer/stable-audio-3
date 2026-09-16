import { type FxImpl, type FxMethod } from './types';
import { Scrambler, LATENCY_SEC } from './scrambleEngine';

/**
 * SCRAMBLE — slices cut on the hits and reordered within the bar.
 *
 * The knob is the chance that any one slice is swapped for an earlier one. The
 * DSP, and the reasoning behind every rule it follows, lives in
 * `scrambleEngine.ts`; this file only plugs it into Web Audio.
 *
 * Unlike Smear and Disperse this is never mixed with the dry signal. The engine
 * runs LATENCY behind its input, so a dry/wet blend would sound as a flam; and
 * probability already IS the amount — at 0.2 most of the original is there,
 * untouched. Engaging or disengaging therefore crossfades straight across.
 */

/** ScriptProcessor buffer. Smaller than the 4096 used elsewhere, because the
 *  processor adds one buffer of its own delay on top of the engine's. */
const BLOCK = 2048;
/** How long the dry/wet switch takes, and how long teardown waits for it. */
const SWITCH_SEC = 0.02;

class Scramble implements FxImpl {
  readonly input: GainNode;
  readonly output: GainNode;
  private ctx: AudioContext;
  private dry: GainNode;
  private wet: GainNode;
  private proc: ScriptProcessorNode | null = null;
  private engine: Scrambler | null = null;
  private sink: GainNode | null = null;
  private amount = 0;
  private bpm: number | null = null;
  private teardownTimer: number | null = null;

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
    const t = this.ctx.currentTime;
    const on = v > 0;
    this.dry.gain.setTargetAtTime(on ? 0 : 1, t, SWITCH_SEC / 3);
    this.wet.gain.setTargetAtTime(on ? 1 : 0, t, SWITCH_SEC / 3);
    if (on) {
      if (this.teardownTimer !== null) { window.clearTimeout(this.teardownTimer); this.teardownTimer = null; }
      if (!this.proc) this.build();
      this.engine!.setProbability(v);
    } else if (this.proc && this.teardownTimer === null) {
      // Let the wet path fade out before pulling it, or the switch clicks.
      this.teardownTimer = window.setTimeout(() => {
        this.teardownTimer = null;
        if (this.amount <= 0) this.teardown();
      }, SWITCH_SEC * 3000);
    }
  }

  setTempo(bpm: number | null) {
    this.bpm = bpm;
    this.engine?.setTempo(bpm);
  }

  private build() {
    const ctx = this.ctx;
    const engine = new Scrambler(ctx.sampleRate);
    engine.setTempo(this.bpm);
    const proc = ctx.createScriptProcessor(BLOCK, 2, 2);
    proc.onaudioprocess = (e) => {
      const ib = e.inputBuffer, ob = e.outputBuffer;
      const inL = ib.getChannelData(0);
      const inR = ib.numberOfChannels > 1 ? ib.getChannelData(1) : inL;
      engine.process([inL, inR], [ob.getChannelData(0), ob.getChannelData(1)]);
    };
    this.input.connect(proc);
    proc.connect(this.wet);
    this.wet.connect(this.output);
    // A ScriptProcessor only fires while it has a path to the destination, and
    // the wet gain sits at zero during the switch, so give it a silent one.
    const sink = ctx.createGain();
    sink.gain.value = 0;
    proc.connect(sink);
    sink.connect(ctx.destination);
    this.proc = proc; this.engine = engine; this.sink = sink;
  }

  private teardown() {
    if (!this.proc) return;
    this.proc.onaudioprocess = null;
    try { this.input.disconnect(this.proc); } catch { /* already */ }
    this.proc.disconnect();
    this.wet.disconnect();
    this.sink?.disconnect();
    this.proc = null; this.engine = null; this.sink = null;
  }

  latency() { return scramble.latencyAt(this.amount); }

  dispose() {
    if (this.teardownTimer !== null) window.clearTimeout(this.teardownTimer);
    this.teardown();
    this.input.disconnect();
    this.dry.disconnect();
    this.output.disconnect();
  }
}

export const scramble: FxMethod = {
  id: 'scramble',
  label: 'Scramble',
  blurb: 'Reorders 1/16-note slices cut on the hits, drawn from the last bar and only from slices that fit the chord. Rhythm changes; pitch and harmony stay.',
  format: (v) => (v <= 0 ? 'off' : `${Math.round(v * 100)}%`),
  latencyAt: (v) => (v > 0 ? LATENCY_SEC : 0),
  create: (ctx) => new Scramble(ctx),
};
