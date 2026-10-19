import type { FxImpl, FxMethod } from './types';
import { smear } from './smear';
import { disperse } from './disperse';
import { scramble } from './scramble';

export type { FxImpl, FxMethod } from './types';

/**
 * Every input pre-FX the rack offers. Adding one is: write a module exporting
 * an `FxMethod`, add it here. Nothing else in the app needs to change.
 */
export const FX_METHODS: FxMethod[] = [smear, disperse, scramble];
export const DEFAULT_FX = smear.id;

export const fxMethod = (id: string): FxMethod =>
  FX_METHODS.find(m => m.id === id) ?? FX_METHODS[0];

/**
 * A stable two-port wrapper around whichever method is selected.
 *
 * The point of it is that the surrounding graph — the box's fader on one side,
 * the track mixes and the monitor on the other — connects to these endpoints
 * once and never again. Switching method rebuilds only the guts in between, so
 * changing the dropdown mid-performance cannot drop a box out of its patch.
 */
export class FxChain {
  readonly input: GainNode;
  readonly output: GainNode;
  private ctx: AudioContext;
  private impl: FxImpl | null = null;
  private id = '';
  private amount = 0;
  private bpm: number | null = null;

  constructor(ctx: AudioContext, method: string, amount: number) {
    this.ctx = ctx;
    this.input = ctx.createGain();
    this.output = ctx.createGain();
    this.setMethod(method);
    this.setAmount(amount);
  }

  setMethod(id: string) {
    if (id === this.id) return;
    if (this.impl) {
      try { this.input.disconnect(this.impl.input); } catch { /* already */ }
      try { this.impl.output.disconnect(this.output); } catch { /* already */ }
      this.impl.dispose();
    }
    this.impl = fxMethod(id).create(this.ctx);
    this.id = id;
    this.input.connect(this.impl.input);
    this.impl.output.connect(this.output);
    // Tempo before amount: a method that builds its guts on the first non-zero
    // amount should build them at the right tempo, not correct itself after.
    this.impl.setTempo?.(this.bpm);
    this.impl.setAmount(this.amount);
  }

  setTempo(bpm: number | null) {
    this.bpm = bpm;
    this.impl?.setTempo?.(bpm);
  }

  setAmount(v: number) {
    this.amount = v;
    this.impl?.setAmount(v);
  }

  latency() { return this.impl?.latency() ?? 0; }

  dispose() {
    this.impl?.dispose();
    this.impl = null;
    this.input.disconnect();
    this.output.disconnect();
  }
}

/**
 * What the master SMEAR and SCRAMBLE buttons set their method to. Fixed rather
 * than another knob: those buttons are a performance gesture, and 0.8 is where
 * both methods read as the effect rather than as a treatment of the input.
 */
export const MASTER_FX_AMOUNT = 0.8;
