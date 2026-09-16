/**
 * Input pre-FX: processing that runs on an input box BEFORE the ring buffer,
 * so the model conditions on the processed audio rather than the raw feed.
 *
 * Every method is a two-port audio block with one scalar control. Adding a new
 * one means writing a module that exports an `FxMethod` and listing it in
 * `index.ts` — nothing in LiveApp needs to know what the methods are.
 *
 * See `analysis/input-prefx.md` for what each method does and why.
 */

export interface FxImpl {
  /** Stable endpoints. The chain wrapper connects these once and never again. */
  readonly input: AudioNode;
  readonly output: AudioNode;
  /**
   * Knob position, 0..1. Zero must be a true bypass: implementations are
   * expected to tear down their guts there rather than run them at zero mix,
   * since an idle convolver or worklet still costs CPU on every box.
   */
  setAmount(v: number): void;
  /**
   * Optional: the app's tempo in BPM, or null when none is set. Methods that
   * work in musical units implement it; the rest simply leave it out. The
   * chain wrapper forwards the tempo to whichever method is selected, including
   * one selected after the tempo was set.
   */
  setTempo?(bpm: number | null): void;
  /** Seconds of bulk delay added at the current amount, for the status line. */
  latency(): number;
  dispose(): void;
}

export interface FxMethod {
  id: string;
  /** Dropdown text. */
  label: string;
  /** One line explaining the algorithm; shown as the dropdown's tooltip. */
  blurb: string;
  /** Readout under the knob — usually the parameter the amount maps to. */
  format(amount: number): string;
  /** Seconds of added delay at a given amount. Pure, so the rack can show it
   *  without waiting for the audio node to catch up with the knob. */
  latencyAt(amount: number): number;
  create(ctx: AudioContext): FxImpl;
}

/** Exponential knob mapping, used by every method that sweeps a time. */
export const expMap = (lo: number, hi: number) => (v: number) =>
  lo * Math.pow(hi / lo, Math.max(0, Math.min(1, v)));

/** Equal-power dry/wet, so a half-turn is half as loud in power, not amplitude. */
export const mixGains = (v: number) => ({
  dry: Math.cos((Math.PI / 2) * v),
  wet: Math.sin((Math.PI / 2) * v),
});
