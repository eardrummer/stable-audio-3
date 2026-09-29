import { DEFAULT_FX } from './inputfx';

/** Shared between LiveApp and each TrackOutput. */

export const API_URL = 'http://localhost:8000';
export const WS_URL = 'ws://localhost:8000';

export const VARIANTS = 5;
// Noise distance between successive variants. The 5-variant ladder spans
// 4 x spread, so 0.08 puts the top variant 0.32 above the base — clearly
// different but still tethered to the input. Past ~0.1 the ladder starts
// clamping at 1.0 at ordinary base levels and the top variants stop varying.
// At 0 every variant gets the same noise level, and the backend already shares
// one seed across the family — so the generations are bit-identical and the only
// remaining effect is the panning spread across the stereo field.
export const SPREAD_MIN = 0;
export const SPREAD_MAX = 0.08;
export const MAX_NUDGE = 5;          // seconds of output delay for alignment

// Crossfade timing. Consecutive generations blend over a fixed window; true
// overlap only exists when Generate Rate is shorter than Input Rate, otherwise
// this is an ease-in on the incoming clip. Switches stay snappy.
export const XF_MAX_CYCLE = 2.0;
export const XF_SWITCH = 0.25;
export const XF_MIN = 0.12;

// Equal-power crossfade curves. sin²+cos² = 1, so summed power stays flat.
// (An exponential ramp is linear in *log* space — at the midpoint both clips
// sit near -37 dB, which is an audible hole.)
const CURVE_N = 128;
export const FADE_IN_CURVE = new Float32Array(CURVE_N);
export const FADE_OUT_CURVE = new Float32Array(CURVE_N);
for (let i = 0; i < CURVE_N; i++) {
  const th = (i / (CURVE_N - 1)) * (Math.PI / 2);
  FADE_IN_CURVE[i] = Math.sin(th);
  FADE_OUT_CURVE[i] = Math.cos(th);
}

/** Everything a track owns independently of the shared input side. */
export interface TrackParams {
  /** Stable identity, so removing a track cannot make React reuse the wrong
      instance (each TrackOutput owns a WebSocket and an audio graph). */
  id: string;
  prompt: string;
  noise: number;
  steps: number;
  cfg: number;
  multiVariant: boolean;
  variant: number;
  /** Noise distance between successive variants (was the fixed VARIANT_DELTA). */
  spread: number;
  /** Play every variant at once, panned across the stereo field. */
  allVariants: boolean;
  loopback: number;
  nudge: number;
  /** Output filters, all browser-side: the model never sees them. Each knob
      is 0..1 and neutral at one end, so a fresh track is unfiltered. */
  lpf: number;      // 1 = wide open
  hpf: number;      // 0 = wide open
  comb: number;     // 0 = bypassed
  outputMonitor: number;
}

let trackSeq = 0;
export const defaultTrack = (): TrackParams => ({
  id: `trk${++trackSeq}`,
  prompt: '',
  noise: 0.35,
  steps: 8,
  cfg: 4.0,
  multiVariant: false,
  variant: 0,
  spread: 0.02,
  allVariants: false,
  loopback: 0,
  nudge: 0,
  lpf: 1,
  hpf: 0,
  comb: 0,
  outputMonitor: 0.8,
});

// ── output filter mappings ───────────────────────────────────────────
// Shared by the knob labels and the audio graph so the number under the knob
// is exactly the number the filter is running at. All three are exponential:
// pitch is logarithmic, so a linear sweep of Hz feels lopsided under the hand.
const expMap = (lo: number, hi: number) => (v: number) => lo * Math.pow(hi / lo, v);

export const LPF_LO = 180, LPF_HI = 20000;
export const HPF_LO = 20, HPF_HI = 6000;
/** Knob 1 leaves the top open; turning down closes the lid. */
export const lpfHz = expMap(LPF_LO, LPF_HI);
/** Knob 0 leaves the bottom open; turning up lifts the floor. */
export const hpfHz = expMap(HPF_LO, HPF_HI);

// The comb is a short feedback delay: its resonance sits at 1/delayTime, so
// the knob reads as a pitch. Zero is a true bypass.
//
// The settings below aim at an effect you would leave in a mix: at full depth
// the comb sits about 7 dB over the dry path, so it plainly rewrites the
// timbre while the source stays audible underneath it, and the resonance rings
// for 0.7 s at the bottom of the sweep rather than droning for seconds.
export const COMB_LO = 40, COMB_HI = 1600;
export const combHz = expMap(COMB_LO, COMB_HI);
/** Fraction of knob travel that ramps the effect in; the rest sweeps pitch.
 *  A quarter of the throw is enough to park at a subtle setting rather than
 *  only choosing between off and full. */
export const COMB_FADE_IN = 0.25;
export const COMB_FEEDBACK = 0.78;  // T60 ≈ 28 delay periods — a resonance, not a drone
export const COMB_WET = 0.7;
export const COMB_DUCK = 0.5;       // the dry path steps back, it does not vanish

/**
 * Gains for the comb at a given knob position.
 *
 * The wet path is a resonator, and how much it multiplies depends on what you
 * ask. A sine parked exactly on a comb peak sees 1 / (1 - fb) — 4.5x at this
 * feedback — but real material is broadband, and its energy gain is
 * 1 / sqrt(1 - fb^2), about 1.6x. Normalising against the sine figure is what
 * made the first version of this knob inaudible, so the gains are set from the
 * broadband figure and the excursions that do overshoot are caught by the soft
 * clipper after the sum (see COMB_KNEE).
 *
 * `pre`/`post` straddle that clipper. With the comb off, pre 0.5 puts the whole
 * signal under the knee — where the curve is a straight line — and post 2.0
 * puts it back, so the block is transparent when you are not using it. As the
 * knob comes up both walk to 1.0 and the clipper takes hold.
 */
export function combGains(v: number) {
  const depth = v > 0 ? Math.min(1, v / COMB_FADE_IN) : 0;
  const fb = COMB_FEEDBACK * depth;
  const dry = 1 - COMB_DUCK * depth;
  const wet = COMB_WET * depth;
  const norm = 1 / (dry + wet / Math.sqrt(1 - fb * fb));
  const pre = 0.5 + 0.5 * depth;
  return {
    fb, dry: dry * norm, wet: wet * norm,
    pre, post: 1 / pre,
    delaySec: 1 / combHz(v),
  };
}

/** DC and subsonic rubbish would pile up inside the loop, so the feedback path
 *  is high-passed — well below COMB_LO, so it never eats the lowest note. */
export const COMB_FB_HPF = 20;

/**
 * Soft clipper for the comb's output. Linear below the knee, so with the comb
 * off the block is transparent; above it, a tanh curve that never reaches 1.0.
 * It is a safety net rather than part of the sound: at these settings only the
 * odd transient reaches it, and it saturates that instead of letting the
 * destination hard-clip.
 */
export const COMB_KNEE = 0.6;
export function softClipCurve(n = 4096): Float32Array<ArrayBuffer> {
  const c = new Float32Array(new ArrayBuffer(n * 4));
  const k = COMB_KNEE, span = 1 - k;
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * 2 - 1;
    const a = Math.abs(x);
    c[i] = a <= k ? x : Math.sign(x) * (k + span * Math.tanh((a - k) / span));
  }
  return c;
}

/** Track 1 keeps the established green; track 2 is cyan so the two output
 *  panes, their tabs and their spectrograms all read as one colour each. */
export const TRACK_THEME = [
  { name: 'Track 1', accent: '#8ad18a', spec: 'green' as const },
  { name: 'Track 2', accent: '#5fc6d8', spec: 'cyan' as const },
  { name: 'Track 3', accent: '#d8c05f', spec: 'green' as const },
  { name: 'Track 4', accent: '#c98ad1', spec: 'cyan' as const },
];

/** Shared, input-side parameters every track generates against. */
export interface SharedParams {
  inputRate: number;
  generateRate: number;
  model: string;
}

/** Props Knob and MidiSlider need to take part in MIDI learn. */
export interface LearnProps {
  id: string;
  learning: boolean;
  armed: boolean;
  bound: string | null;
  onArm: (id: string) => void;
}

// ─────────────────────────────────────────────────────────────────────────
// Input sources and routing
// ─────────────────────────────────────────────────────────────────────────

/** Up to four input boxes and four output tracks. */
export const MAX_SLOTS = 4;

export type SourceKind = 'mic' | 'file' | 'code';
export const SOURCE_KINDS: SourceKind[] = ['mic', 'file', 'code'];

export const SOURCE_META: Record<SourceKind, { name: string; accent: string }> = {
  mic:  { name: 'Live Input', accent: '#8ad18a' },
  file: { name: 'File Loop',  accent: '#d8c05f' },
  code: { name: 'Live Code',  accent: '#b98ad1' },
};

/**
 * One input box. Sources are instances rather than a fixed set of kinds, so two
 * Live Inputs can carry different channels of the same interface.
 *
 * Live Code is the exception: Strudel is a singleton (one repl, one context, one
 * output tap), so at most one instance of that kind can exist.
 */
export interface SourceInstance {
  id: string;
  kind: SourceKind;
  /** Send level into the tracks this box is patched to. */
  level: number;
  /** One flag per track index. */
  to: boolean[];
  /** Mic only: which hardware channel(s) this box takes. */
  chanSel: string;
  /** File only, for display; the File object itself lives in a ref. */
  fileName?: string;
  fileDuration?: number;
  /** File only: the segment that loops, in seconds. Absent means the whole
   *  file. The recorded-take equivalent lives on the box's LoopSlot. */
  fileTrim?: [number, number];
  /** Input pre-FX: which algorithm, and how much of it. This runs BEFORE the
   *  ring buffer, so it changes what the model conditions on, not just what
   *  you hear. See `src/live/inputfx/` and `analysis/input-prefx.md`. */
  fxMethod: string;
  fxAmount: number;
}

let sourceSeq = 0;
export const newSource = (kind: SourceKind, nTracks: number): SourceInstance => ({
  id: `src${++sourceSeq}`,
  kind,
  level: 1,
  to: Array.from({ length: nTracks }, () => true),
  chanSel: '0+1',
  fxMethod: DEFAULT_FX,
  fxAmount: 0,
});

export const defaultSources = (nTracks: number): SourceInstance[] => [newSource('mic', nTracks)];
