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
  outputMonitor: number;
}

export const defaultTrack = (): TrackParams => ({
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
  outputMonitor: 0.8,
});

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
}

let sourceSeq = 0;
export const newSource = (kind: SourceKind, nTracks: number): SourceInstance => ({
  id: `src${++sourceSeq}`,
  kind,
  level: 1,
  to: Array.from({ length: nTracks }, () => true),
  chanSel: '0+1',
});

export const defaultSources = (nTracks: number): SourceInstance[] => [newSource('mic', nTracks)];
