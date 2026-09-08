/** Shared between LiveApp and each TrackOutput. */

export const API_URL = 'http://localhost:8000';
export const WS_URL = 'ws://localhost:8000';

export const VARIANTS = 5;
export const VARIANT_DELTA = 0.02;   // fixed noise step between variants
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
  loopback: 0,
  nudge: 0,
  outputMonitor: 0.8,
});

/** Track 1 keeps the established green; track 2 is cyan so the two output
 *  panes, their tabs and their spectrograms all read as one colour each. */
export const TRACK_THEME = [
  { name: 'Track 1', accent: '#8ad18a', spec: 'green' as const },
  { name: 'Track 2', accent: '#5fc6d8', spec: 'cyan' as const },
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
