/**
 * The DSP half of the SCRAMBLE method, kept free of Web Audio (and of imports)
 * so Node can run it directly — see `scripts/test-scramble.mjs`, which is where
 * every number quoted in this file comes from.
 *
 * The input is cut into rhythmic slices and, slice by slice, either played in
 * place or swapped for an earlier slice from the last bar. Nothing is resampled
 * and nothing is averaged, so each slice keeps its pitch, timbre and attack;
 * only the ORDER changes. That is the difference from a smear: rhythm is
 * rearranged rather than dissolved.
 *
 * How cuts are placed
 * -------------------
 * Slices are a 1/16 note at the current tempo, but a detected hit takes the cut
 * the moment it is heard, and the next grid line is then measured from that hit.
 * The grid follows the playing rather than the other way round:
 *
 * - A hit at least half a slice after the last hit-cut starts a new slice.
 *   Anything closer is a flam or a ripple in the same attack.
 * - A grid cut is a guess made in the absence of a hit. It waits TOL past its
 *   grid line for one to turn up; a hit that arrives later still, by at least
 *   MIN_SLIVER, overrides it. That override is what lets a performance offset
 *   from the engine's grid by half a slice lock on at all — with a symmetric
 *   snap window it never would, because no hit ever falls inside it.
 * - Cuts sit PREROLL ahead of the hit, so the seam crossfade happens in the
 *   quiet just before a transient instead of shaving its front off.
 *
 * What makes a swap sound right
 * -----------------------------
 * - HARMONY. A bar-long pool alone does not protect it: slice k of a bar draws
 *   on average half its candidates from the previous bar, and measured 47% of
 *   slices carrying the right chord across a chord-per-bar progression. So each
 *   slice gets a chroma fingerprint, the last few slices are matched against the
 *   24 major and minor triads, and a candidate is only eligible if its energy
 *   sits in the chord that is sounding now. Material with no clear harmony —
 *   drums, noise — is never gated, and a melody moving through chord tones still
 *   scrambles freely, because the context is a chord rather than a single note.
 * - NO DOUBLE HITS. A swapped slice may be shorter than the slot it fills. Read
 *   on past its end and the next hit in the source plays late inside the slot,
 *   so a swapped slice stops where its source did whenever a hit came next.
 * - NO FLUTTER. On sustained tones a displacement of whole slices lands at an
 *   arbitrary phase, and at a seam two out-of-phase copies cancel: measured as a
 *   sine reading 217.5 Hz, heard as an 8 Hz flutter on a pad. Grid-cut swaps are
 *   therefore nudged by up to ALIGN_REACH to the offset that best continues the
 *   waveform across the seam — the alignment step of WSOLA.
 */

/** How long a grid cut waits past its grid line for a hit to take it. */
export const TOL_MAX_SEC = 0.04;
/** A hit this long after a grid cut replaces it, leaving a short slice behind. */
export const MIN_SLIVER_SEC = 0.03;
/** Budget for the detector to register a hit after it happens. */
export const DETECT_LAG_SEC = 0.01;
/** Seam crossfade between slices that do not continue each other. */
export const RAMP_SEC = 0.005;
/**
 * Cuts land this far ahead of the detected onset. Must be at least RAMP: the
 * outgoing slice keeps playing through the crossfade, and if the crossfade ran
 * past the onset the outgoing side would play the new hit's attack itself.
 */
export const PREROLL_SEC = RAMP_SEC;

/**
 * Only grid cuts constrain latency. One is final at its grid line + TOL + LAG
 * and lands on the grid line itself, so the output must still be short of that
 * point by then. Hit cuts never do: a hit that has just been detected is behind
 * the output already, by far more than PREROLL.
 */
export const LATENCY_SEC = TOL_MAX_SEC + DETECT_LAG_SEC + 0.002;

export const SLICE_MIN_SEC = 0.04;
export const SLICE_MAX_SEC = 0.3;
/** A 1/16 at 120 BPM, used until a tempo is known. */
export const SLICE_DEFAULT_SEC = 0.125;
/** Slices per bar: sixteenth notes in 4/4. */
export const BAR_SLICES = 16;

// ── Onset detector ─────────────────────────────────────────────────────
// Two bands, each compared with ITSELF a moment ago rather than with a long
// average. The obvious design — high-frequency energy over its running mean —
// measured 1 hit in 128 on a pluck sequence: first-difference energy scales
// with frequency squared, so a low note after a high one never clears the
// average the high one left behind, and a dense stream of notes keeps that
// average high anyway. A local rise has neither problem.
//
//   band L: plain energy        — catches notes of any pitch, bass included
//   band H: first difference     — catches quiet bright hits over a loud low end
const ENV_SEC = 0.005;
/** How far back "a moment ago" is. Longer than the envelope's own rise, far
 *  shorter than any crescendo or tremolo. */
const LOOKBACK_SEC = 0.015;
/** A hit is either band's envelope rising this much within LOOKBACK (+6 dB). */
const RISE = 4;
/** Denominator floors, so the rise out of silence into hiss is not a hit.
 *  Band L's corresponds to roughly -45 dBFS. */
const FLOOR_L = 1e-5;
const FLOOR_H = 1e-7;
/**
 * Band H only counts while band L is holding at least this share of where it
 * was. A hard note-off — a gated synth, a chopped sample — is a first-difference
 * spike too, and it measured as a false hit that then blocked the real note
 * behind it. A real hit is never accompanied by falling energy; a note ending
 * always is, since a decaying note is already below its level a moment ago.
 */
const NOT_FALLING = 0.7;
/** One attack must not register as several hits. */
const HOLD_SEC = 0.03;
/** How late the detector fires on a sharp attack. Calibrated by the test. */
export const DETECT_BACKDATE_SEC = 0.001;

// ── Seam alignment (grid cuts only) ────────────────────────────────────
/** How far a grid-cut source may be nudged to continue the waveform. Covers a
 *  full period down to 50 Hz; hit cuts are never nudged, since that would move
 *  the hit. */
const ALIGN_REACH_SEC = 0.01;
/** Waveform compared across the seam. Two periods of a 200 Hz tone. */
const ALIGN_WIN_SEC = 0.01;

// ── Harmony gate ───────────────────────────────────────────────────────
/** Chroma is read over at most this much of each slice's start. */
const CHROMA_WIN_SEC = 0.085;
/** C4 to B6: where chords live. Below it a semitone is narrower than the
 *  analysis can resolve in a slice; above it there is mostly overtone. */
const CHROMA_LOW_HZ = 261.63, CHROMA_OCTAVES = 3;
/** Weight of older slices in the harmonic context, per slice. At 0.5 a chord
 *  change takes over within about two slices. */
const CONTEXT_DECAY = 0.5;
/** Context must look this much like a triad before anything is gated. A lone
 *  note scores 0.58 against its best triad, so melody alone never gates. */
const CHORD_MIN = 0.7;
/** Fraction of a candidate's chroma that must sit in the sounding triad. A
 *  relative minor shares two of three notes and lands at ~0.67, just outside. */
const COVER_MIN = 0.7;
/** A slice whose chroma peak is under this multiple of its mean has no
 *  harmony to clash with (drums, noise) and is never gated. */
const TONAL_PEAK = 2;

interface Slot {
  start: number;
  off: number;
  hit: boolean;
  /** Max-normalised chroma, null when untonal; set when the slice completes. */
  chroma: Float32Array | null;
  /** Swapped slices only: the source frame where playback must stop. */
  stopAt: number;
}

const TRIADS: number[][] = [];
for (let r = 0; r < 12; r++) {
  TRIADS.push([r, (r + 4) % 12, (r + 7) % 12]);   // major
  TRIADS.push([r, (r + 3) % 12, (r + 7) % 12]);   // minor
}

export class Scrambler {
  private sr: number;
  private rng: () => number;
  private hL: Float32Array;
  private hR: Float32Array;
  private hLen: number;

  private pos = 0;              // absolute input frame
  private prob = 0;

  private slice = 0;            // frames
  private tol = 0;
  private range = 0;

  private lat: number;
  private lag: number;
  private pre: number;
  private ramp: number;
  private hold: number;
  private sliver: number;
  private backdate: number;
  private reach: number;
  private alignWin: number;
  private chromaWin: number;

  // detector
  private prev = 0;
  private envL = 0;
  private envH = 0;
  private af: number;
  private backL: Float32Array;
  private backH: Float32Array;
  private backAt = 0;
  private holdUntil = 0;

  // harmony
  private chromaK: Float32Array;       // Goertzel coefficients at sr/2
  private hann: Float32Array;
  private context = new Float32Array(12);
  private chord: number[] | null = null;
  private scratch: Float32Array;

  // slicing
  private nextGrid = 0;
  private slots: Slot[] = [{ start: 0, off: 0, hit: false, chroma: null, stopAt: 0 }];
  private cur = 0;

  /** Every cut taken, as [frame, wasHit, wasSwapped] — only when asked for,
   *  since a live session would grow this for hours. */
  readonly cuts: [number, boolean, boolean][] | null;

  constructor(sampleRate: number, opts: { rng?: () => number; recordCuts?: boolean } = {}) {
    this.sr = sampleRate;
    this.rng = opts.rng ?? Math.random;
    this.cuts = opts.recordCuts ? [] : null;
    const f = (sec: number) => Math.round(sec * sampleRate);
    this.lat = Math.ceil(LATENCY_SEC * sampleRate);
    this.lag = f(DETECT_LAG_SEC);
    this.pre = f(PREROLL_SEC);
    this.ramp = Math.max(1, f(RAMP_SEC));
    this.hold = f(HOLD_SEC);
    this.sliver = f(MIN_SLIVER_SEC);
    this.backdate = f(DETECT_BACKDATE_SEC);
    this.reach = f(ALIGN_REACH_SEC);
    this.alignWin = f(ALIGN_WIN_SEC);
    this.chromaWin = f(CHROMA_WIN_SEC);
    this.af = 1 - Math.exp(-1 / (ENV_SEC * sampleRate));
    const look = Math.max(1, f(LOOKBACK_SEC));
    this.backL = new Float32Array(look);
    this.backH = new Float32Array(look);
    this.scratch = new Float32Array(this.alignWin);

    // Chroma runs on every other sample: Nyquist is still far above B6.
    const half = sampleRate / 2;
    this.chromaK = new Float32Array(12 * CHROMA_OCTAVES);
    for (let o = 0; o < CHROMA_OCTAVES; o++) {
      for (let pc = 0; pc < 12; pc++) {
        const hz = CHROMA_LOW_HZ * Math.pow(2, o + pc / 12);
        this.chromaK[o * 12 + pc] = 2 * Math.cos((2 * Math.PI * hz) / half);
      }
    }
    this.hann = new Float32Array(1024);
    for (let i = 0; i < 1024; i++) this.hann[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / 1023);

    // Allocated once at the largest a bar can be, so a tempo change never
    // reallocates under a running stream.
    const maxRange = BAR_SLICES * SLICE_MAX_SEC;
    this.hLen = Math.ceil((maxRange + 2 * SLICE_MAX_SEC + LATENCY_SEC + ALIGN_REACH_SEC + 0.1) * sampleRate);
    this.hL = new Float32Array(this.hLen);
    this.hR = new Float32Array(this.hLen);
    this.setTempo(null);
    this.nextGrid = this.slice;
  }

  /** @param bpm quarter-note tempo, or null to fall back to a fixed slice. */
  setTempo(bpm: number | null) {
    const sec = bpm && bpm > 0 ? 60 / bpm / 4 : SLICE_DEFAULT_SEC;
    const s = Math.max(SLICE_MIN_SEC, Math.min(SLICE_MAX_SEC, sec));
    this.slice = Math.round(s * this.sr);
    this.tol = Math.min(Math.round(this.slice / 3), Math.round(TOL_MAX_SEC * this.sr));
    this.range = BAR_SLICES * this.slice;
  }

  /** Chance that any one slice is swapped, 0..1. */
  setProbability(p: number) { this.prob = Math.max(0, Math.min(1, p)); }

  latency() { return this.lat / this.sr; }
  sliceSeconds() { return this.slice / this.sr; }
  /** The triad the gate currently believes is sounding, as pitch classes. */
  currentChord() { return this.chord; }

  private read(ch: Float32Array, frame: number) {
    return ch[((frame % this.hLen) + this.hLen) % this.hLen];
  }

  // ── harmony ─────────────────────────────────────────────────────────

  private chromaOf(start: number, end: number): Float32Array | null {
    const n = Math.min(end - start, this.chromaWin) >> 1;
    if (n < 128) return null;
    const c = new Float32Array(12);
    const hs = 1023 / (n - 1);
    for (let b = 0; b < this.chromaK.length; b++) {
      const k = this.chromaK[b];
      let s1 = 0, s2 = 0;
      for (let j = 0; j < n; j++) {
        const fr = start + 2 * j;
        const x = (this.read(this.hL, fr) + this.read(this.hR, fr)) * this.hann[(j * hs) | 0];
        const s0 = x + k * s1 - s2; s2 = s1; s1 = s0;
      }
      const pw = s1 * s1 + s2 * s2 - k * s1 * s2;
      c[b % 12] += Math.sqrt(Math.max(0, pw));
    }
    let max = 0, sum = 0;
    for (let i = 0; i < 12; i++) { max = Math.max(max, c[i]); sum += c[i]; }
    if (max <= 1e-9 || max < TONAL_PEAK * (sum / 12)) return null;
    for (let i = 0; i < 12; i++) c[i] /= max;
    return c;
  }

  /** Fold a finished slice into the context and re-estimate the chord. */
  private listen(chroma: Float32Array | null) {
    for (let i = 0; i < 12; i++) this.context[i] = this.context[i] * CONTEXT_DECAY + (chroma ? chroma[i] : 0);
    let max = 0, sq = 0;
    for (let i = 0; i < 12; i++) max = Math.max(max, this.context[i]);
    if (max <= 1e-9) { this.chord = null; return; }
    for (let i = 0; i < 12; i++) { const v = this.context[i] / max; sq += v * v; }
    let best: number[] | null = null, bestScore = 0;
    for (const t of TRIADS) {
      const score = (this.context[t[0]] + this.context[t[1]] + this.context[t[2]]) / max / Math.sqrt(3 * sq);
      if (score > bestScore) { bestScore = score; best = t; }
    }
    this.chord = bestScore >= CHORD_MIN ? best : null;
  }

  private fits(chroma: Float32Array | null) {
    if (!chroma || !this.chord) return true;
    let inside = 0, total = 0;
    for (let i = 0; i < 12; i++) total += chroma[i];
    for (const pc of this.chord) inside += chroma[pc];
    return inside >= COVER_MIN * total;
  }

  // ── seams ───────────────────────────────────────────────────────────

  /** Nudge a grid-cut source to the offset that best continues the waveform
   *  the outgoing slice is playing across the seam. */
  private align(cut: number, outOff: number, off: number) {
    const W = this.alignWin, x = this.scratch;
    let xx = 0;
    for (let t = 0; t < W; t++) {
      const fr = cut + outOff + t;
      x[t] = this.read(this.hL, fr) + this.read(this.hR, fr);
      xx += x[t] * x[t];
    }
    if (xx <= 1e-9) return off;
    let best = off, bestScore = -Infinity;
    for (let dd = -this.reach; dd <= this.reach; dd += 2) {
      const o = cut + off + dd;
      let xy = 0, yy = 0;
      for (let t = 0; t < W; t++) {
        const y = this.read(this.hL, o + t) + this.read(this.hR, o + t);
        xy += x[t] * y; yy += y * y;
      }
      const score = xy / Math.sqrt(yy + 1e-12);
      if (score > bestScore) { bestScore = score; best = off + dd; }
    }
    return best;
  }

  /** One channel of a slot's output at timeline frame m. */
  private sample(s: Slot, ch: Float32Array, m: number) {
    const fr = m + s.off;
    if (s.off === 0 || s.stopAt === 0) return this.read(ch, fr);
    const left = s.stopAt - fr;
    if (left <= 0) return 0;
    const v = this.read(ch, fr);
    return left < this.ramp ? v * Math.sin((left / this.ramp) * (Math.PI / 2)) : v;
  }

  // ── slicing ─────────────────────────────────────────────────────────

  /** Start a new slice at `cut` and decide whether it plays in place. */
  private cutAt(cut: number, hit: boolean) {
    // The previous slice is complete now: fingerprint it and let it update
    // what the gate believes the harmony is.
    const done = this.slots[this.slots.length - 1];
    done.chroma = this.chromaOf(done.start, cut);
    this.listen(done.chroma);

    let off = 0, stopAt = 0;
    if (this.prob > 0 && this.rng() < this.prob) {
      const n = this.slots.length;
      const pool: number[] = [];
      for (let j = 0; j < n; j++) {
        const s = this.slots[j];
        const end = j + 1 < n ? this.slots[j + 1].start : cut;
        // Within the last bar; not a sliver, which played for a whole slot
        // would read straight on into the hit behind it; and in key.
        if (s.start >= cut - this.range && end - s.start >= this.slice >> 1 && this.fits(s.chroma)) {
          pool.push(j);
        }
      }
      if (pool.length) {
        const j = pool[Math.floor(this.rng() * pool.length)];
        const src = this.slots[j];
        off = src.start - cut;
        const nextHit = j + 1 < n ? this.slots[j + 1].hit : hit;
        const end = j + 1 < n ? this.slots[j + 1].start : cut;
        // What followed the source was a hit: stop there rather than play it.
        if (nextHit) stopAt = end;
        // Sustained material is where out-of-phase seams flutter; a hit masks
        // its own seam, and nudging it would move the hit.
        if (!hit) off = this.align(cut, done.off, off);
      }
    }
    this.slots.push({ start: cut, off, hit, chroma: null, stopAt });
    this.cuts?.push([cut, hit, off !== 0]);
  }

  process(input: Float32Array[], output: Float32Array[]) {
    const inL = input[0], inR = input[1] ?? input[0];
    const outL = output[0], outR = output[1] ?? output[0];
    const n = inL.length;

    for (let i = 0; i < n; i++) {
      const p = this.pos;
      const w = p % this.hLen;
      this.hL[w] = inL[i];
      this.hR[w] = inR[i];

      // ── detector
      const x = 0.5 * (inL[i] + inR[i]);
      const d = x - this.prev;
      this.prev = x;
      this.envL += (x * x - this.envL) * this.af;
      this.envH += (d * d - this.envH) * this.af;
      const thenL = this.backL[this.backAt], thenH = this.backH[this.backAt];
      this.backL[this.backAt] = this.envL;
      this.backH[this.backAt] = this.envH;
      this.backAt = (this.backAt + 1) % this.backL.length;
      const riseL = this.envL > RISE * (thenL + FLOOR_L);
      const riseH = this.envH > RISE * (thenH + FLOOR_H) && this.envL >= NOT_FALLING * thenL;
      if (p >= this.holdUntil && (riseL || riseH)) {
        this.holdUntil = p + this.hold;
        const at = p - this.backdate;
        const last = this.slots[this.slots.length - 1];
        const gap = last.hit ? this.slice >> 1 : this.sliver;
        if (at - this.pre >= last.start + gap) {
          this.cutAt(at - this.pre, true);
          this.nextGrid = at + this.slice;        // the grid now runs from this hit
        }
      }

      // ── no hit turned up for this grid line: cut on the grid
      if (p >= this.nextGrid + this.tol + this.lag) {
        this.cutAt(this.nextGrid, false);
        this.nextGrid += this.slice;
      }

      // ── output, LATENCY behind the input
      const m = p - this.lat;
      if (m < 0) { outL[i] = 0; outR[i] = 0; this.pos++; continue; }
      while (this.cur + 1 < this.slots.length && this.slots[this.cur + 1].start <= m) this.cur++;
      const s = this.slots[this.cur];
      let l = this.sample(s, this.hL, m);
      let r = this.sample(s, this.hR, m);
      const t = m - s.start;
      if (t < this.ramp && this.cur > 0) {
        const ps = this.slots[this.cur - 1];
        // Equal offsets read the very same samples, so a slice that carries on
        // from its neighbour needs no fade at all — which is also what makes
        // probability 0 an exact delayed copy of the input.
        if (ps.off !== s.off) {
          // Unrelated audio either side of the seam: equal power, not equal
          // amplitude, or the join dips.
          const th = (t / this.ramp) * (Math.PI / 2);
          const a = Math.sin(th), b = Math.cos(th);
          l = l * a + this.sample(ps, this.hL, m) * b;
          r = r * a + this.sample(ps, this.hR, m) * b;
        }
      }
      outL[i] = l;
      outR[i] = r;

      // Slots that can no longer be the playing slice, its fade partner, or a
      // source for anything still to come.
      if (this.cur > 2 && this.slots[1].start < m - this.range - 2 * this.slice) {
        this.slots.shift();
        this.cur--;
      }
      this.pos++;
    }
  }
}
