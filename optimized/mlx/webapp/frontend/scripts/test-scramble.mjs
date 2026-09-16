// Measures the SCRAMBLE input effect outside a browser, one check per claim in
// its brief: cuts on the hits, pitch kept, harmony kept, rhythm rearranged but
// not smeared, no clicks or flutter at the seams, level unchanged.
//
// Run: node scripts/test-scramble.mjs      (Node 22.6+ strips the TS types)
import {
  Scrambler, LATENCY_SEC, PREROLL_SEC, BAR_SLICES,
} from '../src/live/inputfx/scrambleEngine.ts';

const SR = 48000, BLOCK = 4096, BPM = 120;
const SLICE = 60 / BPM / 4;                  // a 1/16 note: 0.125 s
const LAT = Math.ceil(LATENCY_SEC * SR);

// Deterministic randomness, so a run can be repeated exactly.
function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function run(signal, prob, { seed = 1, record = false, bpm = BPM } = {}) {
  const s = new Scrambler(SR, { rng: mulberry32(seed), recordCuts: record });
  s.setTempo(bpm);
  s.setProbability(prob);
  const out = new Float32Array(signal.length);
  for (let p = 0; p < signal.length; p += BLOCK) {
    const len = Math.min(BLOCK, signal.length - p);
    const inp = signal.subarray(p, p + len);
    const oL = new Float32Array(len), oR = new Float32Array(len);
    s.process([inp, inp], [oL, oR]);
    out.set(oL, p);
  }
  return { out, cuts: s.cuts };
}

const rms = (a, i0 = 0, i1 = a.length) => {
  let e = 0; const lo = Math.max(0, i0), hi = Math.min(a.length, i1);
  for (let i = lo; i < hi; i++) e += a[i] * a[i];
  return Math.sqrt(e / Math.max(1, hi - lo));
};
/** Power at one frequency over a window (Goertzel). */
function tone(a, f, i0, i1) {
  const k = 2 * Math.cos(2 * Math.PI * f / SR);
  let s1 = 0, s2 = 0;
  for (let i = Math.max(0, i0); i < Math.min(a.length, i1); i++) {
    const s0 = a[i] + k * s1 - s2; s2 = s1; s1 = s0;
  }
  return s1 * s1 + s2 * s2 - k * s1 * s2;
}
const median = xs => { const s = [...xs].sort((a, b) => a - b); return s[s.length >> 1]; };
const pct = x => `${(x * 100).toFixed(0)}%`;
const ms = frames => `${((frames / SR) * 1000).toFixed(1)}ms`;
let failures = 0;
const check = (ok, label) => { console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}`); if (!ok) failures++; };

// ── test signal: 16 plucks per bar, each step its own pitch ──────────────
// All sixteen are C-major chord tones spread over five octaves. A unique pitch
// per step is what lets the output be read back as a sequence; keeping them in
// one chord is what makes this also a test that the harmony gate lets a melody
// over a single chord scramble freely instead of freezing it.
const C_MAJOR = [130.81, 164.81, 196.0, 261.63, 329.63, 392.0, 523.25, 659.26,
  783.99, 1046.5, 1318.51, 1567.98, 2093.0, 2637.02, 3135.96, 4186.01];
const BARS = 8, LEAD = 0.25;
function sequence(jitterSec, seed = 7, hz = C_MAJOR, { tau = 0.045, dur = 0.11 } = {}) {
  const rng = mulberry32(seed);
  const n = Math.ceil((LEAD + BARS * BAR_SLICES * SLICE + 0.5) * SR);
  const a = new Float32Array(n);
  const onsets = [];
  for (let b = 0; b < BARS; b++) {
    for (let k = 0; k < BAR_SLICES; k++) {
      const t0 = LEAD + (b * BAR_SLICES + k) * SLICE + (rng() * 2 - 1) * jitterSec;
      const i0 = Math.round(t0 * SR);
      onsets.push({ at: i0, step: k });
      for (let i = 0; i < Math.round(dur * SR) && i0 + i < n; i++) {
        const t = i / SR;
        a[i0 + i] += 0.6 * Math.sin(2 * Math.PI * hz[k] * t) * Math.exp(-t / tau);
      }
    }
  }
  return { a, onsets };
}

/** Which step's pitch sounds at each input onset, as heard in the output. */
function readSequence(out, onsets, hz = C_MAJOR) {
  const w0 = Math.round(0.008 * SR), w1 = Math.round(0.068 * SR);
  return onsets.map(({ at }) => {
    const p = hz.map(f => tone(out, f, at + LAT + w0, at + LAT + w1));
    const total = p.reduce((x, y) => x + y, 0) || 1;
    let best = 0; for (let k = 1; k < p.length; k++) if (p[k] > p[best]) best = k;
    return { step: best, purity: p[best] / total };
  });
}

// ── 1. cuts land on the hits ────────────────────────────────────────────
console.log('\n1. Cuts land on the hits (±20 ms humanised timing)');
{
  const { a, onsets } = sequence(0.02);
  const { cuts } = run(a, 0, { record: true });
  const pre = Math.round(PREROLL_SEC * SR);
  const hitCuts = cuts.filter(c => c[1]).map(c => c[0]);
  const errs = onsets.map(o => {
    let best = Infinity;
    for (const c of hitCuts) { const e = c + pre - o.at; if (Math.abs(e) < Math.abs(best)) best = e; }
    return best;
  });
  const found = errs.filter(e => Math.abs(e) < 0.01 * SR);
  const mean = found.reduce((x, y) => x + y, 0) / found.length;
  const worst = Math.max(...found.map(Math.abs));
  console.log(`  ${found.length}/${onsets.length} hits cut · onset error mean ${ms(mean)}, worst ${ms(worst)}`);
  console.log(`  ${cuts.length} cuts in total, ${hitCuts.length} on hits, ${cuts.length - hitCuts.length} on the grid`);
  check(found.length >= onsets.length * 0.98, 'at least 98% of hits get their own cut');
  check(Math.abs(mean) <= 0.002 * SR, 'mean onset error within 2 ms (backdate calibrated)');
}

// ── 2. a performance offset by half a slice still locks on ───────────────
console.log('\n2. Grid lock with the performance offset half a slice from the engine');
{
  const { a, onsets } = sequence(0, 7);
  const shift = Math.round(0.06 * SR);     // no hit falls near an engine grid line
  const b = new Float32Array(a.length); b.set(a.subarray(0, a.length - shift), shift);
  const { cuts } = run(b, 0, { record: true });
  const pre = Math.round(PREROLL_SEC * SR);
  const hits = cuts.filter(c => c[1]).map(c => c[0] + pre);
  const locked = onsets.filter(o => hits.some(h => Math.abs(h - (o.at + shift)) < 0.005 * SR)).length;
  console.log(`  ${locked}/${onsets.length} hits cut after shifting 60 ms off-grid`);
  check(locked >= onsets.length - 2, 'locks within the first couple of hits');
}

// ── 3. probability 0 is an exact delayed copy ────────────────────────────
console.log('\n3. Probability 0 is transparent');
{
  const { a } = sequence(0.02);
  const { out } = run(a, 0);
  let err = 0;
  for (let i = LAT; i < a.length; i++) err = Math.max(err, Math.abs(out[i] - a[i - LAT]));
  console.log(`  max |out[n] - in[n - ${ms(LAT)}]| = ${err.toExponential(2)}`);
  check(err === 0, 'bit-exact, delayed by the reported latency');
}

// ── 4. rhythm rearranged, pitch of every hit intact ──────────────────────
console.log('\n4. Rhythm rearranged; every slice still a clean pitch (melody over one chord)');
console.log('  prob   same step as input   pitch purity   level vs input');
{
  const { a, onsets } = sequence(0.02);
  const analysed = onsets.slice(BAR_SLICES);      // bar 1 has nothing earlier to swap in
  const inRms = rms(a, SR, a.length - SR);
  const rows = [];
  for (const prob of [0, 0.25, 0.5, 1]) {
    const { out } = run(a, prob, { seed: 3 });
    const seq = readSequence(out, analysed);
    const same = seq.filter((s, i) => s.step === analysed[i].step).length / seq.length;
    const purity = seq.reduce((x, s) => x + s.purity, 0) / seq.length;
    const db = 20 * Math.log10(rms(out, SR + LAT, out.length - SR) / inRms);
    rows.push({ prob, same, purity, db });
    console.log(`  ${prob.toFixed(2)}   ${pct(same).padStart(10)}           ${pct(purity).padStart(6)}        ${db >= 0 ? '+' : ''}${db.toFixed(2)} dB`);
  }
  check(rows[0].same === 1, 'probability 0 keeps every step in place');
  check(rows[3].same < 0.2, 'probability 1 leaves the order near chance — the gate does not freeze a melody');
  check(rows[1].same > rows[2].same && rows[2].same > rows[3].same, 'order falls steadily with probability');
  check(rows.every(r => r.purity > 0.8), 'every slice stays a single clear pitch — nothing blurred');
  check(rows.every(r => Math.abs(r.db) < 0.5), 'level within ±0.5 dB at every setting');
}

// ── 5. attacks survive (a smear would erase them) ────────────────────────
// Sharpness measured directly, with no detector involved: the level in the
// 10 ms after each hit against the 10 ms before it. A smear pulls this toward
// 0 dB; a scramble that cuts on the hits should leave it where it was.
//
// The notes here ring on into the next one. With notes that stop dead before
// the next hit, the input's "before" is true silence and the ratio measures
// the gap, not the attack — and reordering slices of different lengths
// legitimately changes how long a note rings before the next hit lands.
console.log('\n5. Attacks survive (notes ringing into each other)');
{
  const { a, onsets } = sequence(0.02, 7, C_MAJOR, { tau: 0.15, dur: 0.6 });
  const analysed = onsets.slice(BAR_SLICES);
  const w = Math.round(0.01 * SR);
  const jumps = (x, shift) => analysed.map(({ at }) => {
    const i = at + shift;
    return 20 * Math.log10(rms(x, i, i + w) / Math.max(1e-3, rms(x, i - w, i)));
  });
  const { out } = run(a, 1, { seed: 3 });
  const ji = median(jumps(a, 0)), jo = median(jumps(out, LAT));
  console.log(`  median level jump across a hit: input ${ji.toFixed(1)} dB, scrambled ${jo.toFixed(1)} dB`);
  check(jo > ji - 3, 'attacks within 3 dB of the original sharpness');
}

// ── 6. a sustained tone: pitch kept, no clicks, no flutter ───────────────
console.log('\n6. Sustained 220 Hz sine through probability 1');
{
  const n = SR * 6;
  const a = Float32Array.from({ length: n }, (_, i) => 0.5 * Math.sin(2 * Math.PI * 220 * i / SR));
  const { out, cuts } = run(a, 1, { seed: 5, record: true });
  // Spectral peak, not zero crossings: a phase jump at a seam can add or drop
  // a crossing without moving the pitch at all.
  let hz = 0, best = -1;
  for (let f = 215; f <= 225; f += 0.1) {
    const pw = tone(out, f, SR, n - SR);
    if (pw > best) { best = pw; hz = f; }
  }
  let dIn = 0, dOut = 0;
  for (let i = 1; i < n; i++) dIn = Math.max(dIn, Math.abs(a[i] - a[i - 1]));
  for (let i = SR; i < n; i++) dOut = Math.max(dOut, Math.abs(out[i] - out[i - 1]));
  // Flutter is out-of-phase copies cancelling at a seam: look for the deepest
  // dip in a 2 ms level envelope against the steady level.
  const wd = Math.round(0.002 * SR), steady = rms(a, SR, n - SR);
  let dip = Infinity;
  for (let i = SR; i + wd < n - SR; i += wd >> 1) dip = Math.min(dip, rms(out, i, i + wd));
  const dipDb = 20 * Math.log10(Math.max(1e-6, dip) / steady);
  const swapped = cuts.filter(c => c[2]).length;
  console.log(`  ${swapped}/${cuts.length} slices swapped`);
  console.log(`  frequency ${hz.toFixed(1)} Hz · largest sample step ${dOut.toFixed(4)} vs ${dIn.toFixed(4)} · deepest dip ${dipDb.toFixed(1)} dB`);
  check(Math.abs(hz - 220) < 0.5, 'pitch unchanged');
  check(dOut < dIn * 1.5, 'no step at any seam larger than the signal itself makes');
  check(dipDb > -3, 'no seam dips more than 3 dB — aligned, so nothing cancels');
}

// ── 7. harmony: a chord change every bar ─────────────────────────────────
// Chords with real overtones (2nd and 3rd harmonics), since the 3rd harmonic
// of every chord tone lands on a pitch class outside the chord and pure sines
// would flatter the gate.
console.log('\n7. Harmony across a chord change every bar (C, F, G, Am, with overtones)');
{
  const CHORDS = [[261.63, 329.63, 392.0], [349.23, 440.0, 523.25], [392.0, 493.88, 587.33], [440.0, 523.25, 659.25]];
  const barSec = BAR_SLICES * SLICE;
  const n = Math.ceil((BARS * barSec + 0.5) * SR);
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const c = CHORDS[Math.floor(i / SR / barSec) % 4];
    const t = i / SR;
    let v = 0;
    for (const f of c) v += Math.sin(2 * Math.PI * f * t) + 0.5 * Math.sin(4 * Math.PI * f * t) + 0.33 * Math.sin(6 * Math.PI * f * t);
    a[i] = 0.12 * v;
  }
  const chordOf = (x, i0, i1) => {
    const e = CHORDS.map(c => c.reduce((s, f) => s + tone(x, f, i0, i1), 0));
    return e.indexOf(Math.max(...e));
  };
  for (const prob of [0, 1]) {
    const { out, cuts } = run(a, prob, { seed: 9, record: true });
    let right = 0, total = 0;
    for (let b = 1; b < BARS; b++) for (let k = 0; k < BAR_SLICES; k++) {
      const i0 = Math.round((b * barSec + k * SLICE + 0.02) * SR) + LAT;
      if (i0 + 0.08 * SR >= n) continue;
      total++;
      if (chordOf(out, i0, i0 + Math.round(0.08 * SR)) === b % 4) right++;
    }
    const swapped = cuts.filter(c => c[2]).length / cuts.length;
    console.log(`  probability ${prob}: ${pct(right / total)} of slices carry the bar's own chord · ${pct(swapped)} swapped`);
    if (prob === 1) {
      console.log('  (before the harmony gate this measured 47%: half of any bar-long pool is the previous bar)');
      check(right / total >= 0.85, 'at least 85% of slices carry the current chord');
      check(swapped > 0.5, 'the gate still lets most slices swap — it filters, it does not freeze');
    }
  }
}

// ── 8. drums are never gated ─────────────────────────────────────────────
console.log('\n8. Untonal hits (noise bursts) are never held back by the gate');
{
  const n = Math.ceil((BARS * BAR_SLICES * SLICE + 0.5) * SR);
  const a = new Float32Array(n);
  const rng = mulberry32(11);
  for (let s = 0; s < BARS * BAR_SLICES; s++) {
    const i0 = Math.round((LEAD + s * SLICE) * SR);
    for (let i = 0; i < 0.06 * SR && i0 + i < n; i++) a[i0 + i] = (rng() * 2 - 1) * 0.5 * Math.exp(-i / (0.015 * SR));
  }
  const { cuts } = run(a, 1, { seed: 13, record: true });
  const late = cuts.filter(c => c[0] > (LEAD + BAR_SLICES * SLICE) * SR);
  const swapped = late.filter(c => c[2]).length / late.length;
  console.log(`  ${pct(swapped)} of slices after the first bar swapped`);
  check(swapped > 0.95, 'nearly every slice swaps: no harmony, nothing to protect');
}

console.log(`\n${failures === 0 ? 'All checks passed.' : `${failures} check(s) failed.`}\n`);
process.exit(failures ? 1 : 0);
