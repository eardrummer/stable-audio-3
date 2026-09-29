import {
  forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState,
} from 'react';
import { Plus, Minus, Headphones } from 'lucide-react';
import MidiSlider from './MidiSlider';
import { drawSpectrogram } from '../shared/spectrogram';
import {
  WS_URL, VARIANTS, MAX_NUDGE, XF_MAX_CYCLE, XF_SWITCH, XF_MIN,
  FADE_IN_CURVE, FADE_OUT_CURVE, TRACK_THEME,
  lpfHz, hpfHz, combGains, softClipCurve, COMB_LO, COMB_FB_HPF, SPREAD_MAX,
  type TrackParams, type SharedParams, type LearnProps,
} from './trackTypes';

/**
 * One output stream: its own backend session, its own generation parameters and
 * its own playback chain. Every track generates from the same input snapshot,
 * which the parent pushes to each of them.
 *
 * A track maps one-to-one onto a backend LiveSession, which already keeps its
 * latent cache, loopback memory and output directory per connection — so a
 * track's Loopback feeds back *its own* audio, not the other track's.
 */

export interface TrackHandle {
  push(header: object, pcm: ArrayBuffer): void;
  begin(): void;
  /** Silence what is sounding without ending the session. */
  flush(): void;
  finish(): void;
  sendWeights(w: Float32Array, renorm: boolean): void;
}

interface Props {
  index: number;
  connectionId: string;
  params: TrackParams;
  shared: SharedParams;
  onParam: (patch: Partial<TrackParams>) => void;
  onMidi?: (channel: number, control: number, norm: number) => void;
  onAdd?: () => void;
  onRemove?: () => void;
  onError: (msg: string) => void;
  onShelfDone?: () => void;
  onConnected?: (c: boolean) => void;
  learnProps: (id: string) => LearnProps;
  running: boolean;
  /** The track the knob row is editing. */
  active: boolean;
  onActivate: () => void;
  soloed: boolean;
  /** Another track is soloed, so this one is silenced. */
  muted: boolean;
  onSolo: () => void;
  soloVisible: boolean;
  /** Registers this track's patch anchor with the cable overlay. */
  onNode: (el: HTMLDivElement | null) => void;
  /** A cable is being dragged and could land here. */
  patchTarget: boolean;
  /** Start a cable at this track, to pull one out from the output end. */
  onNodeDown: (e: React.PointerEvent) => void;
}

interface CycleAudio { cycle: number; buffers: (AudioBuffer | null)[]; }

/** One sounding clip. In ALL mode a track runs several of these at once. */
/** How many DISTINCT clips a cycle needs.
 *
 *  At spread 0 every variant would get the same noise level, and the backend
 *  already shares one seed across the family, so all five come out bit-identical
 *  — generating them is five times the work for the same audio. We ask for one
 *  and fan it across the pan positions instead, which sounds the same. */
const familySize = (p: { multiVariant: boolean; spread: number }) =>
  (p.multiVariant && p.spread > 0 ? VARIANTS : 1);

interface Voice {
  src: AudioBufferSourceNode;
  gain: GainNode;
  startedAt: number;
  dur: number;
  /** A silent second copy of the clip, running LOOKAHEAD_FRAC of its length
   *  ahead of the audible one, and the tap that reads it. This is what lets
   *  the display draw audio that has been generated but not yet heard. */
  pre?: AudioBufferSourceNode;
  an?: AnalyserNode;
}

/** Gap between variant lanes, in canvas pixels. It lies outside every lane's
 *  scroll region, so the divider stays put while the lanes fall past it. */
const LANE_GUTTER = 3;

/**
 * Where the playback line sits, as a fraction of the pane height measured from
 * the top, and therefore also how far ahead the display runs. One clip spans
 * the pane, so a third of the height is a third of the clip.
 *
 * Above the line is audio the model has already produced and you have not
 * reached yet; below it is what you have heard. At Generate Rate below Input
 * Rate the tail above the line is material the next cycle will cut off, which
 * is the whole reason to show it.
 */
const LOOKAHEAD_FRAC = 1 / 3;
/** How long Flush takes to reach silence. */
const FLUSH_FADE = 0.03;
/** Dimming over the not-yet-heard region. */
const AHEAD_SCRIM = 'rgba(0, 0, 0, .55)';

/**
 * How tall to draw variant i's spread marker, as a fraction of its gauge.
 *
 * The rungs of the ladder are `i * spread` above the base noise, measured
 * against the widest ladder the Spread knob can build. Straight proportion
 * would make the useful settings invisible — a single knob detent of 0.0025
 * is 3% of full scale — so the bar is that raised to 0.4, which lifts a
 * one-detent ladder to a legible quarter of the gauge, with a floor so no
 * rung above zero disappears entirely. Zero stays exactly zero: a flat row
 * means the variants really are identical, which at spread 0 they are, since
 * one clip is generated and fanned across the pans.
 */
const spreadBar = (i: number, spread: number) => {
  const raw = (i * spread) / (SPREAD_MAX * (VARIANTS - 1));
  return raw <= 0 ? 0 : Math.max(0.09, Math.pow(Math.min(1, raw), 0.4));
};

/** Variants spread evenly left to right; a lone variant sits centre. */
const panFor = (i: number, n: number) => (n <= 1 ? 0 : -1 + (2 * i) / (n - 1));

/** StereoPannerNode's stereo-input law, so the estimate below matches what the
 *  graph will actually produce rather than an idealised equal-power curve. */
function panMatrix(pan: number) {
  if (pan <= 0) {
    const x = (pan + 1) * Math.PI / 2;
    return { ll: 1, lr: Math.cos(x), rl: 0, rr: Math.sin(x) };
  }
  const x = pan * Math.PI / 2;
  return { ll: Math.cos(x), lr: 0, rl: Math.sin(x), rr: 1 };
}

/** Peak of the summed, panned set — so ALL mode plays at the level of a single
 *  variant instead of stacking to N times as loud.
 *
 *  A fixed 1/N or 1/sqrt(N) cannot work here: the variants share a seed and an
 *  input, so at small spread they are nearly identical (amplitudes add, needing
 *  1/N) and at large spread they decorrelate (power adds, needing 1/sqrt(N)).
 *  Measuring the actual sum covers the whole range. */
const NORM_TARGET = 0.89;      // headroom below full scale
const NORM_STRIDE = 2;         // every other frame; peak error is negligible

function sumGain(entries: { buf: AudioBuffer; pan: number }[]): number {
  if (entries.length <= 1) return 1;
  const chans = entries.map(({ buf, pan }) => ({
    L: buf.getChannelData(0),
    R: buf.numberOfChannels > 1 ? buf.getChannelData(1) : buf.getChannelData(0),
    m: panMatrix(pan),
  }));
  const n = Math.min(...entries.map(e => e.buf.length));
  let peak = 0;
  for (let i = 0; i < n; i += NORM_STRIDE) {
    let sl = 0, sr = 0;
    for (const c of chans) {
      const l = c.L[i], r = c.R[i];
      sl += l * c.m.ll + r * c.m.lr;
      sr += l * c.m.rl + r * c.m.rr;
    }
    const a = Math.abs(sl), b = Math.abs(sr);
    if (a > peak) peak = a;
    if (b > peak) peak = b;
  }
  return peak > 1e-6 ? Math.min(1, NORM_TARGET / peak) : 1;
}

const TrackOutput = forwardRef<TrackHandle, Props>(function TrackOutput({
  index, connectionId, params, shared, onParam, onMidi, onAdd, onRemove,
  onError, onShelfDone, onConnected, learnProps, running,
  active, onActivate, soloed, muted, onSolo, soloVisible, onNode, patchTarget, onNodeDown,
}, ref) {
  const theme = TRACK_THEME[index] ?? TRACK_THEME[0];

  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState('Idle');
  const [cycleInfo, setCycleInfo] = useState<{ n: number; ms: number } | null>(null);
  const [dropped, setDropped] = useState(0);
  /** How many variant lanes the spectrogram is currently split into. */
  const [lanes, setLanes] = useState(1);

  const wsRef = useRef<WebSocket | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const delayRef = useRef<DelayNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  // Output filter block: every voice lands on fxIn, whatever the panning.
  const fxInRef = useRef<GainNode | null>(null);
  const lpfNodeRef = useRef<BiquadFilterNode | null>(null);
  const hpfNodeRef = useRef<BiquadFilterNode | null>(null);
  const combDelayRef = useRef<DelayNode | null>(null);
  const combFbRef = useRef<GainNode | null>(null);
  const combWetRef = useRef<GainNode | null>(null);
  const combDryRef = useRef<GainNode | null>(null);
  const combPreRef = useRef<GainNode | null>(null);
  const combPostRef = useRef<GainNode | null>(null);
  /** Silent path to the destination for the look-ahead taps. A subgraph with
   *  no route to the destination is not guaranteed to be rendered at all, so
   *  the previews would read as silence without it. */
  const previewSinkRef = useRef<GainNode | null>(null);
  const specRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const cycleRef = useRef<CycleAudio | null>(null);
  const pendingHeaderRef = useRef<any>(null);
  const voicesRef = useRef<Voice[]>([]);

  // The socket handler is created once; live values reach it through refs.
  const paramsRef = useRef(params);
  useEffect(() => { paramsRef.current = params; }, [params]);
  const mutedRef = useRef(muted);
  useEffect(() => { mutedRef.current = muted; }, [muted]);
  const sharedRef = useRef(shared);
  useEffect(() => { sharedRef.current = shared; }, [shared]);
  const midiRef = useRef(onMidi);
  useEffect(() => { midiRef.current = onMidi; }, [onMidi]);

  const activeVariants = params.multiVariant ? VARIANTS : 1;

  // ── output chain ────────────────────────────────────────────────────
  const ensureCtx = useCallback(() => {
    if (!ctxRef.current) {
      const ctx = new AudioContext();
      const gain = ctx.createGain();
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.6;
      // Nudge sits ahead of the analyser so the spectrogram shows the audio at
      // the moment it is heard, not before the delay.
      const delay = ctx.createDelay(MAX_NUDGE);
      delay.delayTime.value = paramsRef.current.nudge;
      delay.connect(analyser);
      analyser.connect(gain);
      gain.connect(ctx.destination);
      gain.gain.value = mutedRef.current ? 0 : paramsRef.current.outputMonitor;

      // ── filters ──────────────────────────────────────────────────────
      // fxIn -> LPF -> HPF -> [dry + comb] -> nudge -> analyser -> monitor.
      // They sit ahead of the analyser so the spectrogram draws what is
      // actually heard. None of this reaches the backend: filtering here keeps
      // the model's input untouched and costs no generation time.
      const p0 = paramsRef.current;
      const fxIn = ctx.createGain();
      const lp = ctx.createBiquadFilter();
      lp.type = 'lowpass';
      lp.Q.value = 0.9;                 // a hint of resonance, short of a whistle
      lp.frequency.value = lpfHz(p0.lpf);
      const hp = ctx.createBiquadFilter();
      hp.type = 'highpass';
      hp.Q.value = 0.9;
      hp.frequency.value = hpfHz(p0.hpf);

      // The comb is a feedback delay tapped in parallel: its resonance sits at
      // 1 / delayTime, so the knob reads as a pitch. Web Audio allows the
      // feedback cycle because a DelayNode breaks it.
      const combDelay = ctx.createDelay(1 / COMB_LO + 0.01);   // headroom over the lowest note
      const combFb = ctx.createGain();
      const combWet = ctx.createGain();
      const combDry = ctx.createGain();
      // Inside the loop: at 0.95 feedback any DC or subsonic content would be
      // multiplied twentyfold and swallow the resonance.
      const combFbHp = ctx.createBiquadFilter();
      combFbHp.type = 'highpass';
      combFbHp.frequency.value = COMB_FB_HPF;
      // After the sum: linear until the knee, then saturating. This is what
      // absorbs a deep resonator's overshoot, and most of what gives the effect
      // its bite — without it the destination would hard-clip instead.
      const shaper = ctx.createWaveShaper();
      shaper.curve = softClipCurve();
      shaper.oversample = '4x';
      // Trim into the clipper and make it back up on the way out. At comb 0
      // this keeps the signal in the curve's straight section, so the whole
      // block is transparent until you actually turn the knob.
      const combPre = ctx.createGain();
      const combPost = ctx.createGain();

      const g0 = combGains(p0.comb);
      combDelay.delayTime.value = g0.delaySec;
      combFb.gain.value = g0.fb;
      combWet.gain.value = g0.wet;
      combDry.gain.value = g0.dry;
      combPre.gain.value = g0.pre;
      combPost.gain.value = g0.post;

      fxIn.connect(lp);
      lp.connect(hp);
      hp.connect(combDry);
      hp.connect(combDelay);
      combDelay.connect(combFbHp);
      combFbHp.connect(combFb);
      combFb.connect(combDelay);
      combDelay.connect(combWet);
      combDry.connect(combPre);
      combWet.connect(combPre);
      combPre.connect(shaper);
      shaper.connect(combPost);
      combPost.connect(delay);

      const previewSink = ctx.createGain();
      previewSink.gain.value = 0;
      previewSink.connect(ctx.destination);
      previewSinkRef.current = previewSink;

      ctxRef.current = ctx; gainRef.current = gain;
      analyserRef.current = analyser; delayRef.current = delay;
      fxInRef.current = fxIn;
      lpfNodeRef.current = lp; hpfNodeRef.current = hp;
      combDelayRef.current = combDelay; combFbRef.current = combFb;
      combWetRef.current = combWet; combDryRef.current = combDry;
      combPreRef.current = combPre; combPostRef.current = combPost;
    }
    return ctxRef.current!;
  }, []);

  /** Start a set of clips together, equal-power crossfading out whatever is
   *  sounding. One entry in single-variant mode; one per variant in ALL mode,
   *  each panned to its own place in the stereo field. */
  const playSet = useCallback((
    entries: { buf: AudioBuffer; pan: number }[],
    offset = 0,
    maxFade = XF_MAX_CYCLE,
  ) => {
    if (!entries.length) return;
    const ctx = ensureCtx();
    const now = ctx.currentTime;
    const prev = voicesRef.current;

    const shortest = Math.min(...entries.map(e => e.buf.duration));
    const fade = Math.max(XF_MIN, Math.min(maxFade, shortest - offset));
    // One gain for the whole set, measured from the actual summed peak.
    const k = sumGain(entries);
    const fadeIn = k === 1 ? FADE_IN_CURVE : FADE_IN_CURVE.map(v => v * k);

    const next: Voice[] = entries.map(({ buf, pan }) => {
      const gain = ctx.createGain();
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(gain);
      // The display runs ahead of the ears: a second, silent copy of the same
      // clip plays LOOKAHEAD_FRAC of its length early, and the spectrogram is
      // drawn from that. By the time a painted row has scrolled down to the
      // playback line, the audible voice has reached the same moment.
      //
      // This tap is pre-filter — LPF/HPF/comb run on the sum, downstream of
      // here — but it is the only place a variant is still separable, and the
      // only place the future exists at all.
      const an = ctx.createAnalyser();
      an.fftSize = 1024;
      an.smoothingTimeConstant = 0.6;
      const pre = ctx.createBufferSource();
      pre.buffer = buf;
      pre.connect(an);
      // An analyser passes its input through untouched, so this carries the
      // preview to a muted sink: nothing is heard, and the graph has a reason
      // to render it.
      if (previewSinkRef.current) an.connect(previewSinkRef.current);
      // A panner per voice; at pan 0 this is transparent, so the single-variant
      // path behaves exactly as it did before.
      if (pan !== 0) {
        const panner = ctx.createStereoPanner();
        panner.pan.value = Math.max(-1, Math.min(1, pan));
        gain.connect(panner);
        panner.connect(fxInRef.current ?? analyserRef.current!);
      } else {
        gain.connect(fxInRef.current ?? analyserRef.current!);
      }

      try {
        gain.gain.setValueCurveAtTime(fadeIn, now, fade);
      } catch {
        gain.gain.setValueAtTime(0, now);
        gain.gain.linearRampToValueAtTime(k, now + fade);
      }
      const at = Math.min(offset, Math.max(0, buf.duration - 0.05));
      src.start(now, at);
      pre.start(now, Math.min(at + buf.duration * LOOKAHEAD_FRAC, buf.duration - 0.01));
      return { src, gain, startedAt: now - offset, dur: buf.duration, pre, an };
    });

    for (const v of prev) {
      const g = v.gain.gain;
      const from = g.value;
      try {
        g.cancelScheduledValues(now);
        g.setValueCurveAtTime(FADE_OUT_CURVE.map(x => x * from), now, fade);
      } catch {
        g.cancelScheduledValues(now);
        g.setValueAtTime(from, now);
        g.linearRampToValueAtTime(0, now + fade);
      }
      v.src.stop(now + fade + 0.03);
      // The preview is silent, so it can go as soon as its replacement exists.
      try { v.pre?.stop(now); } catch { /* already finished */ }
    }
    voicesRef.current = next;
  }, [ensureCtx]);

  /** Everything a cycle should currently be sounding. */
  const setFromCycle = useCallback((cyc: CycleAudio, offset = 0, maxFade = XF_MAX_CYCLE) => {
    const p = paramsRef.current;
    const lanes = p.multiVariant ? VARIANTS : 1;   // pan positions to fill
    const made = familySize(p);                    // distinct clips available
    if (p.multiVariant && p.allVariants) {
      const entries = [];
      for (let i = 0; i < lanes; i++) {
        // At spread 0 every lane draws the same clip; only the pan differs.
        const buf = cyc.buffers[Math.min(i, made - 1)];
        if (buf) entries.push({ buf, pan: panFor(i, lanes) });
      }
      playSet(entries, offset, maxFade);
    } else {
      const buf = cyc.buffers[Math.min(p.variant, made - 1)];
      if (buf) playSet([{ buf, pan: 0 }], offset, maxFade);
    }
  }, [playSet]);

  useEffect(() => {
    if (gainRef.current) gainRef.current.gain.value = muted ? 0 : params.outputMonitor;
  }, [params.outputMonitor, muted]);

  useEffect(() => {
    const d = delayRef.current, c = ctxRef.current;
    if (!d || !c) return;
    d.delayTime.cancelScheduledValues(c.currentTime);
    d.delayTime.setValueAtTime(d.delayTime.value, c.currentTime);
    d.delayTime.linearRampToValueAtTime(params.nudge, c.currentTime + 0.08);
  }, [params.nudge]);

  // Filter knobs. Every move is a short ramp: stepping an AudioParam under a
  // running signal clicks, and these are meant to be swept during a set.
  useEffect(() => {
    const c = ctxRef.current;
    const lp = lpfNodeRef.current, hp = hpfNodeRef.current;
    if (!c || !lp || !hp) return;
    const t = c.currentTime, glide = 0.05;
    lp.frequency.cancelScheduledValues(t);
    lp.frequency.setValueAtTime(lp.frequency.value, t);
    lp.frequency.exponentialRampToValueAtTime(lpfHz(params.lpf), t + glide);
    hp.frequency.cancelScheduledValues(t);
    hp.frequency.setValueAtTime(hp.frequency.value, t);
    hp.frequency.exponentialRampToValueAtTime(hpfHz(params.hpf), t + glide);
  }, [params.lpf, params.hpf]);

  useEffect(() => {
    const c = ctxRef.current, d = combDelayRef.current;
    const fb = combFbRef.current, wet = combWetRef.current, dry = combDryRef.current;
    const pre = combPreRef.current, post = combPostRef.current;
    if (!c || !d || !fb || !wet || !dry || !pre || !post) return;
    const t = c.currentTime, glide = 0.05;
    const g = combGains(params.comb);
    const ramp = (p: AudioParam, to: number) => {
      p.cancelScheduledValues(t);
      p.setValueAtTime(p.value, t);
      p.linearRampToValueAtTime(to, t + glide);
    };
    // Sweeping the delay time is what makes the comb sing, so it glides too.
    ramp(d.delayTime, g.delaySec);
    ramp(fb.gain, g.fb);
    ramp(wet.gain, g.wet);
    ramp(dry.gain, g.dry);
    ramp(pre.gain, g.pre);
    ramp(post.gain, g.post);
  }, [params.comb]);

  // Switching the variant knob mid-clip swaps lanes at the same position.
  // In ALL mode every variant is already sounding, so there is nothing to swap.
  useEffect(() => {
    const cyc = cycleRef.current, cur = voicesRef.current[0];
    if (!cyc || !cur || !ctxRef.current || params.allVariants) return;
    const buf = cyc.buffers[params.variant];
    if (!buf) return;
    const off = ctxRef.current.currentTime - cur.startedAt;
    if (off >= 0 && off < buf.duration - 0.1) setFromCycle(cyc, off, XF_SWITCH);
  }, [params.variant, params.allVariants, setFromCycle]);

  // Turning ALL on or off re-voices the current cycle in place.
  useEffect(() => {
    const cyc = cycleRef.current, cur = voicesRef.current[0];
    if (!cyc || !cur || !ctxRef.current) return;
    const off = ctxRef.current.currentTime - cur.startedAt;
    setFromCycle(cyc, off > 0 ? off : 0, XF_SWITCH);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params.allVariants]);

  // ── spectrogram ─────────────────────────────────────────────────────
  // One clip spans the pane, so the scroll is driven by the audio clock rather
  // than by frame count: a pixel is always the same number of seconds, which is
  // what makes a fixed playback line mean anything. The rows accumulate on an
  // offscreen canvas; the visible one is that image plus the dimming and the
  // line, neither of which must ever be baked into the scroll.
  const offRef = useRef<HTMLCanvasElement | null>(null);
  const lanesRef = useRef(0);
  const lastTRef = useRef(0);
  const carryRef = useRef(0);
  useEffect(() => {
    const tick = () => {
      rafRef.current = requestAnimationFrame(tick);
      const sc = specRef.current, ctx = ctxRef.current;
      if (!sc || !ctx) return;
      if (!offRef.current) {
        offRef.current = document.createElement('canvas');
        offRef.current.width = sc.width;
        offRef.current.height = sc.height;
      }
      const off = offRef.current;
      const g = sc.getContext('2d');
      if (!g) return;

      const taps = voicesRef.current.filter(v => v.an);
      const n = taps.length || 1;
      if (n !== lanesRef.current) {
        lanesRef.current = n;
        setLanes(n);
        // The old image is at the wrong scale now; wipe rather than let it
        // scroll away over a whole clip.
        off.getContext('2d')?.clearRect(0, 0, off.width, off.height);
      }

      // Pixels per second, from the clip currently playing.
      const dur = taps[0]?.dur || sharedRef.current.inputRate || 10;
      const pps = sc.height / dur;

      const t = ctx.currentTime;
      const dt = lastTRef.current ? t - lastTRef.current : 0;
      lastTRef.current = t;
      // A backgrounded tab returns with a huge dt; a whole pane is the most
      // that can be meaningfully redrawn from a single analyser reading.
      if (!taps.length) carryRef.current = 0;
      else carryRef.current += Math.min(dt, dur) * pps;
      const step = Math.floor(carryRef.current);
      if (step >= 1 && taps.length) {
        carryRef.current -= step;
        if (n === 1) {
          drawSpectrogram(off, taps[0].an!, theme.spec, { shift: step });
        } else {
          const lane = (off.width - LANE_GUTTER * (n - 1)) / n;
          taps.forEach((v, i) => drawSpectrogram(off, v.an!, theme.spec, {
            x: Math.round(i * (lane + LANE_GUTTER)),
            w: Math.round(lane),
            shift: step,
          }));
        }
      }

      // Nudge delays the ears but not the display, so the moment being heard
      // sits that much further down the falling image.
      const lineY = Math.min(
        sc.height - 1,
        Math.round(sc.height * LOOKAHEAD_FRAC + paramsRef.current.nudge * pps),
      );
      g.clearRect(0, 0, sc.width, sc.height);
      g.drawImage(off, 0, 0);
      g.fillStyle = AHEAD_SCRIM;
      g.fillRect(0, 0, sc.width, lineY);
      g.fillStyle = theme.accent;
      g.fillRect(0, lineY - 1, sc.width, 2);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [theme.spec, theme.accent]);

  // ── socket ──────────────────────────────────────────────────────────
  const sendParams = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const p = paramsRef.current, sh = sharedRef.current;
    ws.send(JSON.stringify({
      type: 'params',
      params: {
        prompt: p.prompt, input_rate: sh.inputRate, generate_rate: sh.generateRate,
        noise: p.noise, variant_delta: p.spread, steps: p.steps,
        cfg: p.cfg, variants: familySize(p), model: sh.model,
        loopback: p.loopback, selected_variant: p.variant,
      },
    }));
  }, []);

  useEffect(() => {
    // Each track is its own backend session, so it gets its own connection id.
    const cid = `${connectionId}-t${index}`;
    const ws = new WebSocket(`${WS_URL}/ws/live/${cid}`);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen = () => { setConnected(true); onConnected?.(true); };
    ws.onclose = () => { setConnected(false); onConnected?.(false); };
    ws.onerror = () => onError(`Track ${index + 1}: WebSocket error — is the backend running?`);

    ws.onmessage = async (ev) => {
      if (typeof ev.data !== 'string') {
        const hdr = pendingHeaderRef.current;
        pendingHeaderRef.current = null;
        if (!hdr) return;
        const ctx = ensureCtx();
        const buf = await ctx.decodeAudioData(ev.data as ArrayBuffer);
        let cyc = cycleRef.current;
        if (!cyc || cyc.cycle !== hdr.cycle) {
          cyc = { cycle: hdr.cycle, buffers: new Array(VARIANTS).fill(null) };
          cycleRef.current = cyc;
        }
        cyc.buffers[hdr.index] = buf;

        const pp = paramsRef.current;
        const wantAll = pp.multiVariant && pp.allVariants;
        const family = familySize(pp);

        if (hdr.redecode) {
          onShelfDone?.();
          const cur = voicesRef.current[0];
          const audible = wantAll || hdr.index === pp.variant;
          if (audible && cur && ctxRef.current) {
            const off = ctxRef.current.currentTime - cur.startedAt;
            setFromCycle(cyc, off >= 0 && off < buf.duration - 0.1 ? off : 0, XF_SWITCH);
          }
          return;
        }

        if (wantAll) {
          // Hold until the whole family has landed, so the variants start
          // together as one panned image rather than trickling in.
          const ready = cyc.buffers.slice(0, family).every(Boolean);
          if (ready) setFromCycle(cyc, 0);
        } else if (hdr.index === pp.variant) {
          // Single variant: start as soon as the audible lane lands.
          setFromCycle(cyc, 0);
        }
        return;
      }

      const d = JSON.parse(ev.data);
      switch (d.type) {
        case 'variant': pendingHeaderRef.current = d; break;
        case 'cycle_start': setStatus(`Generating ${d.cycle}…`); break;
        case 'cycle_complete':
          setCycleInfo({ n: d.cycle, ms: d.total_ms });
          setStatus('Listening');
          break;
        case 'dropped': setDropped(n => n + 1); break;
        // The MIDI bridge broadcasts to every session, so only one track may
        // act on it — otherwise a single CC would be applied twice.
        case 'midi_cc': midiRef.current?.(d.channel, d.control, d.norm); break;
        case 'error': onError(`Track ${index + 1}: ${d.error}`); break;
      }
    };

    return () => ws.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connectionId, index, setFromCycle, ensureCtx]);

  useEffect(() => { if (running) sendParams(); }, [
    running, sendParams,
    params.prompt, params.noise, params.steps, params.cfg,
    params.loopback, params.variant, params.multiVariant, params.spread, params.allVariants,
    shared.inputRate, shared.generateRate, shared.model,
  ]);

  // ── handle for the parent ───────────────────────────────────────────
  useImperativeHandle(ref, () => ({
    push(header, pcm) {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      ws.send(JSON.stringify(header));
      ws.send(pcm);
    },
    begin() {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const p = paramsRef.current, sh = sharedRef.current;
      setDropped(0);
      const c = ensureCtx();
      if (c.state === 'suspended') void c.resume();
      ws.send(JSON.stringify({
        type: 'start',
        params: {
          prompt: p.prompt, input_rate: sh.inputRate, generate_rate: sh.generateRate,
          noise: p.noise, variant_delta: p.spread, steps: p.steps, cfg: p.cfg,
          variants: familySize(p), model: sh.model,
          loopback: p.loopback, selected_variant: p.variant,
        },
      }));
    },
    flush() {
      // The backend forgets its own last output too, or Loopback would feed
      // the model the very material just cut.
      wsRef.current?.send(JSON.stringify({ type: 'flush' }));
      const ctx = ctxRef.current;
      if (ctx) {
        const now = ctx.currentTime;
        for (const v of voicesRef.current) {
          v.gain.gain.cancelScheduledValues(now);
          v.gain.gain.setValueAtTime(Math.max(v.gain.gain.value, 0.0001), now);
          // Not an instant cut: stopping mid-waveform clicks, and 30 ms still
          // reads as immediate.
          v.gain.gain.exponentialRampToValueAtTime(0.0001, now + FLUSH_FADE);
          v.src.stop(now + FLUSH_FADE + 0.02);
          try { v.pre?.stop(now); } catch { /* already finished */ }
        }
        voicesRef.current = [];
      }
      // The Variant knob and the ALL toggle both re-voice from the cached
      // cycle, so it has to go as well or a knob touch brings the audio back.
      cycleRef.current = null;
      setCycleInfo(null);
    },
    finish() {
      wsRef.current?.send(JSON.stringify({ type: 'stop' }));
      setStatus('Idle');
      const ctx = ctxRef.current;
      if (ctx) {
        const now = ctx.currentTime;
        for (const v of voicesRef.current) {
          v.gain.gain.cancelScheduledValues(now);
          v.gain.gain.setValueAtTime(Math.max(v.gain.gain.value, 0.0001), now);
          v.gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.2);
          v.src.stop(now + 0.25);
          try { v.pre?.stop(now); } catch { /* already finished */ }
        }
        voicesRef.current = [];
      }
    },
    sendWeights(weights, renorm) {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      ws.send(JSON.stringify({
        type: 'weights',
        weights: Array.from(weights),
        renormalize: renorm,
        redecode_index: paramsRef.current.variant,
      }));
    },
  }), [ensureCtx]);

  return (
    <section
      className={`out-pane ${active ? 'active' : ''} ${muted ? 'muted' : ''}` + (patchTarget ? ' patch-target' : '')}
      style={{ ['--track' as string]: theme.accent }}
      onPointerDown={onActivate}
    >
      {/* Cables from the input sources land here. */}
      <div ref={onNode} className="patch-node in" data-track={index}
           onPointerDown={onNodeDown}
           title="Drag to an input to connect or disconnect" />

      <div className="out-head">
        <span className="out-title">{theme.name}</span>
        <span className="out-stat">
          {connected ? status : 'disconnected'}
          {cycleInfo && <span className="dim"> · {(cycleInfo.ms / 1000).toFixed(2)}s</span>}
          {dropped > 0 && <span className="warn"> · {dropped} dropped</span>}
        </span>
        {soloVisible && (
          <button
            className={`track-btn solo ${soloed ? 'on' : ''}`}
            onClick={onSolo}
            title={soloed ? 'Un-solo' : 'Solo this track (silences the other)'}
          ><Headphones size={12} /></button>
        )}
        {/* Independent, not either/or: the last track both adds and removes. */}
        {onAdd && (
          <button className="track-btn" onClick={onAdd} title="Add another output track">
            <Plus size={13} />
          </button>
        )}
        {onRemove && (
          <button className="track-btn del" onClick={onRemove} title="Remove this track">
            <Minus size={13} />
          </button>
        )}
      </div>

      <input
        className="prompt-box" value={params.prompt} placeholder={`Text prompt — ${theme.name}`}
        onChange={(e) => onParam({ prompt: e.target.value })}
        // Clicking into a track's prompt is how you choose which track the
        // knob row edits — no separate tab strip needed.
        onFocus={onActivate}
      />

      {/* In ALL mode the pane is divided one column per variant; the numbers
          sit outside the canvas so they do not scroll away with the image. */}
      <div className="spec-wrap">
        <canvas ref={specRef} width={900} height={420} className="pane-canvas" />
        {lanes > 1 && (
          <div className={`lane-marks ${params.spread > 0 ? '' : 'flat'}`}>
            {Array.from({ length: lanes }, (_, i) => (
              <div key={i} className={`lane-mark ${i === params.variant ? 'on' : ''}`}>
                <div className="lane-n">{i}</div>
                {/* Height is this variant's rung on the noise ladder. */}
                <div className="lane-gauge">
                  <div className="lane-fill"
                       style={{ height: `${spreadBar(i, params.spread) * 100}%` }} />
                </div>
                <div className="lane-nz">{(params.noise + i * params.spread).toFixed(3)}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      <MidiSlider
        {...learnProps(`outputMonitor${index}`)}
        label="Monitor"
        value={params.outputMonitor}
        onChange={(v) => onParam({ outputMonitor: v })}
      />
      <div className="dim out-foot">
        {activeVariants === 1 ? '1 variant' : `${activeVariants} variants`}
      </div>
    </section>
  );
});

export default TrackOutput;
