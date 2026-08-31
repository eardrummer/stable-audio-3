import { useCallback, useEffect, useRef, useState } from 'react';
import { Play, Square, Sliders, Circle, Mic, Repeat, FolderOpen, PanelRight } from 'lucide-react';
import Knob from './Knob';
import LatentShelf from './LatentShelf';
import { drawSpectrogram } from '../shared/spectrogram';
import './live.css';

const API_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000';

const MODEL_SR = 44100;      // Stable Audio 3 is 44.1 kHz end to end
// 32 beats at 60 BPM is 32 s, so the ring must hold that much for the longest
// beat-denominated window. ~12 MB of Float32 at 48 kHz stereo.
const MAX_BUFFER_SECS = 32;  // largest InputRate the ring buffer must hold
// Crossfade timing. A new cycle overlaps the outgoing clip for as long as that
// clip still has material left (capped below), so consecutive generations blend
// instead of one being cut short. Variant switches stay snappy.
const XF_MAX_CYCLE = 2.0;    // longest overlap between consecutive generations
const XF_SWITCH = 0.25;      // variant switch / shelf re-decode
const XF_MIN = 0.12;         // shortest fade that still avoids a click

// Equal-power crossfade curves. sin²+cos² = 1, so the summed power stays flat
// across the whole transition. (An exponential ramp is linear in *log* space —
// at the midpoint both clips sit near -37 dB, which is an audible hole.)
const CURVE_N = 128;
const FADE_IN_CURVE = new Float32Array(CURVE_N);
const FADE_OUT_CURVE = new Float32Array(CURVE_N);
for (let i = 0; i < CURVE_N; i++) {
  const th = (i / (CURVE_N - 1)) * (Math.PI / 2);
  FADE_IN_CURVE[i] = Math.sin(th);
  FADE_OUT_CURVE[i] = Math.cos(th);
}
const RATES = [1, 3, 5, 10];               // seconds, when Tempo is off
const BEATS = [4, 8, 16, 32];              // beats, when Tempo is on
                                           // (1 beat is too short to diffuse well)
const TAP_RESET_MS = 2500;                 // gap after which a tap starts a new count
const TAP_HISTORY = 8;                     // intervals averaged for the estimate
const CLICK_LOOKAHEAD_MS = 25;             // scheduler wake interval
const CLICK_SCHEDULE_AHEAD = 0.12;         // seconds of clicks queued in advance
const VARIANT_DELTA = 0.02;  // fixed noise step between successive variants

type Bindings = Record<string, { channel: number; control: number }>;

/** "0" = channel 1 as mono; "0+1" = channels 1 and 2 as stereo. */
function parseChannelSel(sel: string, available: number): [number, number] {
  const clamp = (n: number) => Math.max(0, Math.min(available - 1, n));
  if (sel.includes('+')) {
    const [a, b] = sel.split('+').map(Number);
    return [clamp(a), clamp(b)];
  }
  const a = clamp(Number(sel));
  return [a, a];   // mono: same source feeds both sides
}

/** Every mono channel the device exposes, plus each adjacent stereo pair. */
function channelOptions(n: number): { value: string; label: string }[] {
  const opts = [];
  for (let i = 0; i < n; i++) opts.push({ value: String(i), label: `Ch ${i + 1} (mono)` });
  for (let i = 0; i + 1 < n; i += 2) {
    opts.push({ value: `${i}+${i + 1}`, label: `Ch ${i + 1}+${i + 2} (stereo)` });
  }
  return opts;
}

interface CycleAudio {
  cycle: number;
  buffers: (AudioBuffer | null)[];
}

export default function LiveApp() {
  const [connectionId] = useState(() => Math.random().toString(36).slice(2, 12));
  const [running, setRunning] = useState(false);
  const [connected, setConnected] = useState(false);

  // --- parameters -----------------------------------------------------
  const [prompt, setPrompt] = useState('');
  const [inputRate, setInputRate] = useState(5);        // seconds (Tempo off)
  const [generateRate, setGenerateRate] = useState(5);
  const [tempoOn, setTempoOn] = useState(false);
  const [bpm, setBpm] = useState<number | null>(null);
  const [inputBeats, setInputBeats] = useState(8);       // beats  (Tempo on)
  const [generateBeats, setGenerateBeats] = useState(4);
  const tapsRef = useRef<number[]>([]);
  const [bpmText, setBpmText] = useState('');
  const [clickOn, setClickOn] = useState(false);
  const [clickLevel, setClickLevel] = useState(0.5);
  const [loopback, setLoopback] = useState(0);
  const [noise, setNoise] = useState(0.35);
  const [steps, setSteps] = useState(8);
  const [cfg, setCfg] = useState(4.0);
  const [model, setModel] = useState<'sm-music' | 'sm-sfx' | 'medium'>('sm-music');
  const [variant, setVariant] = useState(0);
  const VARIANTS = 5;

  const [inputMonitor, setInputMonitor] = useState(0);   // muted by default
  const [outputMonitor, setOutputMonitor] = useState(0.8);

  // --- latent shelf ----------------------------------------------------
  const [shelfOpen, setShelfOpen] = useState(false);
  const [renormalize, setRenormalize] = useState(false);
  const [shelfBusy, setShelfBusy] = useState(false);

  // --- input source: live hardware, or a file looped as a stand-in ------
  const [inputSource, setInputSource] = useState<'mic' | 'file'>('mic');
  const [fileInfo, setFileInfo] = useState<{ name: string; duration: number } | null>(null);

  // --- devices / status ----------------------------------------------
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState('');
  // Which physical input(s) feed the model. Defaults to the first stereo pair;
  // the option list grows once we see how many channels the device really has.
  const [chanSel, setChanSel] = useState('0+1');
  const [detectedChannels, setDetectedChannels] = useState(2);
  const [captureRate, setCaptureRate] = useState<number | null>(null);
  const [status, setStatus] = useState('Idle');
  const [cycleInfo, setCycleInfo] = useState<{ n: number; ms: number } | null>(null);
  const [dropped, setDropped] = useState(0);
  const [fadeInfo, setFadeInfo] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  // --- MIDI ------------------------------------------------------------
  const [midiLearn, setMidiLearn] = useState(false);
  const [armed, setArmed] = useState<string | null>(null);
  const [midiPorts, setMidiPorts] = useState<string[]>([]);
  const [midiPort, setMidiPort] = useState<string | null>(null);
  const [bindings, setBindings] = useState<Bindings>(() => {
    try { return JSON.parse(localStorage.getItem('sa3_live_midi') || '{}'); }
    catch { return {}; }
  });

  // --- refs ------------------------------------------------------------
  const wsRef = useRef<WebSocket | null>(null);
  const inCtxRef = useRef<AudioContext | null>(null);
  const outCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const inGainRef = useRef<GainNode | null>(null);
  const outGainRef = useRef<GainNode | null>(null);
  const inAnalyserRef = useRef<AnalyserNode | null>(null);
  const outAnalyserRef = useRef<AnalyserNode | null>(null);
  const procRef = useRef<ScriptProcessorNode | null>(null);
  const fileRef = useRef<File | null>(null);
  const fileSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const clickGainRef = useRef<GainNode | null>(null);
  const clickTimerRef = useRef<number | null>(null);
  const nextClickRef = useRef(0);
  const clickBeatRef = useRef(0);
  const bpmRef = useRef<number | null>(null);
  const splitterRef = useRef<ChannelSplitterNode | null>(null);
  const mergerRef = useRef<ChannelMergerNode | null>(null);
  const filePickerRef = useRef<HTMLInputElement | null>(null);
  const waveCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const specCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const timerRef = useRef<number | null>(null);

  const ringRef = useRef<{ l: Float32Array; r: Float32Array; w: number; filled: number; sr: number } | null>(null);
  const cycleRef = useRef<CycleAudio | null>(null);
  const pendingHeaderRef = useRef<any>(null);
  const playingRef = useRef<{ src: AudioBufferSourceNode; gain: GainNode; startedAt: number } | null>(null);

  // With Tempo engaged the rate knobs are denominated in beats; everything
  // downstream still works in seconds, so convert here and clamp the window to
  // what the ring buffer can actually hold.
  const beatSec = tempoOn && bpm ? 60 / bpm : null;
  const effInputRate = beatSec
    ? Math.min(MAX_BUFFER_SECS, inputBeats * beatSec)
    : inputRate;
  const effGenerateRate = beatSec ? generateBeats * beatSec : generateRate;

  /** Tap tempo: average the recent intervals, restarting after a long gap. */
  const tap = useCallback(() => {
    const now = performance.now();
    const taps = tapsRef.current;
    if (taps.length && now - taps[taps.length - 1] > TAP_RESET_MS) taps.length = 0;
    taps.push(now);
    if (taps.length > TAP_HISTORY + 1) taps.shift();
    if (taps.length < 2) return;
    const gaps = [];
    for (let i = 1; i < taps.length; i++) gaps.push(taps[i] - taps[i - 1]);
    const mean = gaps.reduce((a, b) => a + b, 0) / gaps.length;
    const est = 60000 / mean;
    if (est >= 30 && est <= 300) {
      setBpm(Math.round(est * 10) / 10);
      setTempoOn(true);
    }
  }, []);

  const clearTempo = useCallback(() => { tapsRef.current = []; setBpm(null); }, []);

  // Keep the editable field in step with taps, which change bpm from outside it.
  useEffect(() => { setBpmText(bpm !== null ? String(bpm) : ''); }, [bpm]);

  /** Accept a typed BPM. A typed value supersedes the tap history entirely. */
  const commitBpm = useCallback(() => {
    const v = parseFloat(bpmText);
    if (isFinite(v) && v >= 30 && v <= 300) {
      setBpm(Math.round(v * 10) / 10);
      setTempoOn(true);
      tapsRef.current = [];
    } else {
      setBpmText(bpm !== null ? String(bpm) : '');   // reject and restore
    }
  }, [bpmText, bpm]);

  // Params the socket send path reads; refs avoid re-opening the WS per edit.
  const paramsRef = useRef({ prompt, inputRate: effInputRate, generateRate: effGenerateRate,
                             noise, steps, cfg, model, loopback, variant });
  useEffect(() => {
    paramsRef.current = { prompt, inputRate: effInputRate, generateRate: effGenerateRate,
                          noise, steps, cfg, model, loopback, variant };
  }, [prompt, effInputRate, effGenerateRate, noise, steps, cfg, model, loopback, variant]);
  const variantRef = useRef(variant);
  useEffect(() => { variantRef.current = variant; }, [variant]);
  // The canvas loop runs once and reads this, so noise can drive the wash
  // without restarting the animation frame on every knob move.
  const tempoRef = useRef(false);
  useEffect(() => { tempoRef.current = tempoOn && bpm !== null; }, [tempoOn, bpm]);
  const noiseRef = useRef(noise);
  useEffect(() => { noiseRef.current = noise; }, [noise]);
  const runningRef = useRef(running);
  useEffect(() => { runningRef.current = running; }, [running]);

  useEffect(() => { bpmRef.current = bpm; }, [bpm]);
  useEffect(() => {
    if (clickGainRef.current) clickGainRef.current.gain.value = clickLevel;
  }, [clickLevel]);

  // ─────────────────────────────────────────────────────────────────────
  // Metronome. Clicks are generated inside the input context but connected
  // straight to the destination, downstream of the merger that feeds the
  // capture tap — so the performer hears them and the model never does.
  // ─────────────────────────────────────────────────────────────────────
  useEffect(() => {
    const stopScheduler = () => {
      if (clickTimerRef.current) { clearInterval(clickTimerRef.current); clickTimerRef.current = null; }
    };
    if (!running || !clickOn || !bpm) { stopScheduler(); return; }
    const ctx = inCtxRef.current;
    if (!ctx || !clickGainRef.current) return;

    nextClickRef.current = ctx.currentTime + 0.1;
    clickBeatRef.current = 0;

    // Lookahead scheduling: a timer this coarse would jitter audibly, so it
    // only queues clicks that the audio clock then plays sample-accurately.
    clickTimerRef.current = window.setInterval(() => {
      const c = inCtxRef.current;
      const dest = clickGainRef.current;
      if (!c || !dest) return;
      const spb = 60 / (bpmRef.current || 120);
      while (nextClickRef.current < c.currentTime + CLICK_SCHEDULE_AHEAD) {
        const t = nextClickRef.current;
        const accent = clickBeatRef.current % 4 === 0;   // downbeat of each bar
        const osc = c.createOscillator();
        const env = c.createGain();
        osc.frequency.value = accent ? 1600 : 1000;
        osc.connect(env);
        env.connect(dest);
        env.gain.setValueAtTime(accent ? 1 : 0.6, t);
        env.gain.exponentialRampToValueAtTime(0.0008, t + 0.045);
        osc.start(t);
        osc.stop(t + 0.06);
        nextClickRef.current += spb;
        clickBeatRef.current++;
      }
    }, CLICK_LOOKAHEAD_MS);

    return stopScheduler;
  }, [running, clickOn, bpm]);

  // ─────────────────────────────────────────────────────────────────────
  // Device list
  // ─────────────────────────────────────────────────────────────────────
  useEffect(() => {
    navigator.mediaDevices?.enumerateDevices().then(ds => {
      const ins = ds.filter(d => d.kind === 'audioinput');
      setDevices(ins);
      if (ins[0]) setDeviceId(prev => prev || ins[0].deviceId);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    fetch(`${API_URL}/api/live/midi_ports`)
      .then(r => r.json())
      .then(d => { setMidiPorts(d.ports || []); setMidiPort(d.open || null); })
      .catch(() => {});
  }, []);

  const openMidiPort = async (name: string) => {
    try {
      const r = await fetch(`${API_URL}/api/live/midi_port`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      const d = await r.json();
      if (d.status === 'ok') setMidiPort(d.open);
      else setError(d.error || 'Failed to open MIDI port');
    } catch (e: any) { setError(e.message); }
  };

  // ─────────────────────────────────────────────────────────────────────
  // MIDI: apply an incoming CC to whatever it is bound to
  // ─────────────────────────────────────────────────────────────────────
  // CCs arrive on the socket, so the handler must read the *current* armed
  // control and bindings — refs keep it out of the WebSocket's dependency list.
  const armedRef = useRef<string | null>(null);
  const bindingsRef = useRef<Bindings>(bindings);
  useEffect(() => { armedRef.current = armed; }, [armed]);
  useEffect(() => { bindingsRef.current = bindings; }, [bindings]);

  const applyMidi = useCallback((channel: number, control: number, norm: number) => {
    // Learn mode: the next CC claims whichever control is armed.
    const armedId = armedRef.current;
    if (armedId) {
      const next = { ...bindingsRef.current, [armedId]: { channel, control } };
      bindingsRef.current = next;
      setBindings(next);
      localStorage.setItem('sa3_live_midi', JSON.stringify(next));
      setArmed(null);
      return;
    }

    const hit = Object.entries(bindingsRef.current).find(
      ([, b]) => b.channel === channel && b.control === control
    );
    if (!hit) return;

    const [id] = hit;
    const lerp = (lo: number, hi: number) => lo + norm * (hi - lo);
    const pick = (arr: number[]) => arr[Math.min(arr.length - 1, Math.floor(norm * arr.length))];
    if (id === 'inputRate') {
      if (tempoRef.current) setInputBeats(pick(BEATS)); else setInputRate(pick(RATES));
    } else if (id === 'generateRate') {
      if (tempoRef.current) setGenerateBeats(pick(BEATS)); else setGenerateRate(pick(RATES));
    }
    else if (id === 'noise') setNoise(Number(lerp(0.05, 1).toFixed(2)));
    else if (id === 'steps') setSteps(Math.round(lerp(1, 16)));
    else if (id === 'cfg') setCfg(Number(lerp(1, 10).toFixed(1)));
    else if (id === 'variant') setVariant(Math.min(VARIANTS - 1, Math.floor(norm * VARIANTS)));
    else if (id === 'loopback') setLoopback(Number(norm.toFixed(2)));
    else if (id === 'inputMonitor') setInputMonitor(Number(norm.toFixed(2)));
    else if (id === 'outputMonitor') setOutputMonitor(Number(norm.toFixed(2)));
  }, []);

  const bindingLabel = (id: string) => {
    const b = bindings[id];
    return b ? `CC ${b.control}` : null;
  };

  // ─────────────────────────────────────────────────────────────────────
  // Output playback: decode, crossfade, drive the falling spectrogram
  // ─────────────────────────────────────────────────────────────────────
  const ensureOutputCtx = () => {
    if (!outCtxRef.current) {
      const ctx = new AudioContext();
      const gain = ctx.createGain();
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.6;
      analyser.connect(gain);
      gain.connect(ctx.destination);
      gain.gain.value = outputMonitor;
      outCtxRef.current = ctx;
      outGainRef.current = gain;
      outAnalyserRef.current = analyser;
    }
    return outCtxRef.current!;
  };

  /** Start `buf`, equal-power crossfading out whatever is currently playing.
   *  The fade lasts as long as the outgoing clip still has audio left (up to
   *  `maxFade`), so the two genuinely overlap rather than one being truncated. */
  const crossfadeTo = useCallback((buf: AudioBuffer, offset = 0, maxFade = XF_MAX_CYCLE) => {
    const ctx = ensureOutputCtx();
    const now = ctx.currentTime;
    const prev = playingRef.current;

    let fade: number;
    if (prev?.src.buffer) {
      const remaining = prev.src.buffer.duration - (now - prev.startedAt);
      fade = Math.max(XF_MIN, Math.min(maxFade, remaining));
    } else {
      fade = 0.02;   // nothing to fade from — just de-click the attack
    }
    // Never fade for longer than the incoming clip actually lasts.
    fade = Math.min(fade, Math.max(0.02, buf.duration - offset));
    setFadeInfo(fade);

    const gain = ctx.createGain();
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(gain);
    gain.connect(outAnalyserRef.current!);

    try {
      gain.gain.setValueCurveAtTime(FADE_IN_CURVE, now, fade);
    } catch {
      // A curve can be rejected if one is still running on this param.
      gain.gain.setValueAtTime(0, now);
      gain.gain.linearRampToValueAtTime(1, now + fade);
    }

    if (prev) {
      const g = prev.gain.gain;
      const from = g.value;
      try {
        g.cancelScheduledValues(now);
        // Scale the curve so a clip still fading in descends from where it is.
        g.setValueCurveAtTime(FADE_OUT_CURVE.map(v => v * from), now, fade);
      } catch {
        g.cancelScheduledValues(now);
        g.setValueAtTime(from, now);
        g.linearRampToValueAtTime(0, now + fade);
      }
      prev.src.stop(now + fade + 0.03);
    }

    src.start(now, Math.min(offset, Math.max(0, buf.duration - 0.05)));
    playingRef.current = { src, gain, startedAt: now - offset };
  }, []);

  // Switching the variant knob mid-clip swaps lanes at the same position.
  useEffect(() => {
    const cyc = cycleRef.current;
    const cur = playingRef.current;
    if (!cyc || !cur || !outCtxRef.current) return;
    const buf = cyc.buffers[variant];
    if (!buf) return;
    const offset = outCtxRef.current.currentTime - cur.startedAt;
    if (offset >= 0 && offset < buf.duration - 0.1) crossfadeTo(buf, offset, XF_SWITCH);
  }, [variant, crossfadeTo]);

  // Re-pick the input channel(s) without stopping the loop — useful for moving
  // from, say, a guitar on input 1 to a mic on input 2 mid-performance.
  useEffect(() => {
    const sp = splitterRef.current, mg = mergerRef.current;
    if (!sp || !mg) return;
    const [a, b] = parseChannelSel(chanSel, detectedChannels);
    try {
      sp.disconnect();
      sp.connect(mg, a, 0);
      sp.connect(mg, b, 1);
    } catch (e: any) {
      setError(`Could not select channel: ${e.message || e}`);
    }
  }, [chanSel, detectedChannels]);

  useEffect(() => { if (outGainRef.current) outGainRef.current.gain.value = outputMonitor; }, [outputMonitor]);
  useEffect(() => { if (inGainRef.current) inGainRef.current.gain.value = inputMonitor; }, [inputMonitor]);

  // ─────────────────────────────────────────────────────────────────────
  // Canvas render loop: input waveform (left) + output spectrogram (right)
  // ─────────────────────────────────────────────────────────────────────
  useEffect(() => {
    const tick = () => {
      rafRef.current = requestAnimationFrame(tick);

      const wc = waveCanvasRef.current;
      const an = inAnalyserRef.current;
      if (wc && an) {
        const ctx = wc.getContext('2d');
        if (ctx) {
          const w = wc.width, h = wc.height;
          const data = new Uint8Array(an.fftSize);
          an.getByteTimeDomainData(data);
          ctx.fillStyle = '#000';
          ctx.fillRect(0, 0, w, h);

          // Noise wash: the middle rectangle's green bleeds leftward into the
          // input's background, reaching further and sitting heavier as more
          // noise is added — the input is visibly being obscured by it.
          const nz = noiseRef.current;
          const wash = ctx.createLinearGradient(w, 0, 0, 0);
          const reach = Math.min(1, 0.12 + nz * 0.88);
          wash.addColorStop(0, `rgba(140, 205, 150, ${0.08 + nz * 0.42})`);
          wash.addColorStop(reach, 'rgba(140, 205, 150, 0)');
          ctx.fillStyle = wash;
          ctx.fillRect(0, 0, w, h);

          ctx.strokeStyle = '#fff';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          for (let i = 0; i < data.length; i++) {
            const x = (i / data.length) * w;
            const y = (data[i] / 128 - 1) * (h / 2) + h / 2;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
          }
          ctx.stroke();
        }
      }

      const sc = specCanvasRef.current;
      const oa = outAnalyserRef.current;
      if (sc && oa) drawSpectrogram(sc, oa, 'green');
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, []);

  // ─────────────────────────────────────────────────────────────────────
  // WebSocket
  // ─────────────────────────────────────────────────────────────────────
  useEffect(() => {
    const ws = new WebSocket(`${WS_URL}/ws/live/${connectionId}`);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => { setConnected(false); setRunning(false); };
    ws.onerror = () => setError('WebSocket error — is the backend running?');

    ws.onmessage = async (ev) => {
      if (typeof ev.data !== 'string') {
        const hdr = pendingHeaderRef.current;
        pendingHeaderRef.current = null;
        if (!hdr) return;
        const ctx = ensureOutputCtx();
        const buf = await ctx.decodeAudioData(ev.data as ArrayBuffer);
        let cyc = cycleRef.current;
        if (!cyc || cyc.cycle !== hdr.cycle) {
          cyc = { cycle: hdr.cycle, buffers: new Array(VARIANTS).fill(null) };
          cycleRef.current = cyc;
        }
        cyc.buffers[hdr.index] = buf;

        if (hdr.redecode) {
          // Same latent, new channel curve: swap it in at the position we are
          // already at, so shaping the shelf is heard as a morph rather than a
          // restart. Only if this lane is the one currently sounding.
          setShelfBusy(false);
          const cur = playingRef.current;
          if (hdr.index === variantRef.current && cur && outCtxRef.current) {
            const off = outCtxRef.current.currentTime - cur.startedAt;
            crossfadeTo(buf, off >= 0 && off < buf.duration - 0.1 ? off : 0, XF_SWITCH);
          }
          return;
        }

        // Begin the new cycle as soon as the selected lane lands, rather than
        // waiting for the whole family — keeps output continuous.
        if (hdr.index === variantRef.current) crossfadeTo(buf, 0);
        return;
      }

      const d = JSON.parse(ev.data);
      switch (d.type) {
        case 'variant': pendingHeaderRef.current = d; break;
        case 'cycle_start': setStatus(`Generating cycle ${d.cycle}…`); break;
        case 'cycle_complete':
          setCycleInfo({ n: d.cycle, ms: d.total_ms });
          setStatus('Listening');
          break;
        case 'dropped': setDropped(n => n + 1); break;
        case 'midi_cc': applyMidi(d.channel, d.control, d.norm); break;
        case 'error': setError(d.error); break;
      }
    };

    return () => ws.close();
  }, [connectionId, applyMidi, crossfadeTo]);

  /** Push a new channel curve and ask the backend to re-render the audible
   *  variant through it — a decode, not a regeneration. */
  const sendWeights = useCallback((weights: Float32Array, renorm: boolean) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    setShelfBusy(true);
    ws.send(JSON.stringify({
      type: 'weights',
      weights: Array.from(weights),
      renormalize: renorm,
      redecode_index: variantRef.current,
    }));
  }, []);

  const sendParams = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const p = paramsRef.current;
    ws.send(JSON.stringify({
      type: 'params',
      params: {
        prompt: p.prompt, input_rate: p.inputRate, generate_rate: p.generateRate,
        noise: p.noise, variant_delta: VARIANT_DELTA, steps: p.steps,
        cfg: p.cfg, variants: VARIANTS, model: p.model,
        loopback: p.loopback, selected_variant: p.variant,
      },
    }));
  }, []);

  useEffect(() => { if (running) sendParams(); },
    [prompt, noise, steps, cfg, model, effInputRate, effGenerateRate, loopback, variant,
     running, sendParams]);

  // ─────────────────────────────────────────────────────────────────────
  // Transport
  // ─────────────────────────────────────────────────────────────────────
  const pushSnapshot = useCallback(() => {
    const ws = wsRef.current;
    const ring = ringRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || !ring) return;

    const want = Math.min(Math.floor(paramsRef.current.inputRate * ring.sr), ring.filled);
    if (want < ring.sr * 0.25) return;    // less than 250 ms captured; nothing useful yet

    // Unwrap the newest `want` samples out of the circular buffer.
    const inter = new Float32Array(want * 2);
    const start = (ring.w - want + ring.l.length) % ring.l.length;
    for (let i = 0; i < want; i++) {
      const j = (start + i) % ring.l.length;
      inter[i * 2] = ring.l[j];
      inter[i * 2 + 1] = ring.r[j];
    }
    ws.send(JSON.stringify({ type: 'audio_header', channels: 2, sample_rate: ring.sr }));
    ws.send(inter.buffer);
  }, []);

  /** Load an audio file to stand in for the live feed while testing. */
  const pickFile = async (file: File) => {
    setError(null);
    try {
      // Decode once up front purely to validate it and read its duration; the
      // playback buffer is decoded again at start() against the input context.
      const probe = new AudioContext();
      const buf = await probe.decodeAudioData(await file.arrayBuffer());
      await probe.close();
      fileRef.current = file;
      setFileInfo({ name: file.name, duration: buf.duration });
      setInputSource('file');
      // A file can't feed back into the mic, so make it audible by default —
      // unlike the hardware input, which stays muted until you ask for it.
      setInputMonitor(m => (m === 0 ? 0.7 : m));
    } catch (e: any) {
      setError(`Could not decode ${file.name}: ${e.message || e}`);
    }
  };

  const start = async () => {
    setError(null);
    if (inputSource === 'file' && !fileRef.current) {
      setError('Load an audio file first, or switch back to the live input.');
      return;
    }
    try {
      // Run at the device's native rate (48 kHz on most interfaces) and let the
      // backend resample to the model's 44.1 kHz — forcing the hardware to a
      // rate it doesn't run hides a resample we can't inspect.
      const ctx = new AudioContext();
      inCtxRef.current = ctx;
      setCaptureRate(ctx.sampleRate);

      // Either head feeds the identical downstream graph, so the ring buffer,
      // the snapshot timer and everything after it cannot tell the difference.
      let source: AudioNode;
      let fileBuf: AudioBuffer | null = null;
      let nChannels = 2;
      if (inputSource === 'file') {
        fileBuf = await ctx.decodeAudioData(await fileRef.current!.arrayBuffer());
        const node = ctx.createBufferSource();
        node.buffer = fileBuf;
        node.loop = true;     // sample-accurate loop, no gap at the seam
        fileSourceRef.current = node;
        source = node;        // started below, once the seed offset is known
        nChannels = fileBuf.numberOfChannels;
      } else {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            deviceId: deviceId ? { exact: deviceId } : undefined,
            echoCancellation: false, noiseSuppression: false, autoGainControl: false,
            // Ask for everything the interface offers; a 2-in device simply
            // reports 2 back. Without this the browser hands us a stereo
            // downmix and channels 3+ are unreachable.
            channelCount: { ideal: 8 },
          },
        });
        streamRef.current = stream;
        source = ctx.createMediaStreamSource(stream);
        nChannels = stream.getAudioTracks()[0]?.getSettings().channelCount
          ?? (source as MediaStreamAudioSourceNode).channelCount ?? 2;
      }
      nChannels = Math.max(1, nChannels);
      setDetectedChannels(nChannels);

      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      const monitorGain = ctx.createGain();
      monitorGain.gain.value = inputMonitor;
      const proc = ctx.createScriptProcessor(4096, 2, 2);
      // The processor is a recording tap only — it reads inputBuffer and never
      // writes outputBuffer, so its own output is silence. It still has to
      // reach the destination to keep firing, so it runs into a muted sink.
      const tapSink = ctx.createGain();
      tapSink.gain.value = 0;

      const cap = Math.ceil(MAX_BUFFER_SECS * ctx.sampleRate);
      const ring = {
        l: new Float32Array(cap), r: new Float32Array(cap),
        w: 0, filled: 0, sr: ctx.sampleRate,
      };
      ringRef.current = ring;

      // A file is already in memory, so there is no reason to sit through a
      // real InputRate of silence: pre-fill the ring with its opening window
      // and resume playback from exactly where that window ends, keeping the
      // timeline continuous across the seam.
      let seedSeconds = 0;
      if (fileBuf) {
        const seed = Math.min(Math.floor(effInputRate * ctx.sampleRate), fileBuf.length);
        const L = fileBuf.getChannelData(0);
        const R = fileBuf.numberOfChannels > 1 ? fileBuf.getChannelData(1) : L;
        ring.l.set(L.subarray(0, seed), 0);
        ring.r.set(R.subarray(0, seed), 0);
        ring.w = seed % cap;
        ring.filled = seed;
        seedSeconds = seed / ctx.sampleRate;
        fileSourceRef.current!.start(0, seedSeconds % fileBuf.duration);
      }

      proc.onaudioprocess = (e) => {
        const ring = ringRef.current;
        if (!ring) return;
        const L = e.inputBuffer.getChannelData(0);
        const R = e.inputBuffer.numberOfChannels > 1 ? e.inputBuffer.getChannelData(1) : L;
        for (let i = 0; i < L.length; i++) {
          ring.l[ring.w] = L[i];
          ring.r[ring.w] = R[i];
          ring.w = (ring.w + 1) % ring.l.length;
        }
        ring.filled = Math.min(ring.filled + L.length, ring.l.length);
      };

      // Channel selection: split the device's inputs out, then merge exactly
      // the chosen one or two back into a stereo pair. Everything downstream
      // stays 2-channel, and re-picking a channel is just a rewire of these
      // two connections rather than a rebuild of the whole graph.
      const splitter = ctx.createChannelSplitter(nChannels);
      const merger = ctx.createChannelMerger(2);
      source.connect(splitter);
      const [selA, selB] = parseChannelSel(chanSel, nChannels);
      splitter.connect(merger, selA, 0);
      splitter.connect(merger, selB, 1);
      splitterRef.current = splitter;
      mergerRef.current = merger;

      // Capture tap: feeds the waveform display and the ring buffer.
      merger.connect(analyser);
      analyser.connect(proc);
      proc.connect(tapSink);
      tapSink.connect(ctx.destination);

      // Metronome bus: straight to the destination. It deliberately bypasses
      // `merger` (and therefore the analyser and the ring-buffer tap), so the
      // click is audible but can never end up in the audio sent to the model.
      const clickGain = ctx.createGain();
      clickGain.gain.value = clickLevel;
      clickGain.connect(ctx.destination);
      clickGainRef.current = clickGain;

      // Monitoring is its own direct path off the selected channels, so what
      // you hear is what feeds the model, with none of the processor's latency.
      merger.connect(monitorGain);
      monitorGain.connect(ctx.destination);

      inAnalyserRef.current = analyser;
      inGainRef.current = monitorGain;
      procRef.current = proc;

      // Autoplay policy can leave a context suspended when it is created either
      // side of an await; without this the tap never fires and nothing is heard.
      if (ctx.state === 'suspended') await ctx.resume();

      const outCtx = ensureOutputCtx();
      if (outCtx.state === 'suspended') await outCtx.resume();

      wsRef.current?.send(JSON.stringify({
        type: 'start',
        params: {
          prompt, input_rate: effInputRate, generate_rate: effGenerateRate,
          noise, variant_delta: VARIANT_DELTA, steps, cfg,
          variants: VARIANTS, model, loopback, selected_variant: variant,
        },
      }));

      setRunning(true);
      setDropped(0);

      // The mic has to wait a full buffer so the model never sees a short clip;
      // a seeded file already has one, so it can generate right away.
      const seeded = seedSeconds > 0;
      setStatus(seeded ? 'Generating first cycle…'
                       : `Filling ${effInputRate.toFixed(1)}s buffer…`);
      const firstDelay = seeded ? 120 : effInputRate * 1000;
      timerRef.current = window.setTimeout(function fire() {
        if (!runningRef.current) return;
        pushSnapshot();
        timerRef.current = window.setTimeout(fire, paramsRef.current.generateRate * 1000);
      }, firstDelay);
    } catch (e: any) {
      setError(e.message || 'Could not open the audio input');
    }
  };

  const stop = () => {
    setRunning(false);
    setStatus('Idle');
    if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null; }
    wsRef.current?.send(JSON.stringify({ type: 'stop' }));
    procRef.current?.disconnect();
    if (fileSourceRef.current) {
      try { fileSourceRef.current.stop(); } catch { /* never started */ }
      fileSourceRef.current.disconnect();
      fileSourceRef.current = null;
    }
    if (clickTimerRef.current) { clearInterval(clickTimerRef.current); clickTimerRef.current = null; }
    clickGainRef.current?.disconnect();
    clickGainRef.current = null;
    splitterRef.current?.disconnect();
    mergerRef.current?.disconnect();
    splitterRef.current = null;
    mergerRef.current = null;
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    inCtxRef.current?.close().catch(() => {});
    inCtxRef.current = null;
    inAnalyserRef.current = null;
    procRef.current = null;
    const prev = playingRef.current;
    if (prev && outCtxRef.current) {
      const now = outCtxRef.current.currentTime;
      prev.gain.gain.cancelScheduledValues(now);
      prev.gain.gain.setValueAtTime(Math.max(prev.gain.gain.value, 0.0001), now);
      prev.gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.2);
      prev.src.stop(now + 0.25);
      playingRef.current = null;
    }
  };

  useEffect(() => () => { stop(); }, []);   // eslint-disable-line react-hooks/exhaustive-deps

  const arm = (id: string) => setArmed(a => (a === id ? null : id));
  const knobProps = (id: string) => ({
    id, learning: midiLearn, armed: armed === id,
    bound: bindingLabel(id), onArm: arm,
  });

  // The middle bar's thickness is the diffusion step count; noise instead
  // shows up as the wash bleeding into the input pane.
  const barWidth = 18 + (Math.min(steps, 16) / 16) * 104;
  // Variant 0 sits at the top of the bar and the ladder descends from there.
  const variantPct = VARIANTS > 1 ? (variant / (VARIANTS - 1)) * 100 : 0;

  return (
    <div className="live-root">
      {/* ── transport ─────────────────────────────────────────── */}
      <header className="live-bar">
        <button className={`transport ${running ? 'stop' : 'start'}`}
                onClick={running ? stop : start}
                disabled={!connected}>
          {running ? <Square size={16} /> : <Play size={16} />}
          {running ? 'STOP' : 'START'}
        </button>

        <button className={`midi-btn ${midiLearn ? 'on' : ''}`}
                onClick={() => { setMidiLearn(v => !v); setArmed(null); }}>
          <Sliders size={15} /> MIDI
          {midiLearn && <span className="midi-hint">
            {armed ? `move a control to bind “${armed}”` : 'click a knob to arm it'}
          </span>}
        </button>

        <select className="live-select" value={midiPort ?? ''}
                onChange={e => openMidiPort(e.target.value)}>
          <option value="" disabled>MIDI input…</option>
          {midiPorts.map(p => <option key={p} value={p}>{p}</option>)}
        </select>

        <select className="live-select" value={deviceId}
                onChange={e => setDeviceId(e.target.value)} disabled={running}>
          {devices.map(d => (
            <option key={d.deviceId} value={d.deviceId}>
              {d.label || `Input ${d.deviceId.slice(0, 6)}`}
            </option>
          ))}
        </select>

        <button className={`midi-btn ${shelfOpen ? 'on' : ''}`}
                onClick={() => setShelfOpen(v => !v)}
                title="Per-channel weights on the 256 SAME latent dimensions">
          <PanelRight size={15} /> LATENT
        </button>

        <select className="live-select" value={chanSel}
                onChange={e => setChanSel(e.target.value)}
                title="Which physical input(s) feed the model">
          {channelOptions(detectedChannels).map(o => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>

        <select className="live-select" value={model}
                onChange={e => setModel(e.target.value as any)}>
          <option value="sm-music">sm-music</option>
          <option value="sm-sfx">sm-sfx</option>
          <option value="medium">medium</option>
        </select>

        <div className="live-status">
          <Circle size={8} className={connected ? 'dot ok' : 'dot bad'} />
          {status}
          {captureRate && <span className="dim"> · capture {(captureRate / 1000).toFixed(1)}k → 44.1k · {detectedChannels}ch in</span>}
          {cycleInfo && <span className="dim"> · cycle {cycleInfo.n} in {(cycleInfo.ms / 1000).toFixed(2)}s</span>}
          {fadeInfo !== null && <span className="dim"> · xfade {fadeInfo.toFixed(2)}s</span>}
          {beatSec && inputBeats * beatSec > MAX_BUFFER_SECS &&
            <span className="warn"> · {inputBeats}b needs {(inputBeats * beatSec).toFixed(1)}s, clamped to {MAX_BUFFER_SECS}s</span>}
          {dropped > 0 && <span className="warn"> · {dropped} dropped (loop behind)</span>}
        </div>
      </header>

      {error && <div className="live-error" onClick={() => setError(null)}>{error} ✕</div>}

      {/* ── the two halves ─────────────────────────────────────── */}
      <main className="live-main">
        <section className="pane">
          <div className="pane-title">Live-Input Feed</div>
          <input className="prompt-box" value={prompt} placeholder="Text prompt"
                 onChange={e => setPrompt(e.target.value)} />
          <canvas ref={waveCanvasRef} width={900} height={420} className="pane-canvas" />

          <div className="source-row">
            <div className="source-toggle">
              <button className={inputSource === 'mic' ? 'on' : ''}
                      disabled={running}
                      onClick={() => setInputSource('mic')}>
                <Mic size={13} /> Live Input
              </button>
              <button className={inputSource === 'file' ? 'on' : ''}
                      disabled={running}
                      onClick={() => (fileRef.current ? setInputSource('file') : filePickerRef.current?.click())}>
                <Repeat size={13} /> File Loop
              </button>
            </div>

            <button className="file-btn" disabled={running}
                    onClick={() => filePickerRef.current?.click()}>
              <FolderOpen size={13} /> {fileInfo ? 'Change…' : 'Import file…'}
            </button>

            {fileInfo && (
              <span className="file-info" title={fileInfo.name}>
                {fileInfo.name} · {fileInfo.duration.toFixed(1)}s looping
              </span>
            )}

            <input ref={filePickerRef} type="file" accept="audio/*" hidden
                   onChange={e => {
                     const f = e.target.files?.[0];
                     if (f) pickFile(f);
                     e.target.value = '';   // allow re-picking the same file
                   }} />
          </div>

          <div className="monitor">
            <span>Monitor</span>
            <input type="range" min={0} max={1} step={0.01} value={inputMonitor}
                   onChange={e => setInputMonitor(Number(e.target.value))} />
            <span className="dim">{Math.round(inputMonitor * 100)}%</span>
          </div>
        </section>

        <div className="middle" title={`${steps} steps · variant ${variant} at noise ${(noise + variant * VARIANT_DELTA).toFixed(2)}`}>
          <div className="middle-bar" style={{ width: `${barWidth}px` }}>
            <div className="variant-marker" style={{ top: `${2 + variantPct * 0.96}%` }}>
              <span className="variant-marker-tag">{variant}</span>
            </div>
          </div>
        </div>

        <section className="pane">
          <div className="pane-title right">Live-Output Feed</div>
          <canvas ref={specCanvasRef} width={900} height={420} className="pane-canvas" />
          <div className="monitor">
            <span>Monitor</span>
            <input type="range" min={0} max={1} step={0.01} value={outputMonitor}
                   onChange={e => setOutputMonitor(Number(e.target.value))} />
            <span className="dim">{Math.round(outputMonitor * 100)}%</span>
          </div>
        </section>

        <LatentShelf
          open={shelfOpen}
          onClose={() => setShelfOpen(false)}
          onChange={sendWeights}
          renormalize={renormalize}
          onRenormalize={setRenormalize}
          busy={shelfBusy}
        />
      </main>

      {/* ── knobs ──────────────────────────────────────────────── */}
      <footer className={`live-knobs ${midiLearn ? 'learning' : ''}`}>
        <div className="tempo-box">
          <div className="tempo-row">
            <button className={`tempo-btn ${tempoOn ? 'on' : ''}`}
                    onClick={() => setTempoOn(v => !v)}
                    title="Denominate the rate knobs in beats instead of seconds">
              TEMPO
            </button>
            <button className="tap-btn" onClick={tap} title="Tap 4+ times on the beat">
              TAP
            </button>
          </div>
          <div className="tempo-bpm">
            <input
              className="tempo-bpm-input"
              type="text"
              inputMode="decimal"
              value={bpmText}
              placeholder="tap or type"
              onChange={e => setBpmText(e.target.value)}
              onBlur={commitBpm}
              onKeyDown={e => {
                if (e.key === 'Enter') { commitBpm(); (e.target as HTMLInputElement).blur(); }
                if (e.key === 'Escape') { setBpmText(bpm !== null ? String(bpm) : ''); }
              }}
              aria-label="Tempo in BPM"
            />
            <span className="tempo-unit">BPM</span>
            <button
              className={`click-btn ${clickOn ? 'on' : ''}`}
              onClick={() => setClickOn(v => !v)}
              disabled={!bpm}
              title={bpm ? 'Metronome — heard on the monitor only, never sent to the model'
                         : 'Set a tempo first'}
            >CLICK</button>
          </div>

          {clickOn && (
            <div className="click-level">
              <input type="range" min={0} max={1} step={0.01} value={clickLevel}
                     onChange={e => setClickLevel(Number(e.target.value))}
                     aria-label="Click level" />
              <span className="dim">{Math.round(clickLevel * 100)}%</span>
            </div>
          )}
          {bpm && (
            <div className="tempo-meta">
              1 beat = {(60 / bpm).toFixed(3)}s
              <button className="tempo-clear" onClick={clearTempo}>clear</button>
            </div>
          )}
          {tempoOn && !bpm && <div className="tempo-warn">no tempo yet — using seconds</div>}
        </div>

        <div className="knob-group">
          {beatSec ? (
            <>
              <Knob {...knobProps('inputRate')} label="Input Rate" values={BEATS}
                    value={inputBeats} onChange={setInputBeats}
                    format={b => `${b}b · ${(b * beatSec).toFixed(1)}s`} />
              <Knob {...knobProps('generateRate')} label="Generate Rate" values={BEATS}
                    value={generateBeats} onChange={setGenerateBeats}
                    format={b => `${b}b · ${(b * beatSec).toFixed(1)}s`} />
            </>
          ) : (
            <>
              <Knob {...knobProps('inputRate')} label="Input Rate" values={RATES}
                    value={inputRate} onChange={setInputRate} format={v => `${v}s`} />
              <Knob {...knobProps('generateRate')} label="Generate Rate" values={RATES}
                    value={generateRate} onChange={setGenerateRate} format={v => `${v}s`} />
            </>
          )}
        </div>

        <div className="knob-group">
          <Knob {...knobProps('noise')} label="Noise" min={0.05} max={1} step={0.01}
                value={noise} onChange={setNoise} />
          <Knob {...knobProps('steps')} label="Steps" min={1} max={16} step={1}
                value={steps} onChange={setSteps} format={v => String(Math.round(v))} />
          <Knob {...knobProps('cfg')} label="CFG" min={1} max={10} step={0.1}
                value={cfg} onChange={setCfg} format={v => v.toFixed(1)} />
        </div>

        <div className="knob-group">
          <Knob {...knobProps('variant')} label="Variant" min={0} max={VARIANTS - 1} step={1}
                value={variant} onChange={setVariant}
                format={v => `${Math.round(v)} · ${(noise + v * VARIANT_DELTA).toFixed(2)}`} />
        </div>

        <div className="knob-group">
          <Knob {...knobProps('loopback')} label="Loopback" min={0} max={1} step={0.01}
                value={loopback} onChange={setLoopback}
                format={v => `${Math.round(v * 100)}%`} />
        </div>
      </footer>
    </div>
  );
}
