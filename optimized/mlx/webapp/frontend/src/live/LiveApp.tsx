import { useCallback, useEffect, useRef, useState } from 'react';
import { Play, Square, Sliders, Circle, Mic, Repeat, FolderOpen, PanelRight, Code2, Trash2 } from 'lucide-react';
import Knob from './Knob';
import LatentShelf from './LatentShelf';
import StrudelPanel from './StrudelPanel';
import InputSection from './InputSection';
import MidiSlider from './MidiSlider';
import TrackOutput, { type TrackHandle } from './TrackOutput';
import {
  API_URL, VARIANTS, VARIANT_DELTA, MAX_NUDGE, TRACK_THEME,
  SOURCE_IDS, SOURCE_META, defaultSources, defaultTrack,
  type TrackParams, type SourceId, type SourceState,
} from './trackTypes';

interface RingBuf { l: Float32Array; r: Float32Array; w: number; filled: number; sr: number }
interface SourceNodes { head: AudioNode; gain: GainNode; analyser: AnalyserNode }
import type { StrudelHandle } from './strudel/strudelEngine';
import './live.css';


// 32 beats at 60 BPM is 32 s, so the ring must hold that much for the longest
// beat-denominated window. ~12 MB of Float32 at 48 kHz stereo.
const MAX_BUFFER_SECS = 32;  // largest InputRate the ring buffer must hold
const MAX_LOOP_SECS = 60;    // ceiling on a recorded loop

const RATES = [1, 3, 5, 10];               // seconds, when Tempo is off
const BEATS = [4, 8, 16, 32];              // beats, when Tempo is on
                                           // (1 beat is too short to diffuse well)
const TAP_RESET_MS = 2500;                 // gap after which a tap starts a new count
const TAP_HISTORY = 8;                     // intervals averaged for the estimate
const CLICK_LOOKAHEAD_MS = 25;             // scheduler wake interval
const CLICK_SCHEDULE_AHEAD = 0.12;         // seconds of clicks queued in advance

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

/** Pre-rendered grain tiles: per-pixel randomness is too slow to redo each frame. */
function makeGrain() {
  const tiles: HTMLCanvasElement[] = [];
  for (let t = 0; t < 4; t++) {
    const c = document.createElement('canvas');
    c.width = c.height = 128;
    const g = c.getContext('2d')!;
    const img = g.createImageData(128, 128);
    for (let i = 0; i < img.data.length; i += 4) {
      const v = Math.random() * 255;
      img.data[i] = v * 0.72;          // slight green cast, to match the middle bar
      img.data[i + 1] = v;
      img.data[i + 2] = v * 0.72;
      img.data[i + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    tiles.push(c);
  }
  return { tiles, pats: tiles.map(() => null as CanvasPattern | null) };
}

export default function LiveApp() {
  const [connectionId] = useState(() => Math.random().toString(36).slice(2, 12));
  const [running, setRunning] = useState(false);
  const [connected, setConnected] = useState(false);

  // --- parameters -----------------------------------------------------
  // One entry per output track. Track 2 appears when you press + .
  const [tracks, setTracks] = useState<TrackParams[]>(() => [defaultTrack()]);
  const [activeTab, setActiveTab] = useState(0);
  // Index of the soloed track, or null. Solo silences the others.
  const [solo, setSolo] = useState<number | null>(null);
  // Master: knob (and MIDI) moves land on every track at once instead of just
  // the active one, which converges tracks that have drifted apart.
  const [masterMode, setMasterMode] = useState(false);
  const masterRef = useRef(false);
  useEffect(() => { masterRef.current = masterMode; }, [masterMode]);
  const trackRefs = useRef<(TrackHandle | null)[]>([]);
  const activeTabRef = useRef(0);
  useEffect(() => { activeTabRef.current = activeTab; }, [activeTab]);

  const patchTrack = useCallback((i: number, patch: Partial<TrackParams>) => {
    setTracks(ts => ts.map((t, j) => (j === i ? { ...t, ...patch } : t)));
  }, []);
  /** Knobs and MIDI act on the active track — or on every track when MASTER
   *  is engaged. Reads still come from the active track, so a knob shows one
   *  value; turning it in master mode makes the tracks agree. */
  const patchActive = useCallback((patch: Partial<TrackParams>) => {
    setTracks(ts => ts.map((t, j) =>
      (masterRef.current || j === activeTabRef.current ? { ...t, ...patch } : t)));
  }, []);
  const [inputRate, setInputRate] = useState(10);       // seconds (Tempo off)
  const [generateRate, setGenerateRate] = useState(10);
  const [tempoOn, setTempoOn] = useState(false);
  const [bpm, setBpm] = useState<number | null>(null);
  const [inputBeats, setInputBeats] = useState(32);      // beats  (Tempo on)
  const [generateBeats, setGenerateBeats] = useState(32);
  const tapsRef = useRef<number[]>([]);
  const [bpmText, setBpmText] = useState('');
  const [clickOn, setClickOn] = useState(false);
  const [clickLevel, setClickLevel] = useState(0.5);

  // Looper (Live Input only) and the optional live-input blend for the other modes.
  const [loopState, setLoopState] = useState<'idle' | 'recording' | 'stopped' | 'playing'>('idle');
  const [loopLen, setLoopLen] = useState(0);
  const [trim, setTrim] = useState<[number, number]>([0, 0]);
  const [recSecs, setRecSecs] = useState(0);
  const [model, setModel] = useState<'sm-music' | 'sm-sfx' | 'medium'>('sm-music');
  // Off by default: one generation per cycle is 5x faster, and the ladder is
  // only worth paying for when you are actually auditioning across it.
  const cur = tracks[activeTab] ?? tracks[0];
  const activeVariants = cur.multiVariant ? VARIANTS : 1;

  const [inputMonitor, setInputMonitor] = useState(0);   // muted by default

  // --- latent shelf ----------------------------------------------------
  const [shelfOpen, setShelfOpen] = useState(false);
  const [shelfBusy, setShelfBusy] = useState(false);
  // Cable being dragged from a source node, and where the pointer is now.
  const [patchFrom, setPatchFrom] = useState<SourceId | null>(null);
  const [patchTo, setPatchTo] = useState<{ x: number; y: number } | null>(null);
  const [renormalize, setRenormalize] = useState(false);

  // --- input source: live hardware, or a file looped as a stand-in ------
  // Sources are independently enabled and each is patched to a set of tracks.
  const [sources, setSources] = useState<Record<SourceId, SourceState>>(defaultSources);
  const sourcesRef = useRef(sources);
  useEffect(() => { sourcesRef.current = sources; }, [sources]);
  const patchSource = useCallback((id: SourceId, patch: Partial<SourceState>) => {
    setSources(s => ({ ...s, [id]: { ...s[id], ...patch } }));
  }, []);
  const [strudelReady, setStrudelReady] = useState(false);
  const [strudelPlaying, setStrudelPlaying] = useState(false);
  const [strudelBanks, setStrudelBanks] = useState<string[]>([]);
  const [strudelError, setStrudelError] = useState<string | null>(null);
  const strudelRef = useRef<StrudelHandle | null>(null);
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
  const inCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const inGainRef = useRef<GainNode | null>(null);
  const inAnalyserRef = useRef<AnalyserNode | null>(null);
  const procRef = useRef<ScriptProcessorNode | null>(null);
  const fileRef = useRef<File | null>(null);
  const fileSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const clickGainRef = useRef<GainNode | null>(null);
  const clickTimerRef = useRef<number | null>(null);
  const nextClickRef = useRef(0);
  const clickBeatRef = useRef(0);
  const bpmRef = useRef<number | null>(null);
  const srcNodesRef = useRef<Partial<Record<SourceId, SourceNodes>>>({});
  const trackMixRef = useRef<GainNode[]>([]);
  const ringsRef = useRef<RingBuf[]>([]);
  const procsRef = useRef<ScriptProcessorNode[]>([]);
  const recProcRef = useRef<ScriptProcessorNode | null>(null);
  const tapSinkRef = useRef<GainNode | null>(null);
  const fileBufRef = useRef<AudioBuffer | null>(null);
  const srcCanvasRefs = useRef<Partial<Record<SourceId, HTMLCanvasElement | null>>>({});
  const srcNodeEls = useRef<Partial<Record<SourceId, HTMLDivElement | null>>>({});
  const trackNodeEls = useRef<(HTMLDivElement | null)[]>([]);
  const patchLayerRef = useRef<HTMLDivElement | null>(null);
  const micGainRef = useRef<GainNode | null>(null);
  const loopGainRef = useRef<GainNode | null>(null);
  const loopSrcRef = useRef<AudioBufferSourceNode | null>(null);
  const loopBufRef = useRef<AudioBuffer | null>(null);
  const loopPeaksRef = useRef<Float32Array | null>(null);
  const recRef = useRef<{ chunks: Float32Array[][]; frames: number } | null>(null);
  const loopStartedAtRef = useRef(0);
  const dragHandleRef = useRef<0 | 1 | null>(null);
  const splitterRef = useRef<ChannelSplitterNode | null>(null);
  const mergerRef = useRef<ChannelMergerNode | null>(null);
  const filePickerRef = useRef<HTMLInputElement | null>(null);
  const noiseCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const grainRef = useRef<{ tiles: HTMLCanvasElement[]; pats: (CanvasPattern | null)[] } | null>(null);
  const frameRef = useRef(0);
  const rafRef = useRef<number | null>(null);
  const timerRef = useRef<number | null>(null);


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

  // Only shared, input-side values now; each track owns its own generation
  // parameters and sends them over its own socket.
  const paramsRef = useRef({ inputRate: effInputRate, generateRate: effGenerateRate });
  useEffect(() => {
    paramsRef.current = { inputRate: effInputRate, generateRate: effGenerateRate };
  }, [effInputRate, effGenerateRate]);
  // The canvas loop runs once and reads this, so noise can drive the wash
  // without restarting the animation frame on every knob move.
  const tempoRef = useRef(false);
  useEffect(() => { tempoRef.current = tempoOn && bpm !== null; }, [tempoOn, bpm]);
  const loopStateRef = useRef(loopState);
  useEffect(() => { loopStateRef.current = loopState; }, [loopState]);
  const trimRef = useRef(trim);
  useEffect(() => { trimRef.current = trim; }, [trim]);
  const loopLenRef = useRef(loopLen);
  useEffect(() => { loopLenRef.current = loopLen; }, [loopLen]);
  const noiseRef = useRef(cur.noise);
  useEffect(() => { noiseRef.current = cur.noise; }, [cur.noise]);
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
    // Everything below is per track and lands on whichever tab is showing.
    else if (id === 'noise') patchActive({ noise: Number(lerp(0.05, 1).toFixed(2)) });
    else if (id === 'steps') patchActive({ steps: Math.round(lerp(1, 16)) });
    else if (id === 'cfg') patchActive({ cfg: Number(lerp(1, 10).toFixed(1)) });
    else if (id === 'variant') patchActive({ variant: Math.floor(norm * VARIANTS) });
    else if (id === 'loopback') patchActive({ loopback: Number(norm.toFixed(2)) });
    else if (id === 'nudge') patchActive({ nudge: Number((norm * MAX_NUDGE).toFixed(2)) });
    else if (id === 'inputMonitor') setInputMonitor(Number(norm.toFixed(2)));
    else if (id.startsWith('outputMonitor')) {
      const t = Number(id.slice('outputMonitor'.length)) || 0;
      patchTrack(t, { outputMonitor: Number(norm.toFixed(2)) });
    }
  }, [patchActive, patchTrack]);

  const bindingLabel = (id: string) => {
    const b = bindings[id];
    return b ? `CC ${b.control}` : null;
  };

  // ─────────────────────────────────────────────────────────────────────
  // Output playback: decode, crossfade, drive the falling spectrogram
  // ─────────────────────────────────────────────────────────────────────
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

  useEffect(() => { if (inGainRef.current) inGainRef.current.gain.value = inputMonitor; }, [inputMonitor]);

  // ─────────────────────────────────────────────────────────────────────
  // Canvas render loop: input waveform (left) + output spectrogram (right)
  // ─────────────────────────────────────────────────────────────────────
  useEffect(() => {
    const tick = () => {
      rafRef.current = requestAnimationFrame(tick);

      // One scope per enabled source. The mic's canvas switches to the
      // recorded loop (with its trim handles) once a take exists.
      for (const id of SOURCE_IDS) {
        const wc = srcCanvasRefs.current[id];
        if (!wc) continue;
        const ctx2 = wc.getContext('2d');
        if (!ctx2) continue;
        const w = wc.width, h = wc.height, mid = h / 2;

        const peaks = loopPeaksRef.current;
        const loopView = id === 'mic' && peaks &&
          (loopStateRef.current === 'stopped' || loopStateRef.current === 'playing');

        if (loopView && peaks) {
          ctx2.clearRect(0, 0, w, h);
          const [t0, t1] = trimRef.current;
          const len = loopLenRef.current || 1;
          const x0 = (t0 / len) * w, x1 = (t1 / len) * w;
          ctx2.fillStyle = 'rgba(0,0,0,.45)';
          ctx2.fillRect(0, 0, x0, h);
          ctx2.fillRect(x1, 0, w - x1, h);
          for (let c = 0; c < peaks.length; c++) {
            const x = (c / peaks.length) * w;
            const inTrim = x >= x0 && x <= x1;
            ctx2.fillStyle = inTrim ? 'rgba(255,255,255,.85)' : 'rgba(255,255,255,.22)';
            const amp = Math.max(1, peaks[c] * mid);
            ctx2.fillRect(x, mid - amp, Math.max(1, w / peaks.length), amp * 2);
          }
          ctx2.fillStyle = '#8ad18a';
          ctx2.fillRect(x0 - 1, 0, 3, h);
          ctx2.fillRect(x1 - 1, 0, 3, h);
          const c2 = inCtxRef.current;
          if (loopStateRef.current === 'playing' && c2) {
            const span = Math.max(0.05, t1 - t0);
            const pos = t0 + ((c2.currentTime - loopStartedAtRef.current) % span);
            ctx2.fillStyle = 'rgba(138,209,138,.9)';
            ctx2.fillRect((pos / len) * w, 0, 2, h);
          }
          continue;
        }

        const an = srcNodesRef.current[id]?.analyser;
        ctx2.clearRect(0, 0, w, h);
        if (!an) continue;
        const data = new Uint8Array(an.fftSize);
        an.getByteTimeDomainData(data);
        ctx2.strokeStyle = SOURCE_META[id].accent;
        ctx2.lineWidth = 1.5;
        ctx2.beginPath();
        for (let i = 0; i < data.length; i++) {
          const x = (i / data.length) * w;
          const y = (data[i] / 128 - 1) * mid + mid;
          i === 0 ? ctx2.moveTo(x, y) : ctx2.lineTo(x, y);
        }
        ctx2.stroke();
      }

      // Noise texture across the whole input side. It sits behind the waveform
      // and behind the (translucent) code panel, so the amount of noise being
      // added to the source is legible in every input mode.
      const nc = noiseCanvasRef.current;
      if (nc) {
        if (!grainRef.current) grainRef.current = makeGrain();
        const g = nc.getContext('2d');
        const rect = nc.getBoundingClientRect();
        if (g && rect.width > 0) {
          if (nc.width !== Math.round(rect.width) || nc.height !== Math.round(rect.height)) {
            nc.width = Math.round(rect.width);
            nc.height = Math.round(rect.height);
            grainRef.current.pats = grainRef.current.tiles.map(() => null);
          }
          g.clearRect(0, 0, nc.width, nc.height);
          // Perceptual ramp: half-way on the knob should already read as heavy.
          const alpha = 0.78 * Math.pow(noiseRef.current, 0.6);
          if (alpha > 0.012) {
            // Swap tiles every few frames so the grain shimmers rather than
            // sitting still, without paying for per-pixel work each frame.
            const i = Math.floor(frameRef.current / 3) % grainRef.current.tiles.length;
            if (!grainRef.current.pats[i]) {
              grainRef.current.pats[i] = g.createPattern(grainRef.current.tiles[i], 'repeat');
            }
            const pat = grainRef.current.pats[i];
            if (pat) {
              g.globalAlpha = alpha;
              g.fillStyle = pat;
              g.fillRect(0, 0, nc.width, nc.height);
              g.globalAlpha = 1;
            }
          }
        }
      }
      const inCtx = inCtxRef.current;
      if (recRef.current && inCtx) {
        setRecSecs(recRef.current.frames / inCtx.sampleRate);
      }
      frameRef.current++;

    };
    rafRef.current = requestAnimationFrame(tick);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, []);

  // ─────────────────────────────────────────────────────────────────────
  // Transport
  // ─────────────────────────────────────────────────────────────────────
  const pushSnapshot = useCallback(() => {
    const rings = ringsRef.current;
    if (!rings.length) return;

    // Each track has its own mix, so each gets its own window. When two tracks
    // are patched identically the bytes are identical, but they are still sent
    // separately — the backend sessions are independent.
    rings.forEach((ring, ti) => {
      const want = Math.min(Math.floor(paramsRef.current.inputRate * ring.sr), ring.filled);
      if (want < ring.sr * 0.25) return;   // under 250 ms captured; nothing useful

      const inter = new Float32Array(want * 2);
      const start = (ring.w - want + ring.l.length) % ring.l.length;
      for (let i = 0; i < want; i++) {
        const j = (start + i) % ring.l.length;
        inter[i * 2] = ring.l[j];
        inter[i * 2 + 1] = ring.r[j];
      }
      trackRefs.current[ti]?.push(
        { type: 'audio_header', channels: 2, sample_rate: ring.sr },
        inter.buffer,
      );
    });
  }, []);

  const [isStarting, setIsStarting] = useState(false);

  /** Lazily pull in the AGPL Strudel bundle, only when Live Code is used. */
  const ensureStrudel = useCallback(async (): Promise<StrudelHandle | null> => {
    if (strudelRef.current) return strudelRef.current;
    try {
      setStrudelError(null);
      const { loadStrudel } = await import('./strudel/strudelEngine');
      const h = await loadStrudel();
      strudelRef.current = h;
      setStrudelBanks(h.sampleBanks);
      setStrudelReady(true);
      const b = h.getBpm();
      if (b) { setBpm(Math.round(b * 10) / 10); setTempoOn(true); }
      return h;
    } catch (e: any) {
      setStrudelError(e?.message || String(e));
      return null;
    }
  }, []);

  useEffect(() => {
    if (sources.code.enabled) void ensureStrudel();
  }, [sources.code.enabled, ensureStrudel]);

  const runStrudel = useCallback(async (code: string) => {
    const h = strudelRef.current ?? await ensureStrudel();
    if (!h) return;
    try {
      setStrudelError(null);
      await h.evaluate(code);
      setStrudelPlaying(true);
      // Strudel is the tempo authority on evaluate: a setcpm in the pattern
      // re-times the beat-denominated rate knobs to match what you hear.
      const b = h.getBpm();
      if (b && isFinite(b)) { setBpm(Math.round(b * 10) / 10); setTempoOn(true); }
    } catch (e: any) {
      setStrudelError(e?.message || String(e));
    }
  }, [ensureStrudel]);

  const hushStrudel = useCallback(() => {
    strudelRef.current?.hush();
    setStrudelPlaying(false);
  }, []);

  // ...and the other direction: tapping or typing a BPM re-times the pattern.
  useEffect(() => {
    if (!sources.code.enabled || !bpm) return;
    const h = strudelRef.current;
    if (!h) return;
    const cur = h.getBpm();
    if (cur == null || Math.abs(cur - bpm) > 0.05) h.setBpm(bpm);
  }, [bpm, sources.code.enabled]);

  // ── looper (Live Input) ──────────────────────────────────────────────
  const startRec = useCallback(() => {
    recRef.current = { chunks: [], frames: 0 };
    setRecSecs(0);
    setLoopState('recording');
  }, []);

  const stopRec = useCallback(() => {
    const rec = recRef.current;
    recRef.current = null;
    const ctx = inCtxRef.current;
    if (!rec || !ctx || rec.frames === 0) { setLoopState('idle'); return; }

    const buf = ctx.createBuffer(2, rec.frames, ctx.sampleRate);
    const L = buf.getChannelData(0), R = buf.getChannelData(1);
    let off = 0;
    for (const [l, r] of rec.chunks) { L.set(l, off); R.set(r, off); off += l.length; }
    loopBufRef.current = buf;

    // Pre-reduce to per-column peaks; redrawing 60x a second from the raw
    // buffer would be far too slow for a minute of audio.
    const COLS = 900;
    const peaks = new Float32Array(COLS);
    const per = Math.max(1, Math.floor(buf.length / COLS));
    for (let c = 0; c < COLS; c++) {
      let m = 0;
      const s0 = c * per, s1 = Math.min(buf.length, s0 + per);
      for (let i = s0; i < s1; i++) m = Math.max(m, Math.abs(L[i]));
      peaks[c] = m;
    }
    loopPeaksRef.current = peaks;
    setLoopLen(buf.duration);
    setTrim([0, buf.duration]);
    setLoopState('stopped');
  }, []);

  const playLoop = useCallback(() => {
    const ctx = inCtxRef.current, buf = loopBufRef.current, g = loopGainRef.current;
    if (!ctx || !buf || !g) return;
    try { loopSrcRef.current?.stop(); } catch { /* not started */ }
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.loop = true;
    src.loopStart = trim[0];
    src.loopEnd = Math.max(trim[0] + 0.05, trim[1]);
    src.connect(g);
    src.start(0, trim[0]);
    loopSrcRef.current = src;
    loopStartedAtRef.current = ctx.currentTime;
    g.gain.setTargetAtTime(1, ctx.currentTime, 0.01);
    // The loop replaces the hardware input as the model's source.
    micGainRef.current?.gain.setTargetAtTime(0, ctx.currentTime, 0.01);
    setLoopState('playing');
  }, [trim]);

  const stopLoop = useCallback(() => {
    const ctx = inCtxRef.current;
    try { loopSrcRef.current?.stop(); } catch { /* not started */ }
    loopSrcRef.current = null;
    if (ctx) {
      loopGainRef.current?.gain.setTargetAtTime(0, ctx.currentTime, 0.01);
      micGainRef.current?.gain.setTargetAtTime(1, ctx.currentTime, 0.01);
    }
    setLoopState(loopBufRef.current ? 'stopped' : 'idle');
  }, []);

  const deleteLoop = useCallback(() => {
    stopLoop();
    loopBufRef.current = null;
    loopPeaksRef.current = null;
    recRef.current = null;
    setLoopLen(0);
    setTrim([0, 0]);
    setLoopState('idle');
  }, [stopLoop]);

  // Re-arm the loop points live while the trim handles are dragged.
  useEffect(() => {
    const src = loopSrcRef.current;
    if (src && loopState === 'playing') {
      src.loopStart = trim[0];
      src.loopEnd = Math.max(trim[0] + 0.05, trim[1]);
    }
  }, [trim, loopState]);


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
      patchSource('file', { enabled: true });
      // A file can't feed back into the mic, so make it audible by default —
      // unlike the hardware input, which stays muted until you ask for it.
      setInputMonitor(m => (m === 0 ? 0.7 : m));
    } catch (e: any) {
      setError(`Could not decode ${file.name}: ${e.message || e}`);
    }
  };

  /** Build one enabled source: its head node, fader and display analyser. */
  const buildSource = async (id: SourceId, ctx: AudioContext) => {
    let head: AudioNode | null = null;

    if (id === 'mic') {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          deviceId: deviceId ? { exact: deviceId } : undefined,
          echoCancellation: false, noiseSuppression: false, autoGainControl: false,
          // Ask for everything the interface offers; a 2-in device reports 2.
          channelCount: { ideal: 8 },
        },
      });
      streamRef.current = stream;
      const raw = ctx.createMediaStreamSource(stream);
      const nCh = Math.max(1, stream.getAudioTracks()[0]?.getSettings().channelCount ?? 2);
      setDetectedChannels(nCh);
      // Channel selection belongs to the mic alone — it is meaningless once a
      // stereo file or pattern has been summed in.
      const splitter = ctx.createChannelSplitter(nCh);
      const merger = ctx.createChannelMerger(2);
      raw.connect(splitter);
      const [selA, selB] = parseChannelSel(chanSel, nCh);
      splitter.connect(merger, selA, 0);
      splitter.connect(merger, selB, 1);
      splitterRef.current = splitter;
      mergerRef.current = merger;

      // The live mic and the recorded loop share this source's fader; loop
      // playback swaps between them rather than adding a fourth channel.
      const liveGain = ctx.createGain();
      liveGain.gain.value = loopStateRef.current === 'playing' ? 0 : 1;
      merger.connect(liveGain);
      micGainRef.current = liveGain;

      const loopGain = ctx.createGain();
      loopGain.gain.value = loopStateRef.current === 'playing' ? 1 : 0;
      loopGainRef.current = loopGain;

      const bus = ctx.createGain();
      liveGain.connect(bus);
      loopGain.connect(bus);
      head = bus;

      // The looper records the mic alone, so it needs its own tap rather than
      // reading a track mix that may contain a file or a pattern.
      const recProc = ctx.createScriptProcessor(4096, 2, 2);
      recProc.onaudioprocess = (e) => {
        const rec = recRef.current;
        if (!rec) return;
        const Lc = e.inputBuffer.getChannelData(0);
        const Rc = e.inputBuffer.numberOfChannels > 1 ? e.inputBuffer.getChannelData(1) : Lc;
        if (rec.frames < MAX_LOOP_SECS * ctx.sampleRate) {
          rec.chunks.push([new Float32Array(Lc), new Float32Array(Rc)]);
          rec.frames += Lc.length;
        }
      };
      merger.connect(recProc);
      recProcRef.current = recProc;
    }

    if (id === 'file') {
      const buf = await ctx.decodeAudioData(await fileRef.current!.arrayBuffer());
      const node = ctx.createBufferSource();
      node.buffer = buf;
      node.loop = true;            // sample-accurate loop, no gap at the seam
      fileSourceRef.current = node;
      fileBufRef.current = buf;
      head = node;                 // started below, once the seed is known
    }

    if (id === 'code') head = strudelRef.current!.tap;
    if (!head) return null;

    const gain = ctx.createGain();
    gain.gain.value = sourcesRef.current[id].level;
    head.connect(gain);

    // Display tap is PRE-fader, so a source stays visible at level 0.
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    head.connect(analyser);

    return { head, gain, analyser };
  };

  const start = async () => {
    setError(null);
    const active = SOURCE_IDS.filter(id => sources[id].enabled);
    if (!active.length) { setError('Enable at least one input source.'); return; }
    if (sources.file.enabled && !fileRef.current) {
      setError('Load an audio file, or turn off the File Loop source.');
      return;
    }
    setIsStarting(true);
    try {
      // Adopt Strudel's context when Live Code is in play, so pattern playback
      // and capture share one clock; otherwise make our own. Either way we run
      // at the device's native rate and let the backend resample to 44.1 kHz.
      let ctx: AudioContext;
      if (sources.code.enabled) {
        const h = strudelRef.current ?? await ensureStrudel();
        if (!h) { setIsStarting(false); return; }
        ctx = h.ctx;
        h.reattach();
      } else {
        ctx = new AudioContext();
      }
      inCtxRef.current = ctx;
      setCaptureRate(ctx.sampleRate);
      const cap = Math.ceil(MAX_BUFFER_SECS * ctx.sampleRate);

      // A silent sink keeps every ScriptProcessor firing. Processors read their
      // input and never write their output, so this carries no audio.
      const tapSink = ctx.createGain();
      tapSink.gain.value = 0;
      tapSink.connect(ctx.destination);
      tapSinkRef.current = tapSink;

      // One mixer, capture tap and ring per TRACK. Summing happens in the audio
      // graph rather than in JS, so sources routed to a track stay
      // sample-aligned — independent processors would smear them by a block.
      const mixes: GainNode[] = [];
      const rings: RingBuf[] = [];
      const procs: ScriptProcessorNode[] = [];
      for (let i = 0; i < tracks.length; i++) {
        const mix = ctx.createGain();
        const proc = ctx.createScriptProcessor(4096, 2, 2);
        const ring: RingBuf = {
          l: new Float32Array(cap), r: new Float32Array(cap),
          w: 0, filled: 0, sr: ctx.sampleRate,
        };
        proc.onaudioprocess = (e) => {
          const Lc = e.inputBuffer.getChannelData(0);
          const Rc = e.inputBuffer.numberOfChannels > 1 ? e.inputBuffer.getChannelData(1) : Lc;
          for (let k = 0; k < Lc.length; k++) {
            ring.l[ring.w] = Lc[k];
            ring.r[ring.w] = Rc[k];
            ring.w = (ring.w + 1) % ring.l.length;
          }
          ring.filled = Math.min(ring.filled + Lc.length, ring.l.length);
        };
        mix.connect(proc);
        proc.connect(tapSink);
        mixes.push(mix); rings.push(ring); procs.push(proc);
      }
      trackMixRef.current = mixes;
      ringsRef.current = rings;
      procsRef.current = procs;

      // Monitoring scales the same summed mix that is being sent, so what you
      // hear is the balance the model receives. It stays at 0 by default so a
      // mic cannot feed back the moment it is enabled.
      const monitorGain = ctx.createGain();
      monitorGain.gain.value = inputMonitor;
      monitorGain.connect(ctx.destination);
      inGainRef.current = monitorGain;

      // Metronome bus: straight to the destination, bypassing every capture
      // tap, so the click is audible but can never reach the model.
      const clickGain = ctx.createGain();
      clickGain.gain.value = clickLevel;
      clickGain.connect(ctx.destination);
      clickGainRef.current = clickGain;

      const built: Partial<Record<SourceId, SourceNodes>> = {};
      for (const id of active) {
        const n = await buildSource(id, ctx);
        if (!n) continue;
        built[id] = n;
        n.gain.connect(monitorGain);
        // Patch into whichever tracks this source is routed to.
        sourcesRef.current[id].to.forEach((on, ti) => {
          if (on && mixes[ti]) n.gain.connect(mixes[ti]);
        });
      }
      srcNodesRef.current = built;

      // A file is already in memory, so a track fed ONLY by the file need not
      // sit through a real InputRate of silence: pre-fill its ring and resume
      // playback where that window ends. With other sources mixed in the seed
      // would be wrong, so it is skipped.
      let seeded = false;
      const fbuf = fileBufRef.current;
      if (sources.file.enabled && fbuf) {
        const seedLen = Math.min(Math.floor(effInputRate * ctx.sampleRate), fbuf.length);
        const Lc = fbuf.getChannelData(0);
        const Rc = fbuf.numberOfChannels > 1 ? fbuf.getChannelData(1) : Lc;
        rings.forEach((ring, ti) => {
          const only = active.length === 1 ||
            active.every(id => id === 'file' || !sourcesRef.current[id].to[ti]);
          if (!only) return;
          ring.l.set(Lc.subarray(0, seedLen), 0);
          ring.r.set(Rc.subarray(0, seedLen), 0);
          ring.w = seedLen % cap;
          ring.filled = seedLen;
          seeded = true;
        });
        fileSourceRef.current!.start(0, (seedLen / ctx.sampleRate) % fbuf.duration);
      }

      // Autoplay policy can leave a context suspended when it is created either
      // side of an await; without this nothing fires and nothing is heard.
      if (ctx.state === 'suspended') await ctx.resume();

      for (const t of trackRefs.current) t?.begin();

      setRunning(true);
      setIsStarting(false);

      setStatus(seeded ? 'Generating first cycle…'
                       : `Filling ${effInputRate.toFixed(1)}s buffer…`);
      const firstDelay = seeded ? 120 : effInputRate * 1000;
      timerRef.current = window.setTimeout(function fire() {
        if (!runningRef.current) return;
        pushSnapshot();
        timerRef.current = window.setTimeout(fire, paramsRef.current.generateRate * 1000);
      }, firstDelay);
    } catch (e: any) {
      setIsStarting(false);
      setError(e.message || 'Could not open the audio input');
    }
  };

  const stop = () => {
    setRunning(false);
    if (sourcesRef.current.code.enabled) { strudelRef.current?.hush(); setStrudelPlaying(false); }
    setStatus('Idle');
    if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null; }
    for (const t of trackRefs.current) t?.finish();
    procRef.current?.disconnect();
    if (fileSourceRef.current) {
      try { fileSourceRef.current.stop(); } catch { /* never started */ }
      fileSourceRef.current.disconnect();
      fileSourceRef.current = null;
    }
    try { loopSrcRef.current?.stop(); } catch { /* not started */ }
    loopSrcRef.current = null;
    recRef.current = null;
    if (clickTimerRef.current) { clearInterval(clickTimerRef.current); clickTimerRef.current = null; }
    clickGainRef.current?.disconnect();
    clickGainRef.current = null;
    splitterRef.current?.disconnect();
    mergerRef.current?.disconnect();
    splitterRef.current = null;
    mergerRef.current = null;
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    // Strudel owns its context; closing it would kill the engine.
    if (!sourcesRef.current.code.enabled) inCtxRef.current?.close().catch(() => {});
    inCtxRef.current = null;
    inAnalyserRef.current = null;
    procRef.current = null;
  };

  useEffect(() => () => { stop(); }, []);   // eslint-disable-line react-hooks/exhaustive-deps

  const showLoopView = loopState === 'stopped' || loopState === 'playing';
  const posToTime = (e: React.PointerEvent) => {
    const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
    return Math.max(0, Math.min(loopLen, ((e.clientX - r.left) / r.width) * loopLen));
  };
  const applyTrim = (at: number) => {
    const h = dragHandleRef.current;
    if (h === null) return;
    setTrim(([a, b]) => h === 0
      ? [Math.min(at, b - 0.05), b]
      : [a, Math.max(at, a + 0.05)]);
  };

  /** The latent shelf edits whichever track's tab is showing. */
  const sendWeights = useCallback((weights: Float32Array, renorm: boolean) => {
    setShelfBusy(true);
    trackRefs.current[activeTabRef.current]?.sendWeights(weights, renorm);
  }, []);

  // Serialised routing, so the rewire below fires on patch changes only —
  // keying it on `sources` would disconnect and reconnect on every fader move.
  const routingKey = SOURCE_IDS
    .map(id => sources[id].to.slice(0, tracks.length).map(b => (b ? 1 : 0)).join(''))
    .join('|');
  useEffect(() => {
    const mixes = trackMixRef.current;
    if (!mixes.length) return;
    for (const id of SOURCE_IDS) {
      const n = srcNodesRef.current[id];
      if (!n) continue;
      mixes.forEach((mix) => {
        try { n.gain.disconnect(mix); } catch { /* was not connected */ }
      });
      sourcesRef.current[id].to.forEach((on, ti) => {
        if (on && mixes[ti]) n.gain.connect(mixes[ti]);
      });
    }
  }, [routingKey]);

  // --- patch cables ----------------------------------------------------
  const [anchors, setAnchors] = useState<{
    src: Partial<Record<SourceId, { x: number; y: number }>>;
    trk: { x: number; y: number }[];
  }>({ src: {}, trk: [] });

  const measure = useCallback(() => {
    const layer = patchLayerRef.current;
    if (!layer) return;
    const base = layer.getBoundingClientRect();
    const at = (el: HTMLElement | null | undefined) => {
      if (!el) return undefined;
      const r = el.getBoundingClientRect();
      return { x: r.left + r.width / 2 - base.left, y: r.top + r.height / 2 - base.top };
    };
    const src: Partial<Record<SourceId, { x: number; y: number }>> = {};
    for (const id of SOURCE_IDS) {
      const pt = at(srcNodeEls.current[id]);
      if (pt) src[id] = pt;
    }
    setAnchors({ src, trk: trackNodeEls.current.map(el => at(el) ?? { x: 0, y: 0 }) });
  }, []);

  // Anchors move whenever the layout does.
  useEffect(() => {
    measure();
    const t = window.setTimeout(measure, 60);   // after the panes settle
    window.addEventListener('resize', measure);
    return () => { window.clearTimeout(t); window.removeEventListener('resize', measure); };
  }, [measure, sources, tracks.length, shelfOpen, loopLen]);

  const beginPatch = useCallback((id: SourceId, e: React.PointerEvent) => {
    e.preventDefault();
    measure();
    setPatchFrom(id);
    const layer = patchLayerRef.current;
    const base = layer?.getBoundingClientRect();
    if (base) setPatchTo({ x: e.clientX - base.left, y: e.clientY - base.top });
  }, [measure]);

  // While dragging, the loose end follows the pointer; releasing over a track
  // toggles that connection, anywhere else cancels.
  useEffect(() => {
    if (patchFrom === null) return;
    const move = (e: PointerEvent) => {
      const base = patchLayerRef.current?.getBoundingClientRect();
      if (base) setPatchTo({ x: e.clientX - base.left, y: e.clientY - base.top });
    };
    const up = (e: PointerEvent) => {
      const hit = trackNodeEls.current.findIndex(el => {
        if (!el) return false;
        const r = el.getBoundingClientRect();
        const pad = 26;   // generous, so the drop does not demand precision
        return e.clientX >= r.left - pad && e.clientX <= r.right + pad
            && e.clientY >= r.top - pad && e.clientY <= r.bottom + pad;
      });
      if (hit >= 0) {
        setSources(prev => {
          const to = [...prev[patchFrom].to];
          while (to.length < tracks.length) to.push(false);
          to[hit] = !to[hit];
          return { ...prev, [patchFrom]: { ...prev[patchFrom], to } };
        });
      }
      setPatchFrom(null);
      setPatchTo(null);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    return () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
  }, [patchFrom, tracks.length]);

  const addTrack = useCallback(() => {
    setTracks(ts => (ts.length > 1 ? ts : [...ts, defaultTrack()]));
    setActiveTab(1);
  }, []);
  const removeTrack = useCallback(() => {
    trackRefs.current[1]?.finish();
    trackRefs.current = trackRefs.current.slice(0, 1);
    setTracks(ts => ts.slice(0, 1));
    setActiveTab(0);
    // Otherwise a solo on the departed track would silence the survivor.
    setSolo(null);
  }, []);

  const arm = (id: string) => setArmed(a => (a === id ? null : id));
  const knobProps = (id: string) => ({
    id, learning: midiLearn, armed: armed === id,
    bound: bindingLabel(id), onArm: arm,
  });

  // The middle bar's thickness is the diffusion step count; noise instead
  // shows up as the wash bleeding into the input pane.
  const barWidth = 18 + (Math.min(cur.steps, 16) / 16) * 104;
  // Variant 0 sits at the top of the bar and the ladder descends from there.
  const variantPct = activeVariants > 1 ? (cur.variant / (activeVariants - 1)) * 100 : 0;

  return (
    <div className="live-root">
      {/* ── transport ─────────────────────────────────────────── */}
      <header className="live-bar">
        <button className={`transport ${running ? 'stop' : 'start'}`}
                onClick={running ? stop : start}
                disabled={!connected || isStarting || (sources.code.enabled && !strudelReady)}>
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
          {beatSec && inputBeats * beatSec > MAX_BUFFER_SECS &&
            <span className="warn"> · {inputBeats}b needs {(inputBeats * beatSec).toFixed(1)}s, clamped to {MAX_BUFFER_SECS}s</span>}
        </div>
      </header>

      {error && <div className="live-error" onClick={() => setError(null)}>{error} ✕</div>}

      {/* ── the two halves ─────────────────────────────────────── */}
      <main className="live-main" ref={patchLayerRef}>
        {/* Patch cables. Pointer-transparent so the panes stay usable; the
            layer only draws. */}
        <svg className="patch-layer" aria-hidden="true">
          {SOURCE_IDS.flatMap(id => {
            const a = anchors.src[id];
            if (!a || !sources[id].enabled) return [];
            return sources[id].to.slice(0, anchors.trk.length).map((on, ti) => {
              const b = anchors.trk[ti];
              if (!on || !b) return null;
              // Leaves the source horizontally, arrives at the track from
              // directly above, with a little slack so it reads as a cord.
              const dx = Math.max(48, Math.abs(b.x - a.x) * 0.5);
              const dy = Math.max(48, Math.abs(b.y - a.y) * 0.5);
              const sag = Math.min(26, Math.abs(b.x - a.x) * 0.08);
              const d = `M ${a.x} ${a.y} C ${a.x + dx} ${a.y + sag}, `
                      + `${b.x} ${b.y - dy + sag}, ${b.x} ${b.y}`;
              const col = SOURCE_META[id].accent;
              return (
                <g key={`${id}-${ti}`}>
                  <path className="cable-shadow" d={d} />
                  <path className="cable" stroke={col} d={d} />
                  <circle className="cable-plug" cx={a.x} cy={a.y} r={4} fill={col} />
                  <circle className="cable-plug" cx={b.x} cy={b.y} r={4} fill={col} />
                </g>
              );
            });
          })}
          {patchFrom && patchTo && anchors.src[patchFrom] && (
            <g>
              <path
                className="cable dragging"
                stroke={SOURCE_META[patchFrom].accent}
                d={`M ${anchors.src[patchFrom]!.x} ${anchors.src[patchFrom]!.y} L ${patchTo.x} ${patchTo.y}`}
              />
              <circle className="cable-plug" cx={patchTo.x} cy={patchTo.y} r={5}
                      fill={SOURCE_META[patchFrom].accent} />
            </g>
          )}
        </svg>

        <section className="pane input-pane">
          <canvas ref={noiseCanvasRef} className="pane-noise" />
          <div className="pane-title">Live-Input Feed</div>

          {/* One horizontal band per enabled source, each with its own display,
              send fader and patch node. */}
          <div className="in-stack">
            {SOURCE_IDS.filter(id => sources[id].enabled).map(id => (
              <InputSection
                key={id}
                id={id}
                state={sources[id]}
                connected={sources[id].to.slice(0, tracks.length).filter(Boolean).length}
                patching={patchFrom !== null}
                onLevel={(v) => {
                  patchSource(id, { level: v });
                  srcNodesRef.current[id]?.gain.gain.setTargetAtTime(
                    v, inCtxRef.current?.currentTime ?? 0, 0.02);
                }}
                onDisable={() => patchSource(id, { enabled: false })}
                onCanvas={(el) => { srcCanvasRefs.current[id] = el; }}
                onNode={(el) => { srcNodeEls.current[id] = el; }}
                onNodeDown={(e) => beginPatch(id, e)}
                canvasProps={id === 'mic' ? {
                  style: { cursor: showLoopView ? 'ew-resize' : 'default' },
                  onPointerDown: (e) => {
                    if (!showLoopView || !loopLen) return;
                    (e.target as Element).setPointerCapture(e.pointerId);
                    const at = posToTime(e);
                    dragHandleRef.current = Math.abs(at - trim[0]) <= Math.abs(at - trim[1]) ? 0 : 1;
                    applyTrim(at);
                  },
                  onPointerMove: (e) => { if (dragHandleRef.current !== null) applyTrim(posToTime(e)); },
                  onPointerUp: (e) => {
                    dragHandleRef.current = null;
                    try { (e.target as Element).releasePointerCapture(e.pointerId); } catch { /* done */ }
                  },
                } : undefined}
                overlay={id === 'code' ? (
                  <StrudelPanel
                    onEvaluate={runStrudel}
                    onHush={hushStrudel}
                    running={strudelPlaying}
                    ready={strudelReady}
                    banks={strudelBanks}
                    error={strudelError}
                  />
                ) : undefined}
                tools={
                  id === 'mic' ? (
                    <div className="in-tools">
                      {loopState !== 'recording' ? (
                        <button className="tool rec" onClick={startRec} disabled={!running}
                                title={running ? 'Record a loop' : 'Start the transport first'}>
                          <Circle size={10} /> REC
                        </button>
                      ) : (
                        <button className="tool recording" onClick={stopRec}>
                          <Square size={10} /> STOP {recSecs.toFixed(1)}s
                        </button>
                      )}
                      {loopState === 'playing'
                        ? <button className="tool on" onClick={stopLoop}><Square size={10} /> STOP</button>
                        : <button className="tool" onClick={playLoop} disabled={!loopLen}><Play size={10} /> PLAY</button>}
                      <button className="tool del" onClick={deleteLoop} disabled={!loopLen}>
                        <Trash2 size={10} /> DEL
                      </button>
                      {loopLen > 0 && (
                        <span className="tools-meta">
                          loop {(trim[1] - trim[0]).toFixed(2)}s of {loopLen.toFixed(2)}s · drag edges to trim
                        </span>
                      )}
                    </div>
                  ) : id === 'file' ? (
                    <div className="in-tools">
                      <button className="tool" disabled={running}
                              onClick={() => filePickerRef.current?.click()}>
                        <FolderOpen size={10} /> {fileInfo ? 'Change…' : 'Import…'}
                      </button>
                      {fileInfo && (
                        <span className="tools-meta" title={fileInfo.name}>
                          {fileInfo.name} · {fileInfo.duration.toFixed(1)}s
                        </span>
                      )}
                    </div>
                  ) : undefined
                }
              />
            ))}
          </div>

          {/* Turn sources on; they run simultaneously rather than exclusively. */}
          <div className="source-row">
            <div className="source-toggle">
              {SOURCE_IDS.map(id => (
                <button
                  key={id}
                  className={sources[id].enabled ? 'on' : ''}
                  disabled={running}
                  onClick={() => {
                    if (id === 'file' && !fileRef.current) { filePickerRef.current?.click(); return; }
                    patchSource(id, { enabled: !sources[id].enabled });
                  }}
                >
                  {id === 'mic' ? <Mic size={13} /> : id === 'file' ? <Repeat size={13} /> : <Code2 size={13} />}
                  {SOURCE_META[id].name}
                </button>
              ))}
            </div>

            <input ref={filePickerRef} type="file" accept="audio/*" hidden
                   onChange={e => {
                     const f = e.target.files?.[0];
                     if (f) pickFile(f);
                     e.target.value = '';   // allow re-picking the same file
                   }} />
          </div>

          <MidiSlider {...knobProps('inputMonitor')} label="Monitor"
                      value={inputMonitor} onChange={setInputMonitor} />
        </section>

        <div className="middle" title={`${cur.steps} steps · variant ${cur.variant} at noise ${(cur.noise + cur.variant * VARIANT_DELTA).toFixed(2)}`}>
          <div className="middle-bar" style={{ width: `${barWidth}px` }}>
            <div className="variant-marker" style={{ top: `${2 + variantPct * 0.96}%` }}>
              <span className="variant-marker-tag">{cur.variant}</span>
            </div>
          </div>
        </div>

        <div className="out-stack">
          {tracks.map((t, i) => (
            <TrackOutput
              key={i}
              ref={(h) => { trackRefs.current[i] = h; }}
              index={i}
              connectionId={connectionId}
              params={t}
              shared={{ inputRate: effInputRate, generateRate: effGenerateRate, model }}
              onParam={(patch) => patchTrack(i, patch)}
              // The MIDI bridge broadcasts to every session; only track 1 acts
              // on it, or one CC would be applied twice.
              onMidi={i === 0 ? applyMidi : undefined}
              onAdd={i === 0 && tracks.length === 1 ? addTrack : undefined}
              onRemove={i === 1 ? removeTrack : undefined}
              onError={setError}
              onConnected={i === 0 ? setConnected : undefined}
              active={activeTab === i}
              onActivate={() => setActiveTab(i)}
              soloed={solo === i}
              muted={solo !== null && solo !== i}
              onSolo={() => setSolo(cur => (cur === i ? null : i))}
              soloVisible={tracks.length > 1}
              onNode={(el) => { trackNodeEls.current[i] = el; }}
              patchTarget={patchFrom !== null}
              onShelfDone={() => setShelfBusy(false)}
              learnProps={knobProps}
              running={running}
            />
          ))}
        </div>

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
      <footer className={`live-knobs ${midiLearn ? 'learning' : ''} ${masterMode ? 'master' : ''}`}
              style={{ ['--track' as string]: masterMode ? '#e6e6e6' : TRACK_THEME[activeTab]?.accent }}>
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
                value={cur.noise} onChange={v => patchActive({ noise: v })} />
          <Knob {...knobProps('steps')} label="Steps" min={1} max={16} step={1}
                value={cur.steps} onChange={v => patchActive({ steps: v })} format={v => String(Math.round(v))} />
          <Knob {...knobProps('cfg')} label="CFG" min={1} max={10} step={0.1}
                value={cur.cfg} onChange={v => patchActive({ cfg: v })} format={v => v.toFixed(1)} />
        </div>

        <div className="knob-group">
          <div className="variant-cell">
            <Knob {...knobProps('variant')} label="Variant"
                  min={0} max={Math.max(0, activeVariants - 1)} step={1}
                  disabled={!cur.multiVariant}
                  value={cur.variant} onChange={v => patchActive({ variant: v })}
                  format={v => cur.multiVariant
                    ? `${Math.round(v)} · ${(cur.noise + v * VARIANT_DELTA).toFixed(2)}`
                    : 'off'} />
            <label className="variant-toggle" title={`Generate ${VARIANTS} progressively noisier variants each cycle (${VARIANTS}x slower)`}>
              <input type="checkbox" checked={cur.multiVariant}
                     onChange={e => patchActive({ multiVariant: e.target.checked, variant: 0 })} />
              {VARIANTS} variants
            </label>
          </div>
        </div>

        <div className="knob-group">
          <Knob {...knobProps('loopback')} label="Loopback" min={0} max={1} step={0.01}
                value={cur.loopback} onChange={v => patchActive({ loopback: v })}
                format={v => `${Math.round(v * 100)}%`} />
          <Knob {...knobProps('nudge')} label="Nudge" min={0} max={MAX_NUDGE} step={0.01}
                value={cur.nudge} onChange={v => patchActive({ nudge: v })}
                format={v => `${v.toFixed(2)}s`} />
        </div>

        {tracks.length > 1 && (
          <button
            className={`master-btn ${masterMode ? 'on' : ''}`}
            onClick={() => setMasterMode(v => !v)}
            title="Apply knob and MIDI changes to every track at once"
          >
            <span>MASTER</span>
            <span className="master-sub">{masterMode ? 'all tracks' : `track ${activeTab + 1}`}</span>
          </button>
        )}
      </footer>
    </div>
  );
}
