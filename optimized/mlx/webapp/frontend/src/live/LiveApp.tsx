import { useCallback, useEffect, useRef, useState } from 'react';
import { Play, Square, Sliders, Circle, Mic, Repeat, FolderOpen, PanelRight, Code2, Trash2, Plus, Rows2, Columns2, AudioWaveform, Speaker } from 'lucide-react';
import ShowView from './ShowView';
import Knob from './Knob';
import LatentShelf from './LatentShelf';
import StrudelPanel, { type StrudelPanelHandle } from './StrudelPanel';
import InputSection from './InputSection';
import MidiSlider from './MidiSlider';
import TrackOutput, { type TrackHandle } from './TrackOutput';
import { FX_METHODS, FxChain, fxMethod, MASTER_FX_AMOUNT } from './inputfx';
import {
  API_URL, VARIANTS, SPREAD_MIN, SPREAD_MAX, MAX_NUDGE, TRACK_THEME,
  OUT_PAIRS, pairLabel,
  SYNC_POLL_MS, SYNC_DEADBAND,
  SOURCE_KINDS, SOURCE_META, MAX_SLOTS, defaultSources, defaultTrack, newSource,
  lpfHz, hpfHz, combHz,
  type TrackParams, type SourceKind, type SourceInstance,
} from './trackTypes';

/** Cutoffs read better as "1.2k" than as four digits under a knob. */
const hz = (f: number) =>
  (f >= 1000 ? `${(f / 1000).toFixed(f >= 10000 ? 0 : 1)}k` : String(Math.round(f))) + 'Hz';

/** Stacked, side by side, or the bare performance screen. */
type Layout = 'vertical' | 'horizontal' | 'show';
/** The split is only meaningful for the two working layouts. */
const splitKey = (l: Layout): 'vertical' | 'horizontal' =>
  (l === 'horizontal' ? 'horizontal' : 'vertical');

interface RingBuf { l: Float32Array; r: Float32Array; w: number; filled: number; sr: number }
interface SourceNodes {
  head: AudioNode;
  gain: GainNode;
  /** What the track mixes and the monitor listen to: the pre-FX chain's
   *  output, which is the fader when no effect is engaged. Everything
   *  downstream patches here so switching method never rewires the box. */
  out: AudioNode;
  fx: FxChain;
  analyser: AnalyserNode;
  /** Mic boxes only: kept so the channel pick can be rewired live. */
  merger?: ChannelMergerNode;
  /** Taps either side of the pre-FX chain, so a box can show what the model is
   *  being fed against what went in. Both are post-fader, so the only
   *  difference between them is the effect itself. */
  fxIn?: AnalyserNode;
  fxOut?: AnalyserNode;
  /**
   * Listen-only gain for this box, between its output and the monitor bus. In
   * summed mode it sits at 1 and the bus carries the master level; per box it
   * carries the box's own and the bus becomes a trim over all of them. Flipping
   * modes therefore changes gains only — nothing is ever rewired.
   */
  mon?: GainNode;
  /** Mic: the hardware feed and this box's own loop share the box's fader. */
  liveGain?: GainNode;
  loopGain?: GainNode;
  recProc?: ScriptProcessorNode;
  /** File: a fixed head the transport plugs playback nodes into. A buffer
   *  source can only be started once, so PLAY makes a new one each time. */
  fileGain?: GainNode;
  fileBuf?: AudioBuffer;
  fileSrc?: AudioBufferSourceNode | null;
  fileStartedAt?: number;
}

/** Per-box looper. Every input box owns one, so a take recorded in one Live
 *  Input never appears in another. */
interface LoopSlot {
  state: 'idle' | 'recording' | 'stopped' | 'playing';
  len: number;
  trim: [number, number];
  recSecs: number;
}
const IDLE_SLOT: LoopSlot = { state: 'idle', len: 0, trim: [0, 0], recSecs: 0 };

/** Reduce a buffer to per-column peaks. Redrawing 60x a second from the raw
 *  samples would be far too slow for a minute of audio. */
const peaksOf = (buf: AudioBuffer, cols = 900) => {
  const L = buf.getChannelData(0);
  const peaks = new Float32Array(cols);
  const per = Math.max(1, Math.floor(buf.length / cols));
  for (let c = 0; c < cols; c++) {
    let m = 0;
    const s0 = c * per, s1 = Math.min(buf.length, s0 + per);
    for (let i = s0; i < s1; i++) m = Math.max(m, Math.abs(L[i]));
    peaks[c] = m;
  }
  return peaks;
};
import type { StrudelHandle } from './strudel/strudelEngine';
import './live.css';


// 32 beats at 60 BPM is 32 s, so the ring must hold that much for the longest
// beat-denominated window. ~12 MB of Float32 at 48 kHz stereo.
const MAX_BUFFER_SECS = 32;  // largest InputRate the ring buffer must hold
const MAX_LOOP_SECS = 60;    // ceiling on a recorded loop
/**
 * How many hardware channels a Live Input box can choose from — the channel
 * list, the splitter and the clamp all follow it.
 *
 * TWO, AND NOT BECAUSE OF THE INTERFACE. Chrome hardcodes getUserMedia's
 * maximum channel count to 2 (Chromium issue 40403559, open since the API
 * shipped): channels 1 and 2 arrive, and 3 upward produce nothing at all. This
 * was 4 for an 18-in Focusrite, and the result was a picker offering Ch 3 and
 * Ch 4 that could never carry audio — a promise the browser will not keep, and
 * worse than not offering them, because you reach for one mid-set. Raising it
 * again only helps once that limit moves, or once capture leaves the browser
 * for the backend, where all 18 are reachable.
 *
 * The list is still NOT built from what the device reports: Chrome routinely
 * leaves `channelCount` out of a track's settings, and the choice has to be
 * pickable before the transport starts, when no device is open to report
 * anything. The real count is detected and shown in the status line.
 */
const MAX_INPUT_CHANNELS = 2;

// Every whole second, when Tempo is off. The sparse 1/3/5/10 set could not
// express a 3 s window regenerated every 2 s, which is exactly the kind of
// overlap the loop is for.
//
// Generate Rate stops one second short of Input Rate's ceiling, which is the
// longest cycle that still leaves an overlap at the widest window. The list is
// fixed rather than following the Input Rate knob: a short window regenerated
// sparsely -- 3 s every 12 s -- is a legitimate setting, and a moving ceiling
// would take it away and shift the knob's range under your hand mid-set.
// Both stay inside the backend's limits (input_rate 0.25-32 s, generate_rate
// 0.2-40 s) and inside the ring, which holds MAX_BUFFER_SECS.
const wholeSecs = (n: number) => Array.from({ length: n }, (_, i) => i + 1);
const INPUT_RATES = wholeSecs(15);
const GEN_RATES = wholeSecs(14);
const GEN_MIN = GEN_RATES[0], GEN_MAX = GEN_RATES[GEN_RATES.length - 1];
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
   *  value; turning it in master mode makes the tracks agree.
   *
   *  A track with SOLO PROC lit is passed over by the master sweep, but never
   *  when it is the active track: the button is an opt-out from the rack moving
   *  together, not a lock, and a track you cannot edit at all would be a trap. */
  const patchActive = useCallback((patch: Partial<TrackParams>) => {
    setTracks(ts => ts.map((t, j) => (
      j === activeTabRef.current || (masterRef.current && !t.soloProcess)
        ? { ...t, ...patch } : t)));
  }, []);
  /**
   * Hardware output routing. The master pair carries the whole rack and every
   * track feeds it, so a track's own pair is an ADDITIONAL send rather than a
   * move — nothing ever leaves the mix by being routed somewhere.
   *
   * `outChannels` is what the device actually gave us, not what we asked for.
   * It is shown in the panel because a browser that hands back 2 would make
   * every pair collapse onto 1/2 while the pickers still read 5/6.
   */
  const [masterPair, setMasterPair] = useState(0);
  const [outChannels, setOutChannels] = useState<number | null>(null);
  const [mapOpen, setMapOpen] = useState(false);

  /**
   * Master pre-FX: the same Smear and Scramble the boxes offer, but on the
   * summed feed each track sends, so one press processes everything patched in.
   * Latching, and both can be on at once.
   *
   * SCRAMBLE RUNS FIRST. It cuts its slices on detected onsets, and Smear
   * exists to dissolve exactly those onsets, so the other order would leave the
   * detector nothing to find.
   */
  /**
   * FREEZE holds the input the model is working from. The ring simply stops
   * advancing, so the window every snapshot reads — which ends at the write
   * head — is the same audio on every cycle from the press onward. Generation
   * carries on with a fresh seed each time, so you get the model turning one
   * held moment over and over rather than following you.
   *
   * It is the capture that freezes, not the room: monitoring, the waveforms and
   * the input FX all stay live, and whatever FX was engaged at the press is
   * baked into the held buffer.
   */
  const [frozen, setFrozen] = useState(false);
  const frozenRef = useRef(false);
  useEffect(() => { frozenRef.current = frozen; }, [frozen]);

  const [masterFx, setMasterFx] = useState({ scramble: false, smear: false });
  const masterFxRef = useRef(masterFx);
  useEffect(() => { masterFxRef.current = masterFx; }, [masterFx]);
  /** Amount for either method when its button is lit. */
  const masterFxAmt = useCallback(
    (on: boolean) => (on ? MASTER_FX_AMOUNT : 0), []);

  /** Rack-wide output fader. Each track multiplies it into its own monitor
   *  gain, so the per-track balance survives a move of this one. */
  const [masterVolume, setMasterVolume] = useState(1);
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

  // One looper per input box, keyed by source id.
  const [loops, setLoops] = useState<Record<string, LoopSlot>>({});
  /** Which file boxes are currently sounding. */
  const [filePlaying, setFilePlaying] = useState<Record<string, boolean>>({});
  const [model, setModel] = useState<'sm-music' | 'sm-sfx' | 'medium'>('sm-music');
  // Off by default: one generation per cycle is 5x faster, and the ladder is
  // only worth paying for when you are actually auditioning across it.
  const cur = tracks[activeTab] ?? tracks[0];
  const activeVariants = cur.multiVariant ? VARIANTS : 1;

  /**
   * The master monitor level, remembered per mode. Summed monitoring starts
   * muted, because a mic that monitors itself the moment it opens howls. Per-box
   * monitoring starts at unity instead: there the master is only a trim over
   * levels that are themselves all at zero, so unity is still silent — and
   * leaving it at zero would make every per-box fader do nothing, which is the
   * one combination of these two defaults that reads as a bug.
   */
  const [monLevels, setMonLevels] = useState<{ sum: number; box: number }>({ sum: 0, box: 1 });
  const monitorModeRef = useRef<'sum' | 'box'>('sum');
  /** Summed monitoring, or one level per input box. */
  const [monitorMode, setMonitorMode] = useState<'sum' | 'box'>(() => {
    try { return localStorage.getItem('sa3_live_monmode') === 'box' ? 'box' : 'sum'; }
    catch { return 'sum'; }
  });
  useEffect(() => {
    try { localStorage.setItem('sa3_live_monmode', monitorMode); } catch { /* private mode */ }
    monitorModeRef.current = monitorMode;
  }, [monitorMode]);
  const inputMonitor = monLevels[monitorMode === 'box' ? 'box' : 'sum'];
  /** Same signature the slider and MIDI already use; writes whichever mode's
   *  value is live, read from a ref so a stale closure cannot cross them. */
  const setInputMonitor = useCallback((v: number | ((prev: number) => number)) => {
    setMonLevels(m => {
      const key = monitorModeRef.current === 'box' ? 'box' : 'sum';
      return { ...m, [key]: typeof v === 'function' ? v(m[key]) : v };
    });
  }, []);

  // --- latent shelf ----------------------------------------------------
  const [shelfOpen, setShelfOpen] = useState(false);
  const [shelfBusy, setShelfBusy] = useState(false);
  // Cable being dragged from a source node, and where the pointer is now.
  // A drag in progress, from either end. Releasing over the opposite kind
  // of node toggles that connection, so a cable can be removed from either side.
  const [patchFrom, setPatchFrom] = useState<string | null>(null);
  const [patchFromTrack, setPatchFromTrack] = useState<number | null>(null);
  const [patchTo, setPatchTo] = useState<{ x: number; y: number } | null>(null);
  const [renormalize, setRenormalize] = useState(false);

  // --- input source: live hardware, or a file looped as a stand-in ------
  // Input boxes are instances, not a fixed set of kinds, so two Live Inputs can
  // carry different channels and be patched to different tracks.
  const [sources, setSources] = useState<SourceInstance[]>(() => defaultSources(1));
  const sourcesRef = useRef(sources);
  useEffect(() => { sourcesRef.current = sources; }, [sources]);
  const tracksRef = useRef(tracks);
  useEffect(() => { tracksRef.current = tracks; }, [tracks]);
  const patchSource = useCallback((id: string, patch: Partial<SourceInstance>) => {
    setSources(list => list.map(x => (x.id === id ? { ...x, ...patch } : x)));
  }, []);
  const [addMenu, setAddMenu] = useState(false);
  /** Ceiling on both input boxes and output tracks. */
  const [maxSlots, setMaxSlots] = useState(MAX_SLOTS);
  /** The input box the Input FX controls edit, the way the output knobs
   *  follow whichever track is active. */
  const [activeSrc, setActiveSrc] = useState<string | null>(null);
  /** Share of the stage the input side takes. The two views hold it
   *  separately: a comfortable split stacked is not a comfortable one
   *  side by side. */
  const [split, setSplit] = useState<Record<'vertical' | 'horizontal', number>>(() => {
    try {
      const v = JSON.parse(localStorage.getItem('sa3_live_split') || 'null');
      if (v && typeof v.vertical === 'number' && typeof v.horizontal === 'number') return v;
    } catch { /* first run */ }
    return { vertical: 0.5, horizontal: 0.5 };
  });
  /** 'vertical' = inputs above outputs; 'horizontal' = inputs left of outputs. */
  const [layout, setLayout] = useState<Layout>(() => {
    try {
      const v = localStorage.getItem('sa3_live_layout');
      return v === 'horizontal' || v === 'show' ? v : 'vertical';
    }
    catch { return 'vertical'; }
  });
  const measureRef = useRef<(() => void) | null>(null);
  useEffect(() => {
    try { localStorage.setItem('sa3_live_layout', layout); } catch { /* private mode */ }
    // Anchors move with the layout, so re-measure once it has settled.
    const t = window.setTimeout(() => measureRef.current?.(), 80);
    return () => window.clearTimeout(t);
  }, [layout]);
  const [strudelReady, setStrudelReady] = useState(false);
  const [strudelPlaying, setStrudelPlaying] = useState(false);
  const [strudelBanks, setStrudelBanks] = useState<string[]>([]);
  const [strudelError, setStrudelError] = useState<string | null>(null);
  const strudelRef = useRef<StrudelHandle | null>(null);
  const strudelPanelRef = useRef<StrudelPanelHandle | null>(null);

  // --- devices / status ----------------------------------------------
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState('');
  const [detectedChannels, setDetectedChannels] = useState(2);
  const [captureRate, setCaptureRate] = useState<number | null>(null);
  const [status, setStatus] = useState('Idle');
  const [error, setError] = useState<string | null>(null);

  /**
   * Named sets of per-track prompts, kept in a file beside the backend so a
   * combination that worked can be brought back rather than retyped. Recall
   * fills by position and never adds or removes tracks: each track owns a
   * WebSocket and an audio graph, and restructuring the rack is not something
   * a recall should do by accident mid-set.
   */
  const [presets, setPresets] = useState<{ name: string; prompts: string[] }[]>([]);

  useEffect(() => {
    fetch(`${API_URL}/api/live/prompt_presets`)
      .then(r => r.json())
      .then(d => setPresets(d.presets || []))
      .catch(() => {});
  }, []);

  const recallPreset = (name: string) => {
    const p = presets.find(x => x.name === name);
    if (!p) return;
    setTracks(ts => ts.map((t, i) => (
      i < p.prompts.length ? { ...t, prompt: p.prompts[i] } : t)));
  };

  const savePreset = async () => {
    const prompts = tracksRef.current.map(t => t.prompt);
    const name = window.prompt(
      'Save these prompts as', `preset ${presets.length + 1}`)?.trim();
    if (!name) return;
    try {
      const r = await fetch(`${API_URL}/api/live/prompt_presets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, prompts }),
      });
      const d = await r.json();
      if (d.status === 'ok') setPresets(d.presets);
      else setError(`Could not save the preset: ${d.error}`);
    } catch {
      setError('Could not save the preset — is the backend running?');
    }
  };

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
  /** Summed live input, for the show view. */
  const inAnalyserRef = useRef<AnalyserNode | null>(null);
  const showSumRef = useRef<GainNode | null>(null);
  const showBytesRef = useRef<Uint8Array<ArrayBuffer>>(new Uint8Array(new ArrayBuffer(0)));
  const showTmpRef = useRef<Float32Array>(new Float32Array(0));
  const clickGainRef = useRef<GainNode | null>(null);
  const clickTimerRef = useRef<number | null>(null);
  const nextClickRef = useRef(0);
  const clickBeatRef = useRef(0);
  const bpmRef = useRef<number | null>(null);
  const srcNodesRef = useRef<Record<string, SourceNodes>>({});
  // One shared capture of the interface; each mic box picks its own channels
  // out of it rather than opening the device several times.
  const micSplitRef = useRef<ChannelSplitterNode | null>(null);
  const micChanRef = useRef(2);
  const filesRef = useRef<Record<string, File>>({});
  const trackMixRef = useRef<GainNode[]>([]);
  const ringsRef = useRef<RingBuf[]>([]);
  const procsRef = useRef<ScriptProcessorNode[]>([]);
  /** Master pre-FX, one pair per track and index-parallel to the mixes. */
  const mfxRef = useRef<{ scramble: FxChain; smear: FxChain }[]>([]);
  const tapSinkRef = useRef<GainNode | null>(null);
  const srcCanvasRefs = useRef<Record<string, HTMLCanvasElement | null>>({});
  const srcNodeEls = useRef<Record<string, HTMLDivElement | null>>({});
  const trackNodeEls = useRef<Record<string, HTMLDivElement | null>>({});
  const patchLayerRef = useRef<HTMLDivElement | null>(null);
  /** Recorded takes, one per box: the audio, its display peaks, and the node
   *  playing it back. Parallel to `loops`, which holds the same boxes' UI state. */
  const loopAudioRef = useRef<Record<string, {
    buf: AudioBuffer | null;
    peaks: Float32Array | null;
    src: AudioBufferSourceNode | null;
    startedAt: number;
  }>>({});
  const recRef = useRef<Record<string, { chunks: Float32Array[][]; frames: number }>>({});
  /** Imported files get the same waveform display as a recorded take. */
  const filePeaksRef = useRef<Record<string, Float32Array>>({});
  const dragHandleRef = useRef<{ id: string; h: 0 | 1 } | null>(null);
  const splitterRef = useRef<ChannelSplitterNode | null>(null);
  const mergerRef = useRef<ChannelMergerNode | null>(null);
  const filePickerRef = useRef<HTMLInputElement | null>(null);
  const noiseCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const midNoiseRef = useRef<HTMLCanvasElement | null>(null);
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

  /** The Generate Rate to put back when the valve opens again, or null when
   *  freeze did not touch it. */
  const thawRateRef = useRef<number | null>(null);

  /**
   * Open and close the valve. Holding the input only pays off if a cycle lasts
   * long enough to play the held window through: a cycle gets Generate Rate
   * seconds of airtime before the next one restarts it from zero, so at 2 s a
   * 14 s clip is restarted six times before it ever reaches its end and all you
   * hear is the first two seconds, over and over, from six generations. One
   * second under the window is the longest cycle that still leaves an overlap
   * for the crossfade and for the generation latency to land in.
   *
   * Tempo mode is left alone: its rates are in beats (4/8/16/32), where "one
   * second less" has no equivalent and the nearest move is halving, which is a
   * different musical decision and not one to make on someone's behalf.
   */
  const setFreeze = useCallback((on: boolean) => {
    setFrozen(on);
    if (on) {
      if (tempoRef.current) { thawRateRef.current = null; return; }
      thawRateRef.current = paramsRef.current.generateRate;
      const held = paramsRef.current.inputRate;
      setGenerateRate(Math.max(GEN_MIN, Math.min(GEN_MAX, Math.round(held) - 1)));
    } else if (thawRateRef.current !== null) {
      setGenerateRate(thawRateRef.current);
      thawRateRef.current = null;
    }
  }, []);
  // Only a take's *length* moves the layout; its playhead and rec counter do not.
  const loopLensKey = Object.values(loops).map(l => l.len).join(',');
  const activeSrcRef = useRef<string | null>(activeSrc);
  useEffect(() => { activeSrcRef.current = activeSrc; }, [activeSrc]);
  const loopsRef = useRef(loops);
  useEffect(() => { loopsRef.current = loops; }, [loops]);
  const slotOf = (id: string) => loops[id] ?? IDLE_SLOT;
  const patchSlot = useCallback((id: string, patch: Partial<LoopSlot>) => {
    setLoops(m => ({ ...m, [id]: { ...(m[id] ?? IDLE_SLOT), ...patch } }));
  }, []);
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

  /**
   * The learned mapping lives in a file beside the backend, so it survives a
   * different browser, profile, hostname or port — localStorage is per-origin,
   * and opening the app at 127.0.0.1 rather than localhost is a different store
   * that looks exactly like a lost mapping.
   *
   * On the first run after this change the file is empty and the browser may
   * hold a mapping learned before it, so that one is pushed up rather than
   * discarded. After that the file wins, and localStorage is kept in step only
   * as a fallback for when the backend is not up yet.
   */
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const r = await fetch(`${API_URL}/api/live/midi_map`);
        const d = await r.json();
        if (!alive) return;
        const stored: Bindings = d.bindings || {};
        if (Object.keys(stored).length) {
          setBindings(stored);
          bindingsRef.current = stored;
          return;
        }
        // Nothing on the server yet: adopt whatever this browser remembers.
        const local = bindingsRef.current;
        if (Object.keys(local).length) void saveBindings(local);
      } catch { /* backend not up; localStorage carries us until it is */ }
    })();
    return () => { alive = false; };
  }, []);

  /** Write the mapping through to the file, and to localStorage as a fallback. */
  const saveBindings = async (next: Bindings) => {
    try { localStorage.setItem('sa3_live_midi', JSON.stringify(next)); }
    catch { /* private mode */ }
    try {
      await fetch(`${API_URL}/api/live/midi_map`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bindings: next }),
      });
    } catch { /* backend not up; the local copy is pushed on the next load */ }
  };

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
      void saveBindings(next);
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
      if (tempoRef.current) setInputBeats(pick(BEATS)); else setInputRate(pick(INPUT_RATES));
    } else if (id === 'generateRate') {
      if (tempoRef.current) setGenerateBeats(pick(BEATS)); else setGenerateRate(pick(GEN_RATES));
    }
    // Everything below is per track and lands on whichever tab is showing.
    else if (id === 'noise') patchActive({ noise: Number(lerp(0.05, 1).toFixed(2)) });
    else if (id === 'steps') patchActive({ steps: Math.round(lerp(1, 16)) });
    else if (id === 'cfg') patchActive({ cfg: Number(lerp(1, 10).toFixed(1)) });
    else if (id === 'variant') patchActive({ variant: Math.floor(norm * VARIANTS) });
    else if (id === 'loopback') patchActive({ loopback: Number(norm.toFixed(2)) });
    else if (id === 'nudge') patchActive({ nudge: Number((norm * MAX_NUDGE).toFixed(2)) });
    else if (id === 'spread') {
      patchActive({ spread: Number((SPREAD_MIN + norm * (SPREAD_MAX - SPREAD_MIN)).toFixed(3)) });
    }
    // Momentary: fire once as the control crosses halfway, so a pad, a
    // footswitch or a fader all work and holding one down does not repeat.
    else if (id === 'mfxScramble' || id === 'mfxSmear') {
      const down = norm >= 0.5;
      if (down && !triggersRef.current[id]) {
        const k = id === 'mfxScramble' ? 'scramble' : 'smear';
        setMasterFx(v => ({ ...v, [k]: !v[k] }));
      }
      triggersRef.current[id] = down;
    }
    else if (id === 'freeze') {
      const down = norm >= 0.5;
      if (down && !triggersRef.current[id]) setFreeze(!frozenRef.current);
      triggersRef.current[id] = down;
    }
    else if (id === 'inject') {
      const down = norm >= 0.5;
      if (down && !triggersRef.current[id]) actionsRef.current.inject();
      triggersRef.current[id] = down;
    }
    else if (id === 'fxAmount') {
      const sid = activeSrcRef.current ?? sourcesRef.current[0]?.id;
      if (sid) patchSource(sid, { fxAmount: Number(norm.toFixed(3)) });
    }
    else if (id === 'lpf') patchActive({ lpf: Number(norm.toFixed(3)) });
    else if (id === 'hpf') patchActive({ hpf: Number(norm.toFixed(3)) });
    else if (id === 'comb') patchActive({ comb: Number(norm.toFixed(3)) });
    else if (id === 'inputMonitor') setInputMonitor(Number(norm.toFixed(2)));
    else if (id === 'masterVolume') setMasterVolume(Number(norm.toFixed(2)));
    // One binding per box, keyed to the box rather than to a slot number.
    else if (/^inputMonitor@\d+$/.test(id)) {
      const inst = sourcesRef.current[Number(id.slice('inputMonitor@'.length))];
      if (inst) patchSource(inst.id, { monitor: Number(norm.toFixed(2)) });
    }
    else if (/^send\d+$/.test(id)) {
      const inst = sourcesRef.current[Number(id.slice(4))];
      if (inst) patchSource(inst.id, { level: Number(norm.toFixed(2)) });
    }
    else if (id.startsWith('outputMonitor')) {
      const t = Number(id.slice('outputMonitor'.length)) || 0;
      patchTrack(t, { outputMonitor: Number(norm.toFixed(2)) });
    }
  }, [patchActive, patchTrack]);

  /** Which momentary bindings are currently held, for edge detection. */
  const triggersRef = useRef<Record<string, boolean>>({});

  const bindingLabel = (id: string) => {
    const b = bindings[id];
    return b ? `CC ${b.control}` : null;
  };

  // ─────────────────────────────────────────────────────────────────────
  // Output playback: decode, crossfade, drive the falling spectrogram
  // ─────────────────────────────────────────────────────────────────────
  // Re-pick a box's channel(s) without stopping the loop — useful for moving
  // one input from, say, a guitar on channel 1 to a mic on channel 2 mid-set.
  // Disconnecting by target only detaches THIS box, leaving the others alone.
  const chanKey = sources.map(x => `${x.id}:${x.chanSel}`).join('|');
  useEffect(() => {
    const sp = micSplitRef.current;
    if (!sp) return;
    for (const inst of sourcesRef.current) {
      if (inst.kind !== 'mic') continue;
      const mg = srcNodesRef.current[inst.id]?.merger;
      if (!mg) continue;
      const [a, b] = parseChannelSel(inst.chanSel, micChanRef.current);
      try {
        sp.disconnect(mg);
        sp.connect(mg, a, 0);
        sp.connect(mg, b, 1);
      } catch (e: any) {
        setError(`Could not select channel: ${e.message || e}`);
      }
    }
  }, [chanKey]);

  useEffect(() => { if (inGainRef.current) inGainRef.current.gain.value = inputMonitor; }, [inputMonitor]);

  // Per-box monitor levels, and the switch between the two modes. The master
  // slider stays live in both: in box mode it trims the whole monitor bus.
  const levelKey = sources.map(x => `${x.id}:${x.level}`).join(',');
  useEffect(() => {
    const t = inCtxRef.current?.currentTime ?? 0;
    for (const inst of sourcesRef.current) {
      srcNodesRef.current[inst.id]?.gain.gain.setTargetAtTime(inst.level, t, 0.02);
    }
  }, [levelKey]);

  const monKey = `${monitorMode}|${sources.map(x => `${x.id}:${x.monitor}`).join(',')}`;
  useEffect(() => {
    const t = inCtxRef.current?.currentTime ?? 0;
    for (const inst of sourcesRef.current) {
      const g = srcNodesRef.current[inst.id]?.mon;
      // Ramped, so moving a fader or flipping the mode never clicks.
      g?.gain.setTargetAtTime(monitorModeRef.current === 'box' ? inst.monitor : 1, t, 0.02);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [monKey]);

  // Tempo-aware pre-FX (Scramble slices in sixteenths) follow the app's BPM,
  // including the one a Live Code pattern sets on evaluate — which happens
  // after the chains are built, so this has to be live rather than set once.
  useEffect(() => {
    for (const n of Object.values(srcNodesRef.current)) n?.fx.setTempo(bpm);
    for (const m of mfxRef.current) m.scramble.setTempo(bpm);
  }, [bpm]);

  // The buttons are latches, so the amount is the only thing that moves. Both
  // methods are a true bypass at 0 — they tear their guts down rather than run
  // at zero mix — so an unlit button costs nothing per track.
  useEffect(() => {
    for (const m of mfxRef.current) {
      m.scramble.setAmount(masterFxAmt(masterFx.scramble));
      m.smear.setAmount(masterFxAmt(masterFx.smear));
    }
  }, [masterFx, masterFxAmt]);

  // Pre-FX follows its box live. Method changes swap the chain's guts behind
  // stable endpoints, so a box never drops out of its patch mid-set.
  const fxKey = sources.map(x => `${x.id}:${x.fxMethod}:${x.fxAmount}`).join('|');
  useEffect(() => {
    for (const inst of sourcesRef.current) {
      const fx = srcNodesRef.current[inst.id]?.fx;
      if (!fx) continue;
      fx.setMethod(inst.fxMethod);
      fx.setAmount(inst.fxAmount);
    }
  }, [fxKey]);

  // ─────────────────────────────────────────────────────────────────────
  // Canvas render loop: input waveform (left) + output spectrogram (right)
  // ─────────────────────────────────────────────────────────────────────
  /**
   * The strip along the bottom of a box whose input effect is engaged: the
   * signal going into the effect as a ghost, the signal coming out over it.
   * The gap between the two IS the effect — a smear spreads one transient into
   * a swell, a scramble puts it somewhere else entirely, a disperse rings on
   * after it — so this reads without having to label what each one does.
   */
  const drawFxBand = useCallback((
    inst: SourceInstance, g: CanvasRenderingContext2D,
    w: number, h: number, bandH: number,
  ) => {
    const n = srcNodesRef.current[inst.id];
    if (!n?.fxOut) return;
    const top = h - bandH, midY = top + bandH / 2, amp = bandH / 2 - 4;
    g.fillStyle = 'rgba(0,0,0,.5)';
    g.fillRect(0, top, w, bandH);
    g.fillStyle = 'rgba(255,255,255,.12)';
    g.fillRect(0, top, w, 1);
    const trace = (an: AnalyserNode, stroke: string, lw: number) => {
      const d = new Uint8Array(an.fftSize);
      an.getByteTimeDomainData(d);
      g.strokeStyle = stroke;
      g.lineWidth = lw;
      g.beginPath();
      for (let i = 0; i < d.length; i++) {
        const x = (i / d.length) * w;
        const y = midY + (d[i] / 128 - 1) * amp;
        i === 0 ? g.moveTo(x, y) : g.lineTo(x, y);
      }
      g.stroke();
    };
    if (n.fxIn) trace(n.fxIn, 'rgba(255,255,255,.22)', 1.5);
    trace(n.fxOut, SOURCE_META[inst.kind].accent, 2.5);
  }, []);

  useEffect(() => {
    const tick = () => {
      rafRef.current = requestAnimationFrame(tick);

      // Nothing behind the show view is visible, so none of it is worth drawing.
      // The tracks' own spectrograms keep running: they are cheap, and a track
      // cannot be unmounted without losing its session.
      if (layoutRef.current === 'show') { frameRef.current++; return; }

      // One scope per enabled source. A Live Input's canvas switches to its own
      // recorded take once one exists, and a File Loop shows the file it
      // imported; both carry a handle at each end of the loop.
      for (const inst of sourcesRef.current) {
        const id = inst.id;
        const wc = srcCanvasRefs.current[id];
        if (!wc) continue;
        const ctx2 = wc.getContext('2d');
        if (!ctx2) continue;
        const w = wc.width, h = wc.height;
        const accent = SOURCE_META[inst.kind].accent;
        const c2 = inCtxRef.current;
        ctx2.clearRect(0, 0, w, h);

        // An engaged input effect claims a band along the bottom; everything
        // else draws in the height above it. Only once the graph exists,
        // though — before the transport starts there is nothing to show, and a
        // blank strip would just squash the waveform.
        const bandH = inst.fxAmount > 0 && srcNodesRef.current[id]?.fxOut
          ? Math.max(40, Math.round(h * 0.34)) : 0;
        const vh = h - bandH, mid = vh / 2;
        let view: {
          peaks: Float32Array; len: number; t0: number; t1: number; pos: number | null;
        } | null = null;

        if (inst.kind === 'mic') {
          const slot = loopsRef.current[id] ?? IDLE_SLOT;
          const a = loopAudioRef.current[id];
          if (a?.peaks && (slot.state === 'stopped' || slot.state === 'playing')) {
            const [t0, t1] = slot.trim;
            const span = Math.max(0.05, t1 - t0);
            view = {
              peaks: a.peaks, len: slot.len || 1, t0, t1,
              pos: slot.state === 'playing' && c2
                ? t0 + ((c2.currentTime - a.startedAt) % span) : null,
            };
          }
        } else if (inst.kind === 'file') {
          const fp = filePeaksRef.current[id];
          const len = inst.fileDuration ?? 0;
          if (fp && len > 0) {
            const n = srcNodesRef.current[id];
            const [t0, t1] = inst.fileTrim ?? [0, len];
            const span = Math.max(0.05, t1 - t0);
            view = {
              peaks: fp, len, t0, t1,
              pos: n?.fileSrc && c2
                ? t0 + ((c2.currentTime - (n.fileStartedAt ?? 0)) % span) : null,
            };
          }
        }

        if (view) {
          const { peaks, len, t0, t1, pos } = view;
          const x0 = (t0 / len) * w, x1 = (t1 / len) * w;
          // Outside the trim is dimmed rather than hidden, so you can see what
          // you cropped off and drag it back.
          ctx2.fillStyle = 'rgba(0,0,0,.45)';
          ctx2.fillRect(0, 0, x0, vh);
          ctx2.fillRect(x1, 0, w - x1, vh);
          for (let c = 0; c < peaks.length; c++) {
            const x = (c / peaks.length) * w;
            const inTrim = x >= x0 && x <= x1;
            ctx2.fillStyle = inTrim ? 'rgba(255,255,255,.85)' : 'rgba(255,255,255,.22)';
            const amp = Math.max(1, peaks[c] * mid);
            ctx2.fillRect(x, mid - amp, Math.max(1, w / peaks.length), amp * 2);
          }
          ctx2.fillStyle = accent;
          ctx2.fillRect(x0 - 1, 0, 3, vh);
          ctx2.fillRect(x1 - 1, 0, 3, vh);
          if (pos !== null) {
            ctx2.fillStyle = accent;
            ctx2.fillRect((pos / len) * w, 0, 2, vh);
          }
        } else {
          const an = srcNodesRef.current[id]?.analyser;
          if (an) {
            const data = new Uint8Array(an.fftSize);
            an.getByteTimeDomainData(data);
            ctx2.strokeStyle = accent;
            ctx2.lineWidth = 1.5;
            ctx2.beginPath();
            for (let i = 0; i < data.length; i++) {
              const x = (i / data.length) * w;
              const y = (data[i] / 128 - 1) * mid + mid;
              i === 0 ? ctx2.moveTo(x, y) : ctx2.lineTo(x, y);
            }
            ctx2.stroke();
          }
        }

        if (bandH) drawFxBand(inst, ctx2, w, h, bandH);
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
      for (const [id, rec] of Object.entries(recRef.current)) {
        if (!inCtx) continue;
        // The readout shows tenths; updating it every frame would re-render
        // the whole app 60 times a second for no visible gain.
        const secs = rec.frames / inCtx.sampleRate;
        if (secs - ((loopsRef.current[id] ?? IDLE_SLOT).recSecs) >= 0.1) {
          patchSlot(id, { recSecs: secs });
        }
      }
      // The middle bar carries the same grain as the input background.
      const mn = midNoiseRef.current;
      if (mn && grainRef.current) {
        const g2 = mn.getContext('2d');
        const r2 = mn.getBoundingClientRect();
        if (g2 && r2.width > 1 && r2.height > 1) {
          if (mn.width !== Math.round(r2.width) || mn.height !== Math.round(r2.height)) {
            mn.width = Math.round(r2.width);
            mn.height = Math.round(r2.height);
          }
          g2.clearRect(0, 0, mn.width, mn.height);
          const alpha = 0.9 * Math.pow(noiseRef.current, 0.6);
          if (alpha > 0.012) {
            const i2 = Math.floor(frameRef.current / 3) % grainRef.current.tiles.length;
            const pat2 = g2.createPattern(grainRef.current.tiles[i2], 'repeat');
            if (pat2) {
              g2.globalAlpha = alpha;
              g2.fillStyle = pat2;
              g2.fillRect(0, 0, mn.width, mn.height);
              g2.globalAlpha = 1;
            }
          }
        }
      }

      frameRef.current++;

    };
    rafRef.current = requestAnimationFrame(tick);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, []);

  /** One mixer, capture tap and ring for a single track. Summing happens in the
      audio graph rather than in JS, so sources routed to a track stay
      sample-aligned — independent processors would smear them by a block. */
  const makeTrackChain = useCallback((ctx: AudioContext, tapSink: GainNode, cap: number) => {
    const mix = ctx.createGain();
    const proc = ctx.createScriptProcessor(4096, 2, 2);
    const ring: RingBuf = {
      l: new Float32Array(cap), r: new Float32Array(cap),
      w: 0, filled: 0, sr: ctx.sampleRate,
    };
    proc.onaudioprocess = (e) => {
      // Frozen: drop the block on the floor. Leaving `w` and `filled` where
      // they are is the whole mechanism — the window the snapshot reads back
      // from the write head cannot move if the head does not.
      if (frozenRef.current) return;
      const Lc = e.inputBuffer.getChannelData(0);
      const Rc = e.inputBuffer.numberOfChannels > 1 ? e.inputBuffer.getChannelData(1) : Lc;
      for (let k = 0; k < Lc.length; k++) {
        ring.l[ring.w] = Lc[k];
        ring.r[ring.w] = Rc[k];
        ring.w = (ring.w + 1) % ring.l.length;
      }
      ring.filled = Math.min(ring.filled + Lc.length, ring.l.length);
    };
    // Master pre-FX sits here: after the patch sum, before the ring, so it
    // changes what the model conditions on and leaves the monitor path — which
    // is tapped per source, further upstream — hearing the dry feed.
    const st = masterFxRef.current;
    const mfx = {
      scramble: new FxChain(ctx, 'scramble', masterFxAmt(st.scramble)),
      smear: new FxChain(ctx, 'smear', masterFxAmt(st.smear)),
    };
    mfx.scramble.setTempo(bpmRef.current);
    mix.connect(mfx.scramble.input);
    mfx.scramble.output.connect(mfx.smear.input);
    mfx.smear.output.connect(proc);
    proc.connect(tapSink);
    return { mix, ring, proc, mfx };
  }, [masterFxAmt]);

  // ─────────────────────────────────────────────────────────────────────
  // Transport
  // ─────────────────────────────────────────────────────────────────────
  const pushSnapshot = useCallback(() => {
    const rings = ringsRef.current;
    if (!rings.length) return;

    // Each track has its own mix, so each gets its own window. When two tracks
    // are patched identically the bytes are identical, but they are still sent
    // separately — the backend sessions are independent.
    // One stamp for the whole round: the instant this window of input ends.
    // The backend echoes it back with the audio it made, which is how each
    // track times its own snapshot-to-sound delay.
    const tag = performance.now();

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
        { type: 'audio_header', channels: 2, sample_rate: ring.sr, tag },
        inter.buffer,
      );
    });
  }, []);

  /**
   * Arm the generation clock. Every cycle from here counts off `delayMs`, so
   * Inject re-phases the whole loop simply by calling this again.
   */
  const scheduleCycles = useCallback((delayMs: number) => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(function fire() {
      if (!runningRef.current) return;
      pushSnapshot();
      timerRef.current = window.setTimeout(fire, paramsRef.current.generateRate * 1000);
    }, delayMs);
  }, [pushSnapshot]);

  // ── master actions ───────────────────────────────────────────────────
  // Global by design: these always mean every track, whatever MASTER is set
  // to, because that is what makes them usable as a panic and a cue.

  /** Silence every track at once, and have each session forget its own last
   *  output so Loopback cannot bring back what was just cut. */
  /**
   * Generate now from the last Input Rate of audio, the window ending at this
   * instant, and re-phase the cycle clock to the press. The ring is already
   * holding that audio, so nothing is waited for; what is playing keeps
   * playing until the new clip crossfades in.
   */
  const injectNow = useCallback(() => {
    if (!runningRef.current) return;
    pushSnapshot();
    scheduleCycles(paramsRef.current.generateRate * 1000);
  }, [pushSnapshot, scheduleCycles]);

  /**
   * Line the tracks up on the slowest one. Each track measures its own round
   * trip — queue wait, engine lock, generation, transfer, decode — and holding
   * every faster track back by the difference puts one input moment under the
   * same wall-clock instant in all of them. Nudge sits before each track's
   * analyser, so the spectrograms line up as well as the audio.
   *
   * The alignment is only as stable as those timings: the global engine lock
   * can hand the tracks a different order on the next cycle, so this is a
   * button to re-press, not a setting. It acts on every track at once whatever
   * MASTER says — a half-aligned rack is worse than an unaligned one.
   */
  const applySync = useCallback((lats: (number | null)[]) => {
    const known = lats.filter((v): v is number => v !== null);
    if (known.length < 2) return;
    const slowest = Math.max(...known);
    setTracks(ts => ts.map((t, i) => {
      const own = lats[i];
      if (own == null) return t;
      const n = Math.max(0, Math.min(MAX_NUDGE, Number((slowest - own).toFixed(2))));
      // Deadband: the medians wobble by a few ms and rewriting nudge on every
      // tick would keep the delay line gliding for no audible reason.
      return Math.abs(n - t.nudge) < SYNC_DEADBAND ? t : { ...t, nudge: n };
    }));
  }, []);

  /** Latched: SYNC is upkeep, not a one-shot, so it holds the alignment as the
   *  timings move. Also the only window onto whether any round trip has been
   *  measured yet — without it a press on cold tracks looks like a dead button. */
  const [syncOn, setSyncOn] = useState(false);
  const [latencies, setLatencies] = useState<(number | null)[]>([]);

  useEffect(() => {
    if (!running) { setSyncOn(false); setLatencies([]); return; }
    const id = setInterval(() => {
      const lats = trackRefs.current.map(t => t?.latency() ?? null);
      setLatencies(prev => (prev.length === lats.length
        && prev.every((p, i) => p === lats[i]) ? prev : lats));
      if (syncOn) applySync(lats);
    }, SYNC_POLL_MS);
    return () => clearInterval(id);
  }, [running, syncOn, applySync]);

  // MIDI reaches these through a ref: applyMidi is defined further up, and a
  // dependency array is evaluated during render, before these exist.
  const actionsRef = useRef({ inject: injectNow });
  useEffect(() => { actionsRef.current = { inject: injectNow }; }, [injectNow]);

  const [isStarting, setIsStarting] = useState(false);

  // Bulk delay the master FX adds to the feed. Pure functions of the amount,
  // so this needs no round trip through the audio nodes.
  const mfxLatency =
    (masterFx.scramble ? fxMethod('scramble').latencyAt(MASTER_FX_AMOUNT) : 0)
    + (masterFx.smear ? fxMethod('smear').latencyAt(MASTER_FX_AMOUNT) : 0);

  // What SYNC has to work with: how many tracks have been timed, and what the
  // active one measured. Shown under the button so a greyed-out SYNC can be
  // told apart from a broken one.
  const syncReady = latencies.filter(v => v !== null).length;
  const activeLat = latencies[activeTab] ?? null;
  const syncLabel = activeLat != null ? `${activeLat.toFixed(2)}s rt`
    : syncReady ? 'no data' : 'waiting';

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
    if (sources.some(x => x.kind === 'code')) void ensureStrudel();
  }, [sources, ensureStrudel]);

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
    if (!sources.some(x => x.kind === 'code') || !bpm) return;
    const h = strudelRef.current;
    if (!h) return;
    const cur = h.getBpm();
    if (cur == null || Math.abs(cur - bpm) > 0.05) h.setBpm(bpm);
  }, [bpm, sources]);

  // ── looper: one per Live Input box ───────────────────────────────────
  const audioOf = (id: string) => {
    let a = loopAudioRef.current[id];
    if (!a) a = loopAudioRef.current[id] = { buf: null, peaks: null, src: null, startedAt: 0 };
    return a;
  };

  const startRec = useCallback((srcId: string) => {
    recRef.current[srcId] = { chunks: [], frames: 0 };
    patchSlot(srcId, { state: 'recording', recSecs: 0 });
  }, [patchSlot]);

  const stopRec = useCallback((srcId: string) => {
    const rec = recRef.current[srcId];
    delete recRef.current[srcId];
    const ctx = inCtxRef.current;
    if (!rec || !ctx || rec.frames === 0) { patchSlot(srcId, { state: 'idle' }); return; }

    const buf = ctx.createBuffer(2, rec.frames, ctx.sampleRate);
    const L = buf.getChannelData(0), R = buf.getChannelData(1);
    let off = 0;
    for (const [l, r] of rec.chunks) { L.set(l, off); R.set(r, off); off += l.length; }

    const a = audioOf(srcId);
    a.buf = buf;
    a.peaks = peaksOf(buf);
    patchSlot(srcId, { state: 'stopped', len: buf.duration, trim: [0, buf.duration] });
  }, [patchSlot]);

  const playLoop = useCallback((srcId: string) => {
    const ctx = inCtxRef.current;
    const a = loopAudioRef.current[srcId];
    const g = srcNodesRef.current[srcId]?.loopGain;
    const slot = loopsRef.current[srcId] ?? IDLE_SLOT;
    if (!ctx || !a?.buf || !g) return;
    try { a.src?.stop(); } catch { /* not started */ }
    const [t0, t1] = slot.trim;
    const src = ctx.createBufferSource();
    src.buffer = a.buf;
    src.loop = true;
    src.loopStart = t0;
    src.loopEnd = Math.max(t0 + 0.05, t1);
    src.connect(g);
    src.start(0, t0);
    a.src = src;
    a.startedAt = ctx.currentTime;
    g.gain.setTargetAtTime(1, ctx.currentTime, 0.01);
    // The loop replaces the hardware input as this box's source.
    srcNodesRef.current[srcId]?.liveGain?.gain.setTargetAtTime(0, ctx.currentTime, 0.01);
    patchSlot(srcId, { state: 'playing' });
  }, [patchSlot]);

  const stopLoop = useCallback((srcId: string) => {
    const ctx = inCtxRef.current;
    const a = loopAudioRef.current[srcId];
    try { a?.src?.stop(); } catch { /* not started */ }
    if (a) a.src = null;
    if (ctx) {
      const n = srcNodesRef.current[srcId];
      n?.loopGain?.gain.setTargetAtTime(0, ctx.currentTime, 0.01);
      n?.liveGain?.gain.setTargetAtTime(1, ctx.currentTime, 0.01);
    }
    patchSlot(srcId, { state: a?.buf ? 'stopped' : 'idle' });
  }, [patchSlot]);

  const deleteLoop = useCallback((srcId: string) => {
    stopLoop(srcId);
    delete loopAudioRef.current[srcId];
    delete recRef.current[srcId];
    setLoops(m => ({ ...m, [srcId]: IDLE_SLOT }));
  }, [stopLoop]);

  // Re-arm the loop points live while the trim handles are dragged.
  useEffect(() => {
    for (const [id, slot] of Object.entries(loops)) {
      const src = loopAudioRef.current[id]?.src;
      if (src && slot.state === 'playing') {
        src.loopStart = slot.trim[0];
        src.loopEnd = Math.max(slot.trim[0] + 0.05, slot.trim[1]);
      }
    }
  }, [loops]);

  // ── file loop transport ──────────────────────────────────────────────
  /** A file box is a loop, not a one-shot: playback always wraps. A buffer
   *  source cannot be restarted, so each PLAY builds a fresh one. */
  /** The segment of a file box that loops: its handles, clamped to the buffer. */
  const fileSpan = (srcId: string, dur: number): [number, number] => {
    const inst = sourcesRef.current.find(x => x.id === srcId);
    const [a, b] = inst?.fileTrim ?? [0, dur];
    const t0 = Math.max(0, Math.min(a, dur - 0.05));
    return [t0, Math.min(dur, Math.max(t0 + 0.05, b))];
  };

  const playFile = useCallback((srcId: string, offset = 0) => {
    const ctx = inCtxRef.current;
    const n = srcNodesRef.current[srcId];
    if (!ctx || !n?.fileBuf || !n.fileGain) return;
    try { n.fileSrc?.stop(); } catch { /* not started */ }
    const dur = n.fileBuf.duration;
    const [t0, t1] = fileSpan(srcId, dur);
    const src = ctx.createBufferSource();
    src.buffer = n.fileBuf;
    // Loop between the handles rather than over the whole file.
    src.loop = true;
    src.loopStart = t0;
    src.loopEnd = t1;
    src.connect(n.fileGain);
    const into = offset % (t1 - t0);
    src.start(0, t0 + into);
    n.fileSrc = src;
    // Measured from the loop's start, so the playhead is t0 + elapsed % span.
    n.fileStartedAt = ctx.currentTime - into;
    setFilePlaying(m => ({ ...m, [srcId]: true }));
  }, []);

  const stopFile = useCallback((srcId: string) => {
    const n = srcNodesRef.current[srcId];
    try { n?.fileSrc?.stop(); } catch { /* not started */ }
    if (n) n.fileSrc = null;
    setFilePlaying(m => ({ ...m, [srcId]: false }));
  }, []);

  // Dragging a File Loop's handles re-aims the loop without restarting it, the
  // same way the recorded take's handles do.
  const fileTrimKey = sources.map(x => `${x.id}:${x.fileTrim?.join() ?? ''}`).join('|');
  useEffect(() => {
    for (const inst of sourcesRef.current) {
      const n = srcNodesRef.current[inst.id];
      if (inst.kind !== 'file' || !n?.fileSrc || !n.fileBuf) continue;
      const [t0, t1] = fileSpan(inst.id, n.fileBuf.duration);
      n.fileSrc.loopStart = t0;
      n.fileSrc.loopEnd = t1;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fileTrimKey]);


  /** Load an audio file to stand in for the live feed while testing. */
  const filePickTargetRef = useRef<string | null>(null);

  const pickFile = async (file: File) => {
    setError(null);
    try {
      // Decode once up front purely to validate it and read its duration; the
      // playback buffer is decoded again at start() against the input context.
      const probe = new AudioContext();
      const buf = await probe.decodeAudioData(await file.arrayBuffer());
      await probe.close();
      const target = filePickTargetRef.current;
      if (target) {
        filesRef.current[target] = file;
        // Peaks now, from the probe: the box shows its waveform as soon as the
        // file is chosen, without waiting for the transport.
        filePeaksRef.current[target] = peaksOf(buf);
        const n = srcNodesRef.current[target];
        if (n) { n.fileBuf = undefined; n.fileSrc = null; }
        patchSource(target, {
          fileName: file.name,
          fileDuration: buf.duration,
          fileTrim: [0, buf.duration],   // a new file starts untrimmed
        });
      }
      // A file can't feed back into the mic, so make it audible by default —
      // unlike the hardware input, which stays muted until you ask for it.
      setInputMonitor(m => (m === 0 ? 0.7 : m));
    } catch (e: any) {
      setError(`Could not decode ${file.name}: ${e.message || e}`);
    }
  };
  /** Open the interface once; every mic box picks its channels out of it. */
  const ensureMicCapture = async (ctx: AudioContext) => {
    if (micSplitRef.current) return micSplitRef.current;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        echoCancellation: false, noiseSuppression: false, autoGainControl: false,
        // Ask for every channel a box can address; a 2-in device just reports 2.
        channelCount: { ideal: MAX_INPUT_CHANNELS },
      },
    });
    streamRef.current = stream;
    const raw = ctx.createMediaStreamSource(stream);
    // Track settings often omit channelCount entirely, and the old `?? 2`
    // fallback then hid channels 3 and 4 on a four-in interface. The source
    // node knows how wide the stream really is, so take whichever says more.
    const reported = stream.getAudioTracks()[0]?.getSettings().channelCount ?? 0;
    setDetectedChannels(Math.max(1, reported, raw.channelCount || 0));
    // The splitter and the clamp use the addressable width, not the detected
    // one: clamping to a mis-detected 2 would quietly play channel 2 when
    // channel 3 was asked for, which is worse than silence.
    micChanRef.current = MAX_INPUT_CHANNELS;
    const splitter = ctx.createChannelSplitter(MAX_INPUT_CHANNELS);
    raw.connect(splitter);
    micSplitRef.current = splitter;
    return splitter;
  };

  /** Build one input box: its head node, fader and display analyser. */
  const buildSource = async (inst: SourceInstance, ctx: AudioContext): Promise<SourceNodes | null> => {
    let head: AudioNode | null = null;
    let extra: Partial<SourceNodes> = {};

    let micMerger: ChannelMergerNode | undefined;
    if (inst.kind === 'mic') {
      const splitter = await ensureMicCapture(ctx);
      const nCh = micChanRef.current;
      // Per-box channel pick, so two Live Inputs can carry different channels.
      const merger = ctx.createChannelMerger(2);
      micMerger = merger;
      const [selA, selB] = parseChannelSel(inst.chanSel, nCh);
      splitter.connect(merger, selA, 0);
      splitter.connect(merger, selB, 1);

      // The live signal and this box's recorded loop share its fader.
      const playing = (loopsRef.current[inst.id] ?? IDLE_SLOT).state === 'playing';
      const liveGain = ctx.createGain();
      liveGain.gain.value = playing ? 0 : 1;
      merger.connect(liveGain);

      const loopGain = ctx.createGain();
      loopGain.gain.value = playing ? 1 : 0;

      const bus = ctx.createGain();
      liveGain.connect(bus);
      loopGain.connect(bus);
      head = bus;
      extra = { liveGain, loopGain };

      // The looper records this box alone, not a track mix that may also carry
      // a file or a pattern, so it needs its own tap.
      const recProc = ctx.createScriptProcessor(4096, 2, 2);
      recProc.onaudioprocess = (e) => {
        const rec = recRef.current[inst.id];
        if (!rec) return;
        const Lc = e.inputBuffer.getChannelData(0);
        const Rc = e.inputBuffer.numberOfChannels > 1 ? e.inputBuffer.getChannelData(1) : Lc;
        if (rec.frames < MAX_LOOP_SECS * ctx.sampleRate) {
          rec.chunks.push([new Float32Array(Lc), new Float32Array(Rc)]);
          rec.frames += Lc.length;
        }
      };
      merger.connect(recProc);
      // A ScriptProcessor only fires while it has a path to the destination.
      if (tapSinkRef.current) recProc.connect(tapSinkRef.current);
      extra.recProc = recProc;
    }

    if (inst.kind === 'file') {
      const f = filesRef.current[inst.id];
      if (!f) return null;
      const buf = await ctx.decodeAudioData(await f.arrayBuffer());
      // A fixed head, so PLAY and STOP can swap buffer sources underneath it
      // without rewiring the box into the tracks again.
      const fileGain = ctx.createGain();
      head = fileGain;
      extra = { fileGain, fileBuf: buf, fileSrc: null, fileStartedAt: 0 };
      if (!filePeaksRef.current[inst.id]) filePeaksRef.current[inst.id] = peaksOf(buf);
    }

    if (inst.kind === 'code') head = strudelRef.current!.tap;
    if (!head) return null;

    const gain = ctx.createGain();
    gain.gain.value = inst.level;
    head.connect(gain);

    // Pre-FX sits after the fader and before everything downstream, so it is
    // in the model's path as well as the monitor's — the point of an input
    // effect is that the model conditions on the processed audio.
    const fx = new FxChain(ctx, inst.fxMethod, inst.fxAmount);
    fx.setTempo(bpmRef.current);
    gain.connect(fx.input);

    const fxIn = ctx.createAnalyser();
    fxIn.fftSize = 2048;
    gain.connect(fxIn);
    const fxOut = ctx.createAnalyser();
    fxOut.fftSize = 2048;
    fx.output.connect(fxOut);

    const mon = ctx.createGain();
    mon.gain.value = monitorModeRef.current === 'box' ? inst.monitor : 1;

    // Display tap is PRE-fader, so a box stays visible at level 0.
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    head.connect(analyser);

    return { head, gain, out: fx.output, fx, fxIn, fxOut, mon, analyser, merger: micMerger, ...extra };
  };


  const start = async () => {
    setError(null);
    const active = sources;
    if (!active.length) { setError('Add at least one input box.'); return; }
    const missing = active.find(x => x.kind === 'file' && !filesRef.current[x.id]);
    if (missing) { setError('Load an audio file into the File Loop box, or remove it.'); return; }
    setIsStarting(true);
    try {
      // Adopt Strudel's context when Live Code is in play, so pattern playback
      // and capture share one clock; otherwise make our own. Either way we run
      // at the device's native rate and let the backend resample to 44.1 kHz.
      let ctx: AudioContext;
      if (active.some(x => x.kind === 'code')) {
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

      const mixes: GainNode[] = [];
      const rings: RingBuf[] = [];
      const procs: ScriptProcessorNode[] = [];
      const mfxs: { scramble: FxChain; smear: FxChain }[] = [];
      for (let i = 0; i < tracks.length; i++) {
        const c = makeTrackChain(ctx, tapSink, cap);
        mixes.push(c.mix); rings.push(c.ring); procs.push(c.proc); mfxs.push(c.mfx);
      }
      trackMixRef.current = mixes;
      ringsRef.current = rings;
      procsRef.current = procs;
      mfxRef.current = mfxs;

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

      const built: Record<string, SourceNodes> = {};
      for (const inst of active) {
        const n = await buildSource(inst, ctx);
        if (!n) continue;
        built[inst.id] = n;
        // Through the box's own monitor gain, so per-box levels need no rewire.
        if (n.mon) { n.out.connect(n.mon); n.mon.connect(monitorGain); }
        else n.out.connect(monitorGain);
        // Patch into whichever tracks this box is routed to.
        inst.to.forEach((on, ti) => {
          if (on && mixes[ti]) n.out.connect(mixes[ti]);
        });
      }
      srcNodesRef.current = built;

      // Summed input for the show view. monitorGain would be the obvious tap,
      // but its gain is the monitor volume — usually 0 — so an analyser after it
      // reads silence. This is a unity sum of every box's post-FX output, and it
      // ends at the silent sink so the graph has a reason to render it.
      const showSum = ctx.createGain();
      const showAn = ctx.createAnalyser();
      showAn.fftSize = 2048;
      for (const n of Object.values(built)) n.out.connect(showSum);
      showSum.connect(showAn);
      showAn.connect(tapSink);
      showSumRef.current = showSum;
      inAnalyserRef.current = showAn;

      // A file is already in memory, so a track fed ONLY by the file need not
      // sit through a real InputRate of silence: pre-fill its ring and resume
      // playback where that window ends. With other sources mixed in the seed
      // would be wrong, so it is skipped.
      let seeded = false;
      let seedOffset = 0;
      const fileBoxes = active.filter(x => x.kind === 'file' && built[x.id]);
      // Only with a single file box is it clear what the window should hold;
      // two files mixed into one track would seed it with the wrong audio.
      const fbox = fileBoxes.length === 1 ? fileBoxes[0] : undefined;
      const fbuf = fbox ? built[fbox.id].fileBuf : undefined;
      if (fbox && fbuf) {
        // Seed from the loop's start, not the file's: the trimmed segment is
        // what is about to play.
        const [ft0, ft1] = fileSpan(fbox.id, fbuf.duration);
        const from = Math.floor(ft0 * ctx.sampleRate);
        const span = Math.floor((ft1 - ft0) * ctx.sampleRate);
        const seedLen = Math.min(Math.floor(effInputRate * ctx.sampleRate), span);
        const Lc = fbuf.getChannelData(0);
        const Rc = fbuf.numberOfChannels > 1 ? fbuf.getChannelData(1) : Lc;
        rings.forEach((ring, ti) => {
          const only = active.length === 1 ||
            active.every(x => x.kind === 'file' || !x.to[ti]);
          if (!only) return;
          ring.l.set(Lc.subarray(from, from + seedLen), 0);
          ring.r.set(Rc.subarray(from, from + seedLen), 0);
          ring.w = seedLen % cap;
          ring.filled = seedLen;
          seeded = true;
        });
        if (seeded) seedOffset = seedLen / ctx.sampleRate;
      }
      // Every file box starts looping; the seeded one resumes where its
      // pre-filled window ended so the audio joins on rather than repeating.
      fileBoxes.forEach((x, i) => playFile(x.id, i === 0 ? seedOffset : 0));


      // Autoplay policy can leave a context suspended when it is created either
      // side of an await; without this nothing fires and nothing is heard.
      if (ctx.state === 'suspended') await ctx.resume();

      for (const t of trackRefs.current) t?.begin();

      setRunning(true);
      setIsStarting(false);

      // Live Code shares the transport. Evaluating here rather than whenever
      // RUN happened to be clicked is the whole point: the pattern's cycle 0
      // and the generation clock start on the same instant, so an Input Rate
      // of 32 beats really is 8 bars of what you are hearing.
      //
      // It has to come after the graph is wired — the pattern's output lands on
      // Strudel's tap, which only reaches a track once buildSource has run.
      let firstRate = effInputRate;
      if (active.some(x => x.kind === 'code')) {
        // An error in the buffer is the panel's to report; it must not take
        // the transport down with it.
        await strudelPanelRef.current?.run().catch(() => {});
        // A `setcpm` in the pattern may have just redefined the beat. React
        // state has not caught up yet, so ask the engine directly rather than
        // sizing the first window at the previous tempo.
        const b = strudelRef.current?.getBpm();
        if (b && isFinite(b) && b > 0) {
          firstRate = Math.min(MAX_BUFFER_SECS, inputBeats * (60 / b));
        }
      }

      setStatus(seeded ? 'Generating first cycle…'
                       : `Filling ${firstRate.toFixed(1)}s buffer…`);
      scheduleCycles(seeded ? 120 : firstRate * 1000);
    } catch (e: any) {
      setIsStarting(false);
      setError(e.message || 'Could not open the audio input');
    }
  };

  const stop = () => {
    setRunning(false);
    // START builds fresh rings with nothing in them, so a freeze carried across
    // a stop would leave the model with no input and no way to tell why. Going
    // through setFreeze also hands back the Generate Rate it borrowed.
    setFreeze(false);
    // ...and STOP presses HUSH, the other half of the same deal.
    if (sourcesRef.current.some(x => x.kind === 'code')) hushStrudel();
    setStatus('Idle');
    if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null; }
    for (const t of trackRefs.current) t?.finish();
    for (const pr of procsRef.current) pr.disconnect();
    procsRef.current = [];
    for (const m of mfxRef.current) { m.scramble.dispose(); m.smear.dispose(); }
    mfxRef.current = [];
    for (const m of trackMixRef.current) m.disconnect();
    trackMixRef.current = [];
    showSumRef.current?.disconnect();
    showSumRef.current = null;
    inAnalyserRef.current?.disconnect();
    tapSinkRef.current?.disconnect();
    tapSinkRef.current = null;
    for (const n of Object.values(srcNodesRef.current)) {
      n?.gain.disconnect();
      n?.fx.dispose();
      n?.analyser.disconnect();
      n?.recProc?.disconnect();
      if (n?.fileSrc) {
        try { n.fileSrc.stop(); } catch { /* never started */ }
        n.fileSrc.disconnect();
        n.fileSrc = null;
      }
    }
    setFilePlaying({});
    for (const a of Object.values(loopAudioRef.current)) {
      try { a.src?.stop(); } catch { /* not started */ }
      a.src = null;
    }
    // Takes survive a stop; only playback does. A box whose loop was sounding
    // drops back to 'stopped', and a take still being recorded is discarded.
    setLoops(m => Object.fromEntries(Object.entries(m).map(([id, sl]) =>
      [id, { ...sl, state: loopAudioRef.current[id]?.buf ? 'stopped' as const : 'idle' as const }])));
    recRef.current = {};
    srcNodesRef.current = {};
    if (clickTimerRef.current) { clearInterval(clickTimerRef.current); clickTimerRef.current = null; }
    clickGainRef.current?.disconnect();
    clickGainRef.current = null;
    micSplitRef.current?.disconnect();
    micSplitRef.current = null;
    splitterRef.current?.disconnect();
    mergerRef.current?.disconnect();
    splitterRef.current = null;
    mergerRef.current = null;
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    // Strudel owns its context; closing it would kill the engine.
    if (!sourcesRef.current.some(x => x.kind === 'code')) inCtxRef.current?.close().catch(() => {});
    inCtxRef.current = null;
    inAnalyserRef.current = null;
  };

  useEffect(() => () => { stop(); }, []);   // eslint-disable-line react-hooks/exhaustive-deps

  /**
   * What a box's trim handles act on, or null when it has nothing to trim yet.
   * A Live Input trims the take it recorded; a File Loop trims the file it
   * imported. Both loop between the handles, so the gesture is the same.
   */
  const trimTarget = (inst: SourceInstance): { len: number; trim: [number, number] } | null => {
    if (inst.kind === 'mic') {
      const sl = slotOf(inst.id);
      const showing = sl.state === 'stopped' || sl.state === 'playing';
      return showing && sl.len ? { len: sl.len, trim: sl.trim } : null;
    }
    if (inst.kind === 'file') {
      const len = inst.fileDuration ?? 0;
      return len > 0 ? { len, trim: inst.fileTrim ?? [0, len] } : null;
    }
    return null;
  };
  const posToTime = (len: number, e: React.PointerEvent) => {
    const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
    return Math.max(0, Math.min(len, ((e.clientX - r.left) / r.width) * len));
  };
  const applyTrim = (inst: SourceInstance, at: number) => {
    const d = dragHandleRef.current;
    const t = trimTarget(inst);
    if (!d || d.id !== inst.id || !t) return;
    const [a, b] = t.trim;
    // Never let the handles cross; a loop shorter than 50 ms is a click.
    const trim: [number, number] = d.h === 0
      ? [Math.min(at, b - 0.05), b]
      : [a, Math.max(at, a + 0.05)];
    if (inst.kind === 'file') patchSource(inst.id, { fileTrim: trim });
    else setLoops(m => ({ ...m, [inst.id]: { ...(m[inst.id] ?? IDLE_SLOT), trim } }));
  };

  // ── show view ────────────────────────────────────────────────────────
  /** Fill `into` with the summed live input, as -1..1, at whatever resolution
   *  the caller asked for. */
  const readInputWave = useCallback((into: Float32Array) => {
    const an = inAnalyserRef.current;
    if (!an) return false;
    if (showBytesRef.current.length !== an.fftSize) {
      showBytesRef.current = new Uint8Array(new ArrayBuffer(an.fftSize));
    }
    const buf = showBytesRef.current;
    an.getByteTimeDomainData(buf);
    const n = into.length, m = buf.length;
    for (let i = 0; i < n; i++) into[i] = buf[((i * m) / n) | 0] / 128 - 1;
    return true;
  }, []);

  /** Fill `into` with every track's output as it is actually being heard, each
   *  already scaled by its own monitor level. Each track has its own
   *  AudioContext, so this cannot be summed in the audio graph — it is added
   *  here instead, which is visually right but not phase-accurate. */
  const readOutputWave = useCallback((into: Float32Array) => {
    if (showTmpRef.current.length !== into.length) showTmpRef.current = new Float32Array(into.length);
    const tmp = showTmpRef.current;
    into.fill(0);
    let n = 0;
    for (const t of trackRefs.current) {
      if (!t?.readWave(tmp)) continue;
      for (let i = 0; i < into.length; i++) into[i] += tmp[i];
      n++;
    }
    if (!n) return false;
    // Summed rather than averaged, so one track at full monitor draws the same
    // trace whether or not a second track is up. Clamped so several loud tracks
    // cannot draw a trace straight through the captions.
    if (n > 1) for (let i = 0; i < into.length; i++) into[i] = Math.max(-1, Math.min(1, into[i]));
    return true;
  }, []);

  /**
   * The prompts the audience is actually hearing, for the show view's caption.
   * A track contributes nothing if its prompt is empty, if its monitor is all
   * the way down, or if another track's solo has silenced it — in each case
   * there is no sound to attribute to it.
   */
  const audiblePrompts = tracks
    .filter((t, i) => t.prompt.trim() && t.outputMonitor > 0 && (solo === null || solo === i))
    .map(t => t.prompt.trim());

  /** Leaving the show view returns to whichever working layout you came from. */
  const lastFlatRef = useRef<Layout>('vertical');
  useEffect(() => { if (layout !== 'show') lastFlatRef.current = layout; }, [layout]);
  const leaveShow = useCallback(() => setLayout(lastFlatRef.current), []);

  // ── the draggable divider between the two halves ─────────────────────
  const stageRef = useRef<HTMLDivElement | null>(null);
  const splitDragRef = useRef(false);
  const [splitting, setSplitting] = useState(false);
  const layoutRef = useRef(layout);
  useEffect(() => { layoutRef.current = layout; }, [layout]);

  const onSplitDown = useCallback((e: React.PointerEvent) => {
    // No preventDefault: it would swallow the double-click that resets the
    // split. `user-select: none` on the handle deals with text dragging.
    (e.target as Element).setPointerCapture(e.pointerId);
    splitDragRef.current = true;
    setSplitting(true);
  }, []);

  const onSplitMove = useCallback((e: React.PointerEvent) => {
    if (!splitDragRef.current) return;
    const r = stageRef.current?.getBoundingClientRect();
    if (!r) return;
    // Whichever axis the stage runs along is the one the divider slides on.
    const f = layoutRef.current === 'vertical'
      ? (e.clientY - r.top) / r.height
      : (e.clientX - r.left) / r.width;
    // Neither side may be squeezed to nothing: a pane below this still has
    // room for its title and one box.
    const clamped = Math.max(0.18, Math.min(0.82, f));
    setSplit(sp => ({ ...sp, [splitKey(layoutRef.current)]: clamped }));
  }, []);

  const onSplitUp = useCallback((e: React.PointerEvent) => {
    if (!splitDragRef.current) return;
    splitDragRef.current = false;
    setSplitting(false);
    try { (e.target as Element).releasePointerCapture(e.pointerId); } catch { /* done */ }
    setSplit(sp => {
      localStorage.setItem('sa3_live_split', JSON.stringify(sp));
      return sp;
    });
  }, []);

  /** Double-click the divider to go back to even halves. */
  const resetSplit = useCallback(() => {
    setSplit(sp => {
      const next = { ...sp, [splitKey(layoutRef.current)]: 0.5 };
      localStorage.setItem('sa3_live_split', JSON.stringify(next));
      return next;
    });
  }, []);

  /** The latent shelf edits whichever track's tab is showing. */
  const sendWeights = useCallback((weights: Float32Array, renorm: boolean) => {
    setShelfBusy(true);
    trackRefs.current[activeTabRef.current]?.sendWeights(weights, renorm);
  }, []);

  // Serialised routing, so the rewire below fires on patch changes only —
  // keying it on `sources` would disconnect and reconnect on every fader move.
  const routingKey = sources
    .map(x => `${x.id}:${x.to.slice(0, tracks.length).map(b => (b ? 1 : 0)).join('')}`)
    .join('|');
  useEffect(() => {
    const mixes = trackMixRef.current;
    if (!mixes.length) return;
    for (const inst of sourcesRef.current) {
      const n = srcNodesRef.current[inst.id];
      if (!n) continue;
      mixes.forEach((mix) => {
        try { n.out.disconnect(mix); } catch { /* was not connected */ }
      });
      inst.to.forEach((on, ti) => {
        if (on && mixes[ti]) n.out.connect(mixes[ti]);
      });
    }
  }, [routingKey]);

  // --- patch cables ----------------------------------------------------
  const [anchors, setAnchors] = useState<{
    src: Record<string, { x: number; y: number }>;
    trk: ({ x: number; y: number } | undefined)[];
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
    const src: Record<string, { x: number; y: number }> = {};
    for (const inst of sourcesRef.current) {
      const pt = at(srcNodeEls.current[inst.id]);
      if (pt) src[inst.id] = pt;
    }
    // Walk the current track list rather than the element map: a removed
    // track leaves no anchor at all, so its cables cannot be redrawn.
    setAnchors({
      src,
      trk: tracksRef.current.map(t => at(trackNodeEls.current[t.id])),
    });
  }, []);

  // Anchors move whenever the layout does.
  useEffect(() => {
    measure();
    const t = window.setTimeout(measure, 60);   // after the panes settle
    window.addEventListener('resize', measure);
    return () => { window.clearTimeout(t); window.removeEventListener('resize', measure); };
  }, [measure, sources, tracks.length, shelfOpen, loopLensKey, layout, split]);
  useEffect(() => { measureRef.current = measure; }, [measure]);

  const beginPatchFromTrack = useCallback((ti: number, e: React.PointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
    measure();
    setPatchFromTrack(ti);
    const base = patchLayerRef.current?.getBoundingClientRect();
    if (base) setPatchTo({ x: e.clientX - base.left, y: e.clientY - base.top });
  }, [measure]);

  const beginPatch = useCallback((id: string, e: React.PointerEvent) => {
    e.preventDefault();
    measure();
    setPatchFrom(id);
    const layer = patchLayerRef.current;
    const base = layer?.getBoundingClientRect();
    if (base) setPatchTo({ x: e.clientX - base.left, y: e.clientY - base.top });
  }, [measure]);

  /** Flip one source->track connection on or off. Both drag directions land
      here, so a cable can be pulled apart from whichever end you grab. */
  const toggleLink = useCallback((srcId: string, ti: number) => {
    setSources(prev => prev.map(x => {
      if (x.id !== srcId) return x;
      const to = [...x.to];
      while (to.length <= ti) to.push(false);
      to[ti] = !to[ti];
      return { ...x, to };
    }));
  }, []);

  // A drop does not have to be precise: the pad widens every jack.
  const hits = (el: HTMLElement | null | undefined, e: PointerEvent) => {
    if (!el) return false;
    const r = el.getBoundingClientRect();
    const pad = 26;
    return e.clientX >= r.left - pad && e.clientX <= r.right + pad
        && e.clientY >= r.top - pad && e.clientY <= r.bottom + pad;
  };

  // While dragging, the loose end follows the pointer; releasing over the
  // opposite kind of jack toggles that connection, anywhere else cancels.
  useEffect(() => {
    if (patchFrom === null && patchFromTrack === null) return;
    const move = (e: PointerEvent) => {
      const base = patchLayerRef.current?.getBoundingClientRect();
      if (base) setPatchTo({ x: e.clientX - base.left, y: e.clientY - base.top });
    };
    const up = (e: PointerEvent) => {
      if (patchFrom !== null) {
        // Started at a source: look for a track jack under the pointer.
        const ti = tracksRef.current.findIndex(t => hits(trackNodeEls.current[t.id], e));
        if (ti >= 0) toggleLink(patchFrom, ti);
      } else if (patchFromTrack !== null) {
        // Started at a track: look for a source jack instead.
        const inst = sourcesRef.current.find(x => hits(srcNodeEls.current[x.id], e));
        if (inst) toggleLink(inst.id, patchFromTrack);
      }
      setPatchFrom(null);
      setPatchFromTrack(null);
      setPatchTo(null);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    return () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
  }, [patchFrom, patchFromTrack, toggleLink]);

  const addSource = useCallback((kind: SourceKind) => {
    setAddMenu(false);
    setSources(list => (list.length >= maxSlots ? list : [...list, newSource(kind, tracks.length)]));
  }, [maxSlots, tracks.length]);

  /** Removing a box must also silence it and drop its cables. */
  const removeSource = useCallback((id: string) => {
    const inst = sourcesRef.current.find(x => x.id === id);
    // A running Live Code box keeps sounding otherwise: Strudel's scheduler is
    // independent of our graph, so it has to be hushed explicitly.
    if (inst?.kind === 'code') {
      strudelRef.current?.hush();
      setStrudelPlaying(false);
    }
    const n = srcNodesRef.current[id];
    if (n) {
      // Dropping the connections removes this box's cables with it.
      try { n.gain.disconnect(); } catch { /* already detached */ }
      try { n.fx.dispose(); } catch { /* already detached */ }
      try { n.analyser.disconnect(); } catch { /* already detached */ }
      delete srcNodesRef.current[id];
    }
    delete filesRef.current[id];
    delete filePeaksRef.current[id];
    delete srcCanvasRefs.current[id];
    delete srcNodeEls.current[id];
    // The box's take goes with it — nothing else can reach it now.
    const a = loopAudioRef.current[id];
    try { a?.src?.stop(); } catch { /* not started */ }
    delete loopAudioRef.current[id];
    delete recRef.current[id];
    setLoops(m => { const { [id]: _gone, ...rest } = m; return rest; });
    setFilePlaying(m => { const { [id]: _g, ...rest } = m; return rest; });
    setSources(list => list.filter(x => x.id !== id));
  }, []);

  const addTrack = useCallback(() => {
    setTracks(ts => {
      if (ts.length >= maxSlots) return ts;
      // The capture graph is index-parallel to the track list, so it has to
      // grow with it — otherwise the new track would have no ring to read.
      const ctx = inCtxRef.current, tapSink = tapSinkRef.current;
      if (ctx && tapSink) {
        const c = makeTrackChain(ctx, tapSink, Math.ceil(MAX_BUFFER_SECS * ctx.sampleRate));
        trackMixRef.current = [...trackMixRef.current, c.mix];
        ringsRef.current = [...ringsRef.current, c.ring];
        procsRef.current = [...procsRef.current, c.proc];
        mfxRef.current = [...mfxRef.current, c.mfx];
      }
      // Every existing box patches to the new track by default.
      setSources(list => list.map(x => ({ ...x, to: [...x.to, true] })));
      setActiveTab(ts.length);
      return [...ts, defaultTrack()];
    });
  }, [maxSlots, makeTrackChain]);
  const removeTrack = useCallback((i: number) => {
    trackRefs.current[i]?.finish();
    trackRefs.current = trackRefs.current.filter((_, j) => j !== i);
    // Tear the departed track's capture chain out too: the arrays are indexed
    // in parallel, so leaving a stale entry would hand its mix to a survivor.
    const drop = <T,>(arr: T[]) => arr.filter((_, j) => j !== i);
    const dead = trackMixRef.current[i];
    if (dead) {
      // Sources feed the mix; disconnect() only sheds its outputs, so the
      // incoming edges have to be cut from the source side or the dead node
      // stays reachable.
      for (const inst of sourcesRef.current) {
        try { srcNodesRef.current[inst.id]?.out.disconnect(dead); } catch { /* not routed */ }
      }
      try { dead.disconnect(); } catch { /* never connected */ }
    }
    const oldProc = procsRef.current[i];
    if (oldProc) { oldProc.onaudioprocess = null; try { oldProc.disconnect(); } catch { /* idem */ } }
    trackMixRef.current = drop(trackMixRef.current);
    const deadFx = mfxRef.current[i];
    if (deadFx) { deadFx.scramble.dispose(); deadFx.smear.dispose(); }
    procsRef.current = drop(procsRef.current);
    mfxRef.current = drop(mfxRef.current);
    ringsRef.current = drop(ringsRef.current);
    setTracks(ts => {
      const gone = ts[i];
      if (gone) delete trackNodeEls.current[gone.id];
      return drop(ts);
    });
    setSources(list => list.map(x => ({ ...x, to: drop(x.to) })));
    setActiveTab(0);
    // Otherwise a solo on the departed track would silence the survivors.
    setSolo(null);
  }, []);

  const arm = (id: string) => setArmed(a => (a === id ? null : id));
  const knobProps = (id: string) => ({
    id, learning: midiLearn, armed: armed === id,
    bound: bindingLabel(id), onArm: arm,
  });

  // The middle bar's thickness is the diffusion step count; noise instead
  // shows up as the wash bleeding into the input pane.
  // Horizontal bar between the halves: its THICKNESS is the step count.
  const barThickness = 12 + (Math.min(cur.steps, 16) / 16) * 60;
  // Variant 0 sits at the top of the bar and the ladder descends from there.
  const variantPct = activeVariants > 1 ? (cur.variant / (activeVariants - 1)) * 100 : 0;

  /** One input box; rendered from either column. */
  // The Input FX controls edit this box. Falls back to the first one so the
  // rack is never pointing at nothing.
  const fxSrc = sources.find(x => x.id === activeSrc) ?? sources[0];
  const patchFx = useCallback((patch: Partial<SourceInstance>) => {
    const id = activeSrcRef.current ?? sourcesRef.current[0]?.id;
    if (id) patchSource(id, patch);
  }, []);

  const renderSource = (inst: SourceInstance, slot: number) => (
    <InputSection
      key={inst.id}
      src={inst}
      active={fxSrc?.id === inst.id}
      onActivate={() => setActiveSrc(inst.id)}
      connected={inst.to.slice(0, tracks.length).filter(Boolean).length}
      patching={patchFromTrack !== null}
      running={running}
      channelOptions={channelOptions(MAX_INPUT_CHANNELS)}
      sendLearn={knobProps(`send${slot}`)}
      onChanSel={(v) => patchSource(inst.id, { chanSel: v })}
      onLevel={(v) => patchSource(inst.id, { level: v })}
      onRemove={() => removeSource(inst.id)}
      onCanvas={(el) => { srcCanvasRefs.current[inst.id] = el; }}
      onNode={(el) => { srcNodeEls.current[inst.id] = el; }}
      onNodeDown={(e) => beginPatch(inst.id, e)}
      canvasProps={inst.kind === 'code' ? undefined : {
        style: { cursor: trimTarget(inst) ? 'ew-resize' : 'default' },
        onPointerDown: (e) => {
          const t = trimTarget(inst);
          if (!t) return;
          (e.target as Element).setPointerCapture(e.pointerId);
          const at = posToTime(t.len, e);
          // Grab whichever handle is nearer the click.
          dragHandleRef.current = {
            id: inst.id,
            h: Math.abs(at - t.trim[0]) <= Math.abs(at - t.trim[1]) ? 0 : 1,
          };
          applyTrim(inst, at);
        },
        onPointerMove: (e) => {
          const t = trimTarget(inst);
          if (t && dragHandleRef.current) applyTrim(inst, posToTime(t.len, e));
        },
        onPointerUp: (e) => {
          dragHandleRef.current = null;
          try { (e.target as Element).releasePointerCapture(e.pointerId); } catch { /* done */ }
        },
      }}
      monitorSlider={monitorMode === 'box' ? (
        <MidiSlider {...knobProps(`inputMonitor@${slot}`)} label="Mon"
                    value={inst.monitor}
                    onChange={(v) => {
                      patchSource(inst.id, { monitor: v });
                      srcNodesRef.current[inst.id]?.mon?.gain.setTargetAtTime(
                        v, inCtxRef.current?.currentTime ?? 0, 0.02);
                    }} />
      ) : undefined}
      fxBadge={inst.fxAmount > 0
        ? <>{fxMethod(inst.fxMethod).label}<b>{fxMethod(inst.fxMethod).format(inst.fxAmount)}</b></>
        : undefined}
      overlay={inst.kind === 'code' ? (
        <StrudelPanel
          ref={strudelPanelRef}
          linked
          onEvaluate={runStrudel}
          onHush={hushStrudel}
          running={strudelPlaying}
          ready={strudelReady}
          banks={strudelBanks}
          error={strudelError}
        />
      ) : undefined}
      tools={
        inst.kind === 'mic' ? (() => {
          // Every box reads its own slot, so a take made here never shows up
          // in the Live Input next to it.
          const sl = slotOf(inst.id);
          return (
            <div className="in-tools">
              {sl.state !== 'recording' ? (
                <button className="tool rec" onClick={() => startRec(inst.id)} disabled={!running}
                        title={running ? 'Record a loop' : 'Start the transport first'}>
                  <Circle size={10} /> REC
                </button>
              ) : (
                <button className="tool recording" onClick={() => stopRec(inst.id)}>
                  <Square size={10} /> STOP {sl.recSecs.toFixed(1)}s
                </button>
              )}
              {sl.state === 'playing'
                ? <button className="tool on" onClick={() => stopLoop(inst.id)}><Square size={10} /> STOP</button>
                : <button className="tool" onClick={() => playLoop(inst.id)}
                          disabled={!sl.len || !running}
                          title={running ? 'Loop this take' : 'Start the transport first'}>
                    <Play size={10} /> PLAY
                  </button>}
              <button className="tool del" onClick={() => deleteLoop(inst.id)} disabled={!sl.len}>
                <Trash2 size={10} /> DEL
              </button>
              {sl.len > 0 && (
                <span className="tools-meta">
                  loop {(sl.trim[1] - sl.trim[0]).toFixed(2)}s of {sl.len.toFixed(2)}s · drag edges to trim
                </span>
              )}
            </div>
          );
        })() : inst.kind === 'file' ? (
          <div className="in-tools">
            <button className="tool" disabled={running}
                    onClick={() => {
                      filePickTargetRef.current = inst.id;
                      filePickerRef.current?.click();
                    }}>
              <FolderOpen size={10} /> {inst.fileName ? 'Change…' : 'Import…'}
            </button>
            {/* The transport starts every file box looping; these are for
                dropping one out and back in without stopping the whole rig. */}
            {filePlaying[inst.id]
              ? <button className="tool on" onClick={() => stopFile(inst.id)}>
                  <Square size={10} /> STOP
                </button>
              : <button className="tool" onClick={() => playFile(inst.id)}
                        disabled={!running || !inst.fileName}
                        title={running ? 'Loop this file' : 'Start the transport first'}>
                  <Play size={10} /> PLAY
                </button>}
            {inst.fileName && (() => {
              const len = inst.fileDuration ?? 0;
              const [t0, t1] = inst.fileTrim ?? [0, len];
              const cropped = len - (t1 - t0) > 0.01;
              return (
                <span className="tools-meta" title={inst.fileName}>
                  {inst.fileName} · {cropped
                    ? `loop ${(t1 - t0).toFixed(2)}s of ${len.toFixed(2)}s`
                    : `${len.toFixed(2)}s`} · drag edges to trim
                </span>
              );
            })()}
          </div>
        ) : undefined
      }
    />
  );

  return (
    <div className={`live-root ${layout}`}>
      {/* ── transport ─────────────────────────────────────────── */}
      <header className="live-bar">
        <button className={`transport ${running ? 'stop' : 'start'}`}
                onClick={running ? stop : start}
                disabled={!connected || isStarting || (sources.some(x => x.kind === 'code') && !strudelReady)}>
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

        <div className="outmap">
          <button className={`midi-btn ${mapOpen ? 'on' : ''}`}
                  onClick={() => setMapOpen(v => !v)}
                  title="Send the master mix and each track to hardware output pairs">
            <Speaker size={15} /> OUTPUTS
          </button>
          {mapOpen && (
            <div className="outmap-panel">
              <div className="outmap-head">
                <span>OUTPUT MAPPING</span>
                <button className="outmap-x" onClick={() => setMapOpen(false)}
                        aria-label="Close">×</button>
              </div>

              {/* The device's answer, not our request. A 2 here means every
                  pair below is landing on 1/2 whatever it says. */}
              <div className={`outmap-note ${outChannels !== null && outChannels < 4 ? 'warn' : ''}`}>
                {outChannels === null
                  ? 'Press START to open the device and read its channel count'
                  : outChannels < 4
                    ? `Device gave ${outChannels} channels — every pair is landing on 1/2`
                    : `${outChannels} channels available · ${Math.floor(outChannels / 2)} pairs`}
              </div>

              <label className="outmap-row master">
                <span>Master mix</span>
                <select className="live-select" value={masterPair}
                        onChange={e => setMasterPair(Number(e.target.value))}
                        title="Where the whole rack goes, after the master fader">
                  {Array.from({ length: OUT_PAIRS }, (_, p) => (
                    <option key={p} value={p}>{pairLabel(p)}</option>
                  ))}
                </select>
              </label>

              {tracks.map((t, i) => (
                <label key={t.id} className="outmap-row">
                  <span style={{ color: TRACK_THEME[i]?.accent }}>
                    {TRACK_THEME[i]?.name ?? `Track ${i + 1}`}
                  </span>
                  <select className="live-select" value={t.outPair}
                          onChange={e => patchTrack(i, { outPair: Number(e.target.value) })}
                          title="An additional send; this track stays in the master mix either way">
                    {Array.from({ length: OUT_PAIRS }, (_, p) => (
                      <option key={p} value={p}>{pairLabel(p)}</option>
                    ))}
                  </select>
                </label>
              ))}

              <div className="outmap-foot">
                Each track feeds the master pair as well as its own. Two tracks
                on one pair sum; a track already on the master pair stays at unity.
              </div>
            </div>
          )}
        </div>


        <select className="live-select" value={maxSlots}
                onChange={e => setMaxSlots(Number(e.target.value))}
                title="Ceiling on input boxes and output tracks">
          {[1, 2, 3, 4].map(n => <option key={n} value={n}>max {n}</option>)}
        </select>

        <select className="live-select" value={model}
                onChange={e => setModel(e.target.value as any)}>
          <option value="sm-music">sm-music</option>
          <option value="sm-sfx">sm-sfx</option>
          <option value="medium">medium</option>
        </select>

        {/* Prompts come back by position: the first set goes to track 1. The
            picker resets to its label after a choice, so recalling the same
            set twice in a row is two presses rather than none. */}
        <div className="prompt-presets">
          <select
            className="live-select"
            value=""
            disabled={!presets.length}
            title={presets.length
              ? 'Load a saved set of prompts into the tracks, in order'
              : 'No saved prompt sets yet — SAVE one first'}
            onChange={e => { recallPreset(e.target.value); e.currentTarget.value = ''; }}
          >
            <option value="" disabled>
              {presets.length ? 'Recall prompts…' : 'No presets'}
            </option>
            {presets.map(p => <option key={p.name} value={p.name}>{p.name}</option>)}
          </select>
          <button
            className="preset-save"
            onClick={savePreset}
            disabled={!tracks.some(t => t.prompt.trim())}
            title="Save every track's prompt as a named set, in track order. Saving over a name replaces it."
          >
            SAVE
          </button>
        </div>

        <div className="view-toggle" title="Stacked, side by side, or the bare show screen">
          <button className={layout === 'vertical' ? 'on' : ''}
                  onClick={() => setLayout('vertical')} aria-label="Stacked layout">
            <Rows2 size={13} />
          </button>
          <button className={layout === 'horizontal' ? 'on' : ''}
                  onClick={() => setLayout('horizontal')} aria-label="Side-by-side layout">
            <Columns2 size={13} />
          </button>
          <button className={layout === 'show' ? 'on' : ''}
                  onClick={() => setLayout('show')} aria-label="Show view">
            <AudioWaveform size={13} />
          </button>
        </div>

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
          {sources.flatMap(inst => {
            const a = anchors.src[inst.id];
            if (!a) return [];
            return inst.to.slice(0, anchors.trk.length).map((on, ti) => {
              const b = anchors.trk[ti];
              if (!on || !b) return null;
              // Stacked: leave downwards, arrive from above. Side-by-side:
              // leave to the right, still arriving from above.
              const dy = Math.max(40, Math.abs(b.y - a.y) * 0.55);
              const dx = Math.max(48, Math.abs(b.x - a.x) * 0.5);
              const sag = Math.min(22, Math.abs(b.x - a.x) * 0.06);
              const d = layout === 'vertical'
                ? `M ${a.x} ${a.y} C ${a.x} ${a.y + dy + sag}, ${b.x} ${b.y - dy}, ${b.x} ${b.y}`
                : `M ${a.x} ${a.y} C ${a.x + dx} ${a.y + sag}, ${b.x} ${b.y - dy + sag}, ${b.x} ${b.y}`;
              const col = SOURCE_META[inst.kind].accent;
              return (
                <g key={`${inst.id}-${ti}`}>
                  <path className="cable-shadow" d={d} />
                  <path className="cable" stroke={col} d={d} />
                  <circle className="cable-plug" cx={a.x} cy={a.y} r={4} fill={col} />
                  <circle className="cable-plug" cx={b.x} cy={b.y} r={4} fill={col} />
                </g>
              );
            });
          })}
          {patchTo && (() => {
            // The loose end can originate at either a source or a track.
            const o = patchFrom ? anchors.src[patchFrom] : anchors.trk[patchFromTrack ?? -1];
            if (!o) return null;
            const col = patchFrom
              ? SOURCE_META[sources.find(x => x.id === patchFrom)?.kind ?? 'mic'].accent
              : (TRACK_THEME[patchFromTrack ?? 0]?.accent ?? '#ffffff');
            return (
              <g>
                <path className="cable dragging" stroke={col}
                      d={`M ${o.x} ${o.y} L ${patchTo.x} ${patchTo.y}`} />
                <circle className="cable-plug" cx={patchTo.x} cy={patchTo.y} r={5} fill={col} />
              </g>
            );
          })()}
        </svg>

        <div className="stage" ref={stageRef}>
        <section className="pane input-pane"
                 style={{ flexGrow: split[splitKey(layout)], flexBasis: 0 }}>
          <canvas ref={noiseCanvasRef} className="pane-noise" />
          <div className="pane-title">Live-Input Feed</div>

          {/* One horizontal band per enabled source, each with its own display,
              send fader and patch node. */}
          {/* A Live Code box needs room for its editor, so it takes half the
              input area and the remaining boxes share the other half. */}
          <div className="in-stack">
            <div className="in-col main">
            {sources.filter(x => x.kind !== 'code').map(inst => (
              renderSource(inst, sources.indexOf(inst))
            ))}
            </div>
            {sources.some(x => x.kind === 'code') && (
              <div className="in-col code">
                {sources.filter(x => x.kind === 'code')
                  .map(inst => (renderSource(inst, sources.indexOf(inst))))}
              </div>
            )}
          </div>

          {/* Add another input box. Live Code is capped at one because Strudel
              is a singleton — one repl, one context, one output tap. */}
          <div className="add-row">
            <button className="add-btn" disabled={running || sources.length >= maxSlots}
                    onClick={() => setAddMenu(v => !v)}
                    title={sources.length >= maxSlots ? `Limit is ${maxSlots} input boxes` : 'Add an input'}>
              <Plus size={14} /> ADD INPUT
            </button>
            {addMenu && (
              <div className="add-menu">
                {SOURCE_KINDS.map(kind => {
                  const taken = kind === 'code' && sources.some(x => x.kind === 'code');
                  return (
                    <button key={kind} disabled={taken}
                            title={taken ? 'Only one Live Code box is possible' : undefined}
                            onClick={() => addSource(kind)}>
                      {kind === 'mic' ? <Mic size={12} /> : kind === 'file' ? <Repeat size={12} /> : <Code2 size={12} />}
                      {SOURCE_META[kind].name}
                    </button>
                  );
                })}
              </div>
            )}
            <input ref={filePickerRef} type="file" accept="audio/*" hidden
                   onChange={e => {
                     const f = e.target.files?.[0];
                     if (f) pickFile(f);
                     e.target.value = '';   // allow re-picking the same file
                   }} />
          </div>

          <div className="monitor-row">
            <MidiSlider {...knobProps('inputMonitor')} label="Monitor"
                        value={inputMonitor} onChange={setInputMonitor} />
            <button
              className={`mon-mode ${monitorMode === 'box' ? 'on' : ''}`}
              onClick={() => setMonitorMode(m => (m === 'box' ? 'sum' : 'box'))}
              title={monitorMode === 'box'
                ? 'Each box has its own monitor level; Monitor trims all of them. Click for one summed level.'
                : 'Give every input box its own monitor level'}
            >PER BOX</button>
          </div>
        </section>

        <div
          className={`middle ${splitting ? 'dragging' : ''}`}
          onPointerDown={onSplitDown}
          onPointerMove={onSplitMove}
          onPointerUp={onSplitUp}
          onPointerCancel={onSplitUp}
          onDoubleClick={resetSplit}
          role="separator"
          aria-orientation={layout === 'vertical' ? 'horizontal' : 'vertical'}
          title={`Drag to resize · double-click for even halves — ${cur.steps} steps · variant ${cur.variant} at noise ${(cur.noise + cur.variant * cur.spread).toFixed(2)}`}
        >
          {/* Thickness is the step count; which axis that is depends on the
              view — height when the bar lies flat, width when it stands up. */}
          <div
            className="middle-bar"
            style={layout === 'vertical'
              ? { height: `${barThickness}px` }
              : { width: `${barThickness}px` }}
          >
            {/* Noise grain, same texture and the same amount as the input
                background, so the bar shows what is being added. */}
            <canvas ref={midNoiseRef} className="middle-noise" />
            <div className="variant-marker" style={{ left: `${2 + variantPct * 0.96}%` }}>
              <span className="variant-marker-tag">{cur.variant}</span>
            </div>
          </div>
        </div>

        <div className="out-side" style={{ flexGrow: 1 - split[splitKey(layout)], flexBasis: 0 }}>
          <div className="pane-title right">Live-Output Feed</div>
          <div className="out-stack">
          {tracks.map((t, i) => (
            <TrackOutput
              key={t.id}
              ref={(h) => { trackRefs.current[i] = h; }}
              index={i}
              connectionId={connectionId}
              params={t}
              shared={{ inputRate: effInputRate, generateRate: effGenerateRate, model }}
              onParam={(patch) => patchTrack(i, patch)}
              // The MIDI bridge broadcasts to every session; only track 1 acts
              // on it, or one CC would be applied twice.
              onMidi={i === 0 ? applyMidi : undefined}
              onAdd={i === tracks.length - 1 && tracks.length < maxSlots ? addTrack : undefined}
              onRemove={tracks.length > 1 ? () => removeTrack(i) : undefined}
              onError={setError}
              onConnected={i === 0 ? setConnected : undefined}
              // Master drives every track, so light them all: the ring is
              // what tells you which tracks the knobs are touching.
              active={masterMode || activeTab === i}
              onActivate={() => setActiveTab(i)}
              soloed={solo === i}
              muted={solo !== null && solo !== i}
              onSolo={() => setSolo(cur => (cur === i ? null : i))}
              soloVisible={tracks.length > 1}
              onNode={(el) => { trackNodeEls.current[t.id] = el; }}
              patchTarget={patchFrom !== null}
              onNodeDown={(e) => beginPatchFromTrack(i, e)}
              onShelfDone={() => setShelfBusy(false)}
              learnProps={knobProps}
              running={running}
              masterVolume={masterVolume}
              masterPair={masterPair}
              onChannels={i === 0 ? setOutChannels : undefined}
            />
          ))}
          </div>
        </div>
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

        {/* Master actions, left of everything, because they act on the whole
            rig rather than on one track or one box. */}
        <div className="knob-group g-master">
          <div className="group-cap">MASTER</div>
          <div className="knob-row act-row">
            <button
              className={`act-btn freeze ${frozen ? 'on' : ''} ${midiLearn ? 'learning' : ''} ${armed === 'freeze' ? 'armed' : ''}`}
              onClick={() => (midiLearn ? arm('freeze') : setFreeze(!frozen))}
              disabled={!running && !midiLearn}
              title="Hold the input the model is working from. Capture stops advancing, so every cycle from here re-reads the same window — you keep playing and hearing yourself, but the model keeps turning over the moment you froze. Press again to let it move."
            >
              <span className="act-name">FREEZE</span>
              <span className="act-sub">{frozen ? 'holding' : 'hold input'}</span>
              {bindingLabel('freeze') && <span className="act-bind">{bindingLabel('freeze')}</span>}
            </button>
            <button
              className={`act-btn inject ${midiLearn ? 'learning' : ''} ${armed === 'inject' ? 'armed' : ''}`}
              onClick={() => (midiLearn ? arm('inject') : injectNow())}
              disabled={!running && !midiLearn}
              title="Generate now from the last Input Rate of audio, the window ending at this instant, and re-phase the cycle clock to the press"
            >
              <span className="act-name">INJECT</span>
              <span className="act-sub">generate now</span>
              {bindingLabel('inject') && <span className="act-bind">{bindingLabel('inject')}</span>}
            </button>
            {/* The pre-FX latches share the row with the momentary pair. Both
                are in signal order — whatever is lit processes the summed feed
                on its way to the ring, so the model conditions on it while the
                input monitor keeps hearing the dry sources. */}
            <button
              className={`act-btn mfx ${masterFx.scramble ? 'on' : ''} ${midiLearn ? 'learning' : ''} ${armed === 'mfxScramble' ? 'armed' : ''}`}
              onClick={() => (midiLearn ? arm('mfxScramble') : setMasterFx(v => ({ ...v, scramble: !v.scramble })))}
              title="Reorder 1/16-note slices of the whole summed input at 80%, cut on the hits and drawn from the last bar. Runs before SMEAR, which would otherwise dissolve the onsets it cuts on. Latches."
            >
              <span className="act-name">SCRAMBLE</span>
              <span className="act-sub">{masterFx.scramble ? 'on · 80%' : 'off'}</span>
              {bindingLabel('mfxScramble') && <span className="act-bind">{bindingLabel('mfxScramble')}</span>}
            </button>
            <button
              className={`act-btn mfx ${masterFx.smear ? 'on' : ''} ${midiLearn ? 'learning' : ''} ${armed === 'mfxSmear' ? 'armed' : ''}`}
              onClick={() => (midiLearn ? arm('mfxSmear') : setMasterFx(v => ({ ...v, smear: !v.smear })))}
              title="Convolve the whole summed input with a 529 ms Gaussian noise burst at 80%: the spectrum survives, the attacks dissolve. Runs last. Latches."
            >
              <span className="act-name">SMEAR</span>
              <span className="act-sub">{masterFx.smear ? 'on · 80%' : 'off'}</span>
              {bindingLabel('mfxSmear') && <span className="act-bind">{bindingLabel('mfxSmear')}</span>}
            </button>
          </div>
          {mfxLatency > 0 && (
            <div className="mfx-lat">+{(mfxLatency * 1000).toFixed(0)} ms to the feed</div>
          )}

          {/* The fader sits under the pair because it is the same scope: one
              control over everything the rack is putting out. */}
          <div className="master-vol">
            <MidiSlider {...knobProps('masterVolume')} label="Volume"
                        value={masterVolume} onChange={setMasterVolume} />
          </div>
        </div>

        {/* Input pre-FX leads the knobs because it is the only group that runs
            BEFORE the ring buffer: it changes what the model conditions on,
            not just what you hear. Left to right, the row now follows the
            signal — input, rate, generation, variants, feedback, output. */}
        {fxSrc && (() => {
          const m = fxMethod(fxSrc.fxMethod);
          const lat = m.latencyAt(fxSrc.fxAmount);
          return (
            <div className="knob-group g-infx">
              <div className="group-cap">
                INPUT FX
                <span className="group-sub">{SOURCE_META[fxSrc.kind].name}</span>
              </div>
              <div className="knob-row">
                <div className="fx-cell">
                  <select
                    className="fx-method"
                    value={fxSrc.fxMethod}
                    onChange={e => patchFx({ fxMethod: e.target.value })}
                    title={m.blurb}
                    aria-label="Input FX method"
                  >
                    {FX_METHODS.map(x => (
                      <option key={x.id} value={x.id} title={x.blurb}>{x.label}</option>
                    ))}
                  </select>
                  <span className="fx-lat"
                        title="Added input latency — the model's window is this much older">
                    {lat > 0.001 ? `+${(lat * 1000).toFixed(0)}ms` : 'no delay'}
                  </span>
                </div>
                <Knob {...knobProps('fxAmount')} label="Saturate" min={0} max={1} step={0.005}
                      value={fxSrc.fxAmount} onChange={v => patchFx({ fxAmount: v })}
                      format={v => m.format(v)} />
              </div>
            </div>
          );
        })()}

        <div className="knob-group g-rate">
          <div className="group-cap">RATE</div>
          <div className="knob-row">
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
              <Knob {...knobProps('inputRate')} label="Input Rate" values={INPUT_RATES}
                    value={inputRate} onChange={setInputRate} format={v => `${v}s`} />
              <Knob {...knobProps('generateRate')} label="Generate Rate" values={GEN_RATES}
                    value={generateRate} onChange={setGenerateRate} format={v => `${v}s`} />
            </>
          )}
          </div>
        </div>

        <div className="knob-group g-gen">
          <div className="group-cap">GENERATION</div>
          <div className="knob-row">
          <Knob {...knobProps('noise')} label="Noise" min={0.05} max={1} step={0.01}
                value={cur.noise} onChange={v => patchActive({ noise: v })} />
          <Knob {...knobProps('steps')} label="Steps" min={1} max={16} step={1}
                value={cur.steps} onChange={v => patchActive({ steps: v })} format={v => String(Math.round(v))} />
          <Knob {...knobProps('cfg')} label="CFG" min={1} max={10} step={0.1}
                value={cur.cfg} onChange={v => patchActive({ cfg: v })} format={v => v.toFixed(1)} />
          </div>
        </div>

        <div className="knob-group g-var">
          <div className="group-cap">VARIANTS</div>
          <div className="knob-row">
          <div className="variant-cell">
            <Knob {...knobProps('variant')} label="Variant"
                  min={0} max={Math.max(0, activeVariants - 1)} step={1}
                  disabled={!cur.multiVariant}
                  value={cur.variant} onChange={v => patchActive({ variant: v })}
                  format={v => cur.multiVariant
                    ? `${Math.round(v)} · ${(cur.noise + v * cur.spread).toFixed(2)}`
                    : 'off'} />
            <label className="variant-toggle" title={`Generate ${VARIANTS} progressively noisier variants each cycle (${VARIANTS}x slower)`}>
              <input type="checkbox" checked={cur.multiVariant}
                     onChange={e => patchActive({
                       multiVariant: e.target.checked,
                       variant: 0,
                       // ALL is meaningless without a ladder to spread.
                       allVariants: e.target.checked ? cur.allVariants : false,
                     })} />
              {VARIANTS} variants
            </label>
          </div>

          {cur.multiVariant && (
            <div className="variant-cell">
              <Knob {...knobProps('spread')} label="Spread"
                    min={SPREAD_MIN} max={SPREAD_MAX} step={0.0025}
                    value={cur.spread} onChange={v => patchActive({ spread: v })}
                    format={v => v.toFixed(3)} />
              <label className="variant-toggle"
                     title="Play every variant at once, panned left to right">
                <input type="checkbox" checked={cur.allVariants}
                       onChange={e => patchActive({ allVariants: e.target.checked })} />
                all
              </label>
            </div>
          )}
          </div>
        </div>

        <div className="knob-group g-fb">
          <div className="group-cap">FEEDBACK</div>
          <div className="knob-row">
          <Knob {...knobProps('loopback')} label="Loopback" min={0} max={1} step={0.01}
                value={cur.loopback} onChange={v => patchActive({ loopback: v })}
                format={v => `${Math.round(v * 100)}%`} />
          <div className="variant-cell">
            <Knob {...knobProps('nudge')} label="Nudge" min={0} max={MAX_NUDGE} step={0.01}
                  value={cur.nudge} onChange={v => patchActive({ nudge: v })}
                  format={v => `${v.toFixed(2)}s`} />
            <button className={`sync-btn${syncOn ? ' on' : ''}`}
                    onClick={() => {
                      const next = !syncOn;
                      setSyncOn(next);
                      if (next) applySync(trackRefs.current.map(t => t?.latency() ?? null));
                    }}
                    disabled={!running || syncReady < 2}
                    title={syncReady < 2
                      ? 'Waiting on measured round trips — two tracks need to have played a cycle each. If this stays greyed out while both tracks are sounding, the backend is running an older build that does not echo the timing stamp.'
                      : 'Latched: holds every track back to match the slowest round trip, so one input moment lands at the same instant in all of them, and keeps re-aligning as the timings move.'}>
              SYNC
            </button>
            <div className="sync-read">{syncLabel}</div>
          </div>
          </div>
        </div>

        {/* Output filters. Browser-side only, so they respond instantly and
            cost nothing in generation time — sweep them mid-cycle. */}
        <div className="knob-group g-outfx">
          <div className="group-cap">OUTPUT FX</div>
          <div className="knob-row">
          <Knob {...knobProps('lpf')} label="LPF" min={0} max={1} step={0.005}
                value={cur.lpf} onChange={v => patchActive({ lpf: v })}
                format={v => (v >= 1 ? 'open' : hz(lpfHz(v)))} />
          <Knob {...knobProps('hpf')} label="HPF" min={0} max={1} step={0.005}
                value={cur.hpf} onChange={v => patchActive({ hpf: v })}
                format={v => (v <= 0 ? 'open' : hz(hpfHz(v)))} />
          <Knob {...knobProps('comb')} label="Comb" min={0} max={1} step={0.005}
                value={cur.comb} onChange={v => patchActive({ comb: v })}
                format={v => (v <= 0 ? 'off' : hz(combHz(v)))} />
          </div>
        </div>

        {tracks.length > 1 && (
          <button
            className={`master-btn ${masterMode ? 'on' : ''}`}
            onClick={() => setMasterMode(v => !v)}
            title="Apply knob and MIDI changes to every track at once"
          >
            <span>ALL TRACKS</span>
            <span className="master-sub">
              {masterMode ? 'knobs → every track' : `knobs → track ${activeTab + 1}`}
            </span>
          </button>
        )}
      </footer>

      {/* The performance screen sits over everything; the rest of the app stays
          mounted behind it, because unmounting a track would take its socket
          and its audio graph with it. */}
      {layout === 'show' && (
        <ShowView inputWave={readInputWave} outputWave={readOutputWave}
                  prompts={audiblePrompts} onExit={leaveShow} />
      )}
    </div>
  );
}
