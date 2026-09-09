import {
  forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState,
} from 'react';
import { Plus, Minus, Headphones } from 'lucide-react';
import MidiSlider from './MidiSlider';
import { drawSpectrogram } from '../shared/spectrogram';
import {
  WS_URL, VARIANTS, VARIANT_DELTA, MAX_NUDGE, XF_MAX_CYCLE, XF_SWITCH, XF_MIN,
  FADE_IN_CURVE, FADE_OUT_CURVE, TRACK_THEME,
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
}

interface CycleAudio { cycle: number; buffers: (AudioBuffer | null)[]; }

const TrackOutput = forwardRef<TrackHandle, Props>(function TrackOutput({
  index, connectionId, params, shared, onParam, onMidi, onAdd, onRemove,
  onError, onShelfDone, onConnected, learnProps, running,
  active, onActivate, soloed, muted, onSolo, soloVisible, onNode, patchTarget,
}, ref) {
  const theme = TRACK_THEME[index] ?? TRACK_THEME[0];

  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState('Idle');
  const [cycleInfo, setCycleInfo] = useState<{ n: number; ms: number } | null>(null);
  const [dropped, setDropped] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const delayRef = useRef<DelayNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const specRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const cycleRef = useRef<CycleAudio | null>(null);
  const pendingHeaderRef = useRef<any>(null);
  const playingRef = useRef<{ src: AudioBufferSourceNode; gain: GainNode; startedAt: number } | null>(null);

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
      ctxRef.current = ctx; gainRef.current = gain;
      analyserRef.current = analyser; delayRef.current = delay;
    }
    return ctxRef.current!;
  }, []);

  const crossfadeTo = useCallback((buf: AudioBuffer, offset = 0, maxFade = XF_MAX_CYCLE) => {
    const ctx = ensureCtx();
    const now = ctx.currentTime;
    const prev = playingRef.current;
    const fade = Math.max(XF_MIN, Math.min(maxFade, buf.duration - offset));

    const gain = ctx.createGain();
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(gain);
    gain.connect(delayRef.current ?? analyserRef.current!);

    try {
      gain.gain.setValueCurveAtTime(FADE_IN_CURVE, now, fade);
    } catch {
      gain.gain.setValueAtTime(0, now);
      gain.gain.linearRampToValueAtTime(1, now + fade);
    }

    if (prev) {
      const g = prev.gain.gain;
      const from = g.value;
      try {
        g.cancelScheduledValues(now);
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
  }, [ensureCtx]);

  // Solo silences the other track rather than changing its level, so its
  // own monitor slider still reads what it will return to.
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

  // Switching the variant knob mid-clip swaps lanes at the same position.
  useEffect(() => {
    const cyc = cycleRef.current, cur = playingRef.current;
    if (!cyc || !cur || !ctxRef.current) return;
    const buf = cyc.buffers[params.variant];
    if (!buf) return;
    const off = ctxRef.current.currentTime - cur.startedAt;
    if (off >= 0 && off < buf.duration - 0.1) crossfadeTo(buf, off, XF_SWITCH);
  }, [params.variant, crossfadeTo]);

  // ── spectrogram ─────────────────────────────────────────────────────
  useEffect(() => {
    const tick = () => {
      rafRef.current = requestAnimationFrame(tick);
      const sc = specRef.current, an = analyserRef.current;
      if (sc && an) drawSpectrogram(sc, an, theme.spec);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [theme.spec]);

  // ── socket ──────────────────────────────────────────────────────────
  const sendParams = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const p = paramsRef.current, sh = sharedRef.current;
    ws.send(JSON.stringify({
      type: 'params',
      params: {
        prompt: p.prompt, input_rate: sh.inputRate, generate_rate: sh.generateRate,
        noise: p.noise, variant_delta: VARIANT_DELTA, steps: p.steps,
        cfg: p.cfg, variants: p.multiVariant ? VARIANTS : 1, model: sh.model,
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

        if (hdr.redecode) {
          onShelfDone?.();
          const cur = playingRef.current;
          if (hdr.index === paramsRef.current.variant && cur && ctxRef.current) {
            const off = ctxRef.current.currentTime - cur.startedAt;
            crossfadeTo(buf, off >= 0 && off < buf.duration - 0.1 ? off : 0, XF_SWITCH);
          }
          return;
        }
        // Start the new cycle as soon as the audible lane lands.
        if (hdr.index === paramsRef.current.variant) crossfadeTo(buf, 0);
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
  }, [connectionId, index, crossfadeTo, ensureCtx]);

  useEffect(() => { if (running) sendParams(); }, [
    running, sendParams,
    params.prompt, params.noise, params.steps, params.cfg,
    params.loopback, params.variant, params.multiVariant,
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
          noise: p.noise, variant_delta: VARIANT_DELTA, steps: p.steps, cfg: p.cfg,
          variants: p.multiVariant ? VARIANTS : 1, model: sh.model,
          loopback: p.loopback, selected_variant: p.variant,
        },
      }));
    },
    finish() {
      wsRef.current?.send(JSON.stringify({ type: 'stop' }));
      setStatus('Idle');
      const prev = playingRef.current, ctx = ctxRef.current;
      if (prev && ctx) {
        const now = ctx.currentTime;
        prev.gain.gain.cancelScheduledValues(now);
        prev.gain.gain.setValueAtTime(Math.max(prev.gain.gain.value, 0.0001), now);
        prev.gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.2);
        prev.src.stop(now + 0.25);
        playingRef.current = null;
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
      <div ref={onNode} className="patch-node in" data-track={index} />

      <div className="out-head">
        <span className="out-title">{index === 0 ? 'Live-Output Feed' : theme.name}</span>
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
        {index === 0
          ? onAdd && <button className="track-btn" onClick={onAdd} title="Add a second output track"><Plus size={13} /></button>
          : <button className="track-btn del" onClick={onRemove} title="Remove this track"><Minus size={13} /></button>}
      </div>

      <input
        className="prompt-box" value={params.prompt} placeholder={`Text prompt — ${theme.name}`}
        onChange={(e) => onParam({ prompt: e.target.value })}
        // Clicking into a track's prompt is how you choose which track the
        // knob row edits — no separate tab strip needed.
        onFocus={onActivate}
      />

      <canvas ref={specRef} width={900} height={420} className="pane-canvas" />

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
