import { useCallback, useEffect, useRef, useState } from 'react';
import { X, RotateCcw } from 'lucide-react';

export const N_CHANNELS = 256;

interface Props {
  open: boolean;
  onClose: () => void;
  /** Called with the full 256-length curve; throttled during a drag. */
  onChange: (weights: Float32Array, renormalize: boolean) => void;
  renormalize: boolean;
  onRenormalize: (v: boolean) => void;
  busy?: boolean;
}

type Shape = 'all1' | 'all0' | 'invert' | 'rampUp' | 'rampDown' | 'random' | 'comb';

const SEND_MS = 110;   // during a drag, at most one re-decode request this often

/**
 * Direct manipulation of SAME's 256 latent channels. Each row is one channel;
 * its bar length is a multiplier in [0, 1] applied immediately before the
 * decoder. Painting re-decodes the cached latent (~30 ms) rather than
 * regenerating, so the curve is audible almost as fast as you draw it.
 */
export default function LatentShelf({
  open, onClose, onChange, renormalize, onRenormalize, busy,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const weightsRef = useRef<Float32Array>(new Float32Array(N_CHANNELS).fill(1));
  const paintingRef = useRef(false);
  const lastChanRef = useRef<number | null>(null);
  const lastSendRef = useRef(0);
  const sendTimerRef = useRef<number | null>(null);
  const [hover, setHover] = useState<{ ch: number; v: number } | null>(null);
  const [, forceRedraw] = useState(0);

  const W = 200, H = N_CHANNELS;   // one logical pixel row per channel

  const draw = useCallback(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const w = weightsRef.current;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, W, H);
    for (let i = 0; i < N_CHANNELS; i++) {
      const v = w[i];
      const len = Math.max(1, v * W);
      // Dim green at 0, bright at 1 — so an attenuated band reads as a notch.
      const g = Math.round(70 + v * 165);
      const b = Math.round(60 + v * 60);
      ctx.fillStyle = `rgb(${Math.round(20 + v * 40)}, ${g}, ${b})`;
      ctx.fillRect(0, i, len, 1);
      if (v < 0.999) {
        ctx.fillStyle = 'rgba(255,255,255,0.045)';
        ctx.fillRect(len, i, W - len, 1);
      }
    }
    // Ticks every 32 channels for orientation.
    ctx.fillStyle = 'rgba(255,255,255,0.16)';
    for (let i = 32; i < N_CHANNELS; i += 32) ctx.fillRect(0, i, W, 1);
  }, []);

  useEffect(() => { draw(); }, [draw, open]);

  const flush = useCallback((immediate = false) => {
    const now = performance.now();
    if (sendTimerRef.current) { clearTimeout(sendTimerRef.current); sendTimerRef.current = null; }
    if (immediate || now - lastSendRef.current >= SEND_MS) {
      lastSendRef.current = now;
      onChange(weightsRef.current.slice(), renormalize);
    } else {
      sendTimerRef.current = window.setTimeout(
        () => { lastSendRef.current = performance.now(); onChange(weightsRef.current.slice(), renormalize); },
        SEND_MS - (now - lastSendRef.current));
    }
  }, [onChange, renormalize]);

  const posFrom = (e: React.PointerEvent) => {
    const c = canvasRef.current!;
    const r = c.getBoundingClientRect();
    const ch = Math.max(0, Math.min(N_CHANNELS - 1, Math.floor(((e.clientY - r.top) / r.height) * N_CHANNELS)));
    const v = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    return { ch, v };
  };

  const paint = (ch: number, v: number) => {
    const w = weightsRef.current;
    const prev = lastChanRef.current;
    if (prev !== null && Math.abs(ch - prev) > 1) {
      // Interpolate across skipped rows so a fast drag draws a continuous
      // curve instead of a dotted one.
      const step = ch > prev ? 1 : -1;
      for (let i = prev + step; i !== ch; i += step) w[i] = v;
    }
    w[ch] = v;
    lastChanRef.current = ch;
    draw();
  };

  const onDown = (e: React.PointerEvent) => {
    (e.target as Element).setPointerCapture(e.pointerId);
    paintingRef.current = true;
    lastChanRef.current = null;
    const { ch, v } = posFrom(e);
    paint(ch, v);
    flush(true);
  };

  const onMove = (e: React.PointerEvent) => {
    const { ch, v } = posFrom(e);
    setHover({ ch, v: weightsRef.current[ch] });
    if (!paintingRef.current) return;
    paint(ch, v);
    flush();
  };

  const onUp = (e: React.PointerEvent) => {
    if (!paintingRef.current) return;
    paintingRef.current = false;
    lastChanRef.current = null;
    try { (e.target as Element).releasePointerCapture(e.pointerId); } catch { /* already released */ }
    flush(true);   // always end on the true final curve
  };

  const applyShape = (s: Shape) => {
    const w = weightsRef.current;
    for (let i = 0; i < N_CHANNELS; i++) {
      const f = i / (N_CHANNELS - 1);
      if (s === 'all1') w[i] = 1;
      else if (s === 'all0') w[i] = 0;
      else if (s === 'invert') w[i] = 1 - w[i];
      else if (s === 'rampUp') w[i] = f;
      else if (s === 'rampDown') w[i] = 1 - f;
      else if (s === 'random') w[i] = Math.random();
      else if (s === 'comb') w[i] = i % 2 === 0 ? 1 : 0;
    }
    draw();
    forceRedraw(n => n + 1);
    flush(true);
  };

  useEffect(() => () => { if (sendTimerRef.current) clearTimeout(sendTimerRef.current); }, []);

  if (!open) return null;

  const active = weightsRef.current.reduce((n, v) => n + (v > 0.999 ? 0 : 1), 0);

  return (
    <aside className="shelf">
      <div className="shelf-head">
        <span className="shelf-title">Latent Channels</span>
        <button className="shelf-x" onClick={onClose} aria-label="Close latent shelf">
          <X size={14} />
        </button>
      </div>

      <div className="shelf-sub">
        256 dims · {active} altered {busy && <span className="shelf-busy">decoding…</span>}
      </div>

      <canvas
        ref={canvasRef}
        width={W}
        height={H}
        className="shelf-canvas"
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={onUp}
        onPointerLeave={() => setHover(null)}
      />

      <div className="shelf-readout">
        {hover ? `ch ${hover.ch}  ·  ${hover.v.toFixed(2)}` : 'drag to shape'}
      </div>

      <div className="shelf-btns">
        <button onClick={() => applyShape('all1')}><RotateCcw size={11} /> Reset</button>
        <button onClick={() => applyShape('all0')}>All 0</button>
        <button onClick={() => applyShape('invert')}>Invert</button>
        <button onClick={() => applyShape('rampUp')}>Ramp ↓</button>
        <button onClick={() => applyShape('rampDown')}>Ramp ↑</button>
        <button onClick={() => applyShape('comb')}>Comb</button>
        <button onClick={() => applyShape('random')}>Random</button>
      </div>

      <label className="shelf-renorm">
        <input type="checkbox" checked={renormalize}
               onChange={e => { onRenormalize(e.target.checked); }} />
        Renormalize level
        <span className="shelf-hint">
          keeps overall RMS constant, so you hear timbre change rather than level drop
        </span>
      </label>
    </aside>
  );
}
