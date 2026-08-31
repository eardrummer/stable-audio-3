import { useCallback, useEffect, useRef } from 'react';

export interface KnobProps {
  id: string;                 // stable id used for MIDI bindings
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  values?: number[];          // discrete positions (InputRate / GenerateRate)
  format?: (v: number) => string;
  onChange: (v: number) => void;
  // MIDI learn
  learning?: boolean;         // learn mode active -> show the dotted outline
  armed?: boolean;            // this control is the one awaiting a CC
  bound?: string | null;      // e.g. "CC 16" once assigned
  onArm?: (id: string) => void;
}

const SWEEP = 270;            // degrees of travel, classic 7-o'clock to 5-o'clock

/**
 * Rotary control. Drag vertically to change (hold Shift for fine), double-click
 * to jump back to the middle of the range. In MIDI learn mode a click arms the
 * knob so the next incoming CC binds to it.
 */
export default function Knob({
  id, label, value, min = 0, max = 1, step = 0.01, values,
  format, onChange, learning = false, armed = false, bound = null, onArm,
}: KnobProps) {
  const dragRef = useRef<{ y: number; v: number } | null>(null);
  // onChange identity changes every render; keep the pointer handlers stable.
  const onChangeRef = useRef(onChange);
  useEffect(() => { onChangeRef.current = onChange; }, [onChange]);

  const lo = values ? 0 : min;
  const hi = values ? values.length - 1 : max;
  const pos = values ? Math.max(0, values.indexOf(value)) : value;
  const pct = hi === lo ? 0 : (pos - lo) / (hi - lo);

  const commit = useCallback((raw: number) => {
    const clamped = Math.max(lo, Math.min(hi, raw));
    if (values) {
      onChangeRef.current(values[Math.round(clamped)]);
    } else {
      const snapped = Math.round(clamped / step) * step;
      // step is often fractional (0.01), so round away binary float dust.
      onChangeRef.current(Number(snapped.toFixed(6)));
    }
  }, [lo, hi, step, values]);

  const onPointerDown = (e: React.PointerEvent) => {
    if (learning) { onArm?.(id); return; }
    (e.target as Element).setPointerCapture(e.pointerId);
    dragRef.current = { y: e.clientY, v: pos };
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const d = dragRef.current;
    if (!d) return;
    // 180px of travel spans the full range; Shift slows it to a quarter.
    const span = (hi - lo) * ((d.y - e.clientY) / 180) * (e.shiftKey ? 0.25 : 1);
    commit(d.v + span);
  };

  const endDrag = (e: React.PointerEvent) => {
    if (dragRef.current) {
      try { (e.target as Element).releasePointerCapture(e.pointerId); } catch { /* already released */ }
      dragRef.current = null;
    }
  };

  const onWheel = (e: React.WheelEvent) => {
    if (learning) return;
    const inc = values ? 1 : step;
    commit(pos + (e.deltaY < 0 ? inc : -inc));
  };

  const angle = -SWEEP / 2 + pct * SWEEP;
  const R = 26;
  const arc = (from: number, to: number) => {
    const p = (deg: number) => {
      const r = ((deg - 90) * Math.PI) / 180;
      return [32 + R * Math.cos(r), 32 + R * Math.sin(r)];
    };
    const [x1, y1] = p(from);
    const [x2, y2] = p(to);
    return `M ${x1} ${y1} A ${R} ${R} 0 ${Math.abs(to - from) > 180 ? 1 : 0} 1 ${x2} ${y2}`;
  };

  const display = format
    ? format(value)
    : values ? String(value) : value.toFixed(step < 0.1 ? 2 : 1);

  return (
    <div className={`knob ${learning ? 'learning' : ''} ${armed ? 'armed' : ''}`}>
      <svg
        width="64" height="64" viewBox="0 0 64 64"
        className="knob-dial"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onWheel={onWheel}
        onDoubleClick={() => !learning && commit((hi - lo) / 2 + lo)}
        role="slider"
        aria-label={label}
        aria-valuenow={value}
        aria-valuemin={values ? values[0] : min}
        aria-valuemax={values ? values[values.length - 1] : max}
      >
        <path d={arc(-SWEEP / 2, SWEEP / 2)} className="knob-track" />
        {pct > 0.001 && <path d={arc(-SWEEP / 2, angle)} className="knob-fill" />}
        <circle cx="32" cy="32" r="19" className="knob-body" />
        <line
          x1="32" y1="32"
          x2={32 + 15 * Math.cos(((angle - 90) * Math.PI) / 180)}
          y2={32 + 15 * Math.sin(((angle - 90) * Math.PI) / 180)}
          className="knob-pointer"
        />
      </svg>
      <div className="knob-value">{display}</div>
      <div className="knob-label">{label}</div>
      {bound && <div className="knob-binding">{bound}</div>}
    </div>
  );
}
