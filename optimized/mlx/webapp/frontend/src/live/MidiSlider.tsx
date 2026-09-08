/**
 * A monitor level slider that takes part in MIDI learn, exactly like Knob:
 * in learn mode it shows the dotted outline, a click arms it, and the next
 * incoming CC binds to it.
 */
export interface MidiSliderProps {
  id: string;
  label: string;
  value: number;
  onChange: (v: number) => void;
  learning?: boolean;
  armed?: boolean;
  bound?: string | null;
  onArm?: (id: string) => void;
}

export default function MidiSlider({
  id, label, value, onChange, learning = false, armed = false, bound = null, onArm,
}: MidiSliderProps) {
  return (
    <div
      className={`monitor ${learning ? 'learning' : ''} ${armed ? 'armed' : ''}`}
      onPointerDown={learning ? () => onArm?.(id) : undefined}
    >
      <span>{label}</span>
      <input
        type="range" min={0} max={1} step={0.01} value={value}
        aria-label={label}
        // While arming, the click must select the control rather than move it.
        disabled={learning}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <span className="dim">{Math.round(value * 100)}%</span>
      {bound && <span className="slider-binding">{bound}</span>}
    </div>
  );
}
