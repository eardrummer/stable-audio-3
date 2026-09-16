import type { ReactNode } from 'react';
import { X } from 'lucide-react';
import { SOURCE_META, type SourceInstance } from './trackTypes';

/**
 * One input box: its own display, send fader, and a patch node on the right
 * edge that cables run from to the output tracks.
 *
 * A Live Input box also carries its own channel selector, so two boxes can take
 * different channels of the same interface and be patched to different tracks.
 *
 * Presentational only — the canvas is drawn by LiveApp's animation loop, which
 * owns the loop-view and trim state, and the cable overlay positions itself
 * from the node element registered here.
 */
interface Props {
  src: SourceInstance;
  onLevel: (v: number) => void;
  onChanSel: (v: string) => void;
  onRemove: () => void;
  onCanvas: (el: HTMLCanvasElement | null) => void;
  onNode: (el: HTMLDivElement | null) => void;
  onNodeDown: (e: React.PointerEvent) => void;
  /** Channel options offered by the current device. */
  channelOptions: { value: string; label: string }[];
  /** Live Code renders its editor here instead of a waveform. */
  overlay?: ReactNode;
  tools?: ReactNode;
  canvasProps?: React.CanvasHTMLAttributes<HTMLCanvasElement>;
  patching?: boolean;
  connected: number;
  running: boolean;
  /** This box is the one the Input FX controls are editing. */
  active?: boolean;
  onActivate?: () => void;
}

export default function InputSection({
  src, onLevel, onChanSel, onRemove, onCanvas, onNode, onNodeDown,
  channelOptions, overlay, tools, canvasProps, patching, connected, running,
  active, onActivate,
}: Props) {
  const meta = SOURCE_META[src.kind];
  return (
    <section
      className={`in-section ${patching ? 'patching' : ''} ${active ? 'active' : ''}`}
      style={{ ['--src' as string]: meta.accent }}
      // Clicking anywhere in a box is how you choose which one the Input FX
      // controls edit — the same gesture that picks a track below.
      onPointerDown={onActivate}
    >
      <div className="in-head">
        <span className="in-name">{meta.name}</span>

        {src.kind === 'mic' && (
          <select
            className="in-chan"
            value={src.chanSel}
            onChange={(e) => onChanSel(e.target.value)}
            title="Which hardware channel(s) this box takes"
          >
            {channelOptions.map(o => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        )}

        <span className="in-routed">
          {connected === 0 ? 'not patched' : `→ ${connected} track${connected > 1 ? 's' : ''}`}
        </span>
        <button className="in-close" onClick={onRemove} disabled={running}
                title={running ? 'Stop the transport to remove' : `Remove this ${meta.name}`}>
          <X size={12} />
        </button>
      </div>

      <div className="in-body">
        <canvas ref={onCanvas} width={900} height={200} className="in-canvas" {...canvasProps} />
        {overlay}
      </div>

      {tools}

      {/* Cables to the output tracks leave from the bottom edge. */}
      <div ref={onNode} className="patch-node out" onPointerDown={onNodeDown}
           title="Drag to a track to patch" />

      <div className="in-fader">
        <input
          type="range" min={0} max={1} step={0.01} value={src.level}
          aria-label={`${meta.name} send level`}
          onChange={(e) => onLevel(Number(e.target.value))}
        />
        <span className="dim">{Math.round(src.level * 100)}%</span>
      </div>
    </section>
  );
}
