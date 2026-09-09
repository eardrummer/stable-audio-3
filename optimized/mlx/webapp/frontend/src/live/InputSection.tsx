import type { ReactNode } from 'react';
import { X } from 'lucide-react';
import { SOURCE_META, type SourceId, type SourceState } from './trackTypes';

/**
 * One input source as a horizontal band: its own display, its own send fader,
 * and a patch node on the right edge that cables run from to the output tracks.
 *
 * Presentational only — the canvas is drawn by LiveApp's animation loop, which
 * already owns the loop-view and trim state, and the patch overlay positions
 * its cables from the node element registered here.
 */
interface Props {
  id: SourceId;
  state: SourceState;
  onLevel: (v: number) => void;
  onDisable: () => void;
  /** Registers the canvas with LiveApp's draw loop. */
  onCanvas: (el: HTMLCanvasElement | null) => void;
  /** Registers the patch anchor with the cable overlay. */
  onNode: (el: HTMLDivElement | null) => void;
  /** Live Code renders its editor here instead of a waveform. */
  overlay?: ReactNode;
  /** Looper controls, import button, and so on. */
  tools?: ReactNode;
  canvasProps?: React.CanvasHTMLAttributes<HTMLCanvasElement>;
  /** True while a cable drag is looking for a target. */
  patching?: boolean;
  onNodeDown: (e: React.PointerEvent) => void;
  connected: number;
}

export default function InputSection({
  id, state, onLevel, onDisable, onCanvas, onNode,
  overlay, tools, canvasProps, patching, connected, onNodeDown,
}: Props) {
  const meta = SOURCE_META[id];
  return (
    <section
      className={`in-section ${patching ? 'patching' : ''}`}
      style={{ ['--src' as string]: meta.accent }}
    >
      <div className="in-head">
        <span className="in-name">{meta.name}</span>
        <span className="in-routed">
          {connected === 0 ? 'not patched' : `→ ${connected} track${connected > 1 ? 's' : ''}`}
        </span>
        <button className="in-close" onClick={onDisable} title={`Turn off ${meta.name}`}>
          <X size={12} />
        </button>
      </div>

      <div className="in-body">
        <canvas ref={onCanvas} width={900} height={200} className="in-canvas" {...canvasProps} />
        {overlay}
        {/* Patch node: cables to the output tracks start here. */}
        <div ref={onNode} className="patch-node out" onPointerDown={onNodeDown}
             title="Drag to a track to patch" />
      </div>

      {tools}

      <div className="in-fader">
        <input
          type="range" min={0} max={1} step={0.01} value={state.level}
          aria-label={`${meta.name} send level`}
          onChange={(e) => onLevel(Number(e.target.value))}
        />
        <span className="dim">{Math.round(state.level * 100)}%</span>
      </div>
    </section>
  );
}
