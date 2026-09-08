import { useEffect, useRef, useState } from 'react';
import { Play, Square, Info } from 'lucide-react';
import type { EditorHandle } from './strudel/codeEditor';

const DEFAULT_CODE = `// Live Code — this audio becomes the model's input.
// Cmd/Ctrl+Enter to evaluate · Cmd/Ctrl+. to hush
setcpm(120/4)

stack(
  s("bd*2, ~ sd").bank("RolandTR909"),
  note("<c2 g1 a1 f1>").s("sawtooth").cutoff(600).gain(.7)
)`;

interface Props {
  onEvaluate: (code: string) => void;
  onHush: () => void;
  running: boolean;
  ready: boolean;
  banks: string[];
  error?: string | null;
}

/**
 * Code editor overlaying the input waveform. CodeMirror is loaded through a
 * dynamic import so its bundle only arrives when Live Code mode is opened.
 */
export default function StrudelPanel({
  onEvaluate, onHush, running, ready, banks, error,
}: Props) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const edRef = useRef<EditorHandle | null>(null);
  const codeRef = useRef<string>(
    (() => { try { return localStorage.getItem('sa3_strudel_code') || DEFAULT_CODE; }
             catch { return DEFAULT_CODE; } })(),
  );
  const [flash, setFlash] = useState(false);
  const [editorReady, setEditorReady] = useState(false);

  // Callbacks reach the editor's keymap through refs, so remounting the
  // CodeMirror instance on every parent render is never necessary.
  const runRef = useRef(onEvaluate);
  const hushRef = useRef(onHush);
  useEffect(() => { runRef.current = onEvaluate; }, [onEvaluate]);
  useEffect(() => { hushRef.current = onHush; }, [onHush]);

  const fire = () => {
    const code = edRef.current?.getCode() ?? codeRef.current;
    runRef.current(code);
    setFlash(true);
    window.setTimeout(() => setFlash(false), 220);
  };

  useEffect(() => {
    let disposed = false;
    let handle: EditorHandle | null = null;
    (async () => {
      const { createEditor } = await import('./strudel/codeEditor');
      if (disposed || !hostRef.current) return;
      handle = await createEditor({
        parent: hostRef.current,
        doc: codeRef.current,
        onChange: (code) => {
          codeRef.current = code;
          try { localStorage.setItem('sa3_strudel_code', code); } catch { /* private mode */ }
        },
        onRun: () => fire(),
        onHush: () => hushRef.current(),
      });
      if (disposed) { handle.destroy(); return; }
      edRef.current = handle;
      setEditorReady(true);
    })();
    return () => { disposed = true; handle?.destroy(); edRef.current = null; };
  }, []);   // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className={`strudel-panel ${flash ? 'flash' : ''}`}>
      <div className="strudel-bar">
        <button className="strudel-run" onClick={fire} disabled={!ready || !editorReady}>
          <Play size={12} /> RUN
        </button>
        <button className="strudel-hush" onClick={onHush} disabled={!ready}>
          <Square size={12} /> HUSH
        </button>
        <span className="strudel-hint">⌘↵ run · ⌘. hush</span>
        <span className={`strudel-state ${running ? 'on' : ''}`}>
          {!ready ? 'loading…' : running ? 'playing' : 'idle'}
        </span>
      </div>

      <div ref={hostRef} className="strudel-code" />

      {error && <div className="strudel-error">{error}</div>}

      <div className="strudel-foot">
        <Info size={10} />
        {banks.length
          ? `samples: ${banks.join(', ')}`
          : 'no sample banks vendored \u2014 run `npm run fetch-samples` (synths still work)'}
      </div>
    </div>
  );
}
