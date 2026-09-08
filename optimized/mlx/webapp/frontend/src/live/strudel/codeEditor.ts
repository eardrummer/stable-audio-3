/**
 * CodeMirror 6 editor for Strudel patterns.
 *
 * All @codemirror/* packages are MIT, so unlike strudelEngine.ts this module
 * carries no licence constraint. It is still behind a dynamic import so the
 * editor bundle only loads when Live Code mode is opened.
 *
 * Strudel patterns are plain JavaScript, so the stock JS language mode gives
 * correct highlighting for strings, numbers, calls and comments.
 */

export interface EditorHandle {
  getCode(): string;
  setCode(code: string): void;
  focus(): void;
  destroy(): void;
}

export interface EditorOptions {
  parent: HTMLElement;
  doc: string;
  onChange(code: string): void;
  onRun(): void;
  onHush(): void;
}

export async function createEditor(opts: EditorOptions): Promise<EditorHandle> {
  const [{ EditorState }, view, { defaultKeymap, history, historyKeymap },
         { javascript }, { syntaxHighlighting, HighlightStyle, bracketMatching, indentOnInput },
         { closeBrackets, closeBracketsKeymap, autocompletion },
         { tags }] = await Promise.all([
    import('@codemirror/state'),
    import('@codemirror/view'),
    import('@codemirror/commands'),
    import('@codemirror/lang-javascript'),
    import('@codemirror/language'),
    import('@codemirror/autocomplete'),
    import('@lezer/highlight'),
  ]);
  const { EditorView, keymap, lineNumbers, highlightActiveLine, drawSelection } = view;

  // Muted greens to sit against the waveform showing through behind the panel.
  const highlight = HighlightStyle.define([
    { tag: tags.comment, color: '#5c6f5c', fontStyle: 'italic' },
    { tag: tags.string, color: '#9ed89e' },
    { tag: tags.number, color: '#e0c07a' },
    { tag: tags.keyword, color: '#7fb0e0' },
    { tag: tags.operator, color: '#a8a8a8' },
    { tag: [tags.function(tags.variableName), tags.labelName], color: '#8ad18a' },
    { tag: tags.propertyName, color: '#cfe8cf' },
    { tag: tags.variableName, color: '#d8e8d8' },
    { tag: tags.bracket, color: '#7d7d7d' },
  ]);

  const theme = EditorView.theme({
    '&': { height: '100%', fontSize: '26px', backgroundColor: 'transparent', color: '#d8e8d8' },
    '.cm-scroller': {
      fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
      lineHeight: '1.55',
      overflow: 'auto',
    },
    '.cm-content': { caretColor: '#8ad18a' },
    '&.cm-focused': { outline: 'none' },
    '.cm-gutters': { backgroundColor: 'transparent', color: '#3d4a3d', border: 'none' },
    '.cm-activeLine': { backgroundColor: 'rgba(138,209,138,.05)' },
    '.cm-selectionBackground, &.cm-focused .cm-selectionBackground': {
      backgroundColor: 'rgba(138,209,138,.22)',
    },
  }, { dark: true });

  const state = EditorState.create({
    doc: opts.doc,
    extensions: [
      lineNumbers(),
      history(),
      drawSelection(),
      indentOnInput(),
      bracketMatching(),
      closeBrackets(),
      autocompletion(),
      highlightActiveLine(),
      javascript(),
      syntaxHighlighting(highlight),
      theme,
      EditorView.lineWrapping,
      // Run/hush bindings come first so they win over the defaults.
      keymap.of([
        { key: 'Mod-Enter', preventDefault: true, run: () => { opts.onRun(); return true; } },
        { key: 'Mod-.', preventDefault: true, run: () => { opts.onHush(); return true; } },
        ...closeBracketsKeymap, ...defaultKeymap, ...historyKeymap,
      ]),
      EditorView.updateListener.of((u) => { if (u.docChanged) opts.onChange(u.state.doc.toString()); }),
    ],
  });

  const ed = new EditorView({ state, parent: opts.parent });

  return {
    getCode: () => ed.state.doc.toString(),
    setCode: (code: string) =>
      ed.dispatch({ changes: { from: 0, to: ed.state.doc.length, insert: code } }),
    focus: () => ed.focus(),
    destroy: () => ed.destroy(),
  };
}
