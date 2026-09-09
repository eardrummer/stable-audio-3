/**
 * The ONLY module that imports Strudel.
 *
 * Strudel and superdough are AGPL-3.0-or-later, while the rest of this project
 * is not. Every Strudel import is confined here and reached through a dynamic
 * `import()`, so the code is loaded only when Live Code mode is actually used
 * and the licence boundary stays visible in one file. Deleting this module and
 * the Live Code branch in LiveApp removes the dependency entirely.
 *
 * What this does that a plain Strudel embed does not: it captures Strudel's
 * audio instead of letting it reach the speakers directly, so the patterns you
 * write become the *input* to the diffusion model.
 */

/** 1 cycle is read as one 4/4 bar, so the default cps of 0.5 is 120 BPM. */
export const BEATS_PER_CYCLE = 4;

export interface StrudelHandle {
  /** superdough's AudioContext, adopted as the live app's input context. */
  ctx: AudioContext;
  /** Every Strudel voice arrives here instead of at the speakers. */
  tap: GainNode;
  evaluate(code: string): Promise<void>;
  hush(): void;
  /** Tempo in BPM, derived from Strudel's cycles-per-second. */
  getBpm(): number | null;
  setBpm(bpm: number): void;
  /** Re-assert the output interception (superdough restores it on reset). */
  reattach(): void;
  dispose(): void;
  sampleBanks: string[];
}

let handle: StrudelHandle | null = null;
let loading: Promise<StrudelHandle> | null = null;

/** Load Strudel once and wire its output into our graph. Safe to call again. */
export function loadStrudel(): Promise<StrudelHandle> {
  if (handle) return Promise.resolve(handle);
  if (loading) return loading;

  loading = (async () => {
    const S: any = await import('@strudel/web');

    // Adopt superdough's context rather than making our own: one context means
    // one clock and no cross-context resampling between Strudel and capture.
    const ctx: AudioContext = S.getAudioContext();
    const tap = ctx.createGain();
    tap.gain.value = 1;

    // Vendored banks, fetched by `npm run fetch-samples`. Absent is fine —
    // oscillator sounds need no samples at all.
    let sampleBanks: string[] = [];
    const prebake = async () => {
      try {
        const res = await fetch('/strudel-samples/banks.json');
        if (!res.ok) return;
        sampleBanks = await res.json();
        await Promise.all(
          sampleBanks.map((b) => S.samples(`/strudel-samples/${b}.json`)),
        );
      } catch {
        sampleBanks = [];   // offline and un-vendored: synths still work
      }
    };

    const repl = await S.initStrudel({ prebake });

    // Superdough ends its chain at `destinationGain.connect(ctx.destination)`.
    // Re-pointing that public node is the whole interception.
    const reattach = () => {
      const out = S.getSuperdoughAudioController()?.output;
      if (!out?.destinationGain) return;
      try { out.destinationGain.disconnect(); } catch { /* not yet connected */ }
      out.destinationGain.connect(tap);
    };
    reattach();

    handle = {
      ctx,
      tap,
      sampleBanks,
      async evaluate(code: string) { await repl.evaluate(code); reattach(); },
      hush() { try { repl.stop(); } catch { /* not running */ } },
      getBpm() {
        const cps = repl?.scheduler?.cps;
        return typeof cps === 'number' ? cps * 60 * BEATS_PER_CYCLE : null;
      },
      setBpm(bpm: number) {
        const cps = bpm / (60 * BEATS_PER_CYCLE);
        if (repl?.scheduler?.setCps) repl.scheduler.setCps(cps);
        else S.setcps?.(cps);
      },
      reattach,
      dispose() { try { repl.stop(); } catch { /* already stopped */ } },
    };
    return handle;
  })();

  return loading;
}

export function peekStrudel(): StrudelHandle | null { return handle; }
