/**
 * The score recorder: a timestamped account of what was done to the rack during
 * a take, written out as a file you can read.
 *
 * It is NOT an audio recording and not an automation track. The point is to be
 * able to look back at a performance and see what was changed, when, and why —
 * so every entry carries the control's readable name, what it moved from and to,
 * and who moved it: you at a knob, you at a controller, or the system deriving
 * something on your behalf.
 *
 * Framework-free on purpose: all the awkwardness here is in coalescing a knob
 * drag into one entry, and that is worth being able to test without a browser.
 */

export type Src = 'ui' | 'midi' | 'system';

/** How often a moving value is sampled, and how long a control has to be still
 *  before the move counts as finished. A drag emits a change per frame, so
 *  without the first number a four-second sweep would be 240 entries; without
 *  the second, a pause mid-sweep would split one move into two. */
export const SAMPLE_MS = 100;
export const GESTURE_END_MS = 300;

export interface ScoreEvent {
  /** mm:ss.s from the start of the take — the readable one. */
  t: string;
  /** The same instant in milliseconds, for anything that wants to compute. */
  ms: number;
  /** The control's name as it appears in the rack, not its internal key. */
  what: string;
  /** Which track or input box it belongs to, when it belongs to one. */
  scope?: string;
  by: Src;
  /** What triggered a `system` change — FREEZE, SYNC, a preset recall. */
  cause?: string;
  from?: unknown;
  to?: unknown;
  /** Seconds the move took. Continuous controls only. */
  over?: number;
  /** [seconds from the move's start, value] while it was moving. Keeps the
   *  shape of a sweep, so a slow ramp and a jump-then-hold read differently. */
  path?: [number, number][];
}

export interface Take {
  started: string;
  duration: string;
  durationMs: number;
  openingState: unknown;
  events: ScoreEvent[];
}

/** mm:ss.s — the form a score is read in. */
export const stamp = (ms: number): string => {
  const total = Math.max(0, ms) / 1000;
  const m = Math.floor(total / 60);
  const s = total - m * 60;
  return `${String(m).padStart(2, '0')}:${s.toFixed(1).padStart(4, '0')}`;
};

interface Gesture {
  key: string;
  meta: Omit<ScoreEvent, 't' | 'ms' | 'from' | 'to' | 'over' | 'path'>;
  startedAt: number;
  from: number;
  latest: number;
  lastAt: number;
  lastSampleAt: number;
  path: [number, number][];
}

export class Recorder {
  private t0 = 0;
  private startedIso = '';
  private events: ScoreEvent[] = [];
  private open = new Map<string, Gesture>();
  private opening: unknown = null;
  recording = false;

  start(now: number, isoNow: string, openingState: unknown) {
    this.t0 = now;
    this.startedIso = isoNow;
    this.events = [];
    this.open.clear();
    this.opening = openingState;
    this.recording = true;
  }

  /** A discrete change: a button, a toggle, a menu, a prompt. */
  point(
    now: number,
    what: string,
    to: unknown,
    opts: { scope?: string; by?: Src; cause?: string; from?: unknown } = {},
  ) {
    if (!this.recording) return;
    this.events.push({
      t: stamp(now - this.t0),
      ms: Math.round(now - this.t0),
      what,
      ...(opts.scope ? { scope: opts.scope } : {}),
      by: opts.by ?? 'ui',
      ...(opts.cause ? { cause: opts.cause } : {}),
      ...(opts.from !== undefined ? { from: opts.from } : {}),
      to,
    });
  }

  /**
   * A continuous change. Successive calls for the same key inside
   * GESTURE_END_MS are one move; the value is sampled at SAMPLE_MS so the path
   * keeps the shape of the sweep without one entry per frame.
   */
  move(
    now: number,
    key: string,
    what: string,
    from: number,
    to: number,
    opts: { scope?: string; by?: Src; cause?: string } = {},
  ) {
    if (!this.recording) return;
    const g = this.open.get(key);
    if (g && now - g.lastAt <= GESTURE_END_MS) {
      g.latest = to;
      g.lastAt = now;
      if (now - g.lastSampleAt >= SAMPLE_MS) {
        g.path.push([Number(((now - g.startedAt) / 1000).toFixed(3)), to]);
        g.lastSampleAt = now;
      }
      return;
    }
    if (g) this.close(g);          // the gap says the previous move is over
    this.open.set(key, {
      key,
      meta: {
        what,
        ...(opts.scope ? { scope: opts.scope } : {}),
        by: opts.by ?? 'ui',
        ...(opts.cause ? { cause: opts.cause } : {}),
      },
      startedAt: now,
      from,
      latest: to,
      lastAt: now,
      lastSampleAt: now,
      path: [[0, to]],
    });
  }

  /** Close any move that has gone quiet. Called on a timer so a finished sweep
   *  lands in the score without waiting for the next thing to happen. */
  flush(now: number) {
    for (const g of [...this.open.values()]) {
      if (now - g.lastAt > GESTURE_END_MS) this.close(g);
    }
  }

  private close(g: Gesture) {
    this.open.delete(g.key);
    const last = g.path[g.path.length - 1];
    if (!last || last[1] !== g.latest) {
      g.path.push([Number(((g.lastAt - g.startedAt) / 1000).toFixed(3)), g.latest]);
    }
    this.events.push({
      t: stamp(g.startedAt - this.t0),
      ms: Math.round(g.startedAt - this.t0),
      ...g.meta,
      from: g.from,
      to: g.latest,
      over: Number(((g.lastAt - g.startedAt) / 1000).toFixed(3)),
      // A move that never went anywhere is a tap, not a sweep.
      ...(g.path.length > 1 ? { path: g.path } : {}),
    });
  }

  stop(now: number): Take {
    for (const g of [...this.open.values()]) this.close(g);
    // Moves are emitted when they end, so they can land after a later point
    // event; the score has to read in the order things happened.
    const events = [...this.events].sort((a, b) => a.ms - b.ms);
    this.recording = false;
    return {
      started: this.startedIso,
      duration: stamp(now - this.t0),
      durationMs: Math.round(now - this.t0),
      openingState: this.opening,
      events,
    };
  }
}
