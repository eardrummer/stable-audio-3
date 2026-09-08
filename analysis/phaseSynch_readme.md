# Loop Phase Synchronisation

How the live app aligns its analysis window to a looping source, why it is built
the way it is, and what it still does not fix.

Applies to the three input modes that repeat a fixed length of audio: **File
Loop**, a **recorded loop** played back in Live Input, and **Live Code**
(Strudel). It does nothing in plain Live Input, where there is no loop to align
to.

---

## 1. The Problem

The live app runs two clocks that were never connected to each other.

**The snapshot timer is anchored to START.** Pressing START begins a
`setTimeout` chain that fires every `generateRate` seconds. Each firing takes
*"the last `inputRate` seconds ending right now"*, reading backwards from the
ring buffer's write head.

**The loop is anchored to whenever it began** — when PLAY was pressed on a
recording, when the file started, or when the Strudel pattern was evaluated.

Nothing ties the two together, so the window boundary lands at an arbitrary
point inside the loop.

### Two distinct symptoms

They feel like one problem but behave differently, and only one of them is
fixed here.

**A. Constant rotation** — when the loop period divides evenly into the window.
Every window starts at the same wrong place and stays there. A 4 s loop, an 8 s
window and a loop that began 1.3 s after START:

```
push 0: window starts 2.70s into the loop
push 1: window starts 2.70s into the loop
push 2: window starts 2.70s into the loop
```

The result is stable but permanently "turned around" — the model always sees the
loop beginning from its middle.

**B. Drift** — when the period does not divide evenly. The downbeat walks
through the window. A 3.7 s recording in an 8 s window:

```
push 0: window starts 2.40s into the loop, ends 3.00s in
push 1: window starts 3.00s into the loop, ends 3.60s in
push 2: window starts 3.60s into the loop, ends 0.50s in
push 3: window starts 0.50s into the loop, ends 1.10s in
```

**This case is guaranteed for a free-form recorded loop or an arbitrary audio
file**, because neither is a whole number of beats.

---

## 2. The Decision: Align the Window, Not the Timer

The obvious fix is to re-time the generation clock so pushes happen on loop
boundaries. That was rejected: it couples Generate Rate to loop length, so you
lose independent control of how often you generate, and a cycle that overruns
pushes the next one off the grid.

Instead, **the extraction point moves rather than the schedule.** The ring
buffer already holds up to `MAX_BUFFER_SECS` (32 s) of history, so a window can
be read that *ends in the past*, on the most recent loop boundary:

```
phase = (now − loopOrigin) mod loopPeriod
end   = ring.w − phase · sampleRate
start = end − inputRate · sampleRate
```

Verified against case B above — the window now ends on the boundary every time
instead of walking:

```
push 0: phase 3.00s -> rewind 3.00s -> window ends 0.000s into loop
push 1: phase 3.60s -> rewind 3.60s -> window ends 0.000s into loop
push 2: phase 0.50s -> rewind 0.50s -> window ends 0.000s into loop
push 3: phase 1.10s -> rewind 1.10s -> window ends 0.000s into loop
```

### Why this shape

* **The generation cadence is untouched.** Generate Rate stays free and
  independent of loop length; no jitter is introduced when a cycle overruns.
* **One mechanism covers all three sources.** Only the origin and period differ;
  the rewind is identical.
* **It degrades safely.** If the history needed is not available, it falls back
  to the previous behaviour rather than reading stale or wrapped samples.

---

## 3. Phase Anchors Per Source

Each source needed a different answer to "when did the current repetition
start", and one of them had to be newly recorded.

| Source | Origin | Period | Status |
| :--- | :--- | :--- | :--- |
| Recorded loop | `loopStartedAtRef`, set when playback begins | `trim[1] − trim[0]` | already existed |
| File Loop | `ctx.currentTime − startOffset` | `fileBuffer.duration` | **added** — the start time was previously discarded |
| Live Code | `scheduler.now() % 1` | `1 / cps` | **added** via `getCyclePhase()` |

The Strudel case is worth noting: `Cyclist.now()` returns a **fractional cycle
position**, not seconds — its implementation is `lastBegin + t * cps`. So the
phase in seconds is `(now() % 1) / cps`. Reading it as seconds would have
produced a plausible-looking but entirely wrong offset.

---

## 4. Implementation

| Piece | Location |
| :--- | :--- |
| `syncLoop` state and the SYNC toggle | `LiveApp.tsx` |
| `loopPhase()` — per-source phase | `LiveApp.tsx` (~L538) |
| Rewind applied to the ring read | `LiveApp.tsx` (~L562) |
| File loop origin captured at start | `LiveApp.tsx` (~L875) |
| `getCyclePhase()` | `strudel/strudelEngine.ts` (~L98) |

**SYNC is off by default.** The toggle sits beside the source buttons and only
appears when a loop is available. It was left opt-in for two reasons: existing
behaviour stays untouched so the change can be A/B'd by ear, and a deliberate
mis-alignment is occasionally musically interesting.

The status bar reports the current rewind, e.g. `sync −1.10s`, so the amount of
added latency is always visible rather than implicit.

### The guard

```
if (want + back <= ring.filled && want + back <= ring.l.length) back = b;
```

If the rewind would need more history than the ring holds, alignment is skipped
for that cycle. This happens briefly just after START, and permanently if
`inputRate + loopPeriod` exceeds the 32 s ring — for instance a 32-beat window
at a slow tempo combined with a long file.

---

## 5. Limitations

### It fixes phase, not period

Ending on a boundary does not make the window contain a whole number of loops.
A 3.7 s loop inside an 8 s window is 2.16 repetitions, so the window still
*starts* mid-loop even though it now *ends* cleanly. The audible improvement is
that the rotation stops moving; it does not vanish.

Fixing this properly requires snapping Input Rate to a whole number of loop
periods, so the knob would read `2 loops · 7.4s` instead of `16b · 8.0s`. That
was deliberately deferred, because it means Input Rate stops being
beat-denominated whenever a loop is the source — which changes what a
MIDI-mapped Input Rate knob does mid-performance.

Live Code is the easy case: if the pattern is four cycles and the window is
sixteen beats at the same tempo, the periods already agree and only phase was
ever wrong.

### Residual ~85 ms offset

The ring buffer is filled by a `ScriptProcessor` in 4096-sample blocks, so the
write head lags real time by up to one block — **85 ms at 48 kHz**, roughly a
sixteenth note at 120 BPM. This is *not* compensated for. It is a constant
offset rather than a drift, so it can be corrected with a single subtraction
once it is clear by ear which direction it errs; it was left alone rather than
guessed at.

### Assumes a steady period

The mechanism assumes the source actually repeats at a fixed length. A Strudel
pattern using randomness, or one whose musical phrase spans several cycles,
aligns only to the single-cycle grid — which may not be the boundary that
matters musically. Changing the tempo mid-performance also moves the period, so
the phase re-derives on the next push rather than interpolating.

### Not verified by ear

The alignment arithmetic is verified numerically and the per-source anchors are
read from live state, but nothing here has been confirmed against actual audio.
File Loop is the best first test, since its period is known exactly and the
effect should be most obvious.

---

## 6. Summary

SYNC removes the arbitrary relationship between when START was pressed and where
a loop happens to be, at the cost of up to one loop period of extra latency. It
turns a moving misalignment into no misalignment for commensurate material, and
into a stationary one for material that is not a whole number of beats. The
remaining gap — windows that contain fractional loops — is a separate decision
about whether Input Rate should be denominated in loops rather than beats.

---

## 7. Future Work: Inter-Track Synchronisation

Sections 1–6 align a **single** track's analysis window to its input loop. A
second, independent problem appears once there are **two output tracks**: they
start together but do not stay together. This is unrelated to SYNC and is not
addressed by it.

### It is jitter, not accumulating drift

The distinction matters, because it rules out the obvious fix of a fixed offset.

Playback is re-triggered from the start of each clip on every cycle
(`crossfadeTo(buf, 0)`, `TrackOutput.tsx:236`), so nothing carries over between
cycles and the error does not grow. What changes is *where within each cycle*
each track begins — it resets and re-randomises every time. That is harder to
listen past than a steady drift would be, and it cannot be dialled out with a
constant correction.

### Three mechanisms, largest first

**1. Playback is slaved to arrival time.** A track begins the instant its bytes
land on its socket. Both tracks are handed the same snapshot at the same moment,
but their audio returns at different times because generation serializes on the
shared `_ENGINE_LOCK` (`live.py:45`). Measured:

```
1 track : 0.36 s/cycle
2 tracks: 0.87 s/cycle   -> the second waits ~0.51 s for the lock
                         -> ~1.02 beats at 120 BPM
```

**2. The gap does not stay constant.** Python's `threading.Lock` is not FIFO, so
which track wins can flip between cycles. The tracks also have independent
Steps, CFG and variant-count knobs, and a track running the 5-variant ladder
takes roughly five times as long. Both the *order* and the *size* of the offset
therefore vary.

**3. Each track owns a separate AudioContext** (`TrackOutput.tsx:94`). Their
`currentTime` bases are not comparable, so there is no shared clock to schedule
against, and the two can drift slowly relative to one another. This is genuine
drift, but far slower than (1). It was a simplification made while extracting
the component, and it is the item that blocks every real fix.

A fourth, coarser mode: if one track **drops** a snapshot and the other does not,
the dropping track is generating from *older input*. The two then differ by a
whole cycle of content, not merely by timing.

### The content is already aligned

Both tracks generate from the **identical snapshot**, so their audio represents
the same musical moment. Nothing is wrong with what is produced — this is purely
a playback-scheduling problem, which means it is fully solvable rather than
merely approximable.

### Proposed direction

**Step 0 — one shared AudioContext.** A prerequisite for everything below.
Context ownership moves up to `LiveApp` and is passed into each track. Without a
common timebase there is nothing to schedule against.

**Step 1 — client-side barrier with a scheduled start.** Hold each cycle's audio
until every track's audible lane has arrived, then start them all at a single
explicit `ctx.currentTime + lead`. This is sample-accurate, rather than merely
"the same JavaScript tick".

* Cost: the faster track waits for the slower — roughly +0.5 s today.
* Needs a timeout, so one stalled track cannot freeze the other.
* Practical consequence: the wait is bounded by the slowest track, which is a
  concrete argument for keeping the 5-variant ladder off on at least one track.
  With it enabled on both, the barrier would wait on the order of four seconds.

**A complication to design around: cycle numbers can diverge.** `_cycle` is per
session and increments only when a snapshot is actually processed
(`live.py:323`); a dropped snapshot does not advance it. If one track drops, its
numbering falls permanently behind the other's, and a barrier keyed on
`hdr.cycle` would pair clips from different moments. The fix is for the client to
stamp each snapshot with an id in `audio_header` and for the backend to echo it
back in the variant header — a small change, but one the barrier depends on.

### Alternatives considered

| Approach | Why not (yet) |
| :--- | :--- |
| Backend barrier — sessions rendezvous before sending | Breaks the per-session independence that gives each track its own loopback memory and latent cache, and still leaves the two-context problem |
| One session generating both tracks | Deterministic ordering and a single upload, but the largest refactor, and it re-entangles state that was deliberately separated |
| Compensate with the Nudge knob | Only works against a *constant* offset; useless while the offset varies cycle to cycle. Viable only after ordering is made deterministic |
| Force equal generation times on both tracks | Narrows the gap but does not close it, since lock ordering still varies |

### Open question

How tightly do the two tracks actually need to lock? If they are independent
textures, roughly a beat of slop may be acceptable and this is a large amount of
machinery for little gain. If they are meant to be rhythmically locked — which is
what the tempo work and the beat-denominated rate knobs imply — then the barrier
is the right investment.
