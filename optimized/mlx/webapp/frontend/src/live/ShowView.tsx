import { useCallback, useEffect, useRef, useState } from 'react';
import { Minimize2 } from 'lucide-react';

/**
 * The performance view: one screen, a line, and whatever is happening on it.
 *
 * A single red line across the middle is the live input. Every TRACE_EVERY a
 * snapshot of the generated output is laid down on that line and recedes — up
 * the screen and away from the viewer at once — fading as it goes. So the
 * picture reads as cause and effect: what you are playing is the line, and what
 * the model made of it lifts off it and travels away.
 *
 * Running on the line is a stick figure whose pace is your input level. After a
 * couple of minutes of playing, blocks start arriving from the right and it has
 * to jump or duck; it dodges on its own, imperfectly, and a collision drops it
 * back to an empty line to start again. Ghosts of it — smaller, robot-headed —
 * trail behind whenever output is being heard, and pass straight through the
 * blocks.
 *
 * Presentational only. It knows nothing about the audio graph: LiveApp hands it
 * two readers that fill a Float32Array with -1..1 samples.
 */

interface Props {
  /** Fills `into` with the summed live input. False when there is nothing. */
  inputWave: (into: Float32Array) => boolean;
  /** Fills `into` with the summed output of every track. False when silent. */
  outputWave: (into: Float32Array) => boolean;
  /** The prompts actually being heard, in track order. Joined with `+` above
   *  the line; a track turned down or silenced by another's solo is not here. */
  prompts: string[];
  onExit: () => void;
}

// ── the waveform traces ──────────────────────────────────────────────────
// Traces on screen is LIFE_MS / TRACE_EVERY, and each is a POINTS-segment path
// stroked every frame. At 40 x 256 that is ~10k segments, kept deliberately
// modest because this canvas shares a thread with the audio processors.
/** Points per trace — more than the eye resolves at this line weight. */
const POINTS = 256;
/** How often a trace is laid down, and how long it takes to rise and vanish. */
const TRACE_EVERY = 90;
const LIFE_MS = 3600;
/** Share of the half-screen a trace has climbed by the end of its life. */
const TRAVEL = 0.94;
/**
 * How hard the perspective bites. A trace is projected toward a vanishing point
 * above the input line, so `1 / (1 + z·DEPTH)` scales its width, its height and
 * its amplitude together — the real projective falloff, which is why traces
 * bunch up and slow as they approach the horizon instead of sliding off at a
 * constant rate. 0 would be the flat version.
 */
const DEPTH = 1.1;
/** Full scale as a share of the viewport height, for both the line and the traces. */
const AMP = 0.195;
/** Below this peak a trace is silence, and drawing it would stack flat lines up
 *  the screen. After a Flush the screen empties instead. */
const SILENCE = 0.002;

const INK = '214, 40, 40';        // the receding traces, the blocks, the ghosts
const LIVE = '#ff2f2f';           // the input line, always the brightest mark

// ── the runner ───────────────────────────────────────────────────────────
/** Figure height, as a share of the viewport. */
const FIG = 0.10;
/** Where it holds station once the course arms, as a share of the width. The
 *  world scrolls past it, and the blocks are anchored here rather than to the
 *  figure — so when it darts aside it really does move out of their way. */
const MAN_X = 0.5;
/** How often it leaves the station for a dart forward or back, and how far. */
const DART_CHANCE = 0.1, DART_REACH = 0.12;
const DART_MIN_MS = 2500, DART_SPAN_MS = 3500;
/**
 * Pace at silence and at full level, in the figure's own heights per second —
 * not screen widths, which is what makes the gait look right at any aspect
 * ratio. A person is around 1.7 m running 3 m/s, so ~1.8 heights/s. The floor
 * is a slow walk and the ceiling a sprint, and loudness picks the point between
 * them — so a soft touch really does amble and a moderate one really does run.
 */
const SPEED_MIN = 0.9, SPEED_MAX = 4.6;
/**
 * The loudness window the pace is spread across, in dBFS RMS. It is a dB scale
 * because linear RMS spends almost all of its range on loud signals: from -48
 * to -26 dBFS — the whole of soft playing — a linear map moved the pace by
 * about a fifth of its range, which is why it read as unrelated to how hard you
 * were playing. The window is placed so that the quietest thing loud enough to
 * make the figure appear at all is still a walk, and a moderate -14 dBFS RMS is
 * already a hard run.
 */
const DB_SOFT = -42, DB_FULL = -10;
/**
 * Heights covered per leg cycle, walking and running. Keeps the feet from
 * sliding, since the cycle is driven by distance travelled and this is the
 * conversion — and because a real stride lengthens with pace, interpolating
 * between the two by loudness is what makes a slow pace read as a walk rather
 * than as a lope in slow motion.
 */
const STRIDE_WALK = 0.9, STRIDE_RUN = 1.55;
/** A signal quieter than this is not playing. Roughly -40 dBFS. */
const PRESENT = 0.01;
/** Per-frame approach rate for the two fades, about 150 ms. */
const FADE = 0.12;
/** How far the pace drifts on its own, over and above the input level. */
const PACE_DRIFT = 0.3;

// ── the obstacle course ──────────────────────────────────────────────────
/** Seconds of *playing* — not wall clock — before blocks start arriving, and
 *  how long from then to reach RAMP_MAX pace. A collision resets both. */
const CALM_S = 30, RAMP_S = 180, RAMP_MAX = 2;
/** Spacing between blocks in figure heights at ramp 1, tightening as it climbs. */
const GAP_H = 14;
/** Low block: sits on the line, must be jumped. */
const OBS_W = 0.5, OBS_H = 0.35;
/**
 * A floating block hangs from OBS_HIGH_BOTTOM and is OBS_HIGH_H tall. It has to
 * be too tall to jump, or ducking would never be needed: at 0.35 tall the jump
 * sailed straight over it and the crouch was decoration. At 1.4 the only way
 * through is under it, with 0.07 heights to spare.
 */
const OBS_HIGH_BOTTOM = 0.62, OBS_HIGH_H = 1.4;
/** Jump height, crouch height, and how long a stumble lasts. */
const JUMP_H = 1.0, DUCK_SCALE = 0.55, HIT_MS = 900;
/**
 * A move lasts as long as the block is in contact (0.86 heights of travel),
 * rather than a fixed time. A fixed 620 ms jump was shorter than the contact
 * time below about 1.8 heights/s, which made every low block an unavoidable hit
 * at a gentle pace. A jump is only above clearance for 81% of its arc, hence
 * the larger of the two margins.
 */
const MOVE_MIN_MS = 400, MOVE_MAX_MS = 1500;
const JUMP_MARGIN = 1.42, DUCK_MARGIN = 1.3;
/**
 * It commits to a move when the block is this fraction of the move's own
 * duration away, scaled by a per-block jitter — so some moves are early, some
 * late, and a few land badly. MISS_CHANCE is the share it simply fails to see.
 * Together these are what make the respawn ever happen.
 */
const LEAD_JITTER = 0.45, MISS_CHANCE = 0.03;
/** Half-width of the figure's collision box, in heights. */
const MAN_HALF = 0.18;
/** It only breaks into a dance when the next block is at least this far off.
 *  Has to be under the tightest spacing (7 heights at full ramp) or it could
 *  never dance once the course is armed. */
const DANCE_SAFE_H = 6;
const DANCE_MS = 2600, DANCE_GAP_MS = 7000;

// ── the ghosts ───────────────────────────────────────────────────────────
/** Lag behind the runner in seconds, with its alpha. While he is there they sit
 *  on the same line at the distance he covered in that time; with no input they
 *  are on their own and the lag means nothing. */
const GHOSTS: [number, number][] = [
  [0.28, 0.60], [0.56, 0.46], [0.84, 0.35],
  [1.12, 0.26], [1.40, 0.19], [1.68, 0.13],
];
/** Ghosts are smaller than the figure they echo. */
const GHOST_SCALE = 0.7;
/** Trail history: enough frames to cover the longest ghost lag. */
const TRAIL = 128;
/** With no human on the line, a ghost wanders at this share of his pace, and
 *  reconsiders its direction somewhere in this window. */
const WANDER = 0.8;
const TURN_MIN_MS = 1200, TURN_SPAN_MS = 3000;
/** How fast a ghost falls back into formation when he reappears. */
const CATCH_UP = 8;
/**
 * A robot also stands somewhere in depth, in the same units as a trace's age:
 * 0 is the live input line, 1 is the horizon. It drifts toward a fresh target
 * every few seconds, so at any moment some are down on the line and others are
 * back among the receding waveforms, crossing between them. The cap keeps them
 * short of the vanishing point, where they would be a few pixels of nothing.
 */
const GHOST_Z_MAX = 0.82;
const GHOST_Z_EASE = 0.5;                       // per second, toward the target
const GHOST_Z_MIN_MS = 2600, GHOST_Z_SPAN_MS = 5200;
/** Depth dims a robot, but not all the way down to the trace fade: the tail
 *  ghosts are already faint, and the point is to see them back there. */
const GHOST_Z_DIM = 0.55;

interface Trace { pts: Float32Array; born: number }
/** A robot resolved for this frame, held back until the depth-ordered pass. */
interface GhostDraw {
  /** Depth, in the same units as a trace's age: 0 the line, 1 the horizon. */
  z: number;
  x: number; lift: number; h: number;
  pose: Pose; alpha: number; face: number;
}
interface Obstacle {
  /** World distance at which this block meets the runner. */
  at: number;
  kind: 'low' | 'high';
  /** Rolled once: how well it will be read. */
  lead: number;
  miss: boolean;
  seen: boolean;
}

/**
 * A figure's joints, so every behaviour is a set of numbers rather than its own
 * drawing routine. Limb angles are radians from straight down, positive
 * forward; `hip` lifts the body in figure heights; `lean` tilts the torso.
 *
 * The legs are exactly long enough to reach the line when vertical, which is
 * what plants the feet: a swung leg is foreshortened by its own cosine and so
 * lifts clear on its own, with no special case.
 */
interface Pose {
  legs: [number, number];
  arms: [number, number];
  hip: number;
  lean: number;
}

/**
 * Perceived loudness, 0..1, from an RMS. Everything that answers to how hard
 * you are playing goes through here.
 */
const loudness = (rms: number) => (rms <= 0 ? 0 : Math.max(0, Math.min(1,
  (20 * Math.log10(rms) - DB_SOFT) / (DB_FULL - DB_SOFT))));

/** `drive` is that loudness: the limbs swing wider and the torso leans further
 *  the harder you play, so the gait itself moves from a walk to a run rather
 *  than the same walk played faster. */
const runPose = (p: number, drive = 1): Pose => ({
  legs: [(0.45 + 0.4 * drive) * Math.sin(p), (0.45 + 0.4 * drive) * Math.sin(p + Math.PI)],
  arms: [(0.6 + 0.55 * drive) * Math.sin(p + Math.PI), (0.6 + 0.55 * drive) * Math.sin(p)],
  hip: -(0.02 + 0.04 * drive) * Math.abs(Math.sin(p * 2)),
  lean: 0.07 + 0.17 * drive,
});

/**
 * The dance vocabulary. A bout picks one and holds it, so a move reads as a
 * move rather than as a twitch, and the phase is the same one the gait uses —
 * the tempo of a dance is still the tempo of the playing. The view is side-on,
 * so every move has to be legible in profile: "side to side" becomes forward
 * and back, which is why these are built out of fore-and-aft weight shifts.
 *
 * `rate` is cycles per second, because a kick wants longer than a shuffle.
 */
const DANCES: { pose: (p: number) => Pose; rate: number }[] = [
  // Arms overhead, knees bouncing, a step side to side. The original.
  {
    rate: 1.8,
    pose: (p) => ({
      legs: [0.32 * Math.sin(p), -0.32 * Math.sin(p)],
      arms: [2.4 + 0.45 * Math.sin(p * 2), 2.4 - 0.45 * Math.sin(p * 2)],
      hip: -0.08 * Math.abs(Math.sin(p)),
      lean: 0.12 * Math.sin(p),
    }),
  },
  // Two-step: small alternating steps with the arms swinging against them, the
  // way an arm swing actually opposes the leg on the same side.
  {
    rate: 2.3,
    pose: (p) => ({
      legs: [0.52 * Math.sin(p), 0.52 * Math.sin(p + Math.PI)],
      arms: [0.95 * Math.sin(p + Math.PI), 0.95 * Math.sin(p)],
      hip: -0.05 * Math.abs(Math.sin(p)),
      lean: 0.1 * Math.sin(p) + 0.05,
    }),
  },
  // Alternating high kicks. The planted leg stays under the hip and the arms go
  // back as counterweight, or the whole figure reads as falling over.
  {
    rate: 1.2,
    pose: (p) => {
      const k = Math.sin(p), lead = k > 0;
      const swing = 1.35 * Math.abs(k);
      return {
        legs: lead ? [swing, -0.12] : [-0.12, swing],
        arms: [-0.5 - 0.5 * Math.abs(k), 0.8 + 0.5 * Math.abs(k)],
        hip: -0.04 * Math.abs(k),
        lean: -0.14 * Math.abs(k),
      };
    },
  },
  // Stiff right angles held on the beat, nodding at the robots in the tail. The
  // quantised phase is the whole move: it steps between positions instead of
  // sweeping through them, which is what reads as mechanical.
  {
    rate: 2.0,
    pose: (p) => {
      const q = Math.floor((p / (Math.PI * 2)) * 8) % 8;
      const flip = q % 2 ? 1 : -1;
      return {
        legs: [0.16 * flip, -0.16 * flip],
        arms: q < 4 ? [Math.PI / 2, -Math.PI / 2] : [Math.PI * 0.92, 0.1],
        hip: q % 4 < 2 ? 0 : -0.05,
        lean: 0,
      };
    },
  },
  // A jumping jack, read in profile: the legs split fore and aft while both
  // arms sweep from the hips to overhead together, hips lifting on the open.
  {
    rate: 1.5,
    pose: (p) => {
      const open = (1 - Math.cos(p)) / 2;          // 0 closed, 1 open
      return {
        legs: [0.62 * open, -0.62 * open],
        arms: [0.25 + 2.85 * open, 0.25 + 2.85 * open],
        hip: -0.1 * open,
        lean: 0.03,
      };
    },
  },
  // One arm rolling through a full overhead circle while the hips push the
  // other way. The only move where the two arms are doing different jobs.
  {
    rate: 0.9,
    pose: (p) => ({
      legs: [0.2 * Math.sin(p * 2), -0.2 * Math.sin(p * 2)],
      arms: [((p % (Math.PI * 2)) + Math.PI * 0.4), 1.25],
      hip: -0.05 * Math.abs(Math.sin(p * 2)),
      lean: 0.16 * Math.sin(p),
    }),
  },
];

/** Tucked on the way up, reaching on the way down. `k` is 0..1 through the arc. */
const jumpPose = (k: number): Pose => ({
  legs: [0.55 - 0.35 * Math.sin(Math.PI * k), -0.4 + 0.25 * Math.sin(Math.PI * k)],
  arms: [-0.75, -1.05],
  hip: 0,
  lean: 0.22,
});

/** Folded forward over a wide stance; the crouch itself comes from the height. */
const duckPose = (p: number): Pose => ({
  legs: [0.75, -0.75],
  arms: [1.45 + 0.1 * Math.sin(p * 3), 1.6],
  hip: 0,
  lean: 0.5,
});

/** Thrown backwards, flailing. */
const hitPose = (k: number): Pose => ({
  legs: [1.1 - 0.3 * k, -0.45],
  arms: [2.7, -2.5],
  hip: -0.02,
  lean: -0.35 - 0.25 * k,
});

export default function ShowView({ inputWave, outputWave, prompts, onExit }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const tracesRef = useRef<Trace[]>([]);
  const poolRef = useRef<Float32Array[]>([]);
  const lastTraceRef = useRef(0);
  const rafRef = useRef<number | null>(null);

  // The runner, its two fades, the scrolling world, and the course.
  const runRef = useRef({
    phase: 0, gate: 0, echo: 0, level: 0, last: 0,
    outPeak: 0, outRms: 0, outLevel: 0,
    mode: 'run' as 'run' | 'dance' | 'jump' | 'duck' | 'hit',
    until: 0, dur: 0, lastDance: 0, dance: 0,
    world: 0, played: 0, nextSpawn: 0,
    /** An occasional break from the station, when to reconsider it, and which
     *  way it is actually travelling. */
    dart: 0, dartAt: 0, face: 1,
    /** Where it is, as a share of the width. It travels until the course arms,
     *  then holds station while the world scrolls past instead. */
    x: MAN_X,
  });
  const obsRef = useRef<Obstacle[]>([]);
  const trailRef = useRef<
    { t: number; world: number; phase: number; mode: string; lift: number; dance: number }[]
  >([]);
  /**
   * Each ghost keeps its own position, so it can carry on alone when the input
   * goes quiet and slide back into formation when the human returns, rather
   * than teleporting between the two.
   */
  const ghostRef = useRef(GHOSTS.map((_, i) => ({
    x: 0.18 + i * 0.11,
    dir: i % 2 ? -1 : 1,
    phase: i * 1.7,
    turnAt: 0,
    // Spread through depth to begin with, so the first frame already reads as
    // a crowd standing at different distances rather than a row on the line.
    z: (i / Math.max(1, GHOSTS.length - 1)) * GHOST_Z_MAX,
    zWant: (i / Math.max(1, GHOSTS.length - 1)) * GHOST_Z_MAX,
    zAt: 0,
  })));

  /** Space strips the figures back to the bare waveforms, and restores them. */
  const livingRef = useRef(true);

  /** Whether anything is coming out of the model right now, which is what puts
   *  the AI caption on screen. Mirrored in a ref because the animation loop
   *  cannot see state, and set only on the crossing. */
  const [aiLive, setAiLive] = useState(false);
  const aiLiveRef = useRef(false);

  /**
   * Enter plays the announcement, Enter again stops it. A plain audio element
   * rather than a node in the app's graph: it goes to the room, and staying out
   * of the graph is what keeps it from being captured and fed to the model.
   */
  const annRef = useRef<HTMLAudioElement | null>(null);
  const toggleAnnouncement = useCallback(() => {
    if (!annRef.current) {
      annRef.current = new Audio(`${import.meta.env.BASE_URL}assets/announcement.wav`);
      annRef.current.preload = 'auto';
    }
    const a = annRef.current;
    if (!a.paused) {
      a.pause();
      a.currentTime = 0;
      return;
    }
    a.currentTime = 0;
    // Missing file, or a browser that wants a gesture first: nothing to do but
    // stay quiet about it rather than throw into the animation loop.
    void a.play().catch(() => {});
  }, []);

  // Leaving the view takes the announcement with it.
  useEffect(() => () => { annRef.current?.pause(); }, []);

  // Esc leaves as well as the corner button — it costs nothing and adds no mark.
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { onExit(); return; }
      if (e.code !== 'Space' && e.key !== 'Enter') return;
      // Not while the commentary box has focus, where a space is a space and
      // Enter is a new line; and not on a focused button, where Enter is a
      // click on whatever is focused.
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === 'TEXTAREA' || t.tagName === 'INPUT'
        || (e.key === 'Enter' && t.tagName === 'BUTTON'))) return;
      e.preventDefault();
      if (e.key === 'Enter') toggleAnnouncement();
      else livingRef.current = !livingRef.current;
    };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  }, [onExit, toggleAnnouncement]);

  const resize = useCallback(() => {
    const c = canvasRef.current;
    if (!c) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    c.width = Math.round(window.innerWidth * dpr);
    c.height = Math.round(window.innerHeight * dpr);
  }, []);

  useEffect(() => {
    resize();
    window.addEventListener('resize', resize);
    return () => window.removeEventListener('resize', resize);
  }, [resize]);

  useEffect(() => {
    const live = new Float32Array(POINTS);
    const next = new Float32Array(POINTS);

    /** `span` is the trace's width as a share of the screen — 1 for the input
     *  line, less for a trace that has receded — narrowing about the centre. */
    const stroke = (
      g: CanvasRenderingContext2D, pts: Float32Array,
      w: number, midY: number, amp: number, style: string, lw: number, span = 1,
    ) => {
      g.strokeStyle = style;
      g.lineWidth = lw;
      g.beginPath();
      for (let i = 0; i < pts.length; i++) {
        const x = w / 2 + (i / (pts.length - 1) - 0.5) * w * span;
        const y = midY - pts[i] * amp;
        i === 0 ? g.moveTo(x, y) : g.lineTo(x, y);
      }
      g.stroke();
    };

    /** One stick figure in the given pose, feet on `baseY`. `robot` swaps the
     *  round head for a square one with an antenna, which is what tells a ghost
     *  apart from the figure it is echoing. */
    const figure = (
      g: CanvasRenderingContext2D, x: number, baseY: number, h: number,
      pose: Pose, style: string, lw: number, robot = false, face = 1,
    ) => {
      const hipY = baseY - h * 0.46 + pose.hip * h;
      const legLen = baseY - hipY;          // vertical leg exactly reaches the line
      const torso = h * 0.34;
      const sx = x + Math.sin(pose.lean) * torso * face;
      const sy = hipY - Math.cos(pose.lean) * torso;
      const headR = h * 0.11;
      const arm = h * 0.30;
      g.strokeStyle = style;
      g.fillStyle = style;
      g.lineWidth = lw;
      g.lineCap = 'round';

      const hx = sx + Math.sin(pose.lean) * headR * 1.4 * face;
      const hy = sy - headR * 1.4;
      if (robot) {
        g.fillRect(hx - headR, hy - headR, headR * 2, headR * 2);
        g.beginPath();
        g.moveTo(hx, hy - headR);
        g.lineTo(hx, hy - headR * 2.1);
        g.stroke();
        g.beginPath();
        g.arc(hx, hy - headR * 2.4, Math.max(lw * 0.7, headR * 0.28), 0, Math.PI * 2);
        g.fill();
      } else {
        g.beginPath();
        g.arc(hx, hy, headR, 0, Math.PI * 2);
        g.fill();
      }
      g.beginPath();
      g.moveTo(sx, sy);
      g.lineTo(x, hipY);
      g.stroke();

      for (const a of pose.legs) {
        const fx = x + Math.sin(a) * legLen * face;
        const fy = hipY + Math.cos(a) * legLen;
        g.beginPath();
        g.moveTo(x, hipY);
        g.lineTo((x + fx) / 2 + Math.sin(a) * legLen * 0.14 * face, (hipY + fy) / 2);
        g.lineTo(fx, fy);
        g.stroke();
      }
      for (const a of pose.arms) {
        const ex = sx + Math.sin(a) * arm * 0.55 * face;
        const ey = sy + Math.cos(a) * arm * 0.55;
        g.beginPath();
        g.moveTo(sx, sy);
        g.lineTo(ex, ey);
        g.lineTo(ex + Math.sin(a * 1.15) * arm * 0.5 * face, ey + Math.cos(a * 1.15) * arm * 0.5);
        g.stroke();
      }
    };

    /** The pose and the height a given state draws with. */
    const poseOf = (
      mode: string, phase: number, k: number, drive = 1, dance = 0,
    ): [Pose, number] => {
      if (mode === 'dance') return [DANCES[dance % DANCES.length].pose(phase), 1];
      if (mode === 'jump') return [jumpPose(k), 1];
      if (mode === 'duck') return [duckPose(phase), DUCK_SCALE];
      if (mode === 'hit') return [hitPose(k), 1];
      return [runPose(phase, drive), 1];
    };

    const tick = (now: number) => {
      rafRef.current = requestAnimationFrame(tick);
      const c = canvasRef.current;
      const g = c?.getContext('2d');
      if (!c || !g) return;
      const run = runRef.current;
      const w = c.width, H = c.height, midY = H / 2;
      const dpr = H / Math.max(1, window.innerHeight);
      const amp = H * AMP;
      const h = H * FIG;

      // ── lay down a trace of the output on its own clock
      if (now - lastTraceRef.current >= TRACE_EVERY) {
        lastTraceRef.current = now;
        if (outputWave(next)) {
          let peak = 0, osum = 0;
          for (let i = 0; i < next.length; i++) {
            peak = Math.max(peak, Math.abs(next[i]));
            osum += next[i] * next[i];
          }
          run.outPeak = peak;          // also what gates the ghosts
          run.outRms = Math.sqrt(osum / next.length);   // and what paces them alone
          if (peak > SILENCE) {
            const pts = poolRef.current.pop() ?? new Float32Array(POINTS);
            pts.set(next);
            tracesRef.current.push({ pts, born: now });
          }
        } else {
          // Every track turned down: nothing is being heard, so the echo has to
          // close rather than hold the last peak it saw.
          run.outPeak = 0;
          run.outRms = 0;
        }
      }

      // ── expire what has risen out of sight, recycling the buffers
      const traces = tracesRef.current;
      while (traces.length && now - traces[0].born >= LIFE_MS) {
        poolRef.current.push(traces.shift()!.pts);
      }

      g.clearRect(0, 0, w, H);

      // Everything about a trace is one projection toward a vanishing point on
      // the centre line. Solving for the vanishing distance from TRAVEL keeps
      // the end of the journey where it was before the perspective went in.
      const vanish = (TRAVEL * midY) / (1 - 1 / (1 + DEPTH));

      /** A robot standing at `d.z`, projected exactly as a trace of that age
       *  is — same vanishing point, same narrowing toward the centre, same
       *  shrink. That shared projection is what puts it *on* the waveform plane
       *  back there instead of floating in front of it. */
      const paintGhost = (d: GhostDraw) => {
        const s = 1 / (1 + d.z * DEPTH);
        const dim = GHOST_Z_DIM + (1 - GHOST_Z_DIM) * s;
        figure(
          g,
          w / 2 + (d.x - 0.5) * w * s,
          midY - vanish * (1 - s) - d.lift * s,
          d.h * s,
          d.pose,
          `rgba(${INK}, ${d.alpha * dim})`,
          dpr * 1.5 * (0.5 + 0.5 * s),
          true,
          d.face,
        );
      };

      /**
       * The traces, oldest first — and the robots merged into the same walk.
       * Both are ordered deepest first and measured on the same depth scale, so
       * one pass puts every mark behind the ones in front of it: a robot three
       * traces back is drawn before those three and really is behind them.
       *
       * Deferred rather than drawn where it was, because the robots cannot be
       * resolved until the runner has been, and the runner needs the input level
       * that is read below.
       */
      const paintTraces = (ghosts: GhostDraw[]) => {
        const deep = [...ghosts].sort((p, q) => q.z - p.z);
        let gi = 0;
        const upTo = (depth: number) => {
          while (gi < deep.length && deep[gi].z > depth) paintGhost(deep[gi++]);
        };
        for (const t of traces) {
          const age = (now - t.born) / LIFE_MS;
          upTo(age);
          // Fade faster than it recedes, so the horizon stays clean.
          const a = (1 - age) ** 1.7;
          if (a < 0.03) continue;        // invisible; not worth a path
          const s = 1 / (1 + age * DEPTH);
          stroke(
            g, t.pts, w,
            midY - vanish * (1 - s),     // born on the line, ending near the horizon
            amp * s,
            `rgba(${INK}, ${a * 0.85})`,
            dpr * (0.6 + s),             // distant lines thin out too
            s,
          );
        }
        upTo(-1);                        // whatever stands in front of them all
      };

      // ── the live input: read now, painted after the depth pass, because it
      // is the nearest thing on screen and the brightest mark on it
      const hasIn = inputWave(live);
      let inPeak = 0, sum = 0;
      if (hasIn) {
        for (let i = 0; i < live.length; i++) {
          const v = live[i];
          if (Math.abs(v) > inPeak) inPeak = Math.abs(v);
          sum += v * v;
        }
      }
      const paintInput = () => {
        if (hasIn) { stroke(g, live, w, midY, amp, LIVE, dpr * 2.4); return; }
        // Nothing coming in: hold the line rather than blank the screen.
        g.strokeStyle = LIVE;
        g.lineWidth = dpr * 2.4;
        g.beginPath();
        g.moveTo(0, midY);
        g.lineTo(w, midY);
        g.stroke();
      };

      // ── the runner and its course
      const dt = run.last ? Math.min(0.1, (now - run.last) / 1000) : 0;
      run.last = now;
      // Smoothed level, so the pace follows the playing rather than the sample.
      const rms = hasIn ? Math.sqrt(sum / live.length) : 0;
      run.level += (loudness(rms) - run.level) * 0.1;
      // Two independent fades: the figure answers the input, the ghosts the output.
      run.gate += ((inPeak > PRESENT ? 1 : 0) - run.gate) * FADE;
      run.echo += ((run.outPeak > PRESENT ? 1 : 0) - run.echo) * FADE;
      run.outLevel += (loudness(run.outRms) - run.outLevel) * 0.1;

      // The AI caption is on exactly while the model is being heard. Pushed to
      // React only when it changes, so this stays a canvas loop.
      const heard = run.outPeak > SILENCE;
      if (heard !== aiLiveRef.current) { aiLiveRef.current = heard; setAiLive(heard); }

      // The ghosts answer the output and the figure answers the input, so with
      // the input quiet and the model still sounding there is nobody on the line
      // and only the ghosts are left. Both halves run below; only the figure's
      // is skipped when there is no input.
      // Space leaves nothing but the waveforms: the course stops advancing too,
      // so switching back resumes rather than jumping ahead.
      if (!livingRef.current) {
        paintTraces([]);
        paintInput();
        return;
      }

      const gh = h * GHOST_SCALE;
      const stationX = MAN_X * w;
      const human = run.gate > 0.5;
      let manPose: Pose | null = null;
      let manScale = 1, manLift = 0, manX = run.x * w;
      const trail = trailRef.current;
      /** Blocks are collected rather than stroked, so the deferred pass can put
       *  them back over the input line where they were. */
      const boxes: [number, number, number, number][] = [];
      const paintObstacles = () => {
        if (!boxes.length) return;
        g.strokeStyle = `rgba(${INK}, ${0.85 * run.gate})`;
        g.lineWidth = dpr * 2;
        for (const b of boxes) g.strokeRect(b[0], b[1], b[2], b[3]);
      };

      if (run.gate > 0.01) {
        // The course is measured in seconds actually played, not wall clock, so
        // leaving the view idle never advances the difficulty.
        run.played += dt;
        const into = run.played - CALM_S;
        const armed = into > 0;
        const ramp = armed ? 1 + (RAMP_MAX - 1) * Math.min(1, into / RAMP_S) : 1;

        // Pace: level, the ramp, and a slow drift from two detuned sines so it is
        // never metronomic. The leg cycle is driven by distance actually covered,
        // which is what stops the feet sliding when the pace changes.
        const drift = 1 + PACE_DRIFT
          * Math.sin(now / 1000 * 0.55) * Math.sin(now / 1000 * 0.21 + 1.3);
        const stumbling = run.mode === 'hit';
        const hps = (SPEED_MIN + (SPEED_MAX - SPEED_MIN) * run.level) * drift * ramp
          * (stumbling ? 0.25 : 1);
        const speed = hps * h;                      // px per second
        const dist = speed * dt;
        run.world += dist;
        // Stride lengthens with pace, so cadence rises gently while the step
        // itself grows — which is how a real gait scales from walk to run.
        const stride = (STRIDE_WALK + (STRIDE_RUN - STRIDE_WALK) * run.level) * h;
        if (run.mode === 'dance') {
          run.phase += DANCES[run.dance].rate * Math.PI * 2 * dt;
        }
        else if (!stumbling) run.phase += dist / stride * Math.PI * 2;

        // Before the course arms, he actually travels: an empty line gives no
        // parallax, so a figure holding station on it just reads as standing
        // still. He runs off the right edge and comes back on the left — and
        // stays put while dancing rather than sliding through the move. Once
        // blocks are coming he eases into his station and the world scrolls past
        // him instead, which is the only way an obstacle course can work.
        const was = run.x;
        if (!armed) {
          if (run.mode !== 'dance') run.x += dist / w;
          if (run.x > 1.05) run.x = -0.05;
          run.dart = 0;
        } else {
          // Mostly it holds the station, but now and then it breaks off and
          // runs forward or back a little. The blocks stay anchored to the
          // station, so a dart genuinely takes it out of their path.
          if (now > run.dartAt) {
            run.dartAt = now + DART_MIN_MS + Math.random() * DART_SPAN_MS;
            run.dart = Math.random() < DART_CHANCE
              ? (Math.random() < 0.5 ? -DART_REACH : DART_REACH) : 0;
          }
          run.x += (MAN_X + run.dart - run.x) * Math.min(1, dt * 3);
        }
        manX = run.x * w;
        // Face the way it is actually travelling, with a deadband so it does
        // not flip about while settling onto a target.
        const vx = run.x - was;
        if (vx > 0.0004) run.face = 1;
        else if (vx < -0.0004) run.face = -1;

        // ── spawn, and retire what has gone past
        const obs = obsRef.current;
        if (armed && run.world > run.nextSpawn) {
          obs.push({
            at: run.world + (1 - MAN_X) * w + h * 2,   // just off the right edge
            kind: Math.random() < 0.5 ? 'low' : 'high',
            lead: 1 + (Math.random() * 2 - 1) * LEAD_JITTER,
            miss: Math.random() < MISS_CHANCE,
            seen: false,
          });
          run.nextSpawn = run.world + (GAP_H * h) / ramp;
        }
        while (obs.length && obs[0].at - run.world < -h * 2) obs.shift();

        // ── decide a move, early or late depending on how well it was read
        if (run.mode === 'run' || run.mode === 'dance') {
          const o = obs.find(x => x.at - run.world >= 0 && !x.seen);
          if (o) {
            // Long enough to cover the contact, whatever the pace.
            const contact = ((OBS_W + 2 * MAN_HALF) * h / Math.max(1, speed)) * 1000;
            const dur = Math.min(MOVE_MAX_MS, Math.max(MOVE_MIN_MS,
              contact * (o.kind === 'low' ? JUMP_MARGIN : DUCK_MARGIN)));
            if (o.at - run.world <= speed * (dur / 2000) * o.lead) {
              o.seen = true;
              if (!o.miss) {
                run.mode = o.kind === 'low' ? 'jump' : 'duck';
                run.dur = dur;
                run.until = now + dur;
              }
            }
          }
        }

        // ── a move, a stumble, or a dance running out
        let k = 0;
        if (run.mode === 'jump' || run.mode === 'duck' || run.mode === 'hit') {
          const dur = run.mode === 'hit' ? HIT_MS : run.dur;
          k = Math.min(1, Math.max(0, 1 - (run.until - now) / dur));
          if (now >= run.until) run.mode = 'run';
        } else if (run.mode === 'dance' && now >= run.until) {
          run.mode = 'run';
          run.lastDance = now;
        }
        const lift = run.mode === 'jump' ? JUMP_H * h * 4 * k * (1 - k) : 0;

        // ── dance only with the road clear ahead
        if (run.mode === 'run' && now - run.lastDance > DANCE_GAP_MS) {
          const nearest = obs.find(o => o.at - run.world >= 0);
          const clear = !nearest || nearest.at - run.world > DANCE_SAFE_H * h;
          if (clear) {
            run.mode = 'dance';
            run.until = now + DANCE_MS;
            run.dance = Math.floor(Math.random() * DANCES.length);
          }
        }

        // ── blocks, then the collision test against the figure's own box
        const manH = h * (run.mode === 'duck' ? DUCK_SCALE : 1);
        const manTop = midY - lift - manH, manBot = midY - lift;
        for (const o of obs) {
          const x = stationX + (o.at - run.world);
          if (x < -h * 2 || x > w + h * 2) continue;
          const tall = o.kind === 'low' ? OBS_H : OBS_HIGH_H;
          const top = o.kind === 'low'
            ? midY - OBS_H * h
            : midY - (OBS_HIGH_BOTTOM + OBS_HIGH_H) * h;
          boxes.push([x - OBS_W * h / 2, top, OBS_W * h, tall * h]);
          if (run.mode === 'hit') continue;
          const hitX = Math.abs(x - manX) < (OBS_W / 2 + MAN_HALF) * h;
          const hitY = manTop < top + tall * h && manBot > top;
          if (hitX && hitY) {
            // Back to an empty line: the calm stretch and the ramp both restart.
            run.mode = 'hit';
            run.until = now + HIT_MS;
            run.played = 0;
            run.nextSpawn = run.world;
            obsRef.current = [];
            break;
          }
        }

        // ── the trail the ghosts read from
        trail.push({
          t: now, world: run.world, phase: run.phase,
          mode: run.mode, lift, dance: run.dance,
        });
        if (trail.length > TRAIL) trail.shift();

        [manPose, manScale] = poseOf(run.mode, run.phase, k, run.level, run.dance);
        manLift = lift;
      }

      // ── the ghosts: in formation behind him, or out on their own
      // Resolved here and painted in the depth pass, so each one lands among
      // the traces it is standing between. They pass through the blocks either
      // way, and the figure stays on top of all of them.
      const ghostDraws: GhostDraw[] = [];
      if (run.echo > 0.01) {
        for (let i = 0; i < GHOSTS.length; i++) {
          const [lag, fade] = GHOSTS[i];
          const gs = ghostRef.current[i];
          let pose = runPose(gs.phase), scale = 1, lift = 0, face = 1;

          // Depth drifts on its own, in both modes: the point is that they
          // cross between the line and the receding waveforms rather than
          // belonging to one or the other.
          if (now > gs.zAt) {
            gs.zAt = now + GHOST_Z_MIN_MS + Math.random() * GHOST_Z_SPAN_MS;
            gs.zWant = Math.random() * GHOST_Z_MAX;
          }
          gs.z += (gs.zWant - gs.z) * Math.min(1, dt * GHOST_Z_EASE);

          if (human) {
            // Follow: the target is where he was `lag` ago, and the ghost eases
            // onto it rather than snapping, so returning from a solo stretch is
            // a catch-up rather than a jump.
            const want = now - lag * 1000;
            let best: typeof trail[number] | null = null;
            for (const s of trail) if (s.t <= want && (!best || s.t > best.t)) best = s;
            if (best) {
              const tx = (stationX - (run.world - best.world)) / w;
              gs.x += (tx - gs.x) * Math.min(1, dt * CATCH_UP);
              gs.phase = best.phase;
              [pose, scale] = poseOf(best.mode, best.phase, 0.5, run.level, best.dance);
              lift = best.lift * GHOST_SCALE;
            }
          } else {
            // Alone: wander the line, paced by the output rather than the input
            // since the input is what has gone quiet. Turns at the edges and at
            // its own intervals, and faces wherever it is heading.
            const d = (SPEED_MIN + (SPEED_MAX - SPEED_MIN) * run.outLevel)
              * WANDER * gh * dt * gs.dir;
            const gStride = (STRIDE_WALK + (STRIDE_RUN - STRIDE_WALK) * run.outLevel) * gh;
            gs.x += d / w;
            gs.phase += Math.abs(d) / gStride * Math.PI * 2;
            if (gs.x > 0.95) { gs.x = 0.95; gs.dir = -1; }
            if (gs.x < 0.05) { gs.x = 0.05; gs.dir = 1; }
            if (now > gs.turnAt) {
              gs.turnAt = now + TURN_MIN_MS + Math.random() * TURN_SPAN_MS;
              if (Math.random() < 0.5) gs.dir = -gs.dir;
            }
            pose = runPose(gs.phase, run.outLevel);
            face = gs.dir;
          }

          ghostDraws.push({
            z: gs.z, x: gs.x, lift,
            h: gh * scale, pose, alpha: run.echo * fade, face,
          });
        }
      }

      // ── one depth-ordered pass, then the two nearest marks on top of it
      paintTraces(ghostDraws);
      paintInput();
      paintObstacles();

      if (manPose) {
        figure(g, manX, midY - manLift, h * manScale, manPose,
          `rgba(255, 47, 47, ${run.gate})`, dpr * 2, false, run.face);
      }
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [inputWave, outputWave]);

  return (
    <div className="show-view">
      <canvas ref={canvasRef} className="show-canvas" />

      {/* Commentary, for whatever needs saying while playing. Deliberately not
          persisted: nobody wants to walk on stage with yesterday's note up. */}
      <textarea
        id="show-note"
        className="show-note"
        rows={2}
        spellCheck={false}
        placeholder="..."
        aria-label="Performance commentary"
      />

      {/* The captions straddle the input line, in the direction the picture
          moves: what the model made of it above, where it came from below. HTML
          rather than canvas text, so they stay crisp and are not redrawn sixty
          times a second.

          The stack above the line is anchored by its foot, so the prompts stay
          put against the line whether or not the AI caption is showing. */}
      <div className="show-above">
        <div className={`show-ai ${aiLive ? 'on' : ''}`}>AI Interpretation</div>
        {prompts.length > 0 && (
          <div className="show-prompts">{prompts.join('  +  ')}</div>
        )}
      </div>
      <div className="show-caption">Direct Human<br />Audio Steering</div>

      <button className="show-exit" onClick={onExit} title="Leave show view (Esc)"
              aria-label="Leave show view">
        <Minimize2 size={14} />
      </button>
    </div>
  );
}
