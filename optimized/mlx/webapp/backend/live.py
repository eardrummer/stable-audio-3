"""Live audio-to-audio loop for the Stable Audio 3 MLX studio.

The browser owns the input ring buffer and the clock: it pushes the most recent
`input_rate` seconds of captured audio whenever its generate timer fires. This
module turns each snapshot into a family of `variants` progressively-noisier
generations and streams them back for gapless playback.

Threading: FastAPI receives frames on the event loop; a single worker thread
runs the generations. The engine shares model handles and is not thread-safe,
so exactly one generation is ever in flight. If a snapshot arrives while the
worker is busy it *replaces* any waiting snapshot rather than queueing — a live
loop should always work from the freshest audio, never drain a backlog.

MIDI is a dumb input bridge: control-change messages are forwarded verbatim to
the client, which owns all bindings and parameter state. That keeps learn mode
and knob values in one place instead of split across the wire.
"""

from __future__ import annotations

import asyncio
import json
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

BACKEND_DIR = Path(__file__).resolve().parent
SAMPLE_RATE = 44100

router = APIRouter()

# Set from main.py's startup hook so worker threads can drive the socket.
_event_loop: asyncio.AbstractEventLoop | None = None

# Shared across ALL sessions, not per session: MLX model handles live on the
# engine and are not thread-safe, so two tracks generating at once must
# serialize here. Everything else a session owns (latent cache, loopback
# memory, output directory, channel weights) is deliberately per session, which
# is what keeps two tracks independent.
_ENGINE_LOCK = threading.Lock()


def set_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _event_loop
    _event_loop = loop


# ─────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────

DEFAULT_PARAMS: dict = {
    "input_rate": 5.0,       # seconds of input audio used as the source
    "generate_rate": 5.0,    # how often the browser fires a generation
    "noise": 0.35,           # init_noise_level of variant 0
    "variant_delta": 0.02,   # fixed noise step between successive variants
    "steps": 8,
    "cfg": 4.0,
    "variants": 5,
    "prompt": "",
    "negative_prompt": "",
    "model": "sm-music",
    # Fraction of the previous cycle's chosen output folded back into the
    # source. 0 = listen only to the live input, 1 = feed entirely on itself.
    "loopback": 0.0,
    "selected_variant": 0,   # which variant the performer is hearing
}

# Guard rails — a knob (or a stray MIDI CC) must never be able to ask for
# something the engine will choke on.
LIMITS: dict[str, tuple[float, float]] = {
    # Beat-denominated rates: 32 beats at 60 BPM is 32 s, and 1 beat at a fast
    # tempo is a fraction of a second, so both ends are wider than the old
    # seconds-only presets needed.
    "input_rate": (0.25, 32.0),
    "generate_rate": (0.2, 40.0),
    "noise": (0.05, 1.0),
    # Spread: how far apart the variants' noise levels sit. The UI tops out at
    # 0.08 (beyond which the ladder clamps at ordinary base levels); this is the
    # guard rail, not the control.
    "variant_delta": (0.0, 0.2),
    "steps": (1, 32),
    "cfg": (1.0, 15.0),
    "variants": (1, 8),
    "loopback": (0.0, 1.0),
    "selected_variant": (0, 7),
}


def _coerce(params: dict) -> dict:
    """Clamp and type-normalize an incoming parameter patch."""
    out = {}
    for k, v in params.items():
        if k not in DEFAULT_PARAMS:
            continue
        if k in ("prompt", "negative_prompt", "model"):
            out[k] = str(v)
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if k in LIMITS:
            lo, hi = LIMITS[k]
            v = max(lo, min(hi, v))
        out[k] = int(round(v)) if k in ("steps", "variants", "selected_variant") else v
    return out


def resample_to_model_rate(audio: np.ndarray, src_rate: int) -> np.ndarray:
    """Resample (C, T) float32 audio from `src_rate` to the model's 44.1 kHz.

    Capture hardware commonly runs at 48 kHz while Stable Audio 3 is 44.1 kHz
    end to end, so we convert on the way in rather than asking the device for a
    rate it doesn't natively run. 48k -> 44.1k is the exact rational 147/160,
    which `resample_poly` handles without any fractional-delay error.
    """
    if src_rate == SAMPLE_RATE:
        return audio
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(SAMPLE_RATE, src_rate)
    out = resample_poly(audio, SAMPLE_RATE // g, src_rate // g, axis=-1)
    return np.ascontiguousarray(out, dtype=np.float32)


@dataclass
class Snapshot:
    audio: np.ndarray       # (2, T) float32
    seconds: float
    received_at: float = field(default_factory=time.time)
    # The client clock reading when this snapshot was pushed, echoed back
    # untouched so the browser can time the whole round trip itself.
    tag: float = 0.0


# ─────────────────────────────────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────────────────────────────────

class LiveSession:
    def __init__(self, engine, websocket: WebSocket, connection_id: str):
        self.engine = engine
        self.ws = websocket
        self.cid = connection_id

        self.params = dict(DEFAULT_PARAMS)
        self.running = False

        self._lock = threading.Lock()
        self._pending: Snapshot | None = None
        self._wake = threading.Event()
        self._worker: threading.Thread | None = None
        self._cycle = 0

        # Latent shelf: 256 per-channel weights applied immediately before the
        # decoder. Latents from the current cycle are cached so moving the
        # shelf can re-decode (~49 ms) instead of re-generating (~330 ms).
        self.channel_weights: np.ndarray | None = None
        self.renormalize = False
        self._latent_cache: dict[int, "object"] = {}
        self._cache_cycle = -1
        self._cache_seconds = 0.0
        # The previous cycle's audible output, held for the loopback mix.
        self._last_output: np.ndarray | None = None
        # Only one thing may touch the engine at a time — MLX model handles are
        # shared and not thread-safe. Taken per variant, so a re-decode can slip
        # between two variants rather than waiting out the whole cycle.
        self._redecode_req: dict | None = None
        self._redecode_wake = threading.Event()
        self._redecode_thread: threading.Thread | None = None
        self.decoder_name = "same-s"

        # Variants are overwritten in place each cycle, so a long session
        # leaves behind `variants` files rather than thousands.
        self.out_dir = BACKEND_DIR / "live_out" / connection_id
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # -- socket helpers (safe to call from the worker thread) -------------

    def _send_json(self, payload: dict) -> None:
        if _event_loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self.ws.send_json(payload), _event_loop)
        except Exception:
            pass

    def _send_bytes(self, data: bytes) -> None:
        if _event_loop is None:
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(self.ws.send_bytes(data), _event_loop)
            fut.result(timeout=10)   # backpressure: don't outrun the socket
        except Exception:
            pass

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._last_output = None      # never loop back audio from a past run
        self._worker = threading.Thread(target=self._run, daemon=True,
                                        name=f"live-{self.cid[:8]}")
        self._worker.start()
        self._redecode_thread = threading.Thread(target=self._redecode_loop, daemon=True,
                                                 name=f"redecode-{self.cid[:8]}")
        self._redecode_thread.start()
        self._send_json({"type": "started", "params": self.params})

    def flush(self) -> None:
        """Forget the last output, so loopback cannot feed the model audio the
        performer has just silenced. Generation itself carries on."""
        with self._lock:
            self._last_output = None

    def stop(self) -> None:
        self.running = False
        self._wake.set()
        self._redecode_wake.set()
        self._send_json({"type": "stopped"})

    def close(self) -> None:
        self.stop()
        for t in (self._worker, self._redecode_thread):
            if t and t.is_alive():
                t.join(timeout=5)

    # -- latent shelf -----------------------------------------------------

    def set_channel_weights(self, weights, renormalize: bool, redecode_index: int | None) -> None:
        """Store the 256-channel curve, then optionally re-render one cached
        variant through it so the change is audible without a new generation."""
        if weights is None:
            w = None
        else:
            w = np.asarray(weights, dtype=np.float32).reshape(-1)
            if w.shape[0] != 256:
                self._send_json({"type": "error",
                                 "error": f"expected 256 channel weights, got {w.shape[0]}"})
                return
            w = np.clip(w, 0.0, 1.0)
        with self._lock:
            self.channel_weights = w
            self.renormalize = bool(renormalize)
        if redecode_index is not None:
            # Coalesce: only the newest request survives, so dragging across the
            # shelf never builds a backlog of stale decodes.
            with self._lock:
                self._redecode_req = {"index": int(redecode_index)}
            self._redecode_wake.set()

    def _redecode_loop(self) -> None:
        while self.running:
            self._redecode_wake.wait(timeout=0.25)
            self._redecode_wake.clear()
            if not self.running:
                break
            with self._lock:
                req = self._redecode_req
                self._redecode_req = None
                weights = self.channel_weights
                renorm = self.renormalize
                cycle = self._cache_cycle
                latents = self._latent_cache.get(req["index"]) if req else None
                seconds = self._cache_seconds if req else None
            if req is None or latents is None:
                continue
            try:
                out_path = self.out_dir / f"v{req['index']}.wav"
                t0 = time.time()
                with _ENGINE_LOCK:
                    self.engine.decode_latents_to_wav(
                        latents, decoder_name=self.decoder_name, seconds=seconds,
                        out_path=str(out_path), channel_weights=weights,
                        renormalize=renorm,
                    )
                data = out_path.read_bytes()
                self._send_json({
                    "type": "variant", "cycle": cycle, "index": req["index"],
                    "redecode": True, "bytes": len(data),
                    "gen_ms": round((time.time() - t0) * 1000, 1),
                })
                self._send_bytes(data)
            except Exception as e:
                self._send_json({"type": "error", "error": f"re-decode failed: {e}"})

    def update_params(self, patch: dict) -> None:
        clean = _coerce(patch)
        with self._lock:
            self.params.update(clean)
        self._send_json({"type": "params", "params": self.params})

    def submit(self, audio: np.ndarray, seconds: float, tag: float = 0.0) -> None:
        """Hand the worker a fresh snapshot, displacing any waiting one."""
        with self._lock:
            dropped = self._pending is not None
            self._pending = Snapshot(audio=audio, seconds=seconds, tag=tag)
        if dropped:
            # The loop is running behind the requested generate rate.
            self._send_json({"type": "dropped"})
        self._wake.set()

    # -- worker -----------------------------------------------------------

    def _run(self) -> None:
        while self.running:
            self._wake.wait(timeout=0.25)
            self._wake.clear()
            if not self.running:
                break

            with self._lock:
                snap = self._pending
                self._pending = None
                params = dict(self.params)
            if snap is None:
                continue

            try:
                self._generate_cycle(snap, params)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json({"type": "error", "error": f"{type(e).__name__}: {e}"})

    def _generate_cycle(self, snap: Snapshot, params: dict) -> None:
        self._cycle += 1
        cycle = self._cycle
        n = int(params["variants"])
        # One seed per cycle, shared across the variant family, so the five
        # outputs differ only by noise level — the studio matrix's semantics.
        seed = random.randint(0, 2**31 - 1)
        t_cycle = time.time()

        # A new cycle invalidates the shelf's cache; only the newest set of
        # latents is worth re-decoding against.
        with self._lock:
            self._latent_cache = {}
            self._cache_cycle = cycle
            self._cache_seconds = snap.seconds
            weights = self.channel_weights
            renorm = self.renormalize
            self.decoder_name = "same-l" if params["model"] == "medium" else "same-s"
            last_out = self._last_output

        # Loopback: fold the previous cycle's audible output back into the
        # source. At 1.0 the live input is ignored entirely and the model feeds
        # on itself; at 0.0 this is a no-op. Mixed once and shared by every
        # variant, so the whole family descends from the same source.
        source_audio = snap.audio
        loopback = float(params.get("loopback", 0.0))
        if loopback > 0.001 and last_out is not None:
            n_samp = source_audio.shape[-1]
            prev = last_out
            if prev.shape[-1] >= n_samp:
                prev = prev[:, -n_samp:]              # align on the most recent audio
            elif prev.shape[-1] > 0:
                prev = np.pad(prev, ((0, 0), (0, n_samp - prev.shape[-1])))
            if prev.shape[-1] == n_samp:
                source_audio = (1.0 - loopback) * source_audio + loopback * prev
                peak = float(np.max(np.abs(source_audio))) if source_audio.size else 0.0
                if peak > 1.0:
                    source_audio = source_audio / peak    # summing two clips can clip

        self._send_json({
            "type": "cycle_start", "cycle": cycle, "variants": n,
            "seconds": snap.seconds, "seed": seed,
            "queue_latency_ms": round((t_cycle - snap.received_at) * 1000, 1),
        })

        for i in range(n):
            if not self.running:
                break
            noise = params["noise"] + i * params["variant_delta"]
            noise = max(0.05, min(1.0, noise))
            out_path = self.out_dir / f"v{i}.wav"

            t0 = time.time()
            sink: list = []
            with _ENGINE_LOCK:
                self.engine.generate(
                    prompt=params["prompt"],
                    negative_prompt=params["negative_prompt"] or None,
                    init_audio_array=source_audio,
                    init_noise_level=noise,
                    seconds=snap.seconds,
                    steps=int(params["steps"]),
                    seed=seed,
                    dit_name=params["model"],
                    decoder_name="auto",
                    cfg=params["cfg"],
                    apg=1.0,
                    free_models=False,
                    out_path=str(out_path),
                    channel_weights=weights,
                    renormalize=renorm,
                    latent_sink=sink,
                )
            gen_ms = (time.time() - t0) * 1000
            if sink:
                with self._lock:
                    self._latent_cache[i] = sink[0]

            data = out_path.read_bytes()
            # Header first, payload second — WebSocket preserves ordering, so
            # the client can pair them without framing the audio itself.
            self._send_json({
                "type": "variant", "cycle": cycle, "index": i,
                "noise": round(noise, 3), "seed": seed,
                "bytes": len(data), "gen_ms": round(gen_ms, 1),
                "tag": snap.tag,
            })
            self._send_bytes(data)

        # Hold on to the variant the performer is actually hearing; that is what
        # "the last produced clip that was played back" means.
        sel = max(0, min(n - 1, int(params.get("selected_variant", 0))))
        sel_path = self.out_dir / f"v{sel}.wav"
        if self.running and sel_path.exists():
            try:
                self._last_output = self.engine.read_wav(str(sel_path))
            except Exception as e:
                print(f"[Live] loopback capture failed: {e}")

        self._send_json({
            "type": "cycle_complete", "cycle": cycle,
            "total_ms": round((time.time() - t_cycle) * 1000, 1),
        })


# ─────────────────────────────────────────────────────────────────────────
# MIDI bridge — forwards control changes; the client owns all bindings.
# ─────────────────────────────────────────────────────────────────────────

class MidiBridge:
    def __init__(self):
        self._port = None
        self._port_name: str | None = None
        self._subscribers: set = set()
        self._lock = threading.Lock()

    @staticmethod
    def list_ports() -> list[str]:
        try:
            import mido
            return list(mido.get_input_names())
        except Exception:
            return []

    def subscribe(self, session: LiveSession) -> None:
        with self._lock:
            self._subscribers.add(session)

    def unsubscribe(self, session: LiveSession) -> None:
        with self._lock:
            self._subscribers.discard(session)

    def open(self, name: str | None) -> str | None:
        """Open `name` (or the first available port when None). Returns the
        port actually opened, or None if there was nothing to open."""
        import mido
        self.close()
        if name is None:
            ports = self.list_ports()
            if not ports:
                return None
            name = ports[0]
        self._port = mido.open_input(name, callback=self._on_message)
        self._port_name = name
        print(f"[Live] MIDI input opened: {name}")
        return name

    def close(self) -> None:
        if self._port is not None:
            try:
                self._port.close()
            except Exception:
                pass
        self._port = None
        self._port_name = None

    @property
    def port_name(self) -> str | None:
        return self._port_name

    def _on_message(self, msg) -> None:
        # Runs on mido's own thread.
        if msg.type != "control_change":
            return
        payload = {
            "type": "midi_cc",
            "channel": msg.channel,
            "control": msg.control,
            "value": msg.value,
            "norm": msg.value / 127.0,
        }
        with self._lock:
            subs = list(self._subscribers)
        for s in subs:
            s._send_json(payload)


# The remembered control surface: which port to reopen, and what the learned
# CCs are bound to. Kept here rather than in the browser because localStorage
# is per-origin — opening the app at 127.0.0.1 instead of localhost, or in
# another browser, is a different store and looks exactly like a lost mapping.
MIDI_MAP_FILE = Path(__file__).resolve().parent / "midi_map.json"


def load_midi_map() -> dict:
    """{"port": name or None, "bindings": {id: {channel, control}}}."""
    try:
        with open(MIDI_MAP_FILE) as f:
            d = json.load(f)
        return {
            "port": d.get("port") or None,
            "bindings": d.get("bindings") or {},
        }
    except FileNotFoundError:
        return {"port": None, "bindings": {}}
    except Exception as e:
        # A corrupt file must not take the app down with it.
        print(f"[Live] MIDI map unreadable, starting empty: {e}")
        return {"port": None, "bindings": {}}


def save_midi_map(**changes) -> dict:
    """Merge `changes` into the stored map and write it back."""
    cur = load_midi_map()
    cur.update({k: v for k, v in changes.items() if v is not None})
    tmp = MIDI_MAP_FILE.with_suffix(".json.tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(cur, f, indent=2)
        tmp.replace(MIDI_MAP_FILE)      # atomic, so a crash cannot truncate it
    except Exception as e:
        print(f"[Live] could not write the MIDI map: {e}")
    return cur


# Prompt presets: named sets of per-track prompts, so a combination that worked
# can be brought back rather than retyped. Unlike the MIDI map this is content,
# not machine state, so it is kept in the repo and is meant to be hand-editable.
PROMPT_FILE = Path(__file__).resolve().parent / "prompt_presets.json"


def load_presets() -> list[dict]:
    """[{"name": str, "prompts": [str, ...]}, ...], newest last."""
    try:
        with open(PROMPT_FILE) as f:
            d = json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"[Live] prompt presets unreadable, starting empty: {e}")
        return []
    out = []
    for p in d.get("presets") or []:
        # Tolerant of a hand-edited file: skip anything malformed rather than
        # failing the whole list, and coerce prompts to strings.
        if not isinstance(p, dict):
            continue
        name = str(p.get("name") or "").strip()
        if not name:
            continue
        out.append({
            "name": name,
            "prompts": [str(x) for x in (p.get("prompts") or [])],
        })
    return out


def save_preset(name: str, prompts: list[str]) -> list[dict]:
    """Add the set, or replace the one of that name in place -- so saving over
    a name is a deliberate overwrite and does not leave two of them."""
    presets = load_presets()
    entry = {"name": name, "prompts": [str(x) for x in prompts]}
    for i, p in enumerate(presets):
        if p["name"].lower() == name.lower():
            presets[i] = entry
            break
    else:
        presets.append(entry)
    tmp = PROMPT_FILE.with_suffix(".json.tmp")
    try:
        with open(tmp, "w") as f:
            json.dump({"presets": presets}, f, indent=2)
        tmp.replace(PROMPT_FILE)
    except Exception as e:
        print(f"[Live] could not write the prompt presets: {e}")
    return presets


def delete_preset(name: str) -> list[dict]:
    presets = [p for p in load_presets() if p["name"].lower() != name.lower()]
    tmp = PROMPT_FILE.with_suffix(".json.tmp")
    try:
        with open(tmp, "w") as f:
            json.dump({"presets": presets}, f, indent=2)
        tmp.replace(PROMPT_FILE)
    except Exception as e:
        print(f"[Live] could not write the prompt presets: {e}")
    return presets


# Hardware output routing: which device, and which stereo pair the master mix
# and each track are sent to. A patching decision about a room, so it outlives
# the browser session the way the MIDI map does.
ROUTING_FILE = Path(__file__).resolve().parent / "routing.json"

ROUTING_DEFAULT = {"sinkId": "", "sinkLabel": "", "masterPair": 0, "trackPairs": []}


def load_routing() -> dict:
    try:
        with open(ROUTING_FILE) as f:
            d = json.load(f)
    except FileNotFoundError:
        return dict(ROUTING_DEFAULT)
    except Exception as e:
        print(f"[Live] routing unreadable, using defaults: {e}")
        return dict(ROUTING_DEFAULT)
    out = dict(ROUTING_DEFAULT)
    out["sinkId"] = str(d.get("sinkId") or "")
    out["sinkLabel"] = str(d.get("sinkLabel") or "")
    try:
        out["masterPair"] = max(0, int(d.get("masterPair", 0)))
    except (TypeError, ValueError):
        pass
    pairs = d.get("trackPairs")
    if isinstance(pairs, list):
        clean = []
        for v in pairs:
            try:
                clean.append(max(0, int(v)))
            except (TypeError, ValueError):
                clean.append(0)
        out["trackPairs"] = clean
    return out


def save_routing(body: dict) -> dict:
    """Merge whatever the panel sent over what is stored, so a device change and
    a pair change can arrive as separate writes without erasing each other."""
    cur = load_routing()
    for k in ROUTING_DEFAULT:
        if k in body and body[k] is not None:
            cur[k] = body[k]
    tmp = ROUTING_FILE.with_suffix(".json.tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(cur, f, indent=2)
        tmp.replace(ROUTING_FILE)
    except Exception as e:
        print(f"[Live] could not write the routing: {e}")
    return cur


midi_bridge = MidiBridge()


def restore_midi_port() -> str | None:
    """Reopen the remembered port at startup. Only if it is actually present:
    a name that is gone means the surface is unplugged, and guessing another
    port would silently bind the knobs to the wrong device."""
    name = load_midi_map()["port"]
    if not name:
        return None
    ports = MidiBridge.list_ports()
    if name not in ports:
        print(f"[Live] remembered MIDI port not connected: {name!r} (have {ports})")
        return None
    try:
        return midi_bridge.open(name)
    except Exception as e:
        print(f"[Live] could not reopen {name!r}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────

def build_router(engine):
    # build_router is called once, as the app starts.
    restore_midi_port()

    @router.get("/api/live/midi_map")
    def get_midi_map():
        return load_midi_map()

    @router.post("/api/live/midi_map")
    def post_midi_map(body: dict):
        """Save the learned bindings. The browser is the editor; this is only
        the filing cabinet, so the whole set is replaced rather than merged —
        otherwise an unlearned control could never be removed."""
        b = body.get("bindings")
        if not isinstance(b, dict):
            return {"status": "error", "error": "bindings must be an object"}
        return {"status": "ok", **save_midi_map(bindings=b)}

    @router.get("/api/live/routing")
    def get_routing():
        return load_routing()

    @router.post("/api/live/routing")
    def post_routing(body: dict):
        return {"status": "ok", **save_routing(body)}

    @router.get("/api/live/prompt_presets")
    def get_presets():
        return {"presets": load_presets()}

    @router.post("/api/live/prompt_presets")
    def post_preset(body: dict):
        name = str(body.get("name") or "").strip()
        if not name:
            return {"status": "error", "error": "a preset needs a name"}
        prompts = body.get("prompts")
        if not isinstance(prompts, list):
            return {"status": "error", "error": "prompts must be a list"}
        return {"status": "ok", "presets": save_preset(name, prompts)}

    @router.delete("/api/live/prompt_presets")
    def del_preset(name: str):
        return {"status": "ok", "presets": delete_preset(name)}

    @router.get("/api/live/midi_ports")
    def midi_ports():
        return {
            "ports": MidiBridge.list_ports(),
            "open": midi_bridge.port_name,
        }

    @router.post("/api/live/midi_port")
    def midi_port(body: dict):
        try:
            opened = midi_bridge.open(body.get("name"))
            # Remember it, so the next start comes up on the same surface.
            if opened:
                save_midi_map(port=opened)
            return {"status": "ok", "open": opened}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @router.get("/api/live/defaults")
    def defaults():
        return {"params": DEFAULT_PARAMS, "limits": LIMITS}

    @router.websocket("/ws/live/{connection_id}")
    async def live_socket(websocket: WebSocket, connection_id: str):
        await websocket.accept()
        session = LiveSession(engine, websocket, connection_id)
        midi_bridge.subscribe(session)
        print(f"[Live] Session connected: {connection_id}")

        # A binary frame carries PCM for the header that preceded it.
        pending_header: dict | None = None
        try:
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break

                if (text := msg.get("text")) is not None:
                    data = json.loads(text)
                    kind = data.get("type")
                    if kind == "start":
                        session.update_params(data.get("params", {}))
                        session.start()
                    elif kind == "stop":
                        session.stop()
                    elif kind == "flush":
                        session.flush()
                    elif kind == "params":
                        session.update_params(data.get("params", {}))
                    elif kind == "weights":
                        session.set_channel_weights(
                            data.get("weights"),
                            bool(data.get("renormalize", False)),
                            data.get("redecode_index"),
                        )
                    elif kind == "audio_header":
                        pending_header = data
                    elif kind == "ping":
                        await websocket.send_json({"type": "pong"})

                elif (blob := msg.get("bytes")) is not None:
                    if pending_header is None or not session.running:
                        pending_header = None
                        continue
                    ch = int(pending_header.get("channels", 2))
                    sr = int(pending_header.get("sample_rate", SAMPLE_RATE))
                    tag = float(pending_header.get("tag", 0.0))
                    pending_header = None
                    pcm = np.frombuffer(blob, dtype=np.float32)
                    if pcm.size == 0:
                        continue
                    if ch == 2:
                        audio = pcm.reshape(-1, 2).T.copy()
                    else:
                        audio = np.stack([pcm, pcm])
                    # Capture runs at whatever the device's native rate is
                    # (48 kHz on most interfaces); the model is 44.1 kHz.
                    try:
                        audio = resample_to_model_rate(audio, sr)
                    except Exception as e:
                        session._send_json({
                            "type": "error",
                            "error": f"resample {sr} Hz -> {SAMPLE_RATE} Hz failed: {e}",
                        })
                        continue
                    session.submit(audio, audio.shape[-1] / SAMPLE_RATE, tag)

        except WebSocketDisconnect:
            pass
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            midi_bridge.unsubscribe(session)
            session.close()
            print(f"[Live] Session closed: {connection_id}")

    return router
