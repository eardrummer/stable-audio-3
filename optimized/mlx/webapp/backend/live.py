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

    def submit(self, audio: np.ndarray, seconds: float) -> None:
        """Hand the worker a fresh snapshot, displacing any waiting one."""
        with self._lock:
            dropped = self._pending is not None
            self._pending = Snapshot(audio=audio, seconds=seconds)
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


midi_bridge = MidiBridge()


# ─────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────

def build_router(engine):
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
                    session.submit(audio, audio.shape[-1] / SAMPLE_RATE)

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
