import os
import sys
import json
import uuid
import time
import asyncio
import random
from pathlib import Path
from typing import Optional, List, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Setup paths so that engine resolves correctly
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from engine import StableAudioEngine, SAMPLE_RATE

app = FastAPI(title="Stable Audio 3 MLX Local UI Backend")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local development, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Engine
engine = StableAudioEngine()

# Active WebSocket connections for streaming progress
# connection_id -> WebSocket
active_connections = {}
# Active abort flags for stopping background variants generation
active_aborts = set()
# Event loop reference for thread-safe websocket streaming
event_loop = None

@app.on_event("startup")
async def startup_event():
    global event_loop
    event_loop = asyncio.get_running_loop()
    print("[Server] FastAPI Startup complete. Event loop captured.")

# History Database Helper
HISTORY_FILE = BACKEND_DIR / "history.json"

def read_history() -> List[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def write_history(history: List[dict]):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Failed to write history: {e}")

# Request Models
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    init_audio_path: Optional[str] = None
    init_audio_path_2: Optional[str] = None
    blend_ratio: float = 0.5
    blend_method: Optional[str] = None
    init_noise_level: float = 1.0
    inpaint_start: Optional[float] = None
    inpaint_end: Optional[float] = None
    seconds: float = 30.0
    steps: int = 8
    seed: Optional[int] = None
    dit_name: str = "sm-music"
    decoder_name: Optional[str] = None
    cfg: float = 1.0
    apg: float = 1.0
    free_models: bool = False
    connection_id: str
    save_history: bool = True  # Toggle to disable history saving

def resolve_audio_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if "/api/audio/" in path:
        parts = path.split("/api/audio/")[-1].split("/")
        return str(engine.output_dir / Path(*parts))
    elif "/api/uploads/" in path:
        filename = path.split("/api/uploads/")[-1]
        return str(engine.upload_dir / filename)
    return path

@app.get("/api/status")
def get_status():
    """Returns the status of local models (which ones are cached)."""
    try:
        status = engine.get_models_status()
        return {"status": "ok", "models": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PreloadModelRequest(BaseModel):
    dit_name: str = "sm-music"

@app.post("/api/preload_model")
def preload_model(req: PreloadModelRequest):
    """Pre-downloads and caches model weights into local MLX cache."""
    try:
        engine.preload_model_bundle(req.dit_name)
        status = engine.get_models_status()
        return {"status": "ok", "models": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Handles uploading an initial audio file or microphone recording."""
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm')):
        raise HTTPException(status_code=400, detail="Only audio files are supported.")
        
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix or ".wav"
    filename = f"{file_id}{ext}"
    dest_path = engine.upload_dir / filename
    
    try:
        with open(dest_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Verify and get duration using the engine's reader
        # If it's not a WAV, we might need ffmpeg to parse it.
        audio_np = engine.read_wav(str(dest_path))
        duration = audio_np.shape[-1] / SAMPLE_RATE
        
        return {
            "status": "success",
            "filename": filename,
            "local_path": str(dest_path),
            "duration": duration,
            "wav_url": f"/api/uploads/{filename}"
        }
    except Exception as e:
        if dest_path.exists():
            os.remove(dest_path)
        raise HTTPException(status_code=500, detail=f"Failed to process audio: {str(e)}")

def update_history_variant(primary_filename: str, delta: float, seed_idx: int, filename: str, seed: int):
    """Safely updates history.json with a newly completed variant on-the-fly."""
    try:
        history = read_history()
        updated = False
        for item in history:
            if item.get("filename") == primary_filename:
                if "variants_matrix" not in item:
                    item["variants_matrix"] = []
                    
                # Find or create delta slot
                delta_slot = None
                for slot in item["variants_matrix"]:
                    if abs(slot["delta"] - delta) < 1e-4:
                        delta_slot = slot
                        break
                
                if not delta_slot:
                    delta_slot = {
                        "delta": delta,
                        "noise_level": round(item["init_noise_level"] + delta, 2),
                        "seeds": []
                    }
                    item["variants_matrix"].append(delta_slot)
                
                # Add or update seed slot
                seed_slot = None
                for s in delta_slot["seeds"]:
                    if s["seed_index"] == seed_idx:
                        seed_slot = s
                        break
                
                if not seed_slot:
                    seed_slot = {
                        "seed_index": seed_idx,
                        "seed": seed,
                        "filename": filename,
                        "isReady": True
                    }
                    delta_slot["seeds"].append(seed_slot)
                else:
                    seed_slot["filename"] = filename
                    seed_slot["seed"] = seed
                    seed_slot["isReady"] = True
                
                updated = True
                break
                
        if updated:
            write_history(history)
    except Exception as e:
        print(f"[Engine] Failed to update history variant: {e}")

def cleanup_temporary_sessions():
    """Garbage Collector: Deletes old temporary session directories to free up disk space."""
    try:
        history = read_history()
        saved_session_names = {f"session_{item['id']}" for item in history}
        
        if engine.output_dir.exists():
            for path in engine.output_dir.iterdir():
                if path.is_dir() and path.name.startswith("session_"):
                    if path.name not in saved_session_names:
                        print(f"[GarbageCollector] Deleting temporary session directory: {path}")
                        import shutil
                        shutil.rmtree(path, ignore_errors=True)
    except Exception as e:
        print(f"[GarbageCollector] Error cleaning temporary sessions: {e}")

def execute_background_generation_queue(
    tasks: List[Tuple[float, int]],
    req: GenerateRequest,
    session_id: str,
    primary_filename: str,
    center_noise: float,
    actual_seed: int,
    connection_id: str
):
    """
    Executes a custom queue of (delta, seed_idx) variant generation tasks.
    Broadcasting progress and results over WebSockets and saving to history.json.
    """
    # 5 seed variations (Seed offsets)
    seed_offsets = [0, 100001, 200002, 300003, 400004]
    
    # Resolve audio paths to local server paths
    init_audio_path = resolve_audio_path(req.init_audio_path)
    init_audio_path_2 = resolve_audio_path(req.init_audio_path_2)

    inpaint_range = None
    if req.inpaint_start is not None and req.inpaint_end is not None:
        inpaint_range = (req.inpaint_start, req.inpaint_end)

    session_dir = engine.output_dir / f"session_{session_id}"

    # Execute the queue
    for delta, seed_idx in tasks:
        # Safeguard: Abort if client disconnected OR if abort requested!
        if connection_id not in active_connections or connection_id in active_aborts:
            print(f"[Server] Aborting background variants generation for client {connection_id}.")
            if connection_id in active_aborts:
                active_aborts.remove(connection_id)
            break
            
        noise_level = center_noise + delta
        # Clamp between 0.05 and 1.0
        if noise_level < 0.05:
            noise_level = 0.05
        elif noise_level > 1.0:
            noise_level = 1.0
            
        # Skip if clamped noise level is identical to center, unless we are generating alternative seeds
        if seed_idx == 0 and abs(noise_level - center_noise) < 1e-4:
            continue
            
        target_seed = actual_seed + seed_offsets[seed_idx]
        
        # Standardized file naming: e.g. var_+0.02_seed_1.wav
        custom_filename = f"var_{delta:+.2f}_seed_{seed_idx}.wav"
        out_path = session_dir / custom_filename
        
        # Notify WebSocket of generation start!
        if connection_id in active_connections and event_loop:
            ws = active_connections[connection_id]
            asyncio.run_coroutine_threadsafe(
                ws.send_json({
                    "type": "variant_start",
                    "delta": round(delta, 2),
                    "seed_index": seed_idx
                }),
                event_loop
            )
        
        print(f"[Engine] Generating variant: noise_level={noise_level:.2f} (delta={delta:+.2f}), seed_idx={seed_idx} (seed={target_seed})")
        
        try:
            engine.generate(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                init_audio_path=init_audio_path,
                init_audio_path_2=init_audio_path_2,
                blend_ratio=req.blend_ratio,
                blend_method=req.blend_method,
                init_noise_level=noise_level,
                inpaint_range=inpaint_range,
                seconds=req.seconds,
                steps=req.steps,
                seed=target_seed,
                dit_name=req.dit_name,
                decoder_name=req.decoder_name,
                cfg=req.cfg,
                apg=req.apg,
                free_models=req.free_models,
                progress_callback=None,  # Suppress step progress
                out_path=str(out_path)  # Write directly to session folder!
            )
            
            relative_url_path = f"session_{session_id}/{custom_filename}"
            wav_url = f"/api/audio/{relative_url_path}"
            
            # Update history database on-the-fly if save_history is enabled
            if req.save_history:
                update_history_variant(
                    primary_filename=f"session_{session_id}/primary.wav",
                    delta=delta,
                    seed_idx=seed_idx,
                    filename=custom_filename,
                    seed=target_seed
                )
            
            # Send variant_ready over WebSockets
            if connection_id in active_connections and event_loop:
                ws = active_connections[connection_id]
                asyncio.run_coroutine_threadsafe(
                    ws.send_json({
                        "type": "variant_ready",
                        "primary_filename": primary_filename,
                        "noise_level": round(noise_level, 2),
                        "delta": round(delta, 2),
                        "seed_index": seed_idx,
                        "filename": custom_filename,
                        "wav_url": wav_url
                    }),
                    event_loop
                )
                
        except Exception as e:
            print(f"[Engine] Failed to generate background variant (noise={noise_level:.2f}, seed_idx={seed_idx}): {e}")

    # Send queue_complete over WebSockets when the background loop is fully completed or aborted!
    if connection_id in active_connections and event_loop:
        ws = active_connections[connection_id]
        asyncio.run_coroutine_threadsafe(
            ws.send_json({
                "type": "queue_complete"
            }),
            event_loop
        )

def generate_variants_in_background(
    req: GenerateRequest,
    session_id: str,
    primary_filename: str,
    center_noise: float,
    actual_seed: int,
    connection_id: str
):
    """
    Initial background queue: Generates ONLY Seed 0 for all other 20 noise levels.
    This provides instant vertical wheel coverage in less than 20 seconds!
    """
    # 21 noise levels (Center is 0.0)
    deltas = [
        0.0,
        0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20,
        -0.02, -0.04, -0.06, -0.08, -0.10, -0.12, -0.14, -0.16, -0.18, -0.20
    ]
    
    # Initial background queue: Only Seed 0 for all other 20 deltas!
    tasks = []
    for delta in deltas[1:]:
        tasks.append((delta, 0))
        
    execute_background_generation_queue(
        tasks=tasks,
        req=req,
        session_id=session_id,
        primary_filename=primary_filename,
        center_noise=center_noise,
        actual_seed=actual_seed,
        connection_id=connection_id
    )

@app.post("/api/generate")
def generate_audio(req: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Triggers generation.
    Supports structured session folders, deterministic seeds, and memory toggling.
    """
    connection_id = req.connection_id
    
    # 0. Run Garbage Collector to sweep any old temporary sessions
    cleanup_temporary_sessions()
    
    # Progress callback that will be triggered by the engine
    def progress_callback(step: int, total: int, elapsed_ms: float):
        if connection_id in active_connections and event_loop:
            ws = active_connections[connection_id]
            asyncio.run_coroutine_threadsafe(
                ws.send_json({
                    "type": "progress",
                    "step": step,
                    "total": total,
                    "elapsed_ms": elapsed_ms
                }),
                event_loop
            )

    inpaint_range = None
    if req.inpaint_start is not None and req.inpaint_end is not None:
        inpaint_range = (req.inpaint_start, req.inpaint_end)

    # Resolve web URLs to local server paths automatically
    init_audio_path = resolve_audio_path(req.init_audio_path)
    init_audio_path_2 = resolve_audio_path(req.init_audio_path_2)

    # Pre-generate concrete seed here to ensure total determinism
    actual_seed = req.seed
    if actual_seed is None:
        actual_seed = random.randint(0, 2**31 - 1)

    # Generate unique session ID
    session_id = str(uuid.uuid4())
    session_dir = engine.output_dir / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    primary_wav_path = session_dir / "primary.wav"
    relative_primary_url = f"session_{session_id}/primary.wav"
    wav_url = f"/api/audio/{relative_primary_url}"

    try:
        # Run generation directly into session folder (runs inside FastAPI threadpool)
        engine.generate(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            init_audio_path=init_audio_path,
            init_audio_path_2=init_audio_path_2,
            blend_ratio=req.blend_ratio,
            blend_method=req.blend_method,
            init_noise_level=req.init_noise_level,
            inpaint_range=inpaint_range,
            seconds=req.seconds,
            steps=req.steps,
            seed=actual_seed, # Guaranteed concrete seed!
            dit_name=req.dit_name,
            decoder_name=req.decoder_name,
            cfg=req.cfg,
            apg=req.apg,
            free_models=req.free_models,
            progress_callback=progress_callback,
            out_path=str(primary_wav_path)  # Write directly to session/primary.wav!
        )
        
        # Create history item
        history_item = {
            "id": session_id,
            "timestamp": time.time(),
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "init_audio_path": req.init_audio_path,
            "init_noise_level": req.init_noise_level,
            "inpaint_start": req.inpaint_start,
            "inpaint_end": req.inpaint_end,
            "seconds": req.seconds,
            "steps": req.steps,
            "seed": actual_seed,
            "dit_name": req.dit_name,
            "decoder_name": req.decoder_name or "auto",
            "cfg": req.cfg,
            "apg": req.apg,
            "filename": relative_primary_url,
            "wav_url": wav_url,
            "variants_matrix": [
                {
                    "delta": 0.0,
                    "noise_level": req.init_noise_level,
                    "seeds": [
                        {
                            "seed_index": 0,
                            "seed": actual_seed,
                            "filename": "primary.wav",
                            "isReady": True
                        }
                    ]
                }
            ]
        }
        
        # Only save to history.json if save_history toggle is enabled!
        if req.save_history:
            history = read_history()
            history.insert(0, history_item)  # Newest first
            write_history(history)
        
        # Notify WebSocket of completion
        if connection_id in active_connections and event_loop:
            ws = active_connections[connection_id]
            asyncio.run_coroutine_threadsafe(
                ws.send_json({
                    "type": "complete",
                    "wav_url": wav_url,
                    "filename": "primary.wav"
                }),
                event_loop
            )
            
        # Trigger background variant generation if we are in Audio-to-Audio mode
        if req.init_audio_path:
            background_tasks.add_task(
                generate_variants_in_background,
                req=req,
                session_id=session_id,
                primary_filename="primary.wav",
                center_noise=req.init_noise_level,
                actual_seed=actual_seed,
                connection_id=connection_id
            )
            
        return history_item
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # Notify WebSocket of error
        if connection_id in active_connections and event_loop:
            ws = active_connections[connection_id]
            asyncio.run_coroutine_threadsafe(
                ws.send_json({
                    "type": "error",
                    "error": str(e)
                }),
                event_loop
            )
            
        raise HTTPException(status_code=500, detail=str(e))

class AbortRequest(BaseModel):
    connection_id: str

@app.post("/api/abort")
def abort_generation(req: AbortRequest):
    """Signals the backend to stop generating any remaining background variants for this connection."""
    print(f"[Server] Received abort request for connection {req.connection_id}")
    active_aborts.add(req.connection_id)
    return {"status": "aborted"}

class FinerVariantsRequest(BaseModel):
    session_id: str
    connection_id: str
    current_variant_index: int
    center_noise: float
    actual_seed: int
    prompt: str
    negative_prompt: Optional[str] = ""
    seconds: float
    steps: int
    dit_name: str
    decoder_name: str
    cfg: float
    apg: float
    free_models: bool
    save_history: bool

@app.post("/api/generate_finer_variants")
def generate_finer_variants(req: FinerVariantsRequest, background_tasks: BackgroundTasks):
    """
    Triggers on-demand background generation of Seeds 1 to 4 for the active noise level
    and its immediate up/down neighbors.
    """
    connection_id = req.connection_id
    
    # Identify target indices (current, one up, one down)
    v = req.current_variant_index
    target_indices = []
    if 0 <= v <= 20:
        target_indices.append(v)
        if v - 1 >= 0:
            target_indices.append(v - 1)
        if v + 1 <= 20:
            target_indices.append(v + 1)
            
    # Build tasks: seeds 1..4 for each target level
    tasks = []
    for idx in target_indices:
        delta = round(-0.20 + idx * 0.02, 2)
        for seed_idx in range(1, 5):
            tasks.append((delta, seed_idx))
            
    # Map FinerVariantsRequest to GenerateRequest
    gen_req = GenerateRequest(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        init_audio_path=str(engine.output_dir / f"session_{req.session_id}/primary.wav"),  # Absolute path resolved on disk
        init_noise_level=req.center_noise,
        inpaint_start=None,
        inpaint_end=None,
        seconds=req.seconds,
        steps=req.steps,
        dit_name=req.dit_name,
        decoder_name=req.decoder_name,
        cfg=req.cfg,
        apg=req.apg,
        free_models=req.free_models,
        connection_id=req.connection_id,
        save_history=req.save_history
    )
    
    print(f"[Server] Queueing on-demand finer variants for level {v} (neighbors: {target_indices}) - {len(tasks)} tasks.")
    
    background_tasks.add_task(
        execute_background_generation_queue,
        tasks=tasks,
        req=gen_req,
        session_id=req.session_id,
        primary_filename="primary.wav",
        center_noise=req.center_noise,
        actual_seed=req.actual_seed,
        connection_id=connection_id
    )
    return {"status": "queued", "task_count": len(tasks)}

@app.get("/api/audio/{filepath:path}")
def get_audio(filepath: str):
    """Serves generated audio files, supporting nested session directories."""
    file_path = engine.output_dir / filepath
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path=file_path, media_type="audio/wav")

@app.get("/api/uploads/{filename}")
def get_uploaded_audio(filename: str):
    """Serves uploaded audio files."""
    file_path = engine.upload_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Uploaded file not found")
    # Determine media type
    ext = Path(filename).suffix.lower()
    media_type = "audio/wav"
    if ext == ".mp3":
        media_type = "audio/mpeg"
    elif ext == ".m4a":
        media_type = "audio/mp4"
    elif ext == ".flac":
        media_type = "audio/flac"
    elif ext == ".ogg":
        media_type = "audio/ogg"
    return FileResponse(path=file_path, media_type=media_type)

@app.get("/api/history")
def get_history():
    """Returns the history of generations."""
    return read_history()

@app.delete("/api/history/{item_id}")
def delete_history_item(item_id: str):
    """Deletes an item from history and purges its entire session directory."""
    history = read_history()
    new_history = []
    deleted = False
    
    for item in history:
        if item["id"] == item_id:
            # Delete the entire session folder neatly
            try:
                session_dir = engine.output_dir / f"session_{item_id}"
                if session_dir.exists() and session_dir.is_dir():
                    print(f"[Server] Deleting session folder: {session_dir}")
                    import shutil
                    shutil.rmtree(session_dir, ignore_errors=True)
                else:
                    # Fallback to single file deletion for legacy history items
                    wav_path = engine.output_dir / item["filename"]
                    if wav_path.exists():
                        os.remove(wav_path)
            except Exception as e:
                print(f"Failed to delete session folder: {e}")
            deleted = True
        else:
            new_history.append(item)
            
    if not deleted:
        raise HTTPException(status_code=404, detail="History item not found")
        
    write_history(new_history)
    return {"status": "success", "message": "Item deleted"}

@app.websocket("/ws/{connection_id}")
async def websocket_endpoint(websocket: WebSocket, connection_id: str):
    """WebSocket connection for real-time progress updates."""
    await websocket.accept()
    active_connections[connection_id] = websocket
    print(f"[WebSocket] Connected client: {connection_id}")
    try:
        while True:
            # Keep connection alive by waiting for messages (e.g. pings)
            data = await websocket.receive_text()
            # If client sends a ping, respond with pong
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        print(f"[WebSocket] Disconnected client: {connection_id}")
    finally:
        if connection_id in active_connections:
            del active_connections[connection_id]
