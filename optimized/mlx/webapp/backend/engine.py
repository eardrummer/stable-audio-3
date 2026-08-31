import os
import sys
import math
import time
import random
import wave
import subprocess
from pathlib import Path
import numpy as np
import mlx.core as mx

# Setup paths so that imports from the parent mlx project resolve correctly
BACKEND_DIR = Path(__file__).resolve().parent
REPO_DIR = BACKEND_DIR.parent.parent  # optimized/mlx/
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "scripts"))

from models.defs.sa3_pipeline import (
    SecondsTotalEmbedder,
    apply_prompt_padding,
    build_pingpong_schedule,
    sample_flow_pingpong,
    patched_decode,
    load_conditioner_from_npz,
)
from models.defs.t5gemma_mlx import T5Gemma
from weights import ensure_local, is_present

# Global Constants
SAMPLE_RATE = 44100
SAMPLES_PER_LATENT = 4096

# Choices mapping from sa3_mlx.py
DIT_CHOICES = {
    "sm-music": {
        "loader": "models.defs.dit_mlx",
        "ckpt": "models/mlx/dit_sm-music_f16.npz",
        "default_decoder": "same-s"
    },
    "sm-sfx": {
        "loader": "models.defs.dit_mlx",
        "ckpt": "models/mlx/dit_sm-sfx_f16.npz",
        "default_decoder": "same-s"
    },
    "medium": {
        "loader": "models.defs.dit_mlx_medium",
        "ckpt": "models/mlx/dit_medium_f16.npz",
        "default_decoder": "same-l"
    },
}

DECODER_CHOICES = {
    "same-s": ("models.defs.same_s_decoder", "decode_chunked", (8, 2), "models/mlx/same_s_decoder_f32.npz"),
    "same-l": ("models.defs.same_l_decoder", "decode_chunked", (128, 8), "models/mlx/same_l_decoder_f32.npz"),
}

ENCODER_CHOICES = {
    "same-s": ("models.defs.same_s_encoder", 32, "models/mlx/same_s_encoder_f32.npz"),
    "same-l": ("models.defs.same_l_encoder", 16, "models/mlx/same_l_encoder_f32.npz"),
}

T5GEMMA_NPZ_REL = "models/mlx/t5gemma_f16.npz"

class StableAudioEngine:
    def __init__(self):
        self.t5_gemma = None
        self.dit_model = None
        self.dit_name = None
        self.decoder = None
        self.decoder_name = None
        self.encoder = None
        self.encoder_name = None
        
        # Create output directory
        self.output_dir = BACKEND_DIR / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temp upload directory
        self.upload_dir = BACKEND_DIR / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def _free_to_pool(self):
        """Force garbage collection and flush MLX's cache to release unified memory."""
        import gc
        gc.collect()
        fn = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
        if fn:
            fn()

    def get_models_status(self):
        """Check which models are locally downloaded and ready."""
        from weights import bundle_status, DIT_BUNDLES
        status = {}
        for name in DIT_BUNDLES:
            present, total = bundle_status(name)
            status[name] = {
                "downloaded": present == total,
                "present_files": present,
                "total_files": total
            }
        # Check T5Gemma status
        status["t5gemma"] = {
            "downloaded": is_present(T5GEMMA_NPZ_REL)
        }
        return status

    def preload_model_bundle(self, dit_name: str):
        """Downloads/caches all weights needed for the selected DiT model, codec, and text encoder."""
        from weights import ensure_local, DIT_BUNDLES
        print(f"[Engine] Preloading model bundle for '{dit_name}'...")
        # 1. Text Encoder
        ensure_local(T5GEMMA_NPZ_REL)
        # 2. DiT & Codec
        if dit_name in DIT_BUNDLES:
            for item in DIT_BUNDLES[dit_name]:
                rel_path = item[0] if isinstance(item, tuple) else item
                ensure_local(rel_path)
        print(f"[Engine] Model bundle '{dit_name}' successfully cached.")

    def load_t5gemma(self):
        if self.t5_gemma is None:
            print("[Engine] Loading T5Gemma text encoder...")
            t5_path = str(ensure_local(T5GEMMA_NPZ_REL))
            self.t5_gemma = T5Gemma.from_npz(t5_path)
            print("[Engine] T5Gemma loaded.")
        return self.t5_gemma

    def load_dit(self, dit_name: str, T_lat: int, dtype):
        if self.dit_model is None or self.dit_name != dit_name:
            print(f"[Engine] Loading DiT model: {dit_name}...")
            cfg = DIT_CHOICES[dit_name]
            ckpt = ensure_local(cfg["ckpt"])
            import importlib, io, contextlib
            mod = importlib.import_module(cfg["loader"])
            with contextlib.redirect_stdout(io.StringIO()):
                # Load DiT and keep warm
                self.dit_model = mod.load_dit(str(ckpt), T_lat=T_lat, dtype=dtype, compile_=False)
            self.dit_name = dit_name
            print(f"[Engine] DiT {dit_name} loaded.")
        return self.dit_model

    def load_decoder(self, decoder_name: str, dtype):
        if self.decoder is None or self.decoder_name != decoder_name:
            print(f"[Engine] Loading Decoder: {decoder_name}...")
            spec = DECODER_CHOICES[decoder_name]
            weights_path = ensure_local(spec[3])
            import importlib, io, contextlib
            mod = importlib.import_module(spec[0])
            with contextlib.redirect_stdout(io.StringIO()):
                self.decoder = mod.load_model(weights_path=str(weights_path), dtype=dtype, compile_=False)
            self.decoder_name = decoder_name
            print(f"[Engine] Decoder {decoder_name} loaded.")
        
        spec = DECODER_CHOICES[decoder_name]
        import importlib
        mod = importlib.import_module(spec[0])
        chunk_fn = getattr(mod, spec[1])
        chunk_cfg = spec[2]
        return self.decoder, chunk_fn, chunk_cfg

    def load_encoder(self, decoder_name: str, dtype):
        # We pair encoder with decoder
        if self.encoder is None or self.encoder_name != decoder_name:
            print(f"[Engine] Loading Encoder for decoder: {decoder_name}...")
            spec = ENCODER_CHOICES[decoder_name]
            weights_path = ensure_local(spec[2])
            import importlib, io, contextlib
            mod = importlib.import_module(spec[0])
            with contextlib.redirect_stdout(io.StringIO()):
                self.encoder = mod.load_model(weights_path=str(weights_path), dtype=dtype, compile_=False)
            self.encoder_name = decoder_name
            print(f"[Engine] Encoder loaded.")
        
        spec = ENCODER_CHOICES[decoder_name]
        pad_modulo = spec[1]
        return self.encoder, pad_modulo

    def save_wav(self, path: str, audio: np.ndarray):
        """audio: (channels, T) float32 in [-1, 1]. Writes 16-bit PCM stereo WAV."""
        if not np.isfinite(audio).all():
            raise RuntimeError("Audio contains non-finite samples (NaN/Inf)")
        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767.0).astype(np.int16).T  # (T, channels) interleaved
        with wave.open(path, "wb") as w:
            w.setnchannels(audio.shape[0])
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm.tobytes())

    def read_wav(self, path: str) -> np.ndarray:
        """Read a WAV file. Returns (2, T) float32 in [-1, 1]. Mono is duplicated to stereo."""
        try:
            with wave.open(path, "rb") as w:
                nch, sw, sr, nframes = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
                if sr == SAMPLE_RATE and sw == 2:
                    raw = np.frombuffer(w.readframes(nframes), dtype=np.int16).astype(np.float32) / 32767.0
                    if nch == 1:
                        return np.stack([raw, raw], axis=0)
                    return raw.reshape(-1, nch).T[:2]
        except wave.Error:
            pass

        # Fallback to ffmpeg
        try:
            result = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "2", "-"],
                capture_output=True, check=True
            )
            raw = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32767.0
            return raw.reshape(-1, 2).T
        except Exception as e:
            raise RuntimeError(f"Failed to read WAV file {path}: {str(e)}")

    def patch_audio(self, audio: np.ndarray, patch_size: int = 256) -> np.ndarray:
        B, C, T = audio.shape
        assert T % patch_size == 0, f"audio length {T} not multiple of patch_size {patch_size}"
        L = T // patch_size
        x = audio.reshape(B, C, L, patch_size).transpose(0, 1, 3, 2)
        return x.reshape(B, C * patch_size, L)

    def generate(
        self,
        prompt: str,
        negative_prompt: str = None,
        init_audio_path: str = None,
        init_audio_array: np.ndarray = None,
        init_audio_path_2: str = None,
        blend_ratio: float = 0.5,
        blend_method: str = None,
        init_noise_level: float = 1.0,
        inpaint_range: tuple[float, float] = None,
        seconds: float = 30.0,
        steps: int = 8,
        seed: int = None,
        dit_name: str = "sm-music",
        decoder_name: str = "same-s",
        cfg: float = 1.0,
        apg: float = 1.0,
        free_models: bool = False,
        progress_callback = None,
        out_path: str = None,
        channel_weights=None,
        renormalize: bool = False,
        latent_sink: list = None,
    ) -> str:
        """
        Runs the Stable Audio 3 generation pipeline.
        Returns the absolute path to the generated WAV file.
        """
        t_start = time.time()
        
        # Seed setup
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        
        # Map default decoder if not specified or set to "auto"
        if not decoder_name or decoder_name == "auto":
            decoder_name = DIT_CHOICES[dit_name]["default_decoder"]

        has_init = init_audio_path is not None or init_audio_array is not None
        print(f"[Engine] Starting generation: prompt='{prompt}', mode={'inpaint' if inpaint_range else 'audio-to-audio' if has_init else 'text-to-audio'}, seed={seed}")

        # Derivations
        dtype = mx.float16  # Use fp16 for DiT
        T_lat = max(1, math.ceil(seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))
        
        # SAME-S decoder padding requirement (divisible by 2)
        if decoder_name == "same-s" and T_lat % 2 != 0:
            T_lat += 1
            
        target_samples = T_lat * SAMPLES_PER_LATENT
        actual_dur = target_samples / SAMPLE_RATE

        # Output filename setup
        if out_path:
            out_wav_path = Path(out_path)
        else:
            import re
            slug = re.sub(r'[^a-z0-9]+', '_', prompt.lower()).strip('_')[:48]
            filename = f"{slug}_{seed}.wav" if slug else f"out_{seed}.wav"
            out_wav_path = self.output_dir / filename

        # 1. Text encoding (T5Gemma)
        t0 = time.time()
        t5_model = self.load_t5gemma()
        embeds, mask = t5_model.encode([prompt], max_len=256)
        mx.eval(embeds, mask)
        print(f"[Engine] T5Gemma encode finished in {(time.time()-t0)*1000:.0f} ms")
        
        # Free T5Gemma if requested
        if free_models:
            print("[Engine] Freeing T5Gemma model...")
            self.t5_gemma = None
            self._free_to_pool()

        # 2. Conditioning
        t0 = time.time()
        dit_ckpt = str(ensure_local(DIT_CHOICES[dit_name]["ckpt"]))
        padding_emb, secs_embedder = load_conditioner_from_npz(dit_ckpt, prefix="cond.")
        
        embeds = embeds.astype(dtype)
        embeds_padded = apply_prompt_padding(embeds, mask, padding_emb.astype(dtype))
        seconds_embed = secs_embedder(seconds).astype(dtype)
        cross_attn = mx.concatenate([embeds_padded, seconds_embed], axis=1)
        global_cond = seconds_embed[:, 0, :]

        # Negative prompt conditioning for CFG
        null_cross_attn = None
        if cfg != 1.0:
            if negative_prompt:
                # Reload T5Gemma if we freed it earlier (not recommended to free if using CFG + Neg prompt)
                t5_model = self.load_t5gemma()
                neg_embeds, neg_mask = t5_model.encode([negative_prompt], max_len=256)
                mx.eval(neg_embeds, neg_mask)
                neg_embeds = neg_embeds.astype(dtype)
                neg_padded = apply_prompt_padding(neg_embeds, neg_mask, padding_emb.astype(dtype))
                null_cross_attn = mx.concatenate([neg_padded, seconds_embed], axis=1)
            else:
                null_cross_attn = mx.zeros_like(cross_attn)
            mx.eval(null_cross_attn)

        mx.eval(cross_attn, global_cond)
        print(f"[Engine] Conditioning finished in {(time.time()-t0)*1000:.0f} ms")

        # 3a. Audio-to-Audio / Inpaint / Extend encoding
        # `init_audio_array` is the in-memory equivalent of `init_audio_path`:
        # the live loop hands over already-decoded (2, T) float32 audio so a
        # generation every few seconds never has to round-trip through disk.
        init_latents = None
        if init_audio_path or init_audio_array is not None:
            t0 = time.time()
            enc_model, pad_mod = self.load_encoder(decoder_name, mx.float32)

            def prep_audio(src):
                arr = src if isinstance(src, np.ndarray) else self.read_wav(src)
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 1:
                    arr = np.stack([arr, arr], axis=0)
                if arr.shape[-1] >= target_samples:
                    arr = arr[:, :target_samples]
                else:
                    pad = target_samples - arr.shape[-1]
                    arr = np.pad(arr, ((0, 0), (0, pad)), mode="constant")
                return arr

            def encode_waveform_to_latents(arr: np.ndarray):
                arr_batch = arr[None, ...]  # Add batch dimension: (1, 2, T)
                patches = self.patch_audio(arr_batch, patch_size=256)
                assert patches.shape[-1] % pad_mod == 0
                latents = enc_model(mx.array(patches))
                mx.eval(latents)
                return latents.astype(dtype)

            audio_np = prep_audio(init_audio_array if init_audio_array is not None else init_audio_path)

            if init_audio_path_2 and blend_method == "waveform":
                print(f"[Engine] Performing Pre-Encoder Waveform Mix (A vs B ratio = {blend_ratio:.2f})")
                audio_np_2 = prep_audio(init_audio_path_2)
                audio_np = (1.0 - blend_ratio) * audio_np + blend_ratio * audio_np_2
                # Anti-clipping peak normalization
                max_val = np.max(np.abs(audio_np))
                if max_val > 1.0:
                    audio_np = audio_np / max_val
                init_latents = encode_waveform_to_latents(audio_np)
            elif init_audio_path_2 and blend_method == "latent_linear":
                print(f"[Engine] Performing Post-Encoder Latent Linear Mix (A vs B ratio = {blend_ratio:.2f})")
                audio_np_2 = prep_audio(init_audio_path_2)
                z_1 = encode_waveform_to_latents(audio_np)
                z_2 = encode_waveform_to_latents(audio_np_2)
                init_latents = (1.0 - blend_ratio) * z_1 + blend_ratio * z_2
                mx.eval(init_latents)
            else:
                init_latents = encode_waveform_to_latents(audio_np)

            print(f"[Engine] Init audio encode finished in {(time.time()-t0)*1000:.0f} ms")
            
            if free_models:
                print("[Engine] Freeing Encoder model...")
                self.encoder = None
                self._free_to_pool()

        # 3b. DiT sampling Setup
        t0 = time.time()
        dit_model = self.load_dit(dit_name, T_lat=T_lat, dtype=dtype)
        sigmas = build_pingpong_schedule(steps, sigma_max=init_noise_level, use_logsnr_shift=True)
        
        key = mx.random.key(seed)
        pure_noise = mx.random.normal((1, 256, T_lat), dtype=dtype, key=key)
        
        # Set up starting noise
        # If inpainting/extending, we start with pure noise (paste_back handles keeping the unmasked region)
        # If standard audio-to-audio, we mix the init latents with noise
        inpaint_latents_range = None
        if inpaint_range is not None:
            # Convert inpaint range from seconds to latents
            inp_start_lat = max(0, int(round(inpaint_range[0] * SAMPLE_RATE / SAMPLES_PER_LATENT)))
            inp_end_lat = min(T_lat, int(round(inpaint_range[1] * SAMPLE_RATE / SAMPLES_PER_LATENT)))
            inpaint_latents_range = (inp_start_lat, inp_end_lat)

        if init_latents is not None and inpaint_latents_range is None:
            # Linear interpolation for standard audio-to-audio
            noise = init_latents * (1.0 - init_noise_level) + pure_noise * init_noise_level
        else:
            noise = pure_noise
        mx.eval(noise)

        # Build local conditioning for Inpainting / Extend
        local_add_cond = None
        paste_back = None
        if inpaint_latents_range is not None and init_latents is not None:
            s0, s1 = inpaint_latents_range
            # mask: 1 = keep, 0 = regenerate
            mask_np = np.ones((1, 1, T_lat), dtype=np.float32)
            mask_np[:, :, s0:s1] = 0.0
            mask = mx.array(mask_np)
            
            masked_input = init_latents.astype(mx.float32) * mask
            lac = mx.concatenate([mask, masked_input], axis=1).transpose(0, 2, 1).astype(dtype)
            local_add_cond = lac
            paste_back = (init_latents, mask)
            print(f"[Engine] Inpainting mask set for latents {s0}..{s1} (representing seconds {inpaint_range[0]}s..{inpaint_range[1]}s)")

        # DiT model function with CFG & APG
        def model_fn(x, t):
            if cfg == 1.0:
                return dit_model(x, t, cross_attn, global_cond, local_add_cond=local_add_cond)

            # Batched CFG
            x2 = mx.concatenate([x, x], axis=0)
            t2 = mx.concatenate([t, t], axis=0)
            cross2 = mx.concatenate([cross_attn, null_cross_attn], axis=0)
            global2 = mx.concatenate([global_cond, global_cond], axis=0)
            lac2 = None if local_add_cond is None else mx.concatenate([local_add_cond, local_add_cond], axis=0)
            
            v_batched = dit_model(x2, t2, cross2, global2, local_add_cond=lac2)
            cond_v, uncond_v = mx.split(v_batched, 2, axis=0)

            # Convert to denoised space (RF: denoised = x - t * v)
            sigma = t.reshape(-1, 1, 1).astype(mx.float32)
            cond_d = x.astype(mx.float32) - cond_v.astype(mx.float32) * sigma
            uncond_d = x.astype(mx.float32) - uncond_v.astype(mx.float32) * sigma
            diff = cond_d - uncond_d

            if apg <= 0.0:
                cfg_diff = diff
            else:
                # Adaptive Projected Guidance
                norm = mx.sqrt((cond_d * cond_d).sum(axis=(-2, -1), keepdims=True))
                unit = cond_d / mx.maximum(norm, 1e-8)
                parallel = (diff * unit).sum(axis=(-2, -1), keepdims=True) * unit
                diff_orth = diff - parallel
                cfg_diff = diff_orth if apg >= 1.0 else (apg * diff_orth + (1.0 - apg) * diff)

            cfg_d = cond_d + (cfg - 1.0) * cfg_diff
            cfg_v = (x.astype(mx.float32) - cfg_d) / sigma
            return cfg_v.astype(x.dtype)

        # Progress callback hook
        t_step_prev = [time.time()]
        def _on_step(i: int, total: int):
            now = time.time()
            elapsed_ms = (now - t_step_prev[0]) * 1000
            t_step_prev[0] = now
            if progress_callback:
                progress_callback(i, total, elapsed_ms)
            print(f"[Engine] Sampling step {i}/{total} - {elapsed_ms:.0f} ms")

        # 3c. DiT sampling loop
        t_sample_start = time.time()
        latents = sample_flow_pingpong(
            model_fn, noise, sigmas, seed=seed + 1,
            paste_back=paste_back, on_step=_on_step
        )
        mx.eval(latents)
        print(f"[Engine] DiT sampling finished in {time.time()-t_sample_start:.2f} s")

        if free_models:
            print("[Engine] Freeing DiT model...")
            self.dit_model = None
            self._free_to_pool()

        # 4 & 5. Decode -> unpatch -> WAV.
        self.decode_latents_to_wav(
            latents, decoder_name=decoder_name, seconds=seconds,
            out_path=str(out_wav_path), free_models=free_models,
            channel_weights=channel_weights, renormalize=renormalize,
        )

        if latent_sink is not None:
            # Callers that want to re-decode later (the live app's latent shelf)
            # keep the pre-decode latents without changing this method's
            # long-standing return type.
            latent_sink.append(latents)

        total_time = time.time() - t_start
        print(f"[Engine] Generation completed successfully in {total_time:.2f}s. Saved to {out_wav_path}")

        return str(out_wav_path)

    def apply_channel_weights(self, latents: mx.array, weights, renormalize: bool = False) -> mx.array:
        """Scale each of the 256 latent channels by `weights` (a 256-vector in [0,1]).

        This is direct surgery on the latent geometry: channel c is attenuated
        toward silence as weights[c] -> 0. Because SAME's softnorm bottleneck
        expects a particular overall scale, heavily attenuated curves drift off
        the learned manifold; `renormalize` restores the original global RMS so
        timbral change can be judged separately from a simple loss of level.
        """
        if weights is None:
            return latents
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        n_ch = latents.shape[1]
        if w.shape[0] != n_ch:
            raise ValueError(f"expected {n_ch} channel weights, got {w.shape[0]}")
        wm = mx.array(w).reshape(1, n_ch, 1).astype(latents.dtype)
        out = latents * wm
        if renormalize:
            before = mx.sqrt(mx.mean(latents.astype(mx.float32) ** 2))
            after = mx.sqrt(mx.mean(out.astype(mx.float32) ** 2))
            out = out * (before / mx.maximum(after, 1e-6)).astype(out.dtype)
        mx.eval(out)
        return out

    def decode_latents_to_wav(
        self,
        latents: mx.array,
        decoder_name: str,
        seconds: float,
        out_path: str,
        free_models: bool = False,
        channel_weights=None,
        renormalize: bool = False,
    ) -> str:
        """Decode latents to audio and write a WAV. Split out of generate() so the
        live app can re-decode a cached latent under a new channel-weight curve
        without paying for sampling again (~49 ms vs ~330 ms for a 5 s clip)."""
        T_lat = latents.shape[-1]

        latents = self.apply_channel_weights(latents, channel_weights, renormalize)

        t0 = time.time()
        decoder_model, chunk_fn, (chunk, ovl) = self.load_decoder(decoder_name, mx.float32)
        latents_fp32 = latents.astype(mx.float32)

        kernel = chunk + 2 * ovl
        if T_lat > kernel:
            patches = chunk_fn(decoder_model, latents_fp32, chunk, ovl)
        elif T_lat % 2 == 0:
            patches = decoder_model(latents_fp32)
        else:
            patches = chunk_fn(decoder_model, latents_fp32, 2, 2)

        mx.eval(patches)
        print(f"[Engine] Decoding finished in {(time.time()-t0)*1000:.0f} ms")

        if free_models:
            print("[Engine] Freeing Decoder model...")
            self.decoder = None
            self._free_to_pool()

        # Unpatch -> audio + save WAV
        t0 = time.time()
        audio = patched_decode(patches, patch_size=256, channels=2)
        mx.eval(audio)
        audio_np = np.array(audio.astype(mx.float32))[0]

        # Trim to exact requested seconds
        requested_samples = int(round(seconds * SAMPLE_RATE))
        if audio_np.shape[-1] > requested_samples:
            audio_np = audio_np[..., :requested_samples]

        self.save_wav(out_path, audio_np)
        print(f"[Engine] Unpatch & write WAV finished in {(time.time()-t0)*1000:.0f} ms")
        return out_path
