#!/usr/bin/env python3
"""
Batch Audio Variation Processor for Stable Audio 3 (Apple Silicon MLX)

Processes all audio files in an input directory (e.g. Google Drive folder)
and generates a dedicated folder for each file containing:
  - original.wav
  - 5 progressively noisier synthesized variants (+0.02 step)
  - metadata.json
"""

import os
import sys
import time
import json
import shutil
import random
import argparse
from pathlib import Path

# Setup paths to import the MLX Stable Audio Engine
REPO_ROOT = Path(__file__).resolve().parent
MLX_DIR = REPO_ROOT / "optimized" / "mlx"
BACKEND_DIR = MLX_DIR / "webapp" / "backend"

sys.path.insert(0, str(MLX_DIR))
sys.path.insert(0, str(MLX_DIR / "scripts"))
sys.path.insert(0, str(BACKEND_DIR))

from engine import StableAudioEngine, SAMPLE_RATE

SUPPORTED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.aif', '.aiff', '.ogg', '.aac'}

def format_duration(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

def get_audio_files(input_dir: Path, recursive: bool = False):
    pattern = "**/*" if recursive else "*"
    files = [
        f for f in input_dir.glob(pattern)
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS and not f.name.startswith('.')
    ]
    return sorted(files)

def main():
    parser = argparse.ArgumentParser(
        description="Stable Audio 3 - Batch Audio Variation Processor (MLX)"
    )
    parser.add_argument("-i", "--input-dir", required=True, type=str, help="Input directory containing audio files (e.g. Google Drive folder)")
    parser.add_argument("-o", "--output-dir", default="./batch_output", type=str, help="Destination directory for output subfolders (default: ./batch_output)")
    parser.add_argument("-p", "--prompt", default="", type=str, help="Global text prompt to guide synthesis (optional, default: unconditional)")
    parser.add_argument("-n", "--noise", default=0.30, type=float, help="Base variation / noise level (default: 0.30)")
    parser.add_argument("-d", "--delta", default=0.02, type=float, help="Noise increment for each successive variant (default: 0.02)")
    parser.add_argument("-v", "--variants", default=5, type=int, help="Number of noisier variants to generate per file (default: 5)")
    parser.add_argument("-s", "--steps", default=8, type=int, help="Diffusion denoising steps (default: 8)")
    parser.add_argument("-c", "--cfg", default=4.0, type=float, help="Classifier-Free Guidance strength (default: 4.0)")
    parser.add_argument("-m", "--model", default="sm-music", choices=["sm-music", "sm-sfx", "medium"], help="DiT model architecture (default: sm-music)")
    parser.add_argument("--seed", default=None, type=int, help="Base random seed (default: random per file)")
    parser.add_argument("-r", "--recursive", action="store_true", help="Scan input directory recursively")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing completed folders instead of skipping")

    args = parser.parse_args()

    input_path = Path(args.input_dir).expanduser().resolve()
    output_path = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists() or not input_path.is_dir():
        print(f"\n❌ Error: Input directory does not exist: {input_path}")
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    audio_files = get_audio_files(input_path, recursive=args.recursive)
    if not audio_files:
        print(f"\n⚠️  No supported audio files found in: {input_path}")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(0)

    print("\n" + "="*70)
    print(" 🎛️  STABLE AUDIO 3 - BATCH VARIATION PROCESSOR (MLX)")
    print("="*70)
    print(f"📁 Input Folder:    {input_path}")
    print(f"📂 Output Folder:   {output_path}")
    print(f"🎵 Audio Files:     {len(audio_files)} found")
    print(f"💬 Global Prompt:   \"{args.prompt}\"" if args.prompt else "💬 Global Prompt:   (Unconditional)")
    print(f"⚡ Model:           {args.model}")
    print(f"🎚️  Noise Curve:     {args.noise:.2f} base (+{args.delta:.2f} step x {args.variants} variants)")
    print(f"🔢 Steps:           {args.steps} | CFG: {args.cfg}")
    print("="*70 + "\n")

    # 1. Initialize MLX Engine
    print("⏳ Initializing MLX Neural Audio Engine...")
    t_init = time.time()
    engine = StableAudioEngine()
    engine.preload_model_bundle(args.model)
    print(f"✅ MLX Engine ready in {time.time() - t_init:.1f}s.\n")

    total_files = len(audio_files)
    batch_start_time = time.time()
    processed_count = 0
    skipped_count = 0

    for file_idx, audio_file in enumerate(audio_files, 1):
        file_stem = audio_file.stem
        # Clean folder name (alphanumeric and dashes/underscores)
        clean_folder_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in file_stem)
        file_out_dir = output_path / clean_folder_name

        # Calculate noise levels for the 5 variants
        noise_levels = [round(args.noise + i * args.delta, 4) for i in range(args.variants)]

        # Check if already completed
        metadata_file = file_out_dir / "metadata.json"
        all_variants_exist = (
            file_out_dir.exists() and
            (file_out_dir / "original.wav").exists() and
            all((file_out_dir / f"variant_{i+1}_noise_{nl:.2f}.wav").exists() for i, nl in enumerate(noise_levels)) and
            metadata_file.exists()
        )

        if all_variants_exist and not args.overwrite:
            print(f"[{file_idx}/{total_files}] ⏩ Skipping (already processed): {audio_file.name}")
            skipped_count += 1
            continue

        file_out_dir.mkdir(parents=True, exist_ok=True)
        file_start_time = time.time()

        print(f"[{file_idx}/{total_files}] 🎧 Processing: {audio_file.name}")

        # 2. Read and copy original audio
        try:
            audio_arr = engine.read_wav(str(audio_file))
            audio_duration = audio_arr.shape[-1] / SAMPLE_RATE
            # Save standardized original.wav in subfolder
            orig_wav_path = file_out_dir / "original.wav"
            engine.save_wav(str(orig_wav_path), audio_arr)
        except Exception as e:
            print(f"   ⚠️  Failed to read audio file: {e}")
            continue

        # Use same deterministic seed across all 5 variants of this file
        file_seed = args.seed if args.seed is not None else random.randint(1, 2_000_000_000)

        variant_records = []

        # 3. Generate the 5 noisier variants
        for v_idx, noise in enumerate(noise_levels, 1):
            out_variant_name = f"variant_{v_idx}_noise_{noise:.2f}.wav"
            out_variant_path = file_out_dir / out_variant_name

            print(f"   ├─ [{v_idx}/{args.variants}] Generating Variant {v_idx} (Noise: {noise:.2f}, Seed: {file_seed})...", end="", flush=True)
            v_t0 = time.time()

            try:
                res_path, res_dur, res_sr = engine.generate(
                    prompt=args.prompt,
                    negative_prompt="",
                    init_audio_path=str(orig_wav_path),
                    init_noise_level=noise,
                    seconds=audio_duration,
                    steps=args.steps,
                    dit_name=args.model,
                    cfg=args.cfg,
                    seed=file_seed,
                    out_path=str(out_variant_path)
                )
                v_elapsed = time.time() - v_t0
                print(f" Done ({v_elapsed:.2f}s)")

                variant_records.append({
                    "variant_index": v_idx,
                    "filename": out_variant_name,
                    "noise_level": noise,
                    "seed": file_seed,
                    "duration": round(res_dur, 2),
                    "generation_time_seconds": round(v_elapsed, 2)
                })
            except Exception as e:
                print(f" Failed: {e}")

        # 4. Save metadata.json manifest
        metadata = {
            "source_file": audio_file.name,
            "source_path": str(audio_file),
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": args.prompt,
            "model": args.model,
            "steps": args.steps,
            "cfg": args.cfg,
            "base_noise": args.noise,
            "delta_step": args.delta,
            "seed": file_seed,
            "duration_seconds": round(audio_duration, 2),
            "variants": variant_records
        }

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        file_elapsed = time.time() - file_start_time
        print(f"   └─ ✅ Folder complete: {clean_folder_name}/ ({file_elapsed:.1f}s)\n")
        processed_count += 1

    total_elapsed = time.time() - batch_start_time
    print("="*70)
    print(f"🎉 BATCH PROCESSING COMPLETE!")
    print(f"📊 Processed: {processed_count} files ({processed_count * args.variants} variants generated)")
    if skipped_count > 0:
        print(f"⏩ Skipped:   {skipped_count} files (already existed)")
    print(f"⏱️  Total Time: {format_duration(total_elapsed)}")
    print(f"📂 Output:     {output_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
