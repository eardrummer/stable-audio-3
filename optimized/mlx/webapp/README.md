# Stable Audio 3 Local Studio & Audio Blending Workstation

Welcome to the **Stable Audio 3 Local Studio** documentation. This dedicated guide explains the goal of the project, how our workstation fits into the broader Stable Audio 3 codebase, the repository structure, code paths, audio blending experiments, and setup instructions.

---

## 🎯 Goal of the Project

The goal of this project is to build an **ultra-fast, interactive local workstation** for Stable Audio 3 on Apple Silicon (MLX), equipped with advanced experimental features for generative audio creation:

1. **Apple Silicon Hardware Acceleration**: Powered by MLX and Metal GPU unified VRAM, running audio generation at **15x–17x faster than real-time**.
2. **Multi-Source Audio Blending Workstation**: A dedicated dual-deck interface allowing sound designers and musicians to blend, morph, and crossfade between two audio sources (**Source A** and **Source B**) before or during diffusion.
3. **2D Generation Matrix Diagnostics**: Real-time exploration of 21 noise levels and 5 seed variations per level, giving full visibility into model behavior across variation strengths.
4. **Synchronized Multi-Track Tape Deck**: Transport controls (Play/Pause, Rewind to Zero, Looping) driving perfect sample-accurate Web Audio API synchronization across input and output audio waveforms.

---

## 🧩 How Our Project Fits Inside the Stable Audio 3 Repository

The base repository (`Stability-AI/stable-audio-3`) contains reference PyTorch implementation models. Our project introduces a **standalone, full-stack Web Application** built on top of an Apple Silicon native MLX engine.

```
Stable-Audio-experiments/  (Repo Root)
├── README.md                           # Original Stability AI PyTorch README
├── run_studio.sh                       # 🚀 Launcher Script for Backend + Frontend
│
├── analysis/                           # 🧪 Research & Methodologies
│   └── audio_blending_analysis.md      # Detailed math & rankings for 5 blending methods
│
└── optimized/                          # ⚡ MLX Accelerated Suite
    └── mlx/
        ├── .venv/                      # Python virtual environment (MLX, FastAPI, etc.)
        ├── models/defs/                # DiT Transformer MLX model definitions
        │   ├── dit_mlx.py              # DiT Small/Medium architecture
        │   └── dit_mlx_medium.py
        │
        └── webapp/                     # 🎛️ OUR PROJECT WORKSTATION
            ├── README.md               # 👈 (This File)
            ├── backend/                # FastAPI Backend & Engine
            │   ├── main.py             # Server endpoints & WebSocket streaming
            │   ├── engine.py           # MLX Synthesis Engine & Blending algorithms
            │   └── .gitignore
            │
            └── frontend/               # React / Vite / WaveSurfer.js UI
                ├── src/
                │   ├── App.tsx         # Studio UI, dual-source state, transport deck
                │   ├── index.css       # Workstation styling & 2D Matrix panel
                │   └── main.tsx
                ├── package.json
                └── vite.config.ts
```

---

## 🛠️ Code Files & Key Paths

### 1. Workstation Launcher: `run_studio.sh`
- **Location**: `/run_studio.sh`
- **Function**: Automatically checks port availability (`8000` for backend, `5173` for frontend), starts the FastAPI Uvicorn server, launches the Vite dev server, and manages clean shutdown on `Ctrl+C`.

### 2. Backend Server: `main.py`
- **Location**: `/optimized/mlx/webapp/backend/main.py`
- **Responsibilities**:
  - `GenerateRequest` Pydantic model: Validates single and dual-source payloads (`init_audio_path_2`, `blend_ratio`, `blend_method`).
  - `/api/generate`: Primary endpoint triggering generation in a threadpool.
  - `/api/generate_finer_variants`: Background queue generating seed variations.
  - `WebSocket /ws/{connection_id}`: Real-time progress updates and completion events.
  - `resolve_audio_path()`: Maps web URLs (`/api/audio/`, `/api/uploads/`) to local file paths.

### 3. MLX Synthesis Engine: `engine.py`
- **Location**: `/optimized/mlx/webapp/backend/engine.py`
- **Responsibilities**:
  - `StableAudioEngine`: Loads T5Gemma text encoder, SAME VAE encoder/decoder, and DiT models into Apple Silicon VRAM.
  - `generate()`: Main pipeline function handling Rectified Flow sampling.
  - `encode_waveform_to_latents()`: Helper function patching and encoding audio arrays.
  - **Audio Blending Algorithms**: Executes time-domain and latent-domain blending modes.

### 4. Frontend Studio: `App.tsx` & `index.css`
- **Location**: `/optimized/mlx/webapp/frontend/src/App.tsx` & `index.css`
- **Responsibilities**:
  - Dual Deck UI: Single Source vs. Dual Source Blend tabs.
  - Independent Waveform Display: WaveSurfer.js rendering for Source A, Source B, and Generated Output.
  - Real-time Web Audio API: Live microphone recorder with oscilloscope, and spectrogram visualizers.
  - Tape Deck Transport Bar: Synchronized Play/Pause, RTZ (Return to Zero), and Master Volume.
  - 2D Matrix Monitor Panel: Always-visible panel displaying 21 noise levels × 5 seeds.

---

## 🧪 Audio Blending Methodology Experiments

Our workstation is designed to test five distinct ways of blending two audio sources ($x_1$ and $x_2$):

| Method | Status | Description | Formula / Mechanics |
| :--- | :---: | :--- | :--- |
| **A. Pre-Encoder Waveform Mix** | **✅ Implemented** | Time-domain sum before VAE encoding. Includes anti-clipping normalization. | $x_{\text{mix}} = (1-w)x_1 + w x_2$ |
| **B. Post-Encoder Latent Linear** | **✅ Implemented** | Encodes sources separately, then morphs latents linearly. | $z_{\text{mix}} = (1-w)\text{Enc}(x_1) + w \text{Enc}(x_2)$ |
| **C. Spherical Latent (Slerp)** | ⏳ Next Up | Morphing along a constant-energy spherical arc. | $z_{\text{mix}} = \text{Slerp}(z_1, z_2, w)$ |
| **D. Asymmetric Noising** | ⏳ Planned | Uses one source as a structural mold and another for noise injection. | $z_{\text{noisy}} = (1-w)z_1(1-\sigma) + w z_2(1-\sigma) + \epsilon\sigma$ |
| **E. Vector Field Interpolation** | ⏳ Planned | Interpolates DiT velocity fields dynamically at each sampling step. | $v_{\text{mix}} = (1-w)v_1(x_t, t) + w v_2(x_t, t)$ |

For a deep dive into the mathematics and quality rankings, see [analysis/audio_blending_analysis.md](../../analysis/audio_blending_analysis.md).

---

## 📦 Dependencies & Environment Setup

### Prerequisites
- **OS**: macOS (Apple Silicon M1/M2/M3/M4 recommended)
- **Python**: Python 3.10+
- **Node.js**: Node 18+

### Python Virtual Environment (`optimized/mlx/.venv`)
The backend relies on the MLX framework:
```bash
mlx >= 0.20.0
fastapi
uvicorn
numpy
scipy
soundfile
torch  # (For loading PyTorch checkpoints into MLX format)
huggingface_hub
```

### Frontend Dependencies (`optimized/mlx/webapp/frontend`)
```json
{
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "wavesurfer.js": "^7.8.6",
    "lucide-react": "^0.468.0"
  },
  "devDependencies": {
    "vite": "^8.0.16",
    "typescript": "^5.6.3"
  }
}
```

---

## 🚀 How to Run the Workstation

### First-Time Setup on a New Computer
If you have cloned this repository onto a new computer:

1. **One-Time MLX & Weight Download**:
   ```bash
   cd optimized/mlx
   ./install.sh
   cd ../../..
   ```
   *This installs `uv` (if missing), creates a Python virtual environment, installs MLX dependencies, and downloads/caches the Stable Audio 3 weights.*

2. **One-Time Frontend Setup**:
   ```bash
   cd optimized/mlx/webapp/frontend
   npm install
   cd ../../../..
   ```

3. **Launch the Studio**:
   From the repository root:
   ```bash
   ./run_studio.sh
   ```
4. Open your browser to **`http://localhost:5173`**. (FastAPI logs will be written to `optimized/mlx/webapp/backend/backend.log`).

---

## 💡 Guide for Iterating & Adding New Features

When adding new blending methods or UI controls:
1. **Adding a New Blending Method**:
   - Update `analysis/audio_blending_analysis.md` with the mathematical breakdown.
   - Extend `engine.generate()` in `engine.py` under section `3a` or `3b`.
   - Update `<select className="model-select-dropdown">` in `App.tsx` to enable the dropdown option.
2. **Modifying UI Layout**:
   - Make component changes in `App.tsx`.
   - Add styles to `index.css`.
   - Run `npx tsc --noEmit` inside `optimized/mlx/webapp/frontend` to verify TypeScript types cleanly.
