# Visions v2

AI-powered film production system that generates short films from text descriptions. Combines AnimateDiff, LTX Video, motion LoRAs, and LLM-based shot planning into a complete pipeline.

## What It Does

**Write** → **Shoot** → **Cut**: Three-verb workflow to go from concept to finished film.

- **Write**: Describe your film in one sentence. An ensemble of LLMs generates a shot list with camera movements, framing, and scene descriptions.
- **Shoot**: Generate each shot using AnimateDiff + 14 motion LoRAs. RAFT upsampling + handheld camera shake for realism.
- **Cut**: Grade, assemble, and export as MP4.

Example: `"A detective walks through rain-soaked streets at night"` → 6 shots, ~30 seconds, fully rendered.

## Features

- **Stable shot IDs**: Reorder shots without regenerating anything
- **Immutable takes**: Every render is preserved; choose between takes like Final Cut auditions
- **Derived staleness**: Changes to prompt/camera mark shots stale automatically
- **Batched rendering**: Load the model once for N shots instead of N times
- **Effects chain**: Posterize, halftone, dither, film damage, pixel sort
- **Background jobs**: Rendering runs detached; close the UI and come back

## Requirements

### Hardware
- **CPU**: Works on CPU-only systems (tested without NVIDIA GPU)
- **RAM**: 16GB+ recommended (models are ~7GB)
- **Storage**: ~20GB for models, ~500MB per minute of generated footage

### Software
- **Python 3.10 or 3.11** (tested on 3.11)
- **FFmpeg** (required for RAFT upsampling and video export)
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
  - Or: `winget install ffmpeg` / `choco install ffmpeg`

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Emmanuelekpeh/Visionsv2.git
cd Visionsv2
```

### 2. Install Python dependencies

This project uses a custom package installer (`.add_pkgs.py`) that downloads wheels directly. Standard pip commands may not work depending on your environment.

**Core dependencies to install:**
```bash
# Option A: If pip works on your system
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install diffusers==0.32.1 transformers==4.45.2 accelerate==1.2.1
pip install huggingface_hub safetensors sentencepiece tokenizers
pip install streamlit pillow numpy opencv-python scipy
pip install moviepy edge-tts ollama

# Option B: Using the custom installer (if pip fails)
python .add_pkgs.py diffusers==0.32.1 transformers==4.45.2 accelerate==1.2.1
python .add_pkgs.py huggingface_hub safetensors sentencepiece tokenizers
# Continue for remaining packages...
```

**Full dependency list:**
- `torch` (CPU or CUDA)
- `diffusers` >= 0.32.1
- `transformers` >= 4.45.2
- `accelerate` >= 1.2.1
- `streamlit` (for UI)
- `pillow`, `opencv-python` (image processing)
- `numpy`, `scipy` (numerical)
- `moviepy` (video assembly)
- `edge-tts` (text-to-speech, optional)
- `ollama` (LLM inference)
- `huggingface_hub`, `safetensors` (model loading)

### 3. Download models

Models are loaded from the `models/` directory. You need:

#### Required models:
1. **AnimateDiff motion module** (v3)
   - Download: `guoyww/animatediff/mm_sd_v15_v3.ckpt`
   - Place in: `models/animatediff_motion_v3/`

2. **Stable Diffusion 1.5 base**
   - Download: `runwayml/stable-diffusion-v1-5`
   - Place in: `models/sd15/`

3. **14 Motion LoRAs** (for camera movements)
   - Download via `fetch_subject_loras.py` or manually from Hugging Face
   - Place in: `models/subject_loras/`
   - Required: zoom-in, zoom-out, tilt-up, tilt-down, pan-left, pan-right, etc.

4. **RAFT optical flow** (for upsampling)
   - Loaded automatically by `flow_interpolator.py`

5. **Super-resolution models** (optional, for upscaling)
   - `EDSR_x4.pb` and `FSRCNN_x4.pb` (included in repo)

#### Optional models:
- **LTX Video**: For longer-form generation (place in `models/ltx_video/`)
- **CLIP**: For vision analysis (auto-downloads on first use)

**Quick model download script:**
```bash
# Fetch motion LoRAs
python fetch_subject_loras.py

# Download base models (requires huggingface-cli)
huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir models/sd15
huggingface-cli download guoyww/animatediff --include "mm_sd_v15_v3.ckpt" --local-dir models/animatediff_motion_v3
```

### 4. Set up Ollama (for LLM-based shot planning)

The story generation uses Ollama with `qwen2.5:7b` by default.

```bash
# Install Ollama: https://ollama.ai/download
# Then pull the model:
ollama pull qwen2.5:7b
```

### 5. Configure environment (optional)

Create a `.env` file if you want custom paths:
```bash
HF_HOME=./.hf_home
TORCH_HOME=./.torch_home
```

## Usage

### Start the studio UI
```bash
streamlit run ui/studio.py
```

Opens at `http://localhost:8501` (or next available port).

### Three-step workflow:

1. **Write view**: Enter your film concept (e.g., "A lone astronaut explores an abandoned space station"). Click "write the shot list". Takes 4-7 minutes to generate ~6 shots.

2. **Shoot view**: Review each shot. Click "shoot it" to render. Each shot takes ~10 minutes on CPU (with preflight caching).
   - **Tip**: Use "shoot all remaining" to batch-render and save model load time.

3. **Cut view**: Preview the assembled film. Click "export the film" to render the final MP4.

### Command-line rendering

```bash
# Generate a specific shot
python make_film.py --store <project-slug> --only 1

# Render all unshot clips
python make_film.py --store <project-slug>

# Export final film
python make_film.py --store <project-slug> --name final_render
```

## Project Structure

```
visions/
├── ui/
│   └── studio.py          # Streamlit UI (main interface)
├── production_system.py   # Core pipeline: AnimateDiff + LoRAs
├── story.py               # LLM ensemble for shot list generation
├── film_store.py          # Production storage (shots, takes, state)
├── job_runner.py          # Background job management
├── make_film.py           # CLI renderer
├── flow_interpolator.py   # RAFT upsampling
├── handheld.py            # Camera shake simulation
├── effects.py             # Post-processing effects chain
├── looks.py               # Color grading presets
├── models/                # Model weights (not in repo)
├── productions/           # Generated projects (gitignored)
└── output/                # Job logs, temp files (gitignored)
```

## Performance

**Per-shot timing (CPU, 16 frames → RAFT 2x → 51 final frames):**
- Cold start (first shot): ~236s (57s model load + 179s preflight)
- Warm shots (cached): ~86s (preflight skipped)
- Diffusion: ~42s/frame
- RAFT upsampling: ~9s/shot
- **Total per shot: ~10 minutes** (after first)

**Speed optimizations:**
- **Preflight caching**: 148s saved per job after the first
- **Batched rendering**: Load model once for N shots (57s saved per additional shot)
- **12 frames instead of 16**: 27% faster, better stability (configurable in settings)

See `START_HERE.md` for detailed performance measurements.

## Testing

```bash
# Verify store logic
python recon_test/test_film_store.py

# Test rendering pipeline (no diffusion model)
python recon_test/test_store_render.py

# Check UI design rules
python recon_test/test_studio.py

# Measure shot list diversity
python recon_test/idea_diversity.py
```

## Documentation

- **[START_HERE.md](START_HERE.md)**: Detailed system overview, performance analysis
- **[REDESIGN.md](REDESIGN.md)**: UI design decisions (what was taken from CapCut, Premiere, Resolve, etc.)
- **[AUDIT.md](AUDIT.md)**: What was wrong with the old data flow
- **[HANDOFF.md](HANDOFF.md)**: Technical handoff document (architecture, design decisions)

## Known Limitations

1. **No audio**: Exports are silent. `add_narration.py` works but isn't wired to the pipeline.
2. **No timeline scrubbing**: The UI shows 4-frame thumbnails, not a video player.
3. **CPU-only performance**: 10 min/shot is acceptable for a few shots, not for feature-length.
4. **Model size**: ~7GB in RAM; OOM likely with <16GB RAM.

## Troubleshooting

**"Port already in use" when starting Streamlit:**
```bash
# Check what's using the port:
netstat -ano | findstr :8501
# Kill the process or use a different port:
streamlit run ui/studio.py --server.port 8502
```

**"Module not found" errors:**
- Make sure `.ltx_env/` is in your Python path, or install packages globally.
- Check that `sys.path.insert(0, str(ROOT / ".ltx_env"))` runs in the file.

**Models fail to load:**
- Verify models are in the correct `models/` subdirectories.
- Check file permissions (some model files are large and may fail partial downloads).

**Renders are slow:**
- Use "shoot all remaining" to batch-render and skip repeated model loads.
- Reduce frame count to 12 in Settings → "raw frames per shot".
- Close other applications to free RAM.

**LLM generation fails:**
- Make sure Ollama is running: `ollama serve`
- Try a smaller model: `ollama pull qwen2.5:3b` and update `story.py`.

## License

(Add your license here)

## Credits

Built using:
- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion)
- [AnimateDiff](https://github.com/guoyww/AnimateDiff)
- [RAFT](https://github.com/princeton-vl/RAFT)
- [Diffusers](https://github.com/huggingface/diffusers)
- [Streamlit](https://streamlit.io)
- [Ollama](https://ollama.ai)
