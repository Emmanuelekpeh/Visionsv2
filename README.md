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
- **Camera direction**: hybrid/suggestions/director/off — decides which shots move, then continuity-repairs them (R2/R3/R6)
- **Sequence grammar**: shots carry a purpose and a link; scenes must cover one action, not one new place per cut
- **Visual organisation**: the hold/change spine — places and figures held across shots, rapid-change runs declared, anchors injected into the render prompt
- **Automatic sequence direction**: the film is split into movements; each gets hold/coverage or montage by evidence from the shots, with camera moves scoped to the movements that can carry them. No mode picker.
- **Camera moves settle**: one motivated move per hold is applied to the delivered frames in finishing (travel, then hold), because a generated camera LoRA cannot settle. Generation stays still.
- **Previs gate**: rendering is gated on a current previs report; missing/stale previs is started automatically and a measured same-scene break blocks the render.
- **Coverage-first writing**: a cheap scene plan commits the writer to 2-3 places, one recurring figure, and one concrete **light + palette per scene** before any shot; the writer's coverage is measured with the director's own place/figure affinity, and a story that holds but has no scene gets one re-ask for held scenes (a montage stays legitimate — it is a writing demand, not a gate).
- **Length scales by shot count**: target seconds → shots at the renderer's real motion budget; longer films buy more shots, never stretched holds
- **Hold conditioning**: held shots are conditioned on their run's anchor frame with IP-Adapter, so a place/figure can actually stay the same (measured: CLIP to reference 0.88 vs 0.80)
- **Previs**: one keyframe per shot, measured at the cuts (and now per-shot prompt alignment, with the prompt printed under each still), before committing the 45-90 min render
- **Output audit**: `output_audit.py --store <slug>` measures a delivered cut against its own prompts and flags weak prompt→frame alignment, hard/exposure cuts, metronomic pacing and a resolve that introduces imagery no earlier shot had
- **Taste capture**: the A/B pick becomes a labelled row with its measured properties, not a lost conversation
- **Auto-paced cut**: holds are fitted to the motion the renderer actually generated, so shots do not duplicate frames
- **Story gate**: a proposal whose beats go backwards, or that is a row of postcards, is refused at adoption unless you override it
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

The story generation uses an Ollama ensemble by default: three cheap "idea farm"
models brainstorm visual ideas, then one showrunner model (`granite4.1:8b`,
`story.SHOWRUNNER`) authors the final shot list. Swap the showrunner with
`--showrunner` or by editing `SHOWRUNNER_CANDIDATES`; `compare_showrunners.py`
measures candidates on the same concept and the same idea pool.

The writer is a **loop**, not a one-shot guess. It plans 2-3 **held scenes** first
(one place, one recurring figure, 2-4 shots each), then writes the shot count the
target length needs with a numbered beat roster (`story.beat_layout` -- shot 1
establishes, the turn lands near the end, the last shot resolves). If the story
gate rejects a draft, the next pass is handed the validator's exact complaints,
the previous shot list and the beat roster; if the story holds but no scene does
(every cut a new place), it re-asks once for 2-3 held scenes; when a plan exists,
every prompt is bound to its scene's place/figure **anchors** and the film is cut
on the planned scenes, which is what lets the director hold them; a clean-but-short
list is extended with coverage shots inside its existing scenes rather than
re-written; a refused draft is kept at `productions/<slug>/draft_attempt.json` so
pressing "write new ideas" continues from it instead of starting blind; and a
final text-only pass (`story.rewrite_prompts`, measured on the first rendered
film) rewrites every prompt as ONE concrete image, because the renderer draws one
clause of a two-image prompt and the first film's resolve became a car.

```bash
# Install Ollama: https://ollama.ai/download
# Then pull the models:
ollama pull granite4.1:8b
ollama pull qwen2.5:3b
ollama pull phi3
ollama pull mistral
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

1. **Write view**: Enter your film concept (e.g., "A lone astronaut explores an abandoned space station") and a target length; the shot count is derived from the renderer's motion budget. Click "write the shot list". The ensemble runs a few minutes to ~20 minutes on CPU, because the writer loops until the story holds and extends a short list to the requested count.

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
- **12 frames instead of 16** is available in settings: ~27% faster on this machine, but
  the default is 16 (new productions record 16), because AnimateDiff's motion module was
  trained on 16-frame windows and a lower count can read as a still frame. The renderer
  prints a warning below 16.

Performance measurements are in `HANDOFF.md`.

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

- **[HANDOFF.md](HANDOFF.md)**: Technical handoff document (architecture, design decisions)
- **[PACING_AND_FLOW_RESEARCH.md](PACING_AND_FLOW_RESEARCH.md)**: Shot lengths, pacing, watchability - what limits them and what was fixed
- **[PACING_MEASURED_STUDY.md](PACING_MEASURED_STUDY.md)**: Our own measurement of how human short videos are cut (700 MSR-VTT clips, validated cut detector)
- **[SEQUENCE_AB.md](SEQUENCE_AB.md)**: Rendered A/B of the same action cut as a montage vs as coverage, measured and watched
- **[VISUAL_ORGANISATION.md](VISUAL_ORGANISATION.md)**: The hold/change spine — one object for places, figures, palettes, pacing and prompts
- **[STUDIO_LOOP.md](STUDIO_LOOP.md)**: The three render-side layers — IP-Adapter hold conditioning, previs before render, taste capture
- **[SHOWRUNNER_TEST.md](SHOWRUNNER_TEST.md)**: Measured comparison of showrunner models

Pacing tools: `python pacing_audit.py` audits every production's holds against
the motion the renderer actually generated; `python pacing_study.py --corpus msrvtt`
measures human short-video pacing with our own cut detector;
`python camera_ab.py --store <slug>` compares camera-direction policies;
`python camera_ab_visual.py --a <takeA> --b <takeB>` measures two rendered takes;
`python sequence_audit.py` replays the sequence grammar over the archive;
`python sequence_ab.py --current _seq_current --coverage _seq_coverage` measures
two rendered cuts of one action; `python visual_org.py --store <slug>` shows and
diagnoses the hold/change organisation; `python previs.py --store <slug>` renders
one still per shot through the render's own look stage (shared technical baseline
first, then each scene's palette) and reports continuity split into WITHIN-scene
holds (defects) vs ACROSS scene cuts (style), plus per-shot prompt alignment,
before the render;
`python style_pass.py --store <slug>` strips fog/mist from the prompts and gives
each held scene its own light clause and palette (fog out, a legible grade in);
`python retime.py --store <slug>` gives an adopted film a real rhythm (near-cap
holds alternating with floor accents) and re-finishes from the takes on disk --
timing is a finishing input, so no shot is regenerated;
`python probe_sheet.py --store <slug> --shots 1,5,9` builds a contact sheet of
REAL rendered take frames (after `make_film --store <slug> --only 1,5,9`) -- the
look gate, because previs stills are a different pipeline from the film;
`python output_audit.py --store <slug>` measures a DELIVERED cut
against the prompts that made it (prompt→frame alignment/margins, cut continuity,
pacing, the resolve's new imagery) and writes contact/flow sheets under
`output/audit/<slug>/`;
`python taste.py summary` reports the recorded A/B preferences;
`python compare_showrunners.py` measures showrunner candidates on a shared idea pool.

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
- Reduce frame count to 12 in Settings → "raw frames per shot" (faster, but motion may read as still; the default is 16).
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
