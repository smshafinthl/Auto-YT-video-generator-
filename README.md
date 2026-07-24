# Auto-YT-Video-Generator

An automated AI pipeline for generating **faceless YouTube Shorts** at scale. Feed it a list of topics, it produces fully edited MP4 videos — complete with voiceover, stickman animation, and a subscribe CTA — with no manual intervention.

**GitHub:** [smshafinthl/Auto-YT-video-generator-](https://github.com/smshafinthl/Auto-YT-video-generator-)

**Stack:** Python 3.11 · FastAPI · LangGraph · LangChain · ElevenLabs TTS · ComfyUI · Wan 2.2 I2V · FFmpeg · SQLite · React 18 · TypeScript · Vite · Tailwind v4

---

## ✨ What's New (Latest Update)

### 🔁 Bulk Video Generation Pipeline
Pass a list of 30+ topics at once. The system loops through each one, runs the full 5-step pipeline, and exports `video_01.mp4`, `video_02.mp4`, etc. to an output folder automatically.

- VRAM cleanup (`torch.cuda.empty_cache()` + `gc.collect()`) runs after **every** video to prevent OOM crashes during long batch runs
- Optional `--unload-model` flag fully destroys the Wan I2V pipeline cache between runs (for very long batches on small GPUs)
- Errors are logged per-video and the batch continues by default (use `--stop-on-error` to abort)

### 📣 Auto Subscribe CTA
Every generated voiceover script now automatically ends with a short, natural call-to-action. It's injected into the LLM system prompt so the CTA always fits the topic:
- Science → *"Subscribe for more mind-blowing science!"*
- Space → *"Follow for daily space discoveries!"*
- History → *"Like and subscribe for more history!"*

---

## How It Works

```
prompts.json (list of 30+ topics)
        │
        ▼
  batch_main.py  ──────────────────────────────────────────
        │                                                   │
        │  for each topic:                                  │
        ▼                                                   │
  ┌─────────────────────────────────────────────────┐      │
  │         LangGraph Pipeline (5 nodes)            │      │
  │                                                 │      │
  │  1. scripting_node  — LLM: script + CTA         │      │
  │  2. audio_node      — ElevenLabs TTS            │      │
  │  3. image_gen_node  — ComfyUI img2img           │      │
  │  4. video_node      — Wan 2.2 I2V animation     │      │
  │  5. assembly_node   — FFmpeg duration sync      │      │
  └─────────────────────────────────────────────────┘      │
        │                                                   │
        ▼                                                   │
  outputs/batch/video_NN.mp4    ←─────────────────────────┘
        │
        ▼
  _free_vram()  ←  runs after every video (OOM prevention)
```

### Alternate Mode: Interactive Editorial Wizard (Web UI)

```
Research Doc (Markdown)
        │
        ▼
  5-Step Editorial Wizard (Web UI)
  │  1. Angles  — LLM proposes 3 story angles
  │  2. Story   — LLM writes full narration blocks
  │  3. Scenes  — LLM breaks story into video scenes
  │  4. Music   — Choose background track
  │  5. Generate — Pipeline runs, MP4 produced
        │
        ▼
  Final MP4 (voiceover + stickman animation + music)
```

### Visual Style

All generated videos use a locked stickman character style enforced at every stage:

> `flat 2D vector art, unshaded, solid #FFFFFF white background, black lines, zero shadows, minimal character motion, zero camera movement`

Character consistency is achieved via a **two-stage img2img pipeline**: a seed image of the stickman is passed to Flux/ComfyUI for img2img conditioning, and the generated scene image is used as the first-frame anchor for Wan 2.2 I2V.

---

## Architecture

```
Auto-YT-Video-Generator/
├── batch_main.py               # ★ NEW — CLI for bulk video generation (30+ videos)
├── prompts.example.json        # ★ NEW — 30 ready-to-use YouTube Shorts topics
├── main.py                     # CLI entrypoint (single prompt mode)
├── backend/
│   ├── api/routes/             # FastAPI route handlers
│   │   ├── generate.py         # Prompt-based generation (legacy)
│   │   ├── history.py          # Job history
│   │   ├── music.py            # Music track selection
│   │   └── projects.py         # Full project CRUD + editorial endpoints
│   ├── assets/
│   │   ├── music/              # MP3 tracks (gitignored) + tracks.json
│   │   └── personas/
│   │       └── default/
│   │           ├── personality.md   # Narrator voice + tone
│   │           ├── character.md     # Visual character description
│   │           └── seed.png         # Stickman reference image
│   ├── models/                 # Pydantic + SQLModel schemas
│   ├── pipeline/
│   │   ├── batch_runner.py     # ★ NEW — Core batch loop with VRAM cleanup
│   │   ├── editorial.py        # LLM functions (angles, story, scenes, regen)
│   │   ├── graph.py            # LangGraph StateGraph (5 nodes)
│   │   ├── nodes/
│   │   │   ├── scripting.py    # ★ UPDATED — SYSTEM_PROMPT now includes CTA
│   │   │   ├── audio.py        # ElevenLabs TTS
│   │   │   ├── image_gen.py    # ComfyUI img2img
│   │   │   ├── video.py        # Wan 2.2 I2V
│   │   │   ├── assembly.py     # MovieLite + FFmpeg
│   │   │   └── persona.py      # Persona loader
│   │   └── state.py            # PipelineState TypedDict
│   ├── providers/              # TTS, image, video, LLM abstractions
│   ├── storage/                # SQLite DB, sessions, job store, project repo
│   └── tests/                  # Unit tests
├── frontend/
│   └── src/
│       ├── components/wizard/  # AnglesStep, StoryStep, ScenesStep, MusicStep, GenerateStep
│       ├── hooks/              # useGenerate, useHistory
│       ├── lib/api.ts          # Full typed API client
│       └── types.ts            # Shared TypeScript interfaces
├── docs/plans/                 # 9 implementation plans
└── scripts/
    ├── verify_mps.py           # Verify PyTorch MPS works on Apple Silicon
    └── download_model.sh       # Download Wan 2.2 I2V model (~20GB)
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11 | Required — 3.12+ breaks `skia-python` dependency |
| uv | latest | Fast Python package manager |
| FFmpeg | 6+ | Required for audio/video assembly |
| ComfyUI | latest | For image generation (local) |
| Wan 2.2 I2V | 5B | ~20GB model download |
| ElevenLabs | API Key | For TTS voiceover |
| Bifrost / OpenAI | API Key | LLM for script generation |

> **Kaggle / Colab users:** FFmpeg is pre-installed. This project is fully batch-runnable on Kaggle with a T4/P100 GPU.

---

## Setup

### 1. Clone and create venv (Python 3.11 required)

```bash
git clone git@github.com:smshafinthl/Auto-YT-video-generator-.git
cd Auto-YT-video-generator-

# Create venv with Python 3.11 specifically
uv venv --python 3.11
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

uv pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in:
# - BIFROST_API_KEY (LLM gateway) or OPENAI_API_KEY
# - ELEVENLABS_API_KEY + ELEVENLABS_VOICE_ID
# - WAN_MODEL_PATH (after downloading the model)
```

### 3. Set up ComfyUI (Image Generation)

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git external/comfyui
pip install -r external/comfyui/requirements.txt

# Download Flux checkpoint:
huggingface-cli download black-forest-labs/FLUX.1-schnell \
  --include "flux1-schnell.safetensors" \
  --local-dir external/comfyui/models/checkpoints/

# Start ComfyUI, build img2img workflow, export API JSON to:
# backend/assets/comfyui_img2img_workflow.json
python external/comfyui/main.py --port 8188 --listen 127.0.0.1
```

### 4. Download Wan 2.2 I2V model (~20GB)

```bash
bash scripts/download_model.sh
# Then set WAN_MODEL_PATH=./models/wan2.2-i2v-5b in .env
```

### 5. Create stickman seed image

Create `backend/assets/personas/default/seed.png` — a simple stickman on a white background. Size: 512×512 or 1024×1024 PNG.

### 6. Add music tracks (optional)

Place royalty-free MP3 files in `backend/assets/music/` and update `backend/assets/music/tracks.json`.

---

## Running: Batch Mode (⭐ Main Feature)

### Prepare your prompts

Edit or create a JSON file with an array of topic strings:

```json
[
  "Why black holes are invisible",
  "How volcanoes actually form",
  "The deepest part of the ocean",
  "How the ancient Egyptians built the pyramids"
]
```

A ready-to-use example with **30 topics** is included: `prompts.example.json`

### Run the batch

```bash
# Generate all videos from a prompts file:
python batch_main.py --prompts-file prompts.example.json

# Custom output directory:
python batch_main.py --prompts-file prompts.json --output-dir outputs/run_01

# Quick inline test (comma-separated):
python batch_main.py --prompts "Black holes,Volcanoes,Deep sea creatures"

# Recommended for Kaggle T4 GPU (16GB VRAM) — fully unloads model between runs:
python batch_main.py --prompts-file prompts.example.json --unload-model

# Abort on first error instead of continuing:
python batch_main.py --prompts-file prompts.json --stop-on-error
```

Output files: `outputs/batch/video_01.mp4`, `video_02.mp4`, …

### VRAM Cleanup (OOM Prevention)

After every video, the batch runner automatically calls:

```python
gc.collect()
torch.cuda.empty_cache()    # Clear cached CUDA tensors
torch.cuda.ipc_collect()    # Clean up IPC handles
# With --unload-model:
LocalWanBackend._pipeline = None   # Fully release 20GB model weights
```

### Running on Kaggle

```python
# In a Kaggle notebook cell:
import subprocess, sys

subprocess.run([
    sys.executable, "batch_main.py",
    "--prompts-file", "prompts.example.json",
    "--output-dir", "outputs/batch",
    "--unload-model",  # recommended on T4 (16GB VRAM)
])
```

---

## Running: Single Video (CLI)

```bash
python main.py --prompt "Why black holes are completely invisible"
```

---

## Running: Web UI (Editorial Wizard)

```bash
# Backend
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000
# API docs: http://localhost:8000/docs

# Frontend
cd frontend
bun install
bun run dev
# Opens at http://localhost:5173
```

### Using the Wizard

1. **Create a project** — paste a research document (Markdown, 200+ words), set target duration
2. **Choose an angle** — the LLM proposes 3 story angles; pick one or regenerate
3. **Edit story** — drag blocks to reorder, edit text inline; confirm when ready
4. **Edit scenes** — edit dialog, image/video prompts; use ↺ Regen buttons; confirm when ready
5. **Select music** — preview tracks, pick one (or skip)
6. **Generate** — click Generate Video; watch SSE progress stream; download the final MP4

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/api/generate` | Prompt-based generation (legacy quick mode) |
| `GET` | `/api/generate/{id}/stream` | SSE progress stream |
| `GET` | `/api/history` | All generation jobs |
| `POST` | `/api/projects` | Create project |
| `GET` | `/api/projects` | List projects |
| `GET` | `/api/projects/{id}` | Project detail |
| `POST` | `/api/projects/{id}/angles/generate` | Generate 3 story angles |
| `POST` | `/api/projects/{id}/angles/{angle_id}/choose` | Choose an angle |
| `POST` | `/api/projects/{id}/story/generate` | Generate story blocks |
| `POST` | `/api/projects/{id}/story/confirm` | Confirm story |
| `POST` | `/api/projects/{id}/scenes/generate` | Generate scenes from story |
| `POST` | `/api/projects/{id}/scenes/{scene_id}/regenerate` | Regenerate image/video prompt |
| `POST` | `/api/projects/{id}/scenes/confirm` | Confirm scenes |
| `GET` | `/api/music/tracks` | List available music tracks |
| `POST` | `/api/projects/{id}/music/select` | Select music track |
| `POST` | `/api/projects/{id}/generate` | Start project-based generation |

---

## Configuration

All configuration is via `.env`:

```bash
# LLM (Bifrost gateway — model swappable)
BIFROST_BASE_URL=https://opencode.ai/zen/go/v1
BIFROST_API_KEY=your_key
BIFROST_MODEL=gpt-4o-mini         # swap to gpt-4o, claude-3-5-sonnet, etc.

# TTS
ELEVENLABS_API_KEY=your_key
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM  # Rachel voice

# Video backend
VIDEO_BACKEND=local               # or: cloud
WAN_MODEL_PATH=./models/wan2.2-i2v-5b

# Image generation
COMFYUI_BASE_URL=http://127.0.0.1:8188

# Persona
ACTIVE_PERSONA=default            # matches folder under backend/assets/personas/
```

---

## Pipeline Detail

### LangGraph nodes (sequential)

```
START
  │
  ├── [scenes empty]  → scripting_node   ← CTA added to every script ✓
  └── [scenes set]    → load_persona_node
                              │
                          audio_node       (ElevenLabs TTS per scene)
                              │
                          image_gen_node   (ComfyUI img2img with seed.png anchor)
                              │
                          video_node       (Wan 2.2 I2V via diffusers)
                              │
                          assembly_node    (MovieLite + FFmpeg duration sync + music mix)
                              │
                            END → final.mp4
```

### Subscribe CTA — How It Works

The LLM system prompt in `scripting_node` contains a mandatory rule:

> The FINAL sentence of `voiceover_script` MUST be a short (2–6 words) call-to-action inviting the audience to subscribe, relevant to the video topic.

This means every video ends with a naturally written CTA voiced through ElevenLabs TTS — no hard-coded suffix, always topic-matched.

### Duration sync

After TTS synthesis, each clip's duration is measured with `ffprobe`:
- **Video shorter than audio** → FFmpeg `tpad` freeze-last-frame to fill the gap
- **Video longer than audio** → FFmpeg trim to match audio length

### Music mixing

Background music is overlaid at **-18dB** under the voiceover using FFmpeg's `amix` filter with `stream_loop -1` for looping.

---

## Development

### Run tests

```bash
# Windows
.venv\Scripts\pytest.exe backend/tests/ -v

# macOS/Linux
pytest backend/tests/ -v
```

### Lint

```bash
ruff check backend/
```

### Build frontend

```bash
cd frontend
bun run build
```

---

## Known Limitations

- **Video generation is slow** — Wan 2.2 I2V takes ~10–15 minutes per clip on Apple MPS; ~3–8 min on a Kaggle T4 GPU. Plan batch time accordingly.
- **ComfyUI requires manual workflow setup** — the img2img workflow JSON must be exported from the ComfyUI UI. See `docs/comfyui-workflow.md`.
- **Music tracks not included** — source CC0 tracks from [Pixabay Music](https://pixabay.com/music/) or [Free Music Archive](https://freemusicarchive.org/). Only `tracks.json` is committed.
- **`seed.png` must be created manually** — any simple black-on-white stickman drawing works.
- **Python 3.11 required** — `skia-python` (used by `movielite`) does not support Python 3.12+.

---

## Roadmap

- [x] Bulk batch generation pipeline (30+ videos)
- [x] Auto subscribe CTA on every video
- [ ] Multiple persona support (swap character per project)
- [ ] Cloud video backend integration (Replicate / fal.ai)
- [ ] Subtitle overlay via FFmpeg
- [ ] Thumbnail generation for YouTube upload
- [ ] Export `chapters.json` for YouTube chapter markers
- [ ] Scheduled daily publishing queue
