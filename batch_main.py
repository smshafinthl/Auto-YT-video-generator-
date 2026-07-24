#!/usr/bin/env python3
"""
batch_main.py — CLI entrypoint for bulk faceless YouTube Shorts generation.

Reads a list of prompts (from a JSON file or inline), runs the full pipeline
for each one (scripting → audio → image → video → assembly), and exports the
final MP4 files to an output directory as video_01.mp4, video_02.mp4, etc.

GPU/VRAM memory is cleaned between every video using torch.cuda.empty_cache()
and gc.collect() so the batch doesn't crash with OOM errors.

────────────────────────────────────────────────────────────────
Examples
────────────────────────────────────────────────────────────────

  # Run from a prompts JSON file (recommended for 30+ videos):
  python batch_main.py --prompts-file prompts.example.json

  # Quick inline test (comma-separated):
  python batch_main.py --prompts "Black holes explained,How volcanoes form,Deep sea creatures"

  # Custom output directory:
  python batch_main.py --prompts-file prompts.example.json --output-dir outputs/my_batch

  # Stop on first failure instead of continuing:
  python batch_main.py --prompts-file prompts.example.json --stop-on-error

  # Unload Wan I2V model from VRAM between each run (slower but safer for OOM):
  python batch_main.py --prompts-file prompts.example.json --unload-model

────────────────────────────────────────────────────────────────
prompts.json format
────────────────────────────────────────────────────────────────
  [
    "Why black holes are invisible",
    "How volcanoes actually form",
    "The deepest part of the ocean"
  ]
"""

import argparse
import json
import logging
import sys
import subprocess
from pathlib import Path

# ── Auto Dependency Check ──────────────────────────────────────────────────
def ensure_dependencies():
    packages = {
        "gTTS": "gTTS",
        "diffusers": "diffusers",
        "transformers": "transformers",
        "huggingface_hub": "huggingface_hub",
    }
    missing = []
    for mod, pkg in packages.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"[AUTO-INSTALL] Missing required packages: {missing}. Installing via pip...", flush=True)
        cmd = [sys.executable, "-m", "pip", "install"] + missing
        subprocess.run(cmd, check=True)
        print("[AUTO-INSTALL] Dependencies installed successfully.", flush=True)

ensure_dependencies()

# ── Wan 2.2 Model Auto-Downloader ──────────────────────────────────────────
def ensure_wan_model():
    model_dir = Path("./models/wan2.2").resolve()
    if not model_dir.exists() or not any(model_dir.iterdir()):
        print(f"[AUTO-DOWNLOAD] Wan 2.2 model not found at {model_dir}. Downloading Wan-AI/Wan2.1-I2V-14B-480P from HuggingFace...", flush=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="Wan-AI/Wan2.1-I2V-14B-480P",
                local_dir=str(model_dir),
                local_dir_use_symlinks=False
            )
            print(f"[AUTO-DOWNLOAD] Wan 2.2 model downloaded successfully to {model_dir}.", flush=True)
        except Exception as e:
            print(f"[AUTO-DOWNLOAD WARNING] Failed to download Wan model: {e}. Pipeline will fallback to image compilation if needed.", flush=True)

# Load .env before any settings/pipeline imports
from dotenv import load_dotenv
load_dotenv()

from backend.pipeline.batch_runner import run_batch  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_main")


def _load_prompts_from_file(path: str) -> list[str]:
    """Load and validate a JSON array of prompt strings from a file."""
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] Prompts file not found: {p}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Failed to parse JSON in {p}: {exc}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, list):
        print("[ERROR] Prompts file must be a JSON array (list).", file=sys.stderr)
        sys.exit(1)
    invalid = [i for i, item in enumerate(data) if not isinstance(item, str)]
    if invalid:
        print(
            f"[ERROR] All items in the prompts file must be strings. "
            f"Non-string items at indices: {invalid}",
            file=sys.stderr,
        )
        sys.exit(1)
    return [s.strip() for s in data if s.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="batch_main.py",
        description=(
            "faceless-gen batch: Generate multiple faceless YouTube Shorts "
            "from a list of prompts, with VRAM cleanup between each video."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python batch_main.py --prompts-file prompts.example.json\n"
            "  python batch_main.py --prompts \"Black holes,Volcanoes\" --output-dir outputs/test\n"
            "  python batch_main.py --prompts-file prompts.example.json --unload-model\n"
        ),
    )

    # Prompt source — mutually exclusive: file or inline
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--prompts-file",
        metavar="FILE",
        help=(
            "Path to a JSON file containing an array of prompt strings. "
            'E.g.: ["Black holes", "Volcanoes", "Deep sea creatures"]'
        ),
    )
    source.add_argument(
        "--prompts",
        metavar="P1,P2,...",
        help="Comma-separated list of prompts (for quick inline testing).",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/batch",
        metavar="DIR",
        help="Directory to write video_01.mp4, video_02.mp4 … (default: outputs/batch).",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the entire batch on the first pipeline error (default: skip and continue).",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL_NAME",
        help="LLM model name (e.g., Qwen/Qwen2.5-1.5B-Instruct, gemini-2.0-flash, gpt-4o-mini).",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "local", "gemini", "openai"],
        default=None,
        help="LLM provider: 'local' (HuggingFace), 'gemini', 'openai', or 'auto'.",
    )
    parser.add_argument(
        "--image-backend",
        choices=["local", "comfyui"],
        default=None,
        help="Image generation backend: 'local' (HuggingFace diffusers) or 'comfyui'.",
    )
    parser.add_argument(
        "--image-model",
        metavar="IMAGE_MODEL",
        default=None,
        help=(
            "Local diffusers model for image generation "
            "(e.g., stabilityai/sd-turbo, runwayml/stable-diffusion-v1-5). "
            "Only used when --image-backend=local."
        ),
    )
    parser.add_argument(
        "--unload-model",
        action="store_true",
        help=(
            "Fully unload all models (Wan I2V + image) from GPU VRAM between each run. "
            "Safer for large batches on small GPUs but adds reload time per video."
        ),
    )
    parser.add_argument(
        "--download-wan",
        action="store_true",
        help="Force download Wan 2.2 model before running the pipeline.",
    )

    args = parser.parse_args()

    # Override LLM + image settings if CLI flags provided
    from backend.models.config import settings as app_settings
    import os

    if args.provider:
        app_settings.llm_provider = args.provider
    if args.model:
        app_settings.local_llm_model = args.model
        app_settings.gemini_model = args.model
        app_settings.bifrost_model = args.model
    if args.image_backend:
        app_settings.image_backend = args.image_backend
    if args.image_model:
        app_settings.local_image_model = args.image_model

    # Ensure Wan model is available if requested or missing
    if args.download_wan:
        ensure_wan_model()
    else:
        # Also auto-check if model directory doesn't exist
        wan_path = Path(app_settings.wan_model_path)
        if not wan_path.is_absolute():
            wan_path = Path.cwd() / wan_path
        if not wan_path.exists() or not any(wan_path.resolve().iterdir()):
            ensure_wan_model()

    # Resolve output_dir to an ABSOLUTE path under cwd to prevent
    # Kaggle path nesting (Auto-YT.../Auto-YT.../...)
    from pathlib import Path
    raw_out = args.output_dir
    out_dir_path = Path(raw_out)
    if not out_dir_path.is_absolute():
        out_dir_path = Path.cwd() / out_dir_path
    resolved_out_dir = str(out_dir_path.resolve())

    # ── Resolve prompts ──────────────────────────────────────────────────────
    if args.prompts_file:
        prompts = _load_prompts_from_file(args.prompts_file)
    else:
        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]

    if not prompts:
        print("[ERROR] No valid prompts found — nothing to generate.", file=sys.stderr)
        return 1

    logger.info("Loaded %d prompts. Output → %s", len(prompts), resolved_out_dir)

    # ── Run batch ────────────────────────────────────────────────────────────
    results = run_batch(
        prompts=prompts,
        output_dir=resolved_out_dir,
        stop_on_error=args.stop_on_error,
        unload_model_between_runs=args.unload_model,
    )

    # Exit 0 only if every video succeeded
    any_failed = any(r["error"] is not None for r in results)
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
