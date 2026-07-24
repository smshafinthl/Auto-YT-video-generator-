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
from pathlib import Path

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
        "--unload-model",
        action="store_true",
        help=(
            "Fully unload the Wan I2V model from GPU VRAM between each run. "
            "Safer for large batches on small GPUs but adds reload time per video."
        ),
    )

    args = parser.parse_args()

    # ── Resolve prompts ──────────────────────────────────────────────────────
    if args.prompts_file:
        prompts = _load_prompts_from_file(args.prompts_file)
    else:
        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]

    if not prompts:
        print("[ERROR] No valid prompts found — nothing to generate.", file=sys.stderr)
        return 1

    logger.info("Loaded %d prompts. Output → %s", len(prompts), args.output_dir)

    # ── Run batch ────────────────────────────────────────────────────────────
    results = run_batch(
        prompts=prompts,
        output_dir=args.output_dir,
        stop_on_error=args.stop_on_error,
        unload_model_between_runs=args.unload_model,
    )

    # Exit 0 only if every video succeeded
    any_failed = any(r["error"] is not None for r in results)
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
