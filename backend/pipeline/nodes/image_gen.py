import logging
from pathlib import Path

from backend.models.config import settings
from backend.pipeline.editorial import STYLE_CONSTRAINTS
from backend.pipeline.state import PipelineState
from backend.providers.image_backend import get_image_backend

logger = logging.getLogger(__name__)


def ensure_seed_image(seed_path: Path) -> None:
    """Ensure seed.png exists by drawing a default stickman image if missing."""
    if seed_path.exists():
        return
    logger.info("seed.png not found. Generating default stickman seed image...")
    try:
        from PIL import Image, ImageDraw
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(img)
        # Head (circle)
        draw.ellipse([216, 120, 296, 200], outline="black", width=4)
        # Eyes
        draw.ellipse([236, 145, 244, 153], fill="black")
        draw.ellipse([268, 145, 276, 153], fill="black")
        # Body
        draw.line([256, 200, 256, 360], fill="black", width=4)
        # Arms
        draw.line([256, 240, 180, 290], fill="black", width=4)
        draw.line([256, 240, 332, 290], fill="black", width=4)
        # Legs
        draw.line([256, 360, 196, 450], fill="black", width=4)
        draw.line([256, 360, 316, 450], fill="black", width=4)
        img.save(seed_path)
        logger.info("Default seed.png generated at %s", seed_path)
    except Exception as exc:
        logger.error("Failed to generate default seed.png: %s", exc)


def image_gen_node(state: PipelineState) -> dict:
    """LangGraph node: generate one image per video prompt via ComfyUI."""
    if state.get("error"):
        return {}

    # --- Project-based mode ---
    if state.get("project_id") and state.get("scenes"):
        return _image_gen_project(state)

    # --- Legacy prompt mode ---
    personas_dir = settings.personas_dir
    persona = settings.active_persona
    persona_dir = Path(personas_dir) / persona

    seed_image_path = persona_dir / "seed.png"
    ensure_seed_image(seed_image_path)
    if not seed_image_path.exists():
        return {
            "error": (
                f"image_gen_node: seed.png not found at {seed_image_path}. "
                "Please add a seed image for the active persona."
            )
        }

    # Load character visual description
    char_md_path = persona_dir / "character.md"
    char_description = ""
    if char_md_path.exists():
        char_description = char_md_path.read_text().strip()[:500]

    backend = get_image_backend()
    job_id = state["job_id"]
    output_base = Path(settings.outputs_dir) / job_id
    output_base.mkdir(parents=True, exist_ok=True)

    image_paths: list[str] = []
    progress_entries: list[str] = []

    for n, prompt in enumerate(state.get("video_prompts", [])):
        if char_description:
            full_prompt = f"{char_description}. {prompt}"
        else:
            full_prompt = prompt

        output_path = str(output_base / f"scene_{n:02d}.png")

        try:
            saved_path = backend.generate_image(
                prompt=full_prompt,
                seed_image_path=str(seed_image_path),
                output_path=output_path,
            )
            image_paths.append(saved_path)
            progress_entries.append(f"image_gen_node: scene {n:02d} → {saved_path}")
        except Exception as exc:
            return {
                "error": f"image_gen_node: failed on scene {n:02d}: {exc}",
                "image_paths": image_paths,
                "progress_log": progress_entries,
            }

    return {
        "image_paths": image_paths,
        "progress_log": progress_entries,
    }


def _image_gen_project(state: PipelineState) -> dict:
    """Handle per-scene image generation in project mode."""
    from backend.storage.database import get_sync_session
    from backend.storage.project_repo import repo

    project_id = state["project_id"]
    scenes = state["scenes"]

    personas_dir = settings.personas_dir
    active_persona = settings.active_persona
    persona_dir = Path(personas_dir) / active_persona

    seed_image_path = persona_dir / "seed.png"
    ensure_seed_image(seed_image_path)
    if not seed_image_path.exists():
        return {
            "error": (
                f"image_gen_node: seed.png not found at {seed_image_path}. "
                "Please add a seed image for the active persona."
            )
        }

    # Load character description from state persona (preferred) or from file
    persona = state.get("persona") or {}
    char_description = ""
    if persona.get("character"):
        char_description = persona["character"][:300]
    else:
        char_md_path = persona_dir / "character.md"
        if char_md_path.exists():
            char_description = char_md_path.read_text().strip()[:300]

    backend = get_image_backend()
    output_base = Path(settings.outputs_dir) / project_id
    output_base.mkdir(parents=True, exist_ok=True)

    progress_entries: list[str] = []
    image_paths: list[str] = []

    sorted_scenes = sorted(scenes, key=lambda s: s["order"])

    for i, scene in enumerate(sorted_scenes):
        prompt = scene.get("image_prompt", "")
        if char_description:
            full_prompt = f"{char_description}. {prompt}"
        else:
            full_prompt = prompt

        output_path = str(output_base / f"scene_{scene['order']:02d}.png")

        try:
            saved_path = backend.generate_image(
                prompt=full_prompt,
                seed_image_path=str(seed_image_path),
                output_path=output_path,
            )
        except Exception as exc:
            return {
                "error": f"image_gen_node: failed on scene {i:02d}: {exc}",
                "image_paths": image_paths,
                "progress_log": progress_entries,
            }

        scene["image_path"] = saved_path
        image_paths.append(saved_path)

        # Persist to DB
        try:
            with get_sync_session() as session:
                repo.update_scene(session, project_id, scene["id"], image_path=saved_path)
        except Exception as db_exc:
            progress_entries.append(f"image_gen_node: DB update warning for scene {i}: {db_exc}")

        msg = f"image_gen_node: scene {scene['order']:02d} → {saved_path}"
        progress_entries.append(msg)

    return {
        "scenes": sorted_scenes,
        "image_paths": image_paths,
        "progress_log": progress_entries,
    }
