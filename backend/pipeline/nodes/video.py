import logging
import subprocess
from pathlib import Path

from backend.models.config import settings
from backend.pipeline.state import PipelineState

logger = logging.getLogger(__name__)


def _wan_model_available() -> bool:
    """Return True if the Wan 2.2 I2V model directory exists and is non-empty."""
    model_path = Path(settings.wan_model_path)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    model_path = model_path.resolve()
    if not model_path.exists():
        return False
    try:
        return any(model_path.iterdir())
    except PermissionError:
        return False


def _ffmpeg_image_to_clip(
    image_path: str,
    output_path: str,
    duration_seconds: int = 5,
) -> str:
    """
    Compile a single static image into a video clip using FFmpeg.
    Loops the image at 30fps for duration_seconds.
    Output is an H.264 MP4 compatible with downstream assembly.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", image_path,
        "-c:v", "libx264",
        "-t", str(duration_seconds),
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(Path(output_path).resolve())


def _extract_thumbnail(video_path: str, thumbnail_path: str) -> None:
    """Extract the first frame of a video as a JPEG thumbnail via ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        thumbnail_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _generate_clip_with_fallback(
    prompt: str,
    image_path: str,
    clip_path: str,
    duration_seconds: int = 5,
) -> tuple[str, str]:
    """
    Try Wan 2.2 I2V first. If model is absent or generation fails, fall back
    to FFmpeg image-to-video compilation.

    Returns:
        (saved_path, method) where method is "wan" or "ffmpeg"
    """
    if _wan_model_available():
        try:
            from backend.providers.video_backend import get_video_backend
            backend = get_video_backend()
            saved_path = backend.generate_clip(
                prompt=prompt,
                first_frame_image_path=image_path,
                output_path=clip_path,
                duration_seconds=duration_seconds,
            )
            return saved_path, "wan"
        except Exception as exc:
            logger.warning(
                "Wan I2V generation failed for clip %s (Error: %s). "
                "Falling back to FFmpeg image-to-video compilation.",
                clip_path, exc
            )

    # Wan model not available or failed — compile image into video with FFmpeg
    logger.info(
        "Wan 2.2 model not found at %s. Using FFmpeg image-to-video fallback for: %s",
        settings.wan_model_path, clip_path
    )
    saved_path = _ffmpeg_image_to_clip(
        image_path=image_path,
        output_path=clip_path,
        duration_seconds=duration_seconds,
    )
    return saved_path, "ffmpeg"


def video_node(state: PipelineState) -> dict:
    """
    LangGraph node: generate one video clip per scene.

    If the Wan 2.2 I2V model exists, uses LocalWanBackend for animated clips.
    If the model is absent or fails, gracefully falls back to compiling each
    scene image into a static clip via FFmpeg so the batch always completes.
    """
    if state.get("error"):
        return {}

    # --- Project-based mode ---
    if state.get("project_id") and state.get("scenes"):
        return _video_node_project(state)

    # --- Legacy prompt mode ---
    image_paths = state.get("image_paths", [])
    if not image_paths:
        return {"error": "video_node: image_paths is empty — image_gen_node must run first"}

    job_id = state["job_id"]
    output_base = Path(settings.outputs_dir) / job_id
    output_base.mkdir(parents=True, exist_ok=True)

    video_paths: list[str] = []
    scene_thumbnails: list[str] = []
    progress_entries: list[str] = []

    video_prompts = state.get("video_prompts", [])

    for n, (prompt, image_path) in enumerate(zip(video_prompts, image_paths)):
        clip_path = str(output_base / f"clip_{n:02d}.mp4")
        thumb_path = str(output_base / f"thumb_{n:02d}.jpg")

        try:
            saved_path, method = _generate_clip_with_fallback(
                prompt=prompt,
                image_path=image_path,
                clip_path=clip_path,
            )
            video_paths.append(saved_path)
            clip_msg = f"video_node: clip {n:02d} [{method}] → {saved_path}"
            progress_entries.append(clip_msg)
            if state.get("progress_queue") is not None:
                state["progress_queue"].put(clip_msg)

            # Extract thumbnail
            _extract_thumbnail(saved_path, thumb_path)
            scene_thumbnails.append(thumb_path)
            thumb_msg = f"video_node: thumbnail {n:02d} → {thumb_path}"
            progress_entries.append(thumb_msg)
            if state.get("progress_queue") is not None:
                state["progress_queue"].put(thumb_msg)

        except Exception as exc:
            return {
                "error": f"video_node: failed on clip {n:02d}: {exc}",
                "video_paths": video_paths,
                "scene_thumbnails": scene_thumbnails,
                "progress_log": progress_entries,
            }

    return {
        "video_paths": video_paths,
        "scene_thumbnails": scene_thumbnails,
        "progress_log": progress_entries,
    }


def _video_node_project(state: PipelineState) -> dict:
    """Handle per-scene video generation in project mode with graceful Wan fallback."""
    from backend.storage.database import get_sync_session
    from backend.storage.project_repo import repo

    project_id = state["project_id"]
    scenes = state["scenes"]

    output_base = Path(settings.outputs_dir) / project_id
    output_base.mkdir(parents=True, exist_ok=True)

    video_paths: list[str] = []
    scene_thumbnails: list[str] = []
    progress_entries: list[str] = []

    sorted_scenes = sorted(scenes, key=lambda s: s["order"])

    for i, scene in enumerate(sorted_scenes):
        prompt = scene.get("video_prompt", "")
        image_path = scene.get("image_path", "")
        clip_path = str(output_base / f"clip_{scene['order']:02d}.mp4")
        thumb_filename = f"thumb_{scene['order']:02d}.jpg"
        thumb_path = str(output_base / thumb_filename)

        try:
            saved_path, method = _generate_clip_with_fallback(
                prompt=prompt,
                image_path=image_path,
                clip_path=clip_path,
            )
            video_paths.append(saved_path)
            scene["video_clip_path"] = saved_path

            clip_msg = f"video_node: clip {scene['order']:02d} [{method}] → {saved_path}"
            progress_entries.append(clip_msg)
            if state.get("progress_queue") is not None:
                state["progress_queue"].put(clip_msg)

            # Extract thumbnail
            _extract_thumbnail(saved_path, thumb_path)
            scene["thumbnail_path"] = thumb_filename
            scene_thumbnails.append(thumb_path)

            thumb_msg = f"video_node: thumbnail {scene['order']:02d} → {thumb_path}"
            progress_entries.append(thumb_msg)
            if state.get("progress_queue") is not None:
                state["progress_queue"].put(thumb_msg)

        except Exception as exc:
            return {
                "error": f"video_node: failed on clip {i:02d}: {exc}",
                "video_paths": video_paths,
                "scene_thumbnails": scene_thumbnails,
                "progress_log": progress_entries,
            }

        # Persist to DB
        try:
            with get_sync_session() as session:
                repo.update_scene(
                    session,
                    project_id,
                    scene["id"],
                    video_clip_path=saved_path,
                    thumbnail_path=thumb_filename,
                )
        except Exception as db_exc:
            progress_entries.append(f"video_node: DB update warning for scene {i}: {db_exc}")

    return {
        "scenes": sorted_scenes,
        "video_paths": video_paths,
        "scene_thumbnails": scene_thumbnails,
        "progress_log": progress_entries,
    }
