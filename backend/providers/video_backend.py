import abc
import logging
from pathlib import Path

from backend.models.config import settings
from backend.pipeline.editorial import STYLE_CONSTRAINTS

logger = logging.getLogger(__name__)


class VideoBackend(abc.ABC):
    """Abstract base class for video generation backends."""

    @abc.abstractmethod
    def generate_clip(
        self,
        prompt: str,
        first_frame_image_path: str,
        output_path: str,
        duration_seconds: int = 5,
    ) -> str:
        """
        Generate a video clip from a first-frame image and a prompt.

        Args:
            prompt: Text prompt describing the clip's motion.
            first_frame_image_path: Path to the anchor image (first frame).
            output_path: Destination path for the output MP4.
            duration_seconds: Desired clip length in seconds.

        Returns:
            Absolute path to the saved MP4.
        """
        ...


class LocalWanBackend(VideoBackend):
    """I2V backend using diffusers WanImageToVideoPipeline, with automatic FFmpeg fallback."""

    _pipeline = None  # Class-level lazy-loaded pipeline cache

    @classmethod
    def _get_pipeline(cls):
        if cls._pipeline is None:
            import torch
            from diffusers import WanImageToVideoPipeline

            logger.info("Loading WanImageToVideoPipeline from %s", settings.wan_model_path)
            pipe = WanImageToVideoPipeline.from_pretrained(
                settings.wan_model_path,
                torch_dtype=torch.float16,
            )
            pipe.enable_model_cpu_offload()
            cls._pipeline = pipe
        return cls._pipeline

    def generate_clip(
        self,
        prompt: str,
        first_frame_image_path: str,
        output_path: str,
        duration_seconds: int = 5,
    ) -> str:
        try:
            import imageio
            from PIL import Image

            full_prompt = f"{prompt}, {STYLE_CONSTRAINTS}"
            pipe = self._get_pipeline()

            image = Image.open(first_frame_image_path).convert("RGB")

            result = pipe(
                image=image,
                prompt=full_prompt,
                num_frames=duration_seconds * 8,  # ~8 fps
            )
            frames = result.frames[0]  # list of PIL Images

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(output_path, fps=8, format="ffmpeg", codec="libx264")
            try:
                import numpy as np
                for frame in frames:
                    writer.append_data(np.array(frame))
            finally:
                writer.close()

            return str(Path(output_path).resolve())

        except Exception as exc:
            logger.warning(
                "Wan video generation model failed or is not downloaded yet (Error: %s). "
                "Automatically falling back to FFmpeg image-to-video compilation...",
                exc
            )
            try:
                import subprocess
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                # Loop the static scene image into a 5-second video at 30 fps using FFmpeg
                cmd = [
                    "ffmpeg", "-y",
                    "-loop", "1",
                    "-i", first_frame_image_path,
                    "-c:v", "libx264",
                    "-t", str(duration_seconds),
                    "-pix_fmt", "yuv420p",
                    "-vf", "scale=832:480",
                    output_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                logger.info("Successfully compiled fallback video using FFmpeg → %s", output_path)
                return str(Path(output_path).resolve())
            except Exception as ffmpeg_exc:
                raise RuntimeError(
                    f"Wan video generation failed and FFmpeg video fallback compiler failed: {ffmpeg_exc}"
                ) from exc


class CloudVideoBackend(VideoBackend):
    """Stub cloud video backend — not yet implemented."""

    def generate_clip(
        self,
        prompt: str,
        first_frame_image_path: str,
        output_path: str,
        duration_seconds: int = 5,
    ) -> str:
        raise NotImplementedError(
            "CloudVideoBackend is not yet implemented. "
            "Set VIDEO_BACKEND=local to use the local Wan I2V backend."
        )


def get_video_backend() -> VideoBackend:
    """Factory — returns the configured video backend."""
    if settings.video_backend == "cloud":
        return CloudVideoBackend()
    return LocalWanBackend()
