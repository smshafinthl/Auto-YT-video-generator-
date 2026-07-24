import abc
import json
import time
import logging
from pathlib import Path

import requests

from backend.models.config import settings
from backend.pipeline.editorial import STYLE_CONSTRAINTS

logger = logging.getLogger(__name__)

# Minimal ComfyUI workflow stub used when the real workflow JSON is absent
_MINIMAL_WORKFLOW_STUB: dict = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 0.75,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    },
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned-emaonly.ckpt"}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"batch_size": 1, "height": 512, "width": 512}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "POSITIVE_PROMPT", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "ugly, blurry", "clip": ["4", 1]}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "output", "images": ["8", 0]}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
}


def _load_workflow() -> dict:
    """Load ComfyUI workflow JSON from assets, or return the minimal stub."""
    workflow_path = Path(__file__).parent.parent.parent / "assets" / "comfyui_img2img_workflow.json"
    if workflow_path.exists():
        with open(workflow_path) as f:
            return json.load(f)
    return _MINIMAL_WORKFLOW_STUB.copy()


class ImageBackend(abc.ABC):
    """Abstract base class for image generation backends."""

    @abc.abstractmethod
    def generate_image(self, prompt: str, seed_image_path: str, output_path: str) -> str:
        """
        Generate an image and save to output_path.

        Args:
            prompt: Text prompt for the image.
            seed_image_path: Path to the seed/reference image.
            output_path: Destination path for the output image.

        Returns:
            Absolute path to the saved image.
        """
        ...


class LocalDiffusersImageBackend(ImageBackend):
    """Local image generator using HuggingFace diffusers on GPU."""

    _pipeline = None

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.local_image_model or "stabilityai/sd-turbo"

    @classmethod
    def _get_pipeline(cls, model_name: str):
        if cls._pipeline is None:
            import torch
            from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline

            logger.info("Loading local diffusers image pipeline: %s", model_name)
            dtype = (
                torch.bfloat16
                if (torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
                else torch.float16
            )

            try:
                # sd-turbo or sdxl-turbo works best with AutoPipelineForText2Image
                cls._pipeline = AutoPipelineForText2Image.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    variant="fp16" if dtype == torch.float16 else None,
                )
            except Exception:
                # Fallback to general StableDiffusionPipeline
                cls._pipeline = StableDiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                )

            if torch.cuda.is_available():
                cls._pipeline = cls._pipeline.to("cuda")
                logger.info("Local image pipeline loaded on CUDA GPU")
            
        return cls._pipeline

    def generate_image(self, prompt: str, seed_image_path: str, output_path: str) -> str:
        import torch

        full_prompt = f"{prompt}, {STYLE_CONSTRAINTS}"
        pipe = self._get_pipeline(self.model_name)

        logger.info("Generating local image using prompt: %s", full_prompt)
        
        num_inference_steps = 1 if "turbo" in self.model_name.lower() else 30
        guidance_scale = 0.0 if "turbo" in self.model_name.lower() else 7.5

        with torch.no_grad():
            image = pipe(
                prompt=full_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return str(Path(output_path).resolve())


def unload_local_image_model() -> None:
    """Unload the local image diffusers model from VRAM."""
    if LocalDiffusersImageBackend._pipeline is not None:
        del LocalDiffusersImageBackend._pipeline
        LocalDiffusersImageBackend._pipeline = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass
        logger.info("Local image diffusers model unloaded from VRAM")


class ComfyUIImageBackend(ImageBackend):
    """Image backend that drives a running ComfyUI server. Fails directly without fallbacks."""

    POLL_INTERVAL = 2  # seconds
    TIMEOUT = 120  # seconds

    def generate_image(self, prompt: str, seed_image_path: str, output_path: str) -> str:
        full_prompt = f"{prompt}, {STYLE_CONSTRAINTS}"
        workflow = _load_workflow()

        # Inject prompt into workflow (update CLIPTextEncode positive node)
        for node in workflow.values():
            if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode":
                if "POSITIVE_PROMPT" in str(node["inputs"].get("text", "")):
                    node["inputs"]["text"] = full_prompt

        payload = {"prompt": workflow}
        base_url = settings.comfyui_base_url.rstrip("/")

        resp = requests.post(f"{base_url}/prompt", json=payload, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"ComfyUI /prompt returned {resp.status_code}: {resp.text}")

        prompt_id = resp.json()["prompt_id"]

        # Poll until complete
        elapsed = 0
        while elapsed < self.TIMEOUT:
            time.sleep(self.POLL_INTERVAL)
            elapsed += self.POLL_INTERVAL

            hist_resp = requests.get(f"{base_url}/history/{prompt_id}", timeout=10)
            if hist_resp.status_code != 200:
                continue

            history = hist_resp.json()
            if prompt_id not in history:
                continue

            job_data = history[prompt_id]
            outputs = job_data.get("outputs", {})

            # Find the first image output
            for node_output in outputs.values():
                images = node_output.get("images", [])
                if images:
                    filename = images[0]["filename"]
                    view_resp = requests.get(
                        f"{base_url}/view",
                        params={"filename": filename},
                        timeout=30,
                    )
                    if view_resp.status_code != 200:
                        raise RuntimeError(
                            f"ComfyUI /view returned {view_resp.status_code}"
                        )
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "wb") as f:
                        f.write(view_resp.content)
                    return str(Path(output_path).resolve())

        raise RuntimeError(f"ComfyUI timed out after {self.TIMEOUT}s for prompt_id={prompt_id}")


def get_image_backend() -> ImageBackend:
    """Factory — returns the configured image backend."""
    if settings.image_backend.lower().strip() in ("local", "diffusers"):
        return LocalDiffusersImageBackend()
    return ComfyUIImageBackend()
