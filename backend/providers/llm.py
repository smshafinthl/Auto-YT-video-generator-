import os
import gc
import logging
from typing import Any

from backend.models.config import settings

logger = logging.getLogger(__name__)


class LocalHuggingFaceLLM:
    """Local HuggingFace LLM wrapper for offline text generation (Qwen, Llama, etc.)."""

    _model = None
    _tokenizer = None

    def __init__(self, model_name: str | None = None, temperature: float = 0.7):
        self.model_name = model_name or settings.local_llm_model or "Qwen/Qwen2.5-1.5B-Instruct"
        self.temperature = temperature

    def _load_model(self):
        if LocalHuggingFaceLLM._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("Loading local HuggingFace LLM: %s", self.model_name)
            dtype = (
                torch.bfloat16
                if (torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
                else torch.float16
            )

            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            LocalHuggingFaceLLM._tokenizer = tokenizer
            LocalHuggingFaceLLM._model = model
        return LocalHuggingFaceLLM._model, LocalHuggingFaceLLM._tokenizer

    def invoke(self, messages: list) -> Any:
        import torch

        model, tokenizer = self._load_model()

        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            elif hasattr(msg, "content"):
                msg_type = getattr(msg, "type", "user")
                role = "system" if msg_type == "system" else "user"
                content = msg.content
            else:
                role = "user"
                content = str(msg)
            formatted_messages.append({"role": role, "content": content})

        if hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in formatted_messages])

        inputs = tokenizer(prompt_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=max(self.temperature, 0.1),
                do_sample=self.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        class Response:
            def __init__(self, content: str):
                self.content = content

        return Response(generated_text)


def unload_local_llm() -> None:
    """Unload the local HuggingFace LLM model from GPU VRAM."""
    if LocalHuggingFaceLLM._model is not None:
        del LocalHuggingFaceLLM._model
        del LocalHuggingFaceLLM._tokenizer
        LocalHuggingFaceLLM._model = None
        LocalHuggingFaceLLM._tokenizer = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass
        logger.info("Local HuggingFace LLM unloaded from VRAM")


def get_llm(temperature: float = 0.7) -> Any:
    """
    Return an LLM instance based on available configuration.
    
    Supports:
      - 'local' / 'huggingface': Local HuggingFace model (Qwen, Llama, etc.)
      - 'gemini': Google Gemini API
      - 'openai' / 'auto': OpenAI / Bifrost gateway
    """
    provider = settings.llm_provider.lower().strip()

    # 1. Local HuggingFace LLM
    if provider in ("local", "huggingface", "hf"):
        logger.info("Using Local HuggingFace LLM: %s", settings.local_llm_model)
        return LocalHuggingFaceLLM(model_name=settings.local_llm_model, temperature=temperature)

    # 2. Gemini check
    gemini_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or settings.gemini_api_key
        or settings.google_api_key
    )

    use_gemini = False
    if provider == "gemini":
        use_gemini = True
    elif provider == "auto":
        if gemini_key and gemini_key.strip() and gemini_key.strip() != "placeholder":
            use_gemini = True

    if use_gemini:
        if not gemini_key or gemini_key == "placeholder":
            logger.warning("Gemini selected but key is missing. Ensure GEMINI_API_KEY is set.")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            model_name = settings.gemini_model or "gemini-2.0-flash"
            logger.info("Initializing ChatGoogleGenerativeAI with model=%s", model_name)
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=gemini_key,
                temperature=temperature,
            )
        except ImportError:
            logger.warning("langchain-google-genai not installed. Falling back to ChatOpenAI.")

    # 3. Fallback to OpenAI / Bifrost gateway
    from langchain_openai import ChatOpenAI

    logger.info(
        "Initializing ChatOpenAI with model=%s, base_url=%s",
        settings.bifrost_model,
        settings.bifrost_base_url,
    )
    return ChatOpenAI(
        model=settings.bifrost_model,
        api_key=settings.bifrost_api_key,
        base_url=settings.bifrost_base_url,
        temperature=temperature,
    )
