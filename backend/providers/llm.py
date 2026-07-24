import os
import logging
from typing import Any

from backend.models.config import settings

logger = logging.getLogger(__name__)


def get_llm(temperature: float = 0.7) -> Any:
    """
    Return a LangChain chat model instance based on available configuration.
    
    Prefers Google Gemini if GEMINI_API_KEY or GOOGLE_API_KEY is present or if
    llm_provider is set to 'gemini' / 'auto'. Falls back to ChatOpenAI if no
    Gemini key is found.
    """
    gemini_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or settings.gemini_api_key
        or settings.google_api_key
    )

    provider = settings.llm_provider.lower().strip()

    use_gemini = False
    if provider == "gemini":
        use_gemini = True
    elif provider == "auto":
        # If gemini_key exists and is not a placeholder, use Gemini
        if gemini_key and gemini_key.strip() and gemini_key.strip() != "placeholder":
            use_gemini = True
        elif not settings.bifrost_api_key or settings.bifrost_api_key == "placeholder":
            # If bifrost key is missing/placeholder, default to Gemini check
            use_gemini = True

    if use_gemini:
        if not gemini_key or gemini_key == "placeholder":
            logger.warning(
                "Gemini selected/detected but GEMINI_API_KEY is missing. "
                "Ensure GEMINI_API_KEY or GOOGLE_API_KEY is set in environment."
            )
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

    # Fallback to OpenAI / Bifrost gateway
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
