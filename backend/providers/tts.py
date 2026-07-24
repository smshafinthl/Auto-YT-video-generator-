import abc
import os
import uuid
import logging

from elevenlabs import ElevenLabs

from backend.models.config import settings

logger = logging.getLogger(__name__)


class TTSProvider(abc.ABC):
    """Abstract base class for Text-to-Speech providers."""

    @abc.abstractmethod
    def synthesize(self, text: str, output_dir: str) -> str:
        """
        Synthesize speech from text and save to output_dir.

        Args:
            text: The text to synthesize.
            output_dir: Directory where the audio file will be saved.

        Returns:
            Absolute path to the saved .mp3 file.
        """
        ...


class gTTSTSProvider(TTSProvider):
    """gTTS (Google Translate) TTS Provider for free voiceover generation."""

    def synthesize(self, text: str, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"voiceover_gtts_{uuid.uuid4().hex[:8]}.mp3"
        output_path = os.path.join(output_dir, filename)
        abs_path = os.path.abspath(output_path)

        try:
            from gtts import gTTS
            logger.info("Synthesizing voiceover using gTTS...")
            tts = gTTS(text=text, lang="en", tld="com")
            tts.save(abs_path)
            logger.info("Voiceover synthesized successfully via gTTS → %s", abs_path)
            return abs_path
        except ImportError as exc:
            raise ImportError(
                "gTTS library is not installed. To use free local TTS, "
                "please run: pip install gTTS"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"gTTS voiceover synthesis failed: {exc}") from exc


class ElevenLabsTTSProvider(TTSProvider):
    """ElevenLabs TTS provider falling back to gTTS if API key is missing or invalid."""

    def __init__(self) -> None:
        self._client = ElevenLabs(api_key=settings.elevenlabs_api_key)

    def synthesize(self, text: str, output_dir: str) -> str:
        """Synthesize text to speech. Falls back to gTTS if key is missing/invalid."""
        os.makedirs(output_dir, exist_ok=True)
        filename = f"voiceover_{uuid.uuid4().hex[:8]}.mp3"
        output_path = os.path.join(output_dir, filename)
        abs_path = os.path.abspath(output_path)

        # Check if ElevenLabs key is valid / not placeholder
        key = settings.elevenlabs_api_key
        use_fallback = False
        if not key or key == "placeholder" or "your_elevenlabs_api_key" in key:
            use_fallback = True

        if use_fallback:
            logger.warning("ElevenLabs key is missing or set to placeholder. Falling back to gTTS...")
            fallback = gTTSTSProvider()
            return fallback.synthesize(text, output_dir)

        try:
            audio_generator = self._client.text_to_speech.convert(
                voice_id=settings.elevenlabs_voice_id,
                text=text,
                model_id="eleven_monolingual_v1",
                output_format="mp3_44100_128",
            )

            with open(abs_path, "wb") as f:
                for chunk in audio_generator:
                    if isinstance(chunk, bytes):
                        f.write(chunk)
            return abs_path

        except Exception as exc:
            logger.warning(
                "ElevenLabs API conversion failed (Error: %s). "
                "Switching to gTTS provider...",
                exc
            )
            fallback = gTTSTSProvider()
            return fallback.synthesize(text, output_dir)


def get_tts_provider() -> TTSProvider:
    """Factory — returns the ElevenLabs provider (which contains gTTS fallback)."""
    return ElevenLabsTTSProvider()
