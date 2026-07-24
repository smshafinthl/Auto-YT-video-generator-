import abc
import os
import uuid
import logging
from pathlib import Path

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


class FallbackTTSProvider(TTSProvider):
    """Free Fallback TTS provider using gTTS (Google Translate) or silent audio via FFmpeg."""

    def synthesize(self, text: str, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"voiceover_fallback_{uuid.uuid4().hex[:8]}.mp3"
        output_path = os.path.join(output_dir, filename)
        abs_path = os.path.abspath(output_path)

        # Method 1: Try gTTS (Google Text-to-Speech)
        try:
            from gtts import gTTS
            logger.info("Synthesizing fallback TTS voiceover using free gTTS...")
            tts = gTTS(text=text, lang="en", tld="com")
            tts.save(abs_path)
            logger.info("Fallback TTS synthesized successfully → %s", abs_path)
            return abs_path
        except Exception as exc:
            logger.warning("gTTS fallback failed: %s. Trying offline fallback...", exc)

        # Method 2: Synthesize a silent audio track matching text length (~2.5 words/sec) via FFmpeg
        try:
            import subprocess
            words_count = len(text.split())
            duration = max(3.0, words_count / 2.5)
            logger.info("Generating silent audio fallback (duration=%.2fs) using FFmpeg...", duration)
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                "-t", f"{duration:.2f}",
                "-q:a", "9", "-acodec", "libmp3lame",
                abs_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info("Silent fallback audio generated successfully → %s", abs_path)
            return abs_path
        except Exception as exc:
            raise RuntimeError(
                f"TTS generation failed. ElevenLabs failed and offline fallbacks (gTTS/FFmpeg) failed: {exc}"
            )


class ElevenLabsTTSProvider(TTSProvider):
    """ElevenLabs TTS provider with automatic local fallback on key/quota failures."""

    def __init__(self) -> None:
        self._client = ElevenLabs(api_key=settings.elevenlabs_api_key)

    def synthesize(self, text: str, output_dir: str) -> str:
        """Synthesize text to speech and save as MP3. Falls back if key is invalid/401/429."""
        os.makedirs(output_dir, exist_ok=True)
        filename = f"voiceover_{uuid.uuid4().hex[:8]}.mp3"
        output_path = os.path.join(output_dir, filename)

        # Ensure directory exists
        abs_path = os.path.abspath(output_path)

        # Check if ElevenLabs key is valid / not placeholder
        key = settings.elevenlabs_api_key
        if not key or key == "placeholder" or "your_elevenlabs_api_key" in key:
            logger.warning("ElevenLabs key is missing or set to placeholder. Directing to fallback TTS.")
            fallback = FallbackTTSProvider()
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
                "Automatically switching to Fallback TTS Provider (gTTS/FFmpeg)...",
                exc
            )
            fallback = FallbackTTSProvider()
            return fallback.synthesize(text, output_dir)


def get_tts_provider() -> TTSProvider:
    """Factory — returns the ElevenLabs provider (which contains automatic fallback)."""
    return ElevenLabsTTSProvider()
