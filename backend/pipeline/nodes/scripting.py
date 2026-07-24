import json
import re
import logging

from backend.pipeline.state import PipelineState
from backend.providers.llm import get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a creative director for short-form faceless YouTube Shorts video content. "
    "Given a user topic, produce a valid single-line JSON object with exactly two keys:\n"
    '  "video_prompts": a list of 4 to 6 short cinematic scene descriptions (strings)\n'
    '  "voiceover_script": a single string containing 3 to 5 sentences of narration\n'
    "\n"
    "CRITICAL RULE — Subscribe CTA:\n"
    "  The FINAL sentence of voiceover_script MUST be a short (2 to 6 words), natural, "
    "  conversational call-to-action that invites the audience to subscribe or follow. "
    "  It must feel relevant to the video topic, not generic. "
    "  Examples by topic:\n"
    '    Science  → "Subscribe for more mind-blowing science!"\n'
    '    Space    → "Follow for daily space discoveries!"\n'
    '    History  → "Like and subscribe for more history!"\n'
    '    Animals  → "Subscribe for more amazing animal facts!"\n'
    '    Tech     → "Follow for more tech explained simply!"\n'
    "  NEVER skip this CTA. It is mandatory on every single video.\n"
    "\n"
    "Output ONLY valid JSON — no markdown fences, no linebreaks inside text values, no extra text."
)


def _parse_json_lenient(text: str) -> dict:
    """Parse JSON string leniently, fixing unescaped newlines or raw markdown fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass

    # Extract JSON object substring if model added leading/trailing prose
    match = re.search(r"(\{[\s\S]*\})", text)
    if match:
        extracted = match.group(1)
        try:
            return json.loads(extracted, strict=False)
        except json.JSONDecodeError:
            pass

    # Flatten unescaped literal newlines inside JSON strings
    flattened = text.replace("\r\n", " ").replace("\n", " ")
    return json.loads(flattened, strict=False)


def scripting_node(state: PipelineState) -> dict:
    """LangGraph node: generate video_prompts and voiceover_script from user_prompt."""
    if state.get("error"):
        return {}

    llm = get_llm()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state["user_prompt"]},
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content if hasattr(response, "content") else str(response)

        data = _parse_json_lenient(raw)

        video_prompts = data["video_prompts"]
        voiceover_script = data["voiceover_script"]

        msg = "scripting_node: script generated successfully"
        if state.get("progress_queue") is not None:
            state["progress_queue"].put(msg)
        return {
            "video_prompts": video_prompts,
            "voiceover_script": voiceover_script,
            "progress_log": [msg],
        }

    except (json.JSONDecodeError, KeyError) as exc:
        raw_snippet = raw[:200] if "raw" in dir() else "(no response)"
        return {
            "error": f"scripting_node error: {type(exc).__name__}: {exc}. Raw: {raw_snippet}",
        }
    except Exception as exc:
        return {
            "error": f"scripting_node unexpected error: {exc}",
        }
