import json
import re

from backend.pipeline.state import PipelineState
from backend.providers.llm import get_llm

SYSTEM_PROMPT = (
    "You are a creative director for short-form faceless YouTube Shorts video content. "
    "Given a user topic, produce a JSON object with exactly two keys:\n"
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
    "Output ONLY valid JSON — no markdown fences, no extra text, no explanations."
)


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if present."""
    text = text.strip()
    # Remove opening fence (```json or ```)
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    # Remove closing fence
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


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

        cleaned = _strip_markdown_fences(raw)
        data = json.loads(cleaned)

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
