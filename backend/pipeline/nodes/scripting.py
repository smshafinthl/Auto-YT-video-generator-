import json
import re
import logging

from backend.pipeline.state import PipelineState
from backend.providers.llm import get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a master horror storyteller and creative director for short-form YouTube Shorts (analog horror, dark suspense, creepypasta).\n"
    "Given a specific user prompt or topic, write a complete, captivating horror narration script based strictly on that prompt.\n"
    "Produce a valid single-line JSON object with exactly two keys:\n"
    '  "video_prompts": a list of 4 to 6 short cinematic scene descriptions (strings)\n'
    '  "voiceover_script": a single string containing 4 to 6 eerie, suspenseful narration sentences\n'
    "\n"
    "STORYTELLING & NARRATION RULES:\n"
    "1. Complete Unique Story: Write a full, immersive, self-contained story tailored strictly to the user's prompt. Do NOT use generic or repeated templates.\n"
    "2. Natural Prompt-Specific Hook/Ending: End the narration naturally or with a terrifying prompt-specific climax/hook. NEVER hardcode generic strings like 'Watch Part 2' or 'Stay tuned for Part 2'.\n"
    "3. Dark Horror Visual Scenes: Every string in video_prompts MUST focus on intense visual suspense, creepy analog horror details, eerie lighting, dark mist, or nightmare imagery.\n"
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
