import json
import re
import logging

from backend.pipeline.state import PipelineState
from backend.providers.llm import get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a master horror storyteller and creative director for short-form YouTube Shorts (analog horror, dark suspense, creepypasta).\n"
    "Given a specific user prompt or topic, write a concise, captivating horror narration script based strictly on that prompt.\n"
    "Produce a valid JSON object with EXACTLY two keys:\n"
    '  "video_prompts": a list of 4 to 5 short cinematic scene descriptions (strings)\n'
    '  "voiceover_script": a single string containing 4 to 5 eerie, suspenseful narration sentences\n'
    "\n"
    "STRICT CONCISENESS & JSON RULES:\n"
    "1. Keep keys and strings concise, direct, and complete so the JSON object never gets truncated.\n"
    "2. Complete Unique Story: Write a full, self-contained story tailored strictly to the user's prompt without generic templates.\n"
    "3. Natural Ending: End narration naturally or with a prompt-specific hook. NEVER hardcode generic strings like 'Watch Part 2'.\n"
    "4. Dark Horror Scenes: Every video_prompt string must focus on atmospheric suspense, dark mist, or nightmare imagery.\n"
    "\n"
    "Output ONLY valid JSON — no markdown fences, no linebreaks inside text values, no extra text."
)


def _repair_and_parse_json(text: str) -> dict:
    """Parse JSON string, using json_repair (if available) or regex string auto-closing cleanup."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    # 1. Try standard json.loads
    try:
        return json.loads(text, strict=False)
    except Exception:
        pass

    # 2. Try json_repair library if installed
    try:
        import json_repair
        repaired = json_repair.repair_json(text, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
    except Exception:
        pass

    # 3. Extract JSON object substring if model added leading/trailing prose
    match = re.search(r"(\{[\s\S]*\})", text)
    if match:
        extracted = match.group(1)
        try:
            return json.loads(extracted, strict=False)
        except Exception:
            pass
    else:
        extracted = text

    # 4. Fallback repair: Fix truncated unclosed strings and braces
    # Count open quotes, add missing trailing quote if odd count
    clean = extracted.replace("\r\n", " ").replace("\n", " ")
    quote_count = clean.count('"') - clean.count('\\"')
    if quote_count % 2 != 0:
        clean += '"'
    if not clean.endswith("}"):
        clean += "}"
    if not clean.startswith("{"):
        clean = "{" + clean

    return json.loads(clean, strict=False)


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

        data = _repair_and_parse_json(raw)

        video_prompts = data.get("video_prompts")
        voiceover_script = data.get("voiceover_script")

        # Fallback if keys missing or malformed
        if not video_prompts or not isinstance(video_prompts, list):
            video_prompts = [state["user_prompt"]] * 4
        if not voiceover_script or not isinstance(voiceover_script, str):
            voiceover_script = str(state["user_prompt"])

        msg = "scripting_node: script generated successfully"
        if state.get("progress_queue") is not None:
            state["progress_queue"].put(msg)
        return {
            "video_prompts": video_prompts,
            "voiceover_script": voiceover_script,
            "progress_log": [msg],
        }

    except (json.JSONDecodeError, KeyError, Exception) as exc:
        raw_snippet = raw[:200] if "raw" in dir() else "(no response)"
        return {
            "error": f"scripting_node error: {type(exc).__name__}: {exc}. Raw: {raw_snippet}",
        }
