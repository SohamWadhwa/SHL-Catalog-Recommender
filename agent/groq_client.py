"""
Thin wrapper around the Groq SDK.
Handles: JSON-mode calls, markdown-fence stripping, one repair retry
on invalid JSON, and a safe fallback if both attempts fail.
"""

import os
import json
import re

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "openai/gpt-oss-120b"


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if the model added them."""
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def call_json(
    system_prompt: str,
    user_prompt: str,
    fallback: dict,
    temperature: float = 0.2,
    reasoning_effort: str = "low",
) -> dict:
    """
    Calls Groq in JSON mode, returns a parsed dict.
    On failure (bad JSON twice, API error), returns `fallback`.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(2):
        try:
            resp = _client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                response_format={"type": "json_object"},
                max_completion_tokens=1500,
            )
            raw = resp.choices[0].message.content
            cleaned = _strip_fences(raw)
            return json.loads(cleaned)

        except json.JSONDecodeError:
            # Repair attempt: tell the model exactly what went wrong.
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON. "
                    "Return ONLY a valid JSON object, no markdown, no explanation, "
                    "no trailing commentary."
                ),
            })
            continue

        except Exception as e:
            print(f"[groq_client] API error: {e}")
            return fallback

    print("[groq_client] Gave up after 2 attempts, using fallback.")
    return fallback