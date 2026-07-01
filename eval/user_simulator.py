"""
Simulated user for trace replay.

Mirrors the described grading harness: "simulates a user using an LLM that
is given the trace's persona and facts and runs a real multi-turn
conversation... The simulated user answers your agent's questions truthfully
from its facts, says it has no preference when asked something outside its
facts, and ends the conversation when the agent provides a shortlist."

We can't know exactly how the real harness prompts its simulator, so this is
our best-effort reconstruction — documented as an approximation in the
approach doc, not a guarantee of matching the real harness's exact wording.
"""

from agent.groq_client import call_text

SIMULATED_USER_SYSTEM_PROMPT = """You are roleplaying as a hiring manager talking to an SHL
assessment recommendation chatbot. You are NOT an AI assistant in this exercise — stay fully
in character as the hiring manager.

Here are the facts about your situation and requirements (this is everything you know —
do not invent details beyond this):
{facts_text}

Rules:
- Reply in 1-2 short, natural sentences, like a busy hiring manager typing quickly.
- If the chatbot asks something covered by your facts, answer truthfully and specifically.
- If the chatbot asks something NOT covered by your facts, say you have no strong preference
  or let them use their judgment — do not invent new requirements.
- Never mention that you are an AI, a simulation, or reference "facts" explicitly.
- Do not thank the chatbot excessively or add filler.
"""


def _conversation_to_text(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = "You (hiring manager)" if m["role"] == "user" else "Chatbot"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def simulate_user_reply(facts_text: str, conversation_so_far: list[dict]) -> str:
    """
    Given the persona's facts and the conversation so far (ending with the
    chatbot's latest reply), generate the next natural user message.
    """
    system_prompt = SIMULATED_USER_SYSTEM_PROMPT.format(facts_text=facts_text)
    user_prompt = (
        f"Conversation so far:\n{_conversation_to_text(conversation_so_far)}\n\n"
        f"Write your next reply as the hiring manager. Output ONLY the reply text, "
        f"nothing else."
    )
    return call_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        fallback="That sounds fine, go ahead.",
    )