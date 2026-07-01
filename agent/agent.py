"""
Agent orchestration.

Entry point: run_agent(messages: list[dict], retriever: Retriever) -> dict
matching the exact API response schema:
    {"reply": str, "recommendations": [...], "end_of_conversation": bool}

Design:
  - Stateless: recomputes everything from full message history each call.
  - Max 2 Groq calls per turn (router, then finalize) to stay well under
    the 30s per-call timeout.
  - Recommendation name/url/test_type ALWAYS come from our catalog metadata,
    never from LLM text, so hallucinated URLs are structurally impossible.
  - Turn-cap safety net: if we're near the 8-turn cap and still clarifying,
    force a recommendation using best-effort context instead of risking
    a conversation that never produces a shortlist.
"""

import re

from agent.groq_client import call_json
from agent.prompts import (
    ROUTER_SYSTEM_PROMPT,
    FINALIZE_RECOMMEND_SYSTEM_PROMPT,
    FINALIZE_COMPARE_SYSTEM_PROMPT,
    build_router_user_prompt,
    build_finalize_recommend_prompt,
    build_finalize_compare_prompt,
)

COMPARE_KEYWORDS = re.compile(
    r"\b(difference|compare|comparison|versus|vs\.?|better than|different from)\b",
    re.IGNORECASE,
)


def _detect_compare_targets(user_text: str, retriever) -> list[str] | None:
    """
    Deterministic safety net for the 'compare' behavior.
    The router LLM only sees conversation text, not the catalog, so it can
    mistake a real SHL assessment name (e.g. ".NET MVC (New)") for a generic
    tech question and refuse. Here we scan for literal catalog name matches
    ourselves — cheap (377 substring checks) and far more reliable than
    asking the LLM to recognize obscure product names it's never seen.
    """
    if not COMPARE_KEYWORDS.search(user_text):
        return None

    text_lower = user_text.lower()
    matches = []
    for item in retriever.metadata:
        name = item["name"]
        if len(name) < 4:  # skip trivially short names to avoid false positives
            continue
        if name.lower() in text_lower:
            matches.append(name)

    # dedupe, prefer longer (more specific) names first
    matches = sorted(set(matches), key=len, reverse=True)
    return matches[:2] if len(matches) >= 2 else None


MAX_TURNS = 8
FORCE_RECOMMEND_AFTER = 6  # if we hit this many messages and still haven't recommended, force it

ROUTER_FALLBACK = {
    "mode": "clarify",
    "reply": "Could you tell me a bit more about the role or skills you're hiring for?",
    "retrieval_queries": [],
    "test_type_filter": [],
    "compare_targets": [],
    "refusal_reason": "",
}


def _conversation_to_text(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "user").capitalize()
        lines.append(f"{role}: {m.get('content', '')}")
    return "\n".join(lines)


def _already_recommended(messages: list[dict]) -> bool:
    """Heuristic: did a prior assistant turn already deliver a shortlist?
    We don't have access to prior structured output (API is stateless and
    only conversation text is passed), so we look for assistant turns that
    look like they listed assessments."""
    for m in messages:
        if m.get("role") == "assistant" and (
            "here are" in m.get("content", "").lower()
            or "assessment" in m.get("content", "").lower()
            and any(ch.isdigit() for ch in m.get("content", ""))
        ):
            return True
    return False


def _merge_candidates(retriever, queries: list[str], test_type_filter: list[str], top_k_each: int = 10) -> list[dict]:
    """Run retrieval for each query facet, merge + dedupe by name, optionally filter by type."""
    seen = {}
    for q in queries:
        for item in retriever.query(q, top_k=top_k_each):
            name = item["name"]
            # keep the highest-scoring occurrence
            if name not in seen or item["_score"] > seen[name]["_score"]:
                seen[name] = item

    results = list(seen.values())

    if test_type_filter:
        filtered = [r for r in results if any(t in r["test_type"].split(",") for t in test_type_filter)]
        # only apply the filter if it doesn't wipe out everything (safety net)
        if filtered:
            results = filtered

    results.sort(key=lambda x: x["_score"], reverse=True)
    return results


def _to_recommendation(item: dict) -> dict:
    """Map internal metadata to the exact public schema."""
    return {
        "name": item["name"],
        "url": item["url"],
        "test_type": item["test_type"],
    }


def run_agent(messages: list[dict], retriever) -> dict:
    if not messages:
        return {
            "reply": "Hi! Tell me about the role you're hiring for and I'll help you find the right SHL assessments.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    conversation_text = _conversation_to_text(messages)
    last_user_msg = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")

    # ---- Deterministic compare pre-check (skips the router entirely if it fires) ----
    forced_compare_targets = _detect_compare_targets(last_user_msg, retriever)
    if forced_compare_targets:
        mode = "compare"
        router_out = {
            "mode": "compare",
            "reply": "",
            "retrieval_queries": [],
            "test_type_filter": [],
            "compare_targets": forced_compare_targets,
            "refusal_reason": "",
        }
    else:
        # ---- Call 1: Router ----
        router_out = call_json(
            system_prompt=ROUTER_SYSTEM_PROMPT,
            user_prompt=build_router_user_prompt(conversation_text),
            fallback=ROUTER_FALLBACK,
        )
        mode = router_out.get("mode", "clarify")

    # ---- Turn-cap safety net ----
    if mode == "clarify" and len(messages) >= FORCE_RECOMMEND_AFTER and not _already_recommended(messages):
        mode = "recommend"
        if not router_out.get("retrieval_queries"):
            # best-effort: use the last user message as the query
            last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
            router_out["retrieval_queries"] = [last_user] if last_user else ["general assessment"]

    # ---- CLARIFY ----
    if mode == "clarify":
        return {
            "reply": router_out.get("reply") or ROUTER_FALLBACK["reply"],
            "recommendations": [],
            "end_of_conversation": False,
        }

    # ---- REFUSE ----
    if mode == "refuse":
        reply = router_out.get("reply") or (
            "I can only help with finding SHL assessments — I'm not able to help with that. "
            "Want help finding a test for a specific role or skill?"
        )
        return {
            "reply": reply,
            "recommendations": [],
            "end_of_conversation": False,
        }

    # ---- COMPARE ----
    if mode == "compare":
        targets = router_out.get("compare_targets", [])
        if len(targets) < 2:
            return {
                "reply": "Could you name the two specific assessments you'd like me to compare?",
                "recommendations": [],
                "end_of_conversation": False,
            }
        item_a = retriever.get_by_name(targets[0])
        item_b = retriever.get_by_name(targets[1])
        if not item_a or not item_b:
            missing = targets[0] if not item_a else targets[1]
            return {
                "reply": f"I couldn't find \"{missing}\" in the SHL catalog. Could you double-check the name?",
                "recommendations": [],
                "end_of_conversation": False,
            }
        finalize_out = call_json(
            system_prompt=FINALIZE_COMPARE_SYSTEM_PROMPT,
            user_prompt=build_finalize_compare_prompt(conversation_text, item_a, item_b),
            fallback={
                "reply": f"{item_a['name']}: {item_a['description'][:150]} | {item_b['name']}: {item_b['description'][:150]}",
                "end_of_conversation": False,
            },
        )
        return {
            "reply": finalize_out.get("reply", ""),
            "recommendations": [],
            "end_of_conversation": bool(finalize_out.get("end_of_conversation", False)),
        }

    # ---- RECOMMEND / REFINE ----
    queries = router_out.get("retrieval_queries") or [messages[-1].get("content", "")]
    test_type_filter = router_out.get("test_type_filter", [])
    candidates = _merge_candidates(retriever, queries, test_type_filter)

    if not candidates:
        return {
            "reply": "I couldn't find any matching assessments in the catalog for that. Could you describe the role or skills differently?",
            "recommendations": [],
            "end_of_conversation": False,
        }

    candidates = candidates[:20]  # cap what we show the LLM to keep prompt small

    finalize_out = call_json(
        system_prompt=FINALIZE_RECOMMEND_SYSTEM_PROMPT,
        user_prompt=build_finalize_recommend_prompt(conversation_text, candidates),
        fallback={
            "reply": f"Here are {min(5, len(candidates))} assessments that fit your requirements.",
            "selected_names": [c["name"] for c in candidates[:5]],
            "end_of_conversation": True,
        },
    )

    selected_names = finalize_out.get("selected_names", [])
    by_name = {c["name"]: c for c in candidates}
    recommendations = []
    for name in selected_names:
        if name in by_name:
            recommendations.append(_to_recommendation(by_name[name]))

    # safety net: if LLM picked nothing valid, fall back to top candidates
    if not recommendations:
        recommendations = [_to_recommendation(c) for c in candidates[:5]]

    recommendations = recommendations[:10]

    return {
        "reply": finalize_out.get("reply", "Here are some assessments that match your requirements."),
        "recommendations": recommendations,
        "end_of_conversation": bool(finalize_out.get("end_of_conversation", True)),
    }