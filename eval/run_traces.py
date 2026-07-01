"""
Replay harness — runs all 10 gold-standard traces against your LIVE /chat
endpoint (local or deployed), using a Groq-driven simulated user, and scores
Recall@10 against each trace's labeled expected shortlist.

Also performs hard-eval checks along the way:
  - Turn cap (<=8 total messages) respected
  - Every returned URL exists in catalog.json (no hallucinated URLs)
  - Every response has the required schema keys

Usage:
  python eval/run_traces.py                          # against local server
  python eval/run_traces.py --base-url https://your-app.onrender.com
"""

import sys
import json
import pathlib
import argparse

import requests

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from eval.user_simulator import simulate_user_reply

ROOT = pathlib.Path(__file__).parent.parent
TRACES_PATH = pathlib.Path(__file__).parent / "traces.json"
CATALOG_PATH = ROOT / "catalog" / "shl_product_catalog.json"
RESULTS_PATH = pathlib.Path(__file__).parent / "results.json"

MAX_MESSAGES = 8  # matches the evaluator's turn cap (user + assistant combined)
REQUIRED_KEYS = {"reply", "recommendations", "end_of_conversation"}


def load_valid_urls() -> set[str]:
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    return {item["link"] for item in catalog}


def post_chat(base_url: str, messages: list[dict]) -> dict:
    resp = requests.post(f"{base_url}/chat", json={"messages": messages}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Response missing required keys: {missing}")
    return data


def run_single_trace(base_url: str, trace: dict, valid_urls: set[str]) -> dict:
    conversation = []
    turn_user_msg = trace["opening_message"]
    final_recommendations = []
    hit_turn_cap_without_shortlist = False
    invalid_urls_seen = []
    api_errors = []

    while True:
        conversation.append({"role": "user", "content": turn_user_msg})

        try:
            resp = post_chat(base_url, conversation)
        except Exception as e:
            api_errors.append(str(e))
            break

        conversation.append({"role": "assistant", "content": resp["reply"]})

        for rec in resp.get("recommendations", []):
            if rec.get("url") not in valid_urls:
                invalid_urls_seen.append(rec.get("url"))

        if resp.get("recommendations"):
            final_recommendations = resp["recommendations"]
            break  # mirrors: harness ends the conversation once a shortlist is given

        if len(conversation) >= MAX_MESSAGES:
            hit_turn_cap_without_shortlist = True
            break

        if resp.get("end_of_conversation"):
            break

        turn_user_msg = simulate_user_reply(trace["facts_text"], conversation)

    recommended_names = {r["name"].strip().lower() for r in final_recommendations}
    expected_names = {n.strip().lower() for n in trace["expected_names"]}

    if expected_names:
        recall = len(recommended_names & expected_names) / len(expected_names)
    else:
        recall = None

    return {
        "trace_id": trace["trace_id"],
        "recall_at_10": recall,
        "recommended_names": [r["name"] for r in final_recommendations],
        "expected_names": trace["expected_names"],
        "matched_names": sorted(recommended_names & expected_names),
        "missed_names": sorted(expected_names - recommended_names),
        "num_messages_used": len(conversation),
        "hit_turn_cap_without_shortlist": hit_turn_cap_without_shortlist,
        "invalid_urls_seen": invalid_urls_seen,
        "api_errors": api_errors,
        "transcript": conversation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    with open(TRACES_PATH, encoding="utf-8") as f:
        traces = json.load(f)
    valid_urls = load_valid_urls()

    print(f"Checking {args.base_url}/health ...")
    try:
        health = requests.get(f"{args.base_url}/health", timeout=120)
        print(f"Health check: {health.status_code} {health.json()}")
    except Exception as e:
        print(f"WARNING: health check failed: {e}")

    results = []
    for trace in traces:
        print(f"\nRunning {trace['trace_id']} ...")
        result = run_single_trace(args.base_url, trace, valid_urls)
        results.append(result)
        recall_str = f"{result['recall_at_10']:.2f}" if result["recall_at_10"] is not None else "N/A"
        print(f"  recall@10={recall_str}  messages_used={result['num_messages_used']}  "
              f"turn_cap_violation={result['hit_turn_cap_without_shortlist']}  "
              f"invalid_urls={len(result['invalid_urls_seen'])}  api_errors={len(result['api_errors'])}")

    valid_recalls = [r["recall_at_10"] for r in results if r["recall_at_10"] is not None]
    mean_recall = sum(valid_recalls) / len(valid_recalls) if valid_recalls else 0.0

    turn_cap_violations = sum(1 for r in results if r["hit_turn_cap_without_shortlist"])
    total_invalid_urls = sum(len(r["invalid_urls_seen"]) for r in results)
    total_api_errors = sum(len(r["api_errors"]) for r in results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Mean Recall@10:            {mean_recall:.3f}")
    print(f"Turn cap violations:       {turn_cap_violations} / {len(results)}")
    print(f"Hallucinated URLs seen:    {total_invalid_urls}")
    print(f"API errors:                {total_api_errors}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "mean_recall_at_10": mean_recall,
            "turn_cap_violations": turn_cap_violations,
            "total_invalid_urls": total_invalid_urls,
            "total_api_errors": total_api_errors,
            "per_trace": results,
        }, f, indent=2)
    print(f"\nFull results (with transcripts) saved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()