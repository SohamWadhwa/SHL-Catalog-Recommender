"""
Parses the 10 gold-standard conversation traces (C1.md ... C10.md) into a
clean traces.json used by the replay harness.

For each trace we extract:
  - facts_text: every "User" quote across the whole trace, concatenated in
    order. This is fed to the user-simulator as "everything this persona
    knows" — it's what a truthful simulated user could answer from.
  - expected_names: the assessment names from the FINAL turn's table (the
    turn where end_of_conversation is true) — our Recall@10 ground truth.

Run: python eval/parse_traces.py
Output: eval/traces.json
"""

import re
import json
import pathlib

TRACES_DIR = pathlib.Path(__file__).parent / "traces_raw"
OUTPUT_PATH = pathlib.Path(__file__).parent / "traces.json"

TURN_SPLIT_RE = re.compile(r"### Turn \d+")
USER_BLOCK_RE = re.compile(r"\*\*User\*\*\s*\n((?:>.*\n?)+)", re.MULTILINE)
END_OF_CONV_RE = re.compile(r"`end_of_conversation`:\s*\*\*(true|false)\*\*", re.IGNORECASE)
TABLE_ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|(.+)\|\s*$", re.MULTILINE)


def extract_user_quotes(turn_text: str) -> list[str]:
    """Pull the blockquoted lines following **User** in a turn block."""
    quotes = []
    for match in USER_BLOCK_RE.finditer(turn_text):
        raw = match.group(1)
        lines = [ln.lstrip(">").strip() for ln in raw.strip().split("\n")]
        lines = [ln for ln in lines if ln]
        if lines:
            quotes.append(" ".join(lines))
    return quotes


def extract_table_names(turn_text: str) -> list[str]:
    """Pull assessment names (2nd column) from a markdown table in a turn block."""
    names = []
    for match in TABLE_ROW_RE.finditer(turn_text):
        cols = [c.strip() for c in match.group(2).split("|")]
        if not cols:
            continue
        name = cols[0].strip()
        # skip separator rows like ---- and header remnants
        if not name or set(name) <= {"-", ":"}:
            continue
        if name.lower() == "name":
            continue
        names.append(name)
    return names


def parse_trace_file(path: pathlib.Path) -> dict:
    text = path.read_text(encoding="utf-8")
    turn_texts = TURN_SPLIT_RE.split(text)[1:]  # drop the preamble before Turn 1

    facts_quotes = []
    expected_names = []
    final_turn_found = False

    for turn_text in turn_texts:
        facts_quotes.extend(extract_user_quotes(turn_text))

        eoc_match = END_OF_CONV_RE.search(turn_text)
        is_final = bool(eoc_match and eoc_match.group(1).lower() == "true")
        if is_final:
            names = extract_table_names(turn_text)
            if names:
                expected_names = names
                final_turn_found = True

    return {
        "trace_id": path.stem,
        "facts_text": "\n".join(f"- {q}" for q in facts_quotes),
        "opening_message": facts_quotes[0] if facts_quotes else "",
        "expected_names": expected_names,
        "num_turns": len(turn_texts),
        "final_turn_found": final_turn_found,
    }


def main():
    trace_files = sorted(TRACES_DIR.glob("C*.md"), key=lambda p: int(p.stem[1:]))
    traces = []
    for path in trace_files:
        trace = parse_trace_file(path)
        traces.append(trace)
        status = "OK" if trace["final_turn_found"] and trace["expected_names"] else "⚠ MISSING DATA"
        print(f"{trace['trace_id']}: {len(trace['expected_names'])} expected assessments, "
              f"{trace['num_turns']} turns, facts_text={len(trace['facts_text'])} chars  [{status}]")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2)
    print(f"\nSaved {len(traces)} traces -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()