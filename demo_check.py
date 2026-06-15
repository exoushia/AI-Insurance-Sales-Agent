"""
demo_check.py — scripted end-to-end smoke test of the live agent brain.
=======================================================================
Drives a full discovery → recommendation → Q&A → closure conversation through the
real LLM path (OpenAI + RAG) and asserts the things a demo depends on:

  - intent is classified (never silently stuck at "unrecognised") on clear turns,
  - a single multi-field utterance is parsed into multiple schema fields,
  - a spelled-out age ("twenty five") is captured as 25,
  - the conversation advances S1 → S2 and surfaces a recommendation,
  - a policy question reaches the Q&A / RAG path.

Run from the AI-Insurance-Sales-Agent dir:
    ../.venv/bin/python demo_check.py

Requires OPENAI_API_KEY (and SARVAM_API_KEY for translation) in .env. This calls
the live API, so it costs a few cents and needs a network connection. For a free,
offline wiring check use scenarios.py instead.
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

load_dotenv()

from orchestrator import ConversationOrchestrator


# Each step: (user message, list of assertions). An assertion is
# (label, callable(result, schema) -> bool). All must hold to pass.
_SCRIPT = [
    (
        "Hi, I'm a 25 year old woman looking for health insurance for myself",
        [
            ("intent classified (not unrecognised)",
             lambda r, s: r["intent"] != "unrecognised"),
            ("age captured = 25", lambda r, s: s.age == 25),
            ("gender captured = female", lambda r, s: s.gender == "female"),
            ("multi-field: buyer_type captured",
             lambda r, s: s.buyer_type is not None),
        ],
    ),
    (
        "no pre-existing conditions, all healthy",
        [
            ("has_ped captured = False", lambda r, s: s.has_ped is False),
        ],
    ),
    (
        "mainly hospitalisation cover",
        [
            ("primary_need = hospitalisation",
             lambda r, s: s.primary_need == "hospitalisation"),
        ],
    ),
    (
        "what's a good plan for me?",
        [
            ("reached recommendation (S2+)",
             lambda r, s: r["state"] in {"S2_RECOMMENDATION", "S3_POLICY_QA"}),
            ("a plan was resolved",
             lambda r, s: s.resolved_plan_id is not None or s.resolved_product_id is not None),
            ("reply is non-empty", lambda r, s: bool(r["assistant_text"].strip())),
        ],
    ),
    (
        "what is the waiting period for pre-existing diseases?",
        [
            ("policy question handled (Q&A/RAG fired)",
             lambda r, s: any(a in r["agents_fired"] for a in ("M_08", "M_09"))),
            ("reply is non-empty", lambda r, s: bool(r["assistant_text"].strip())),
        ],
    ),
]

# A separate spelled-out-age check (fresh session, age is the expected field).
_WORD_AGE_SCRIPT = [
    ("I need insurance just for myself", []),
    ("twenty five", [("spelled-out age -> 25", lambda r, s: s.age == 25)]),
]


def _run_script(title: str, steps) -> bool:
    print(f"\n=== {title} ===")
    orch = ConversationOrchestrator(session_id="demo_check")
    print(f"[llm_enabled={orch.llm.is_available}]")
    if not orch.llm.is_available:
        print("  ! OPENAI_API_KEY not set — this test needs the live LLM path. Skipping.")
        return False

    ok = True
    for message, checks in steps:
        result = orch.process_message(message)
        schema = orch.record.schema
        print(f"\n> {message}")
        print(f"  assistant> {result['assistant_text'][:120]}"
              f"{'…' if len(result['assistant_text']) > 120 else ''}")
        print(f"  [state={result['state']} intent={result['intent']} "
              f"conf={result['confidence']} fired={result['agents_fired']}]")
        for label, predicate in checks:
            try:
                passed = bool(predicate(result, schema))
            except Exception as exc:  # a thrown predicate is a failure, not a crash
                passed = False
                label = f"{label} (raised {type(exc).__name__})"
            mark = "PASS" if passed else "FAIL"
            print(f"    [{mark}] {label}")
            ok = ok and passed
    return ok


def main() -> int:
    happy = _run_script("happy path (discovery → recommendation → Q&A)", _SCRIPT)
    word_age = _run_script("spelled-out age", _WORD_AGE_SCRIPT)

    print("\n" + "=" * 50)
    if happy and word_age:
        print("ALL DEMO CHECKS PASSED ✅")
        return 0
    print("SOME DEMO CHECKS FAILED ❌  (see [FAIL] lines above)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
