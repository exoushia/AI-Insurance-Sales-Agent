"""
Agentic demo runner
===================
Drives the AgenticOrchestrator (M_16 native tool-calling) through a few scripted
local demos in text mode — the same brain the voice stack uses, minus the audio.

Run:
    ORCHESTRATION_MODE=agentic ../.venv/bin/python demo_agentic.py
    ../.venv/bin/python demo_agentic.py --scenario price_anxiety
    ../.venv/bin/python demo_agentic.py --debug   # show advisory guardrail flags

Scenarios (the demo types):
  1. price_anxiety   — customer balks at the premium; agent fetches real
                       treatment costs (estimate_value_vs_cost) to justify value
                       and closes the sale.
  2. deep_policy     — customer asks specific coverage questions; agent grounds
                       answers in product features / policy wording, then pivots
                       to a recommendation and closes.
  3. general_qa      — general health-insurance / IRDAI question answered via
                       answer_general_question (regulations RAG).
  4. hinglish        — Hindi/Hinglish discovery; proves language detection and
                       code-mixed replies through the same agentic brain, and
                       closes the sale in Hinglish.
  5. low_budget_niche — budget-constrained gig worker; agent matches a niche
                       low-cost product (e.g. Hospital Daily Cash) within budget
                       and closes the sale.
  6. off_topic       — customer asks unrelated questions; agent stays polite,
                       declines off-topic asks and ends the call gracefully.

Each turn prints the spoken reply plus the tools the model called and the
detected language, so you can see the agentic reasoning trace (also visible in
logs/llm_calls.log). Each scenario ends with an outcome summary.
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Force the agentic engine for this runner regardless of the ambient flag.
os.environ.setdefault("ORCHESTRATION_MODE", "agentic")

from settings import AppConfig
from agentic_orchestrator import AgenticOrchestrator


SCENARIOS: dict[str, list[str]] = {
    "price_anxiety": [
        "Hi, I'm 32 and looking for health insurance just for myself.",
        "Mainly for maternity, we're planning a baby next year.",
        "Okay what would something like that cost me?",
        "Honestly that feels expensive, why should I pay that much?",
        "Alright that makes sense. Let's go with the recommended plan.",
    ],
    "deep_policy": [
        "I'm 41, need a family floater for me, my wife and one kid.",
        "I have diabetes, will that be covered?",
        "What's the waiting period for pre-existing conditions on that plan?",
        "And does it cover day-care procedures like cataract surgery?",
        "Okay, that sounds reassuring. Let's go ahead with the recommended plan.",
        "Yes, please go ahead and finalize it.",
    ],
    "general_qa": [
        "Before I decide, I have a general question.",
        "If I buy now and switch insurers next year, do I lose my waiting period?",
        "Got it. And what's a free-look period?",
    ],
    "hinglish": [
        "Hello, mujhe apne liye ek health insurance chahiye, main 30 saal ka hoon.",
        "Mainly hospitalisation ke liye, sirf mere liye, koi pre-existing bimari nahi hai.",
        "Mera budget around 10,000 rupees saal ka hai. Iska premium kitna hoga?",
        "Theek hai, recommended plan le lete hain.",
        "Haan, please ise abhi finalize kar dijiye.",
    ],
    "low_budget_niche": [
        "Hi, I'm a delivery rider, 27. Money's tight but I want some cover, just for me.",
        "I won't have a regular salary, so I just want cash if I'm hospitalised.",
        "Honestly I can't afford much — maybe two, three thousand a year at most.",
        "Okay, show me the best plan you've got for me and what it costs.",
        "That sounds doable. Let's go ahead and finalize it.",
    ],
    "off_topic": [
        "Hey, quick one — who do you think wins the cricket world cup this year?",
        "Haha fair. Also what's the weather like in Mumbai today?",
        "Got it. Can you also recommend a good biryani place near me?",
        "Alright, no insurance for me today. Thanks for your help, bye!",
    ],
}


def run_scenario(name: str, turns: list[str], debug: bool = False) -> None:
    print("=" * 70)
    print(f"SCENARIO: {name}")
    print("=" * 70)
    orch = AgenticOrchestrator(session_id=f"demo_{name}", config=AppConfig())
    print(f"[llm_enabled={orch.llm.is_available} "
          f"mode={orch.config.orchestration_mode}]\n")
    if not orch.llm.is_available:
        print("  LLM unavailable — set OPENAI_API_KEY to run the agentic demo.")
        return

    last = {}
    for user_text in turns:
        print(f"user> {user_text}")
        out = orch.process_message(user_text)
        last = out
        tools = [t["name"] for t in out.get("tool_trace", [])]
        print(f"agent> {out['assistant_text']}")
        trace = " → ".join(tools) if tools else "(no tools)"
        meta = f"lang={out.get('language', 'english')} | tools: {trace}"
        if debug and not out.get("guardrail_ok", True):
            meta += " | GUARDRAIL_FLAG"
        print(f"  [{meta}]\n")

    # End-of-scenario summary: where the agent landed.
    summary = []
    if last.get("resolved_product_id"):
        summary.append(f"product={last['resolved_product_id']}")
    if last.get("resolved_plan_id"):
        summary.append(f"plan={last['resolved_plan_id']}")
    summary.append("PURCHASED" if last.get("purchased") else "not purchased")
    print(f"--- outcome: {' | '.join(summary)} ---")


def main() -> int:
    parser = argparse.ArgumentParser(description="Agentic demo runner")
    parser.add_argument(
        "--scenario", choices=list(SCENARIOS) + ["all"], default="all",
        help="Which demo to run (default: all).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Show advisory guardrail flags in the per-turn trace.",
    )
    args = parser.parse_args()

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for name in names:
        run_scenario(name, SCENARIOS[name], debug=args.debug)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
