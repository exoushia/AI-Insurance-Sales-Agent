"""
Command-line entrypoint for the AI Insurance Sales Agent scaffold.

Text-mode harness for testing the full brain (FSM + sub-agents + LLM + RAG)
without the voice stack. Type a message, see the reply plus a debug line showing
the state, intent, which agents fired, and the discovery fields captured so far —
handy for verifying intent classification and NER extraction per turn.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from orchestrator import ConversationOrchestrator
from agentic_orchestrator import build_orchestrator

# Discovery fields surfaced each turn so you can watch extraction happen.
_WATCH_FIELDS = ("buyer_type", "age", "gender", "family_cover", "primary_need",
                 "has_ped", "budget_band", "si_preference")


def _schema_snapshot(orchestrator: ConversationOrchestrator):
    schema = orchestrator.record.schema
    captured = {f: getattr(schema, f) for f in _WATCH_FIELDS
                if getattr(schema, f, None) is not None}
    return captured or "(nothing captured yet)"


def main() -> int:
    orchestrator = build_orchestrator()
    mode = getattr(orchestrator.config, "orchestration_mode", "fsm")
    print("AI Insurance Sales Agent ready. Type a message and press Enter. Type 'exit' to quit.")
    print(f"[llm_enabled={orchestrator.llm.is_available} orchestration_mode={mode}]\n")

    while True:
        try:
            message = input("> ").strip()
        except EOFError:
            break

        if not message:
            continue
        if message.lower() in {"exit", "quit"}:
            break

        result = orchestrator.process_message(message)
        print(f"assistant> {result['assistant_text']}")
        print(
            f"  [state={result['state']} intent={result['intent']} "
            f"conf={result['confidence']} fired={result['agents_fired']}]"
        )
        print(f"  [schema={_schema_snapshot(orchestrator)}]")

        if result["state"] in {"S4_CLOSURE", "S5_HUMAN_HANDOFF", "S6_BLOCKED"}:
            print(f"\n[conversation ended in {result['state']}]")
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())