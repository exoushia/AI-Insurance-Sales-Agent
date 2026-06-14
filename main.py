"""
Command-line entrypoint for the AI Insurance Sales Agent scaffold.
"""

from __future__ import annotations

from orchestrator import ConversationOrchestrator


def main() -> int:
    orchestrator = ConversationOrchestrator()
    print("AI Insurance Sales Agent ready. Type a message and press Enter. Type 'exit' to quit.")

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
            f"[state={result['state']} intent={result['intent']} "
            f"llm_enabled={result['llm_enabled']} model={result['active_model']}]"
        )

        if result["state"] in {"S4_CLOSURE", "S5_HUMAN_HANDOFF", "S6_BLOCKED"}:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())