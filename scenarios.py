"""
scenarios.py — deterministic end-to-end harness (Phase 0)
========================================================
Runs a fully scripted conversation (no API key, no network) through the
orchestrator and asserts:
  * the FSM state path advances S0 → S1 → S2 → S3 → S4,
  * the expected sub-agents (M_*) fire at each stage,
  * a conversation_<session_id>.json analytics log is written.

Run:  ../.venv/bin/python scenarios.py
"""

from __future__ import annotations

import json
import os

from orchestrator import ConversationOrchestrator
from subagents.analytics_logger import _log_path


# Scripted user turns and what we expect after each.
SCRIPT = [
    {
        "say": "Hi, I want to buy maternity health insurance just for myself.",
        "expect_state": "S1_DISCOVERY",
        "must_fire": ["M_01", "M_04"],
    },
    {
        "say": "I'm a 30 year old woman buying only for myself.",
        "expect_state": "S2_RECOMMENDATION",
        "must_fire": ["M_06", "M_07"],
    },
    {
        "say": "What is the waiting period for maternity under this plan?",
        "expect_state": "S3_POLICY_QA",
        "must_fire": ["M_08"],
    },
    {
        "say": "Great, this sounds perfect. I'd like to buy this plan.",
        "expect_state": "S4_CLOSURE",
        "must_fire": ["M_03"],
    },
]


def run() -> None:
    session_id = "scenario_happy_path"
    log_path = _log_path(session_id)
    if os.path.exists(log_path):
        os.remove(log_path)

    orch = ConversationOrchestrator(session_id=session_id)
    print(f"=== Scenario: happy path ({session_id}) ===")

    path: list[str] = []
    for i, step in enumerate(SCRIPT, start=1):
        result = orch.process_message(step["say"])
        path.append(result["state"])

        print(f"\nTurn {i}")
        print(f"  user      : {step['say']}")
        print(f"  intent    : {result['intent']} (conf={result['confidence']})")
        print(f"  state     : {result['state']}")
        print(f"  fired     : {result['agents_fired']}")
        print(f"  assistant : {result['assistant_text'][:140]}")

        assert result["state"] == step["expect_state"], (
            f"Turn {i}: expected state {step['expect_state']}, got {result['state']}"
        )
        for agent_id in step["must_fire"]:
            assert agent_id in result["agents_fired"], (
                f"Turn {i}: expected {agent_id} to fire, fired={result['agents_fired']}"
            )

    # State path assertion.
    assert path == ["S1_DISCOVERY", "S2_RECOMMENDATION", "S3_POLICY_QA", "S4_CLOSURE"], path
    assert orch.record.schema.resolved_product_id == "SP007", orch.record.schema.resolved_product_id
    assert orch.record.schema.resolved_plan_id is not None
    assert orch.record.schema.purchased is True

    # Analytics log written with one entry per turn.
    assert os.path.exists(log_path), f"missing analytics log: {log_path}"
    with open(log_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    assert len(doc["turns"]) == len(SCRIPT), doc["turns"]
    assert doc["final_state"] == "S4_CLOSURE"

    print("\n--- conversation log ---")
    print(f"  path  : {' -> '.join(t['state'] for t in doc['turns'])}")
    print(f"  log   : {log_path}")
    print("\nHAPPY-PATH SCENARIO PASSED.")


# Escalation branch: the user asks for a human → S5 → closure handoff.
ESCALATION_SCRIPT = [
    {"say": "Hi, I'm looking at maternity insurance for myself.",
     "expect_state": "S1_DISCOVERY", "must_fire": ["M_01", "M_04"]},
    {"say": "Actually, can you connect me to a human agent please?",
     "expect_state": "S5_HUMAN_HANDOFF", "must_fire": ["M_02", "M_03"]},
]


def run_escalation() -> None:
    session_id = "scenario_escalation"
    log_path = _log_path(session_id)
    if os.path.exists(log_path):
        os.remove(log_path)

    orch = ConversationOrchestrator(session_id=session_id)
    print("\n=== Scenario: human handoff (S5) ===")

    path: list[str] = []
    for i, step in enumerate(ESCALATION_SCRIPT, start=1):
        result = orch.process_message(step["say"])
        path.append(result["state"])
        print(f"\nTurn {i}")
        print(f"  user      : {step['say']}")
        print(f"  intent    : {result['intent']}")
        print(f"  state     : {result['state']}")
        print(f"  fired     : {result['agents_fired']}")
        print(f"  assistant : {result['assistant_text'][:140]}")

        assert result["state"] == step["expect_state"], (
            f"Turn {i}: expected {step['expect_state']}, got {result['state']}"
        )
        for agent_id in step["must_fire"]:
            assert agent_id in result["agents_fired"], (
                f"Turn {i}: expected {agent_id} to fire, fired={result['agents_fired']}"
            )

    assert path == ["S1_DISCOVERY", "S5_HUMAN_HANDOFF"], path
    assert orch.record.schema.drop_off_reason is not None
    assert os.path.exists(log_path), f"missing analytics log: {log_path}"
    print("\nESCALATION SCENARIO PASSED.")


if __name__ == "__main__":
    run()
    run_escalation()
    print("\nALL SCENARIOS PASSED.")
