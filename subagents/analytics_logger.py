"""
M_13 — AnalyticsLogger
======================
Role     : the ONLY agent permitted to write to the conversation analytics sink.
           Appends one turn record per turn to logs/conversation_<session_id>.json
           and refreshes the rolling schema snapshot.
Trigger  : end of every turn (last step of the orchestrator pipeline).
Tools    : file I/O (conversation log).
Output   : no spoken text; meta {log_path, turn_index}.

Payload contract (populated by the orchestrator):
  ctx.payload["agents_fired"]        — list[str] of M_ ids fired this turn.
  ctx.payload["assistant_response"]  — final response string for this turn.
  ctx.message                        — the user message for this turn.
  ctx.intent / ctx.confidence        — classified intent + confidence.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

try:
    from .base import AgentContext, AgentResult, AgentID
except ImportError:  # pragma: no cover - script execution
    from base import AgentContext, AgentResult, AgentID  # type: ignore

_LOG_DIR = os.path.join(_PKG_DIR, "logs")


def _log_path(session_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(session_id))
    return os.path.join(_LOG_DIR, f"conversation_{safe}.json")


def _load(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            pass
    return {}


class AnalyticsLogger:
    agent_id = AgentID.ANALYTICS_LOGGER

    def run(self, ctx: AgentContext) -> AgentResult:
        schema = ctx.record.schema
        session_id = getattr(schema, "session_id", None) or "unknown"
        path = _log_path(session_id)

        doc = _load(path)
        if not doc:
            doc = {
                "session_id": session_id,
                "language": getattr(schema, "language", None),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "turns": [],
            }

        intent = ctx.intent
        turn = {
            "turn": len(doc["turns"]) + 1,
            "state": ctx.record.state.value,
            "intent": intent.value if hasattr(intent, "value") else intent,
            "confidence": ctx.confidence,
            "user_message": ctx.message,
            "assistant_response": ctx.payload.get("assistant_response", ""),
            "agents_fired": list(ctx.payload.get("agents_fired", [])),
            "consecutive_failures": ctx.record.consecutive_failures,
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        # Tool-calling telemetry (agentic path) — the ordered sequence of tools
        # the SalesAgent invoked this turn, plus the full {name,args,result} trace.
        tool_sequence = ctx.payload.get("tool_sequence")
        if tool_sequence:
            turn["tool_sequence"] = list(tool_sequence)
        tool_trace = ctx.payload.get("tool_trace")
        if tool_trace:
            turn["tool_trace"] = tool_trace
        doc["turns"].append(turn)
        doc["final_state"] = ctx.record.state.value
        doc["schema_snapshot"] = schema.to_analytics_record()
        doc["updated_at"] = datetime.now(timezone.utc).isoformat()

        os.makedirs(_LOG_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False, default=str)

        return AgentResult(
            agent_id=self.agent_id,
            output_text=None,
            tool_calls=["conversation_log"],
            meta={"log_path": path, "turn_index": turn["turn"]},
        )


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fsm import ConversationRecord, FSMState, IntentSignal

    agent = AnalyticsLogger()
    rec = ConversationRecord.new(session_id="m13_test")
    path = _log_path("m13_test")
    if os.path.exists(path):
        os.remove(path)

    # Turn 1.
    rec.state = FSMState.S1_DISCOVERY
    ctx = AgentContext(
        record=rec, message="I want maternity cover", intent=IntentSignal.PROVIDE_INFO,
        confidence=0.95,
        payload={"assistant_response": "Got it — for whom is the cover?",
                 "agents_fired": ["M_01", "M_04", "M_05"]},
    )
    r1 = agent.run(ctx)
    assert os.path.exists(path), "log file should be written"
    assert r1.meta["turn_index"] == 1

    # Turn 2 appends.
    rec.state = FSMState.S2_RECOMMENDATION
    ctx2 = AgentContext(
        record=rec, message="just for me", intent=IntentSignal.PROVIDE_INFO,
        confidence=1.0,
        payload={"assistant_response": "Here's the best match...",
                 "agents_fired": ["M_01", "M_04", "M_06", "M_07"]},
    )
    r2 = agent.run(ctx2)
    assert r2.meta["turn_index"] == 2

    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    assert len(doc["turns"]) == 2, doc["turns"]
    assert doc["final_state"] == "S2_RECOMMENDATION"
    assert doc["schema_snapshot"]["session_id"] == "m13_test"
    print("analytics_logger.py (M_13) self-test passed.")
    print("  log:", path, "turns:", len(doc["turns"]))
