"""
M_11 — ResponseQueue
====================
Role     : assemble the final spoken turn from ordered response segments.
Trigger  : outbound stage, final step before the turn is returned.
Tools    : none (deterministic FIFO join).
Output   : output_text — the single, ordered response string.

Payload contract:
  ctx.payload["segments"]      — ordered list of response fragments (optional).
  ctx.payload["response_text"] — fallback single response if no segments given.
"""

from __future__ import annotations

import os
import re
import sys

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

try:
    from .base import AgentContext, AgentResult, AgentID
except ImportError:  # pragma: no cover - script execution
    from base import AgentContext, AgentResult, AgentID  # type: ignore

_WS_RE = re.compile(r"\s+")


class ResponseQueue:
    agent_id = AgentID.RESPONSE_QUEUE

    def run(self, ctx: AgentContext) -> AgentResult:
        segments = ctx.payload.get("segments")
        if not segments:
            single = ctx.payload.get("response_text", "") or ""
            segments = [single] if single else []

        ordered = [self._normalise(s) for s in segments if s and str(s).strip()]
        combined = " ".join(ordered).strip()
        combined = _WS_RE.sub(" ", combined)

        return AgentResult(
            agent_id=self.agent_id,
            output_text=combined,
            meta={"segment_count": len(ordered)},
        )

    @staticmethod
    def _normalise(segment: str) -> str:
        return _WS_RE.sub(" ", str(segment).strip())


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fsm import ConversationRecord

    agent = ResponseQueue()

    def assemble(payload):
        rec = ConversationRecord.new(session_id="m11_test")
        return agent.run(AgentContext(record=rec, payload=payload))

    # Multiple ordered segments → joined in FIFO order.
    r = assemble({"segments": ["Great choice.", "Here's how the plan works.", "Shall I proceed?"]})
    assert r.output_text == "Great choice. Here's how the plan works. Shall I proceed?", r.output_text
    assert r.meta["segment_count"] == 3

    # Single response_text fallback.
    r = assemble({"response_text": "  Your   plan is   ready.  "})
    assert r.output_text == "Your plan is ready.", repr(r.output_text)

    # Empty payload → empty string, zero segments.
    r = assemble({})
    assert r.output_text == "" and r.meta["segment_count"] == 0

    print("response_queue.py (M_11) self-test passed.")
