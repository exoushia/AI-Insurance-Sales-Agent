"""
M_03 — ClosureAgent
===================
Role     : end the conversation gracefully when it reaches S4_CLOSURE (a natural
           "done", or following an M_02 human handoff). 
           If the user intent is frustrated or they expect to talk to a human, 
           Reassure the user that their concerns have been addressed before closure, and record a drop-off reason.
Trigger  : state == S4_CLOSURE, or handoff from M_02.
Tools    : none (template text).
Output   :
  - a closing message tailored to whether the user purchased,
  - schema_updates["drop_off_reason"] when closing without a purchase,
  - handoff to M_14 (WhatsAppAgent) when the user purchased (send policy docs).
"""

from __future__ import annotations

import os
import sys

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

try:
    from .base import AgentContext, AgentResult, AgentID
except ImportError:  # pragma: no cover - script execution
    from base import AgentContext, AgentResult, AgentID  # type: ignore

from fsm import IntentSignal

_PURCHASED = (
    "Wonderful — your policy is all set! I'll send the full policy documents and "
    "a summary to your WhatsApp right away. Thank you for choosing us, and stay well."
)
_HANDED_OFF = (
    "You're in good hands now — a human advisor will continue from here. "
    "Thank you for your time today."
)
_NOT_READY = (
    "No problem at all — take your time to decide. I've noted your preferences, "
    "so we can pick up right where we left off whenever you're ready. Take care!"
)

# Map the closing intent to an analytics drop-off reason (no purchase only).
_DROP_OFF_BY_INTENT = {
    IntentSignal.DONE: "not_ready",
    IntentSignal.FRUSTRATED: "unknown",
    IntentSignal.WANT_HUMAN: "unknown",
}


class ClosureAgent:
    agent_id = AgentID.CLOSURE

    def run(self, ctx: AgentContext) -> AgentResult:
        schema = ctx.record.schema
        purchased = bool(getattr(schema, "purchased", False))

        if purchased:
            return AgentResult(
                agent_id=self.agent_id,
                output_text=_PURCHASED,
                handoff_to=AgentID.WHATSAPP,
                meta={"closure_type": "purchased"},
            )

        # Closed without purchase — record a drop-off reason for analytics.
        intent = ctx.intent
        came_from_handoff = ctx.record.state == _state_s5()
        if came_from_handoff or intent in (IntentSignal.WANT_HUMAN, IntentSignal.FRUSTRATED):
            text = _HANDED_OFF
        else:
            text = _NOT_READY
        reason = _DROP_OFF_BY_INTENT.get(intent, "unknown")

        return AgentResult(
            agent_id=self.agent_id,
            output_text=text,
            schema_updates={"drop_off_reason": reason},
            meta={"closure_type": "no_purchase", "drop_off_reason": reason},
        )


def _state_s5():
    from fsm import FSMState
    return FSMState.S5_HUMAN_HANDOFF


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fsm import ConversationRecord, FSMState

    agent = ClosureAgent()

    def close(purchased=False, intent=None, state=FSMState.S4_CLOSURE):
        rec = ConversationRecord.new(session_id="m03_test")
        rec.state = state
        if purchased:
            rec.schema.purchased = True
        return agent.run(AgentContext(record=rec, intent=intent))

    # Purchased → WhatsApp handoff.
    r = close(purchased=True)
    assert r.handoff_to == AgentID.WHATSAPP and "policy is all set" in r.output_text
    assert r.meta["closure_type"] == "purchased"

    # Done without purchase → drop_off_reason not_ready.
    r = close(intent=IntentSignal.DONE)
    assert r.schema_updates.get("drop_off_reason") == "not_ready", r.schema_updates
    assert "take your time" in r.output_text.lower()

    # Came from human handoff state → handed-off closing text.
    r = close(intent=IntentSignal.WANT_HUMAN, state=FSMState.S5_HUMAN_HANDOFF)
    assert "human advisor" in r.output_text
    assert r.schema_updates.get("drop_off_reason") == "unknown"

    print("closure.py (M_03) self-test passed.")
