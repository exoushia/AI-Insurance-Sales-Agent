"""
M_02 — EscalationAgent
======================
Role     : produce the human-handoff message when the conversation enters
           S5_HUMAN_HANDOFF (explicit request, repeated frustration, or the
           failure counter tripping).
Trigger  : state == S5_HUMAN_HANDOFF.
Tools    : none (canned, deterministic text).
Output   : handoff message; hands off to M_03 (ClosureAgent) to wrap up.
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

_WANT_HUMAN = (
    "Of course — I'm connecting you to one of our human advisors right now. "
    "They'll already have your details, so you won't need to repeat anything."
)
_FRUSTRATED = (
    "I'm sorry this has been frustrating. Let me bring in a human advisor who "
    "can help you directly from here."
)
_FAILSAFE = (
    "I want to make sure you get the right help — let me connect you with a "
    "human advisor who can take it from here."
)


class EscalationAgent:
    agent_id = AgentID.ESCALATION

    def run(self, ctx: AgentContext) -> AgentResult:
        intent = ctx.intent
        failures = getattr(ctx.record, "consecutive_failures", 0)
        max_failures = getattr(ctx.record, "MAX_FAILURES", 3)

        if intent == IntentSignal.WANT_HUMAN:
            text, reason = _WANT_HUMAN, "want_human"
        elif intent == IntentSignal.FRUSTRATED or failures >= max_failures:
            text, reason = _FRUSTRATED, "frustrated"
        else:
            text, reason = _FAILSAFE, "failsafe"

        return AgentResult(
            agent_id=self.agent_id,
            output_text=text,
            handoff_to=AgentID.CLOSURE,
            meta={"escalation_reason": reason},
        )


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fsm import ConversationRecord, FSMState

    agent = EscalationAgent()

    def esc(intent=None, failures=0):
        rec = ConversationRecord.new(session_id="m02_test")
        rec.state = FSMState.S5_HUMAN_HANDOFF
        rec.consecutive_failures = failures
        return agent.run(AgentContext(record=rec, intent=intent))

    r = esc(intent=IntentSignal.WANT_HUMAN)
    assert "connecting you" in r.output_text and r.handoff_to == AgentID.CLOSURE
    assert r.meta["escalation_reason"] == "want_human"

    r = esc(intent=IntentSignal.FRUSTRATED)
    assert r.meta["escalation_reason"] == "frustrated"

    r = esc(intent=None, failures=3)
    assert r.meta["escalation_reason"] == "frustrated"

    r = esc(intent=None, failures=0)
    assert r.meta["escalation_reason"] == "failsafe"

    print("escalation.py (M_02) self-test passed.")
