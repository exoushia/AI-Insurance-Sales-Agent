"""
M_12 — ResponseValidator
========================
Role     : last-line outbound safety check before a response is spoken.
Trigger  : outbound stage, after a candidate response is assembled.
Check    : Denylist scan (deterministic) — block over-promising / unsafe sales
           phrases and replace them with a safe fallback line.
Output   : output_text (sanitised if a denylist phrase was found) + meta
           {valid, violations}.

Payload contract:
  ctx.payload["response_text"] — the candidate response (required).
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

# Phrases an insurance agent must never use (over-promising / non-compliant).
_DENYLIST = [
    "guaranteed returns", "guaranteed profit", "100% approval", "no questions asked",
    "risk free", "risk-free", "double your money", "tax free guaranteed",
    "we never reject", "always approved",
]

_SAFE_FALLBACK = (
    "Let me give you the accurate details — I'll stick to exactly what your policy "
    "documents say so there's no confusion."
)


class ResponseValidator:
    agent_id = AgentID.RESPONSE_VALIDATOR

    def run(self, ctx: AgentContext) -> AgentResult:
        text = ctx.payload.get("response_text", "") or ""

        violations: list[str] = []
        lowered = text.lower()
        for phrase in _DENYLIST:
            if phrase in lowered:
                violations.append(f"denylist:{phrase}")

        denylisted = bool(violations)
        safe_text = _SAFE_FALLBACK if denylisted else text

        return AgentResult(
            agent_id=self.agent_id,
            output_text=safe_text,
            meta={"valid": not denylisted, "violations": violations},
        )


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fsm import ConversationRecord

    agent = ResponseValidator()

    def validate(text, user_message="", intent=None):
        rec = ConversationRecord.new(session_id="m12_test")
        ctx = AgentContext(record=rec, message=user_message, intent=intent,
                           payload={"response_text": text})
        return agent.run(ctx)

    # Clean response → valid, unchanged.
    r = validate("The waiting period for diabetes is 12 months under this plan.")
    assert r.meta["valid"] is True and r.output_text.startswith("The waiting"), r.meta

    # Denylist phrase → blocked and replaced with safe fallback.
    r = validate("This plan offers guaranteed returns with 100% approval.")
    assert r.meta["valid"] is False, r.meta
    assert r.output_text == _SAFE_FALLBACK
    assert any(v.startswith("denylist:") for v in r.meta["violations"])

    print("response_validator.py (M_12) self-test passed.")
