"""
M_14 — WhatsAppAgent
====================
Role     : deliver the post-purchase follow-up (policy documents + summary) over
           WhatsApp. Sends the message via the Twilio WhatsApp sandbox when
           credentials are configured; always appends the outbound message to
           logs/wa_outbox.json and echoes it to stdout (so the demo still works
           offline / if the send fails).
Trigger  : handoff from M_03 (ClosureAgent) when the user has purchased.
Tools    : Twilio WhatsApp API + file I/O (outbox).
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

try:
    from twilio.rest import Client
except ImportError:  # pragma: no cover - optional dependency
    Client = None  # type: ignore

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PKG_DIR, ".env"))
except ImportError:  # pragma: no cover - optional dependency
    pass

# Twilio WhatsApp sandbox configuration. Credentials come from .env; the
# recipient is hardcoded to the demo owner's verified sandbox number.
_TWILIO_FROM = "whatsapp:+14155238886"   # Twilio sandbox sender
DEMO_WA_NUMBER = "+918167265517"          # hardcoded demo recipient (verified)
_LOG_DIR = os.path.join(_PKG_DIR, "logs")
_OUTBOX_PATH = os.path.join(_LOG_DIR, "wa_outbox.json")


def _send_via_twilio(to_number: str, body: str) -> dict:
    """Send a WhatsApp message through the Twilio sandbox.
    Returns {sent: bool, sid|error: ...}. Never raises — the outbox log and the
    rest of the turn must proceed even if delivery fails."""
    if Client is None:
        return {"sent": False, "error": "twilio sdk not installed"}
    account_sid = os.getenv("Account_SID", "").strip()
    auth_token = os.getenv("Auth_token", "").strip()
    if not (account_sid and auth_token):
        return {"sent": False, "error": "twilio credentials not configured"}
    try:
        client = Client(account_sid, auth_token)
        msg = client.messages.create(
            from_=_TWILIO_FROM,
            to=f"whatsapp:{to_number}",
            body=body,
        )
        return {"sent": True, "sid": msg.sid, "status": msg.status}
    except Exception as exc:  # pragma: no cover - network/credential errors
        return {"sent": False, "error": str(exc)}


def _append_outbox(record: dict) -> None:
    os.makedirs(_LOG_DIR, exist_ok=True)
    outbox = []
    if os.path.exists(_OUTBOX_PATH):
        try:
            with open(_OUTBOX_PATH, "r", encoding="utf-8") as f:
                outbox = json.load(f)
        except (OSError, ValueError):
            outbox = []
    outbox.append(record)
    with open(_OUTBOX_PATH, "w", encoding="utf-8") as f:
        json.dump(outbox, f, indent=2, ensure_ascii=False)


class WhatsAppAgent:
    agent_id = AgentID.WHATSAPP

    def run(self, ctx: AgentContext) -> AgentResult:
        schema = ctx.record.schema
        pid = schema.resolved_product_id or "your policy"
        plan = schema.resolved_plan_id or ""

        body = (
            f"Hi! Your {pid}"
            + (f" ({plan})" if plan else "")
            + " policy is confirmed. We've attached the full policy wording and a "
              "one-page summary for your records. Reply here anytime if you have "
              "questions — welcome aboard, and thank you for choosing us!"
        )

        delivery = _send_via_twilio(DEMO_WA_NUMBER, body)

        record = {
            "to": DEMO_WA_NUMBER,
            "session_id": getattr(schema, "session_id", None),
            "product_id": schema.resolved_product_id,
            "plan_id": schema.resolved_plan_id,
            "body": body,
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "delivery": delivery,
        }
        _append_outbox(record)

        status = "sent" if delivery.get("sent") else f"logged-only ({delivery.get('error')})"
        print(f"[WhatsApp → {DEMO_WA_NUMBER}] ({status}) {body}")

        return AgentResult(
            agent_id=self.agent_id,
            output_text=None,   # outbound side-channel, not spoken on the call
            tool_calls=["twilio_whatsapp", "wa_outbox"],
            meta={
                "wa_sent": bool(delivery.get("sent")),
                "to": DEMO_WA_NUMBER,
                "outbox_path": _OUTBOX_PATH,
                "delivery": delivery,
            },
        )


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fsm import ConversationRecord

    agent = WhatsAppAgent()
    rec = ConversationRecord.new(session_id="m14_test")
    rec.schema.set("resolved_product_id", "SP015")
    rec.schema.resolved_plan_id = "SP015_GOLD"

    res = agent.run(AgentContext(record=rec))
    # wa_sent reflects Twilio delivery; the outbox must be written either way.
    assert "wa_sent" in res.meta
    assert os.path.exists(_OUTBOX_PATH), "outbox file should be written"

    with open(_OUTBOX_PATH, "r", encoding="utf-8") as f:
        outbox = json.load(f)
    assert outbox and outbox[-1]["product_id"] == "SP015", outbox[-1]
    print("whatsapp.py (M_14) self-test passed.")
    print("  delivery:", res.meta["delivery"])
    print("  outbox entries:", len(outbox))
