"""
Agentic conversation orchestrator
=================================
The LLM-tool-calling counterpart to the deterministic FSM ConversationOrchestrator
in orchestrator.py. It is a drop-in replacement: same constructor fields, same
`process_message(message) -> dict` contract, same `record`/`llm`/`sarvam`
attributes — so the voice processor and main.py can swap between them via the
ORCHESTRATION_MODE flag without any other change.

Per turn:
  1. detect language (so the reply is voiced back in the user's language),
  2. run M_16 SalesAgent — the native OpenAI tool-calling loop drives discovery,
     recommendation, plan selection, value framing and closing,
  3. normalise the reply for TTS via M_11 (₹→rupees, markdown strip, etc.),
  4. record the turn.

If the LLM is unavailable or the tool loop fails, the turn is delegated to the
deterministic FSM orchestrator (sharing the SAME ConversationRecord) so the call
never dead-ends. This is the safety net that lets us ship the agentic path with
confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fsm import ConversationRecord, FSMState, detect_input_language, classify_intent
from settings import AppConfig
from llm_gateway import LLMGateway, SarvamGateway
from subagents import AGENT_REGISTRY, AgentID, AgentContext
from orchestrator import ConversationOrchestrator


@dataclass
class AgenticOrchestrator:
    """Agentic (LLM tool-calling) orchestration layer for one session."""

    session_id: str = "session_001"
    config: AppConfig = field(default_factory=AppConfig)
    record: ConversationRecord = field(init=False)
    llm: LLMGateway = field(init=False)
    sarvam: SarvamGateway = field(init=False)

    def __post_init__(self) -> None:
        self.record = ConversationRecord.new(session_id=self.session_id)
        self.llm = LLMGateway(self.config)
        self.sarvam = SarvamGateway(self.config)
        # M_16 now routes through the openai-agents implementation.
        self._sales = AGENT_REGISTRY[AgentID.SALES_AGENT]
        self._queue = AGENT_REGISTRY[AgentID.RESPONSE_QUEUE]
        self._analytics = AGENT_REGISTRY[AgentID.ANALYTICS_LOGGER]
        self._whatsapp = AGENT_REGISTRY[AgentID.WHATSAPP]
        self._fallback: ConversationOrchestrator | None = None

    # ── public contract ────────────────────────────────────────────────────
    def process_message(self, message: str) -> dict[str, Any]:
        rec = self.record

        # If there's no LLM at all, the agentic path can't run — hand the whole
        # turn to the deterministic FSM orchestrator.
        if not self.llm.is_available:
            return self._delegate_to_fsm(message)

        # Language label for THIS message (voice TTS reads schema.language).
        detected_language = detect_input_language(message)
        if detected_language != "english":
            rec.schema.set("language", detected_language)

        if rec.state == FSMState.S0_START:
            rec.transition_to(FSMState.S1_DISCOVERY, reason="session_started")

        ctx = AgentContext(
            record=rec, message=message, config=self.config,
            llm=self.llm, sarvam=self.sarvam,
        )

        result = self._sales.run(ctx)
        if not result.meta.get("agentic_ok"):
            # Tool loop unavailable/empty → deterministic fallback for this turn.
            return self._delegate_to_fsm(message)

        rec.schema.increment_turn()
        rec.add_message("user", message)

        spoken = self._normalise_for_voice(ctx, result.output_text)
        rec.add_message("assistant", spoken)
        rec.last_agent_response = spoken

        # Classify the user intent (deterministic, no extra LLM call) so the
        # agentic path persists a real intent rather than a hardcoded label.
        intent = classify_intent(message, rec.schema)

        # Ordered list of tool names the SalesAgent invoked this turn — the
        # "sequence of tools called" we surface in the demo and persist.
        tool_trace = result.meta.get("tool_trace", [])
        tool_sequence = [t.get("name") for t in tool_trace if t.get("name")]

        agents_fired = [AgentID.SALES_AGENT, AgentID.RESPONSE_QUEUE, AgentID.ANALYTICS_LOGGER]

        # Fire M_14 on the turn where finalize_purchase is called — same
        # behaviour as the FSM path (M_03 → M_14). Guard on tool_sequence so
        # it fires exactly once (not on subsequent turns after purchase).
        wa_status: str | None = None
        if rec.schema.purchased and "finalize_purchase" in tool_sequence:
            wa_result = self._whatsapp.run(ctx)
            wa_status = wa_result.meta.get("delivery_status")
            agents_fired.append(AgentID.WHATSAPP)

        # Trace correlation: carry sdk_trace_id through to the analytics log and
        # the return dict so it can be matched to the OpenAI Traces UI entry.
        sdk_trace_id = result.meta.get("sdk_trace_id")

        # Persist the turn via M_13 (the only agent that writes the conversation
        # log) so intent + tool sequence land in logs/conversation_<session>.json.
        ctx.intent = intent
        ctx.confidence = 1.0
        ctx.payload["assistant_response"] = spoken
        ctx.payload["agents_fired"] = list(agents_fired)
        ctx.payload["tool_trace"] = tool_trace
        ctx.payload["tool_sequence"] = tool_sequence
        if sdk_trace_id:
            ctx.payload["sdk_trace_id"] = sdk_trace_id
        self._analytics.run(ctx)

        backend = result.meta.get("backend", "agents_sdk")
        return {
            "session_id": self.session_id,
            "state": rec.state.value,
            "intent": intent.value if hasattr(intent, "value") else intent,
            "confidence": 1.0,
            "assistant_text": spoken,
            "agents_fired": agents_fired,
            "tool_sequence": tool_sequence,
            "resolved_product_id": rec.schema.resolved_product_id,
            "resolved_plan_id": rec.schema.resolved_plan_id,
            "purchased": rec.schema.purchased,
            "language": rec.schema.language or "english",
            "llm_enabled": self.llm.is_available,
            "orchestration_mode": "agentic",
            "agentic_backend": backend,
            "tool_trace": result.meta.get("tool_trace", []),
            "guardrail_ok": result.meta.get("guardrail_ok", True),
            "messages": rec.messages[-6:],
            # Present only when agentic_backend=agents_sdk + OPENAI_AGENTS_TRACE=1.
            "sdk_trace_id": sdk_trace_id,
            # Present only on the finalize_purchase turn.
            "wa_status": wa_status,
        }

    # ── helpers ─────────────────────────────────────────────────────────────
    def _normalise_for_voice(self, ctx: AgentContext, text: str) -> str:
        """Run the assembled reply through M_11 so it is speech-ready."""
        if not text:
            return text
        ctx.payload["response_text"] = text
        r = self._queue.run(ctx)
        return r.output_text or text

    def _delegate_to_fsm(self, message: str) -> dict[str, Any]:
        """Run this turn through the deterministic FSM orchestrator, sharing the
        SAME ConversationRecord so state stays consistent across mixed turns."""
        if self._fallback is None:
            fb = ConversationOrchestrator(session_id=self.session_id, config=self.config)
            fb.record = self.record          # share state
            fb.llm = self.llm
            fb.sarvam = self.sarvam
            self._fallback = fb
        out = self._fallback.process_message(message)
        out["orchestration_mode"] = "fsm_fallback"
        return out


def build_orchestrator(session_id: str = "session_001", config: AppConfig | None = None):
    """Factory: return the orchestrator selected by ORCHESTRATION_MODE.

    "agentic" → AgenticOrchestrator (M_16 tool-calling, FSM fallback).
    anything else → deterministic ConversationOrchestrator.

    Both expose the same `process_message()` contract and `record`/`llm`/`sarvam`
    attributes, so callers (main.py, the voice processor) are agnostic.
    """
    cfg = config or AppConfig()
    if cfg.agentic_enabled:
        return AgenticOrchestrator(session_id=session_id, config=cfg)
    return ConversationOrchestrator(session_id=session_id, config=cfg)
