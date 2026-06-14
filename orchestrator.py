"""
Lightweight conversation orchestrator for the AI Insurance Sales Agent.

This is a small runnable scaffold around the existing finite state machine.
It keeps the project structure requested by the assignment and provides a
single place to wire the higher-level application flow later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from fsm import (
    ConversationRecord,
    FSMState,
    IntentSignal,
    _next_state,
    detect_input_language,
)
from prompts_template import DATA_DIR
from settings import AppConfig
from llm_gateway import LLMGateway, SarvamGateway
from subagents import AGENT_REGISTRY, AgentID, AgentContext


# Constant fallbacks for states with no routed sub-agent.
_BLOCKED_TEXT = (
    "I'm sorry, but I can't help with that request. "
    "I'm here for health-insurance questions whenever you're ready."
)
_DISCOVERY_FALLBACK = "Could you share the main thing you'd like this insurance to cover?"

# Which agent leads each state's response chain. Handoffs inside the chain are
# followed automatically by the orchestrator (e.g. M_06→M_07, M_02→M_03→M_14).
_STATE_ENTRY_AGENT = {
    FSMState.S1_DISCOVERY:      AgentID.PROBING,          # M_05
    FSMState.S2_RECOMMENDATION: AgentID.POLICY_SUMMARY,   # M_07 (retrieval pre-run)
    FSMState.S3_POLICY_QA:      AgentID.POLICY_QA,        # M_08 → M_09 → …
    FSMState.S4_CLOSURE:        AgentID.CLOSURE,          # M_03 → M_14
    FSMState.S5_HUMAN_HANDOFF:  AgentID.ESCALATION,       # M_02 → M_03
}


@dataclass
class ConversationOrchestrator:
    """Minimal orchestration layer for one conversation session."""

    session_id: str = "session_001"
    config: AppConfig = field(default_factory=AppConfig)
    record: ConversationRecord = field(init=False)
    llm: LLMGateway = field(init=False)
    sarvam: SarvamGateway = field(init=False)

    def __post_init__(self) -> None:
        self.record = ConversationRecord.new(session_id=self.session_id)
        self.llm = LLMGateway(self.config)
        self.sarvam = SarvamGateway(self.config)

    def load_reference_data(self) -> dict[str, Any]:
        """Load the JSON assets from the new data directory."""
        with open(DATA_DIR / "policy_regulations_rag_ready.json", "r", encoding="utf-8") as file_handle:
            policy_regulations = json.load(file_handle)
        with open(DATA_DIR / "treatment_costs.json", "r", encoding="utf-8") as file_handle:
            treatment_costs = json.load(file_handle)
        return {
            "policy_regulations": policy_regulations,
            "treatment_costs": treatment_costs,
        }

    def process_message(self, message: str) -> dict[str, Any]:
        """
        Run one full turn through the deterministic sub-agent pipeline.

        Pipeline (deterministic core; LLM reasoning lives inside agents):
          1. bootstrap: increment turn, record user message, S0→S1.
          2. M_01  understanding: intent + NER in one LLM call → apply fields.
          3. M_04  deterministic schema top-up for fields M_01 missed.
          4. M_06  retrieval (while unresolved) to inform the transition gate.
          5. deterministic transition via fsm._next_state.
          6. route to the state's entry agent and follow its handoff chain.
          7. outbound: M_12 validate → M_10 translate → M_11 assemble.
          8. M_13 log the turn; record assistant message.
        """
        rec = self.record
        rec.schema.increment_turn()
        rec.add_message("user", message)

        # Cheap deterministic language label for THIS message so the reply (M_10)
        # is rendered back in the user's language. Understanding stays with the
        # multilingual LLM; this only sets schema.language (no translation hop).
        detected_language = detect_input_language(message)
        if detected_language != "english":
            rec.schema.set("language", detected_language)

        if rec.state == FSMState.S0_START:
            rec.transition_to(FSMState.S1_DISCOVERY, reason="session_started")

        ctx = AgentContext(record=rec, message=message, config=self.config,
                           llm=self.llm, sarvam=self.sarvam)
        agents_fired: list[str] = []

        # 2 + 3. M_01 (understanding) does intent + NER in ONE LLM call and
        # returns the extracted fields; M_04 then deterministically tops up any
        # fields the LLM missed (and is the sole extractor when the LLM is down).
        r_intent, _ = self._run_intent_and_schema(ctx, agents_fired)
        intent = r_intent.meta["intent"]
        confidence = r_intent.meta["confidence"]
        ctx.intent = intent
        ctx.confidence = confidence
        rec.schema.set("user_intent", intent.value)
        rec.register_intent_outcome(intent)

        # 4. Retrieval (M_06) while no product is resolved — populates
        #    ctx.retrieval_result so the transition gate can see candidates.
        if (
            rec.schema.resolved_product_id is None
            and rec.schema.sufficient_for_retrieval()
            and rec.state in (FSMState.S1_DISCOVERY, FSMState.S2_RECOMMENDATION)
        ):
            self._run_agent(AgentID.POLICY_RETRIEVAL, ctx, agents_fired)

        # 5. Deterministic transition.
        next_state = _next_state(
            rec.state, intent, rec.schema, rec.retrieval_result, rec.consecutive_failures
        )
        if next_state != rec.state:
            rec.transition_to(next_state, reason=f"intent={intent.value}")

        # A user keen to buy who has reached closure has effectively purchased.
        if rec.state == FSMState.S4_CLOSURE and intent == IntentSignal.PROSPECTIVE \
                and rec.schema.sufficient_for_closure():
            rec.schema.set("purchased", True)

        # 6. Route to the state's entry agent and follow its handoff chain.
        segments, guard_context = self._route(ctx, agents_fired)

        # 7. Outbound: validate → translate → assemble.
        final_text = self._finalize_response(ctx, segments, guard_context, agents_fired)

        # 8. Analytics (M_13) — the only agent that writes the conversation log.
        ctx.payload["assistant_response"] = final_text
        ctx.payload["agents_fired"] = agents_fired
        self._run_agent(AgentID.ANALYTICS_LOGGER, ctx, agents_fired)

        rec.add_message("assistant", final_text)
        rec.last_agent_response = final_text

        return {
            "session_id": self.session_id,
            "state": rec.state.value,
            "intent": intent.value,
            "confidence": confidence,
            "assistant_text": final_text,
            "agents_fired": agents_fired,
            "resolved_product_id": rec.schema.resolved_product_id,
            "resolved_plan_id": rec.schema.resolved_plan_id,
            "purchased": rec.schema.purchased,
            "llm_enabled": self.llm.is_available,
            "messages": rec.messages[-6:],
        }

    # ── AGENT EXECUTION ───────────────────────────────────────────────────────

    def _run_intent_and_schema(self, ctx: AgentContext, fired: list[str]):
        """Run M_01 (understanding: intent + NER) then M_04 (deterministic top-up).

        M_01's single LLM call returns the intent AND the structured fields it
        extracted; those are applied first. M_04 then fills only the fields M_01
        missed (it never overwrites a set field), so it both backstops the LLM and
        is the sole extractor when the LLM is unavailable. One network call total.
        """
        intent_agent = AGENT_REGISTRY[AgentID.INTENT_CLASSIFIER]
        schema_agent = AGENT_REGISTRY[AgentID.SCHEMA_EXTRACTOR]

        r_intent = intent_agent.run(ctx)
        fired.append(AgentID.INTENT_CLASSIFIER)
        for key, value in r_intent.schema_updates.items():
            ctx.record.schema.set(key, value)

        r_schema = schema_agent.run(ctx)
        fired.append(AgentID.SCHEMA_EXTRACTOR)
        for key, value in r_schema.schema_updates.items():
            ctx.record.schema.set(key, value)
        return r_intent, r_schema

    def _run_agent(self, agent_id: str, ctx: AgentContext, fired: list[str]):
        """Run one agent, record it, and apply its schema/retrieval side effects."""
        agent = AGENT_REGISTRY[agent_id]
        result = agent.run(ctx)
        fired.append(agent_id)

        for key, value in result.schema_updates.items():
            ctx.record.schema.set(key, value)

        retrieval = result.meta.get("retrieval_result")
        if retrieval is not None:
            ctx.retrieval_result = retrieval
            ctx.record.retrieval_result = retrieval
        return result

    def _route(self, ctx: AgentContext, fired: list[str]) -> tuple[list[str], str]:
        """Run the entry agent for the current state and follow its handoff chain."""
        state = ctx.record.state

        if state == FSMState.S6_BLOCKED:
            return [_BLOCKED_TEXT], ""

        entry = _STATE_ENTRY_AGENT.get(state)
        if entry is None:
            return [_DISCOVERY_FALLBACK], ""

        segments, guard_context, last_meta = self._run_chain(entry, ctx, fired)

        # Once a recommendation puts a plan on the table, mark it resolved so a
        # subsequent "I want to buy" can finalise closure.
        if state == FSMState.S2_RECOMMENDATION:
            plan = last_meta.get("plan_presented")
            if plan and ctx.record.schema.resolved_plan_id is None:
                ctx.record.schema.set("resolved_plan_id", plan)

        if not segments:
            if state == FSMState.S1_DISCOVERY:
                segments = [self._next_discovery_question()]
            else:
                segments = [_DISCOVERY_FALLBACK]
        return segments, guard_context

    def _run_chain(
        self, start_id: str, ctx: AgentContext, fired: list[str], max_hops: int = 6
    ) -> tuple[list[str], str, dict]:
        """Execute an agent handoff chain, collecting spoken segments + context."""
        segments: list[str] = []
        guard_context = ""
        last_meta: dict[str, Any] = {}
        agent_id: str | None = start_id
        visited: set[str] = set()

        while agent_id and agent_id not in visited and len(visited) < max_hops:
            visited.add(agent_id)
            result = self._run_agent(agent_id, ctx, fired)
            if result.output_text:
                segments.append(result.output_text)
            if result.meta.get("_retrieved_context"):
                guard_context = result.meta["_retrieved_context"]
            last_meta = result.meta
            agent_id = result.handoff_to
        return segments, guard_context, last_meta

    # ── OUTBOUND ──────────────────────────────────────────────────────────────

    def _finalize_response(
        self, ctx: AgentContext, segments: list[str], guard_context: str, fired: list[str]
    ) -> str:
        """Validate (M_12) → translate (M_10) → assemble (M_11) the response."""
        candidate = " ".join(s.strip() for s in segments if s and s.strip()).strip()

        ctx.payload["response_text"] = candidate
        r_valid = self._run_agent(AgentID.RESPONSE_VALIDATOR, ctx, fired)
        ctx.payload["response_text"] = r_valid.output_text or candidate

        r_trans = self._run_agent(AgentID.TRANSLATOR, ctx, fired)
        ctx.payload["response_text"] = r_trans.output_text or ctx.payload["response_text"]

        ctx.payload.pop("segments", None)
        r_queue = self._run_agent(AgentID.RESPONSE_QUEUE, ctx, fired)
        return r_queue.output_text or ctx.payload["response_text"]

    def _next_discovery_question(self) -> str:
        """Return next deterministic discovery question from the schema queue."""
        next_field = self.record.schema.next_missing_field()
        if isinstance(next_field, dict):
            question_text = next_field.get("question_text")
            if isinstance(question_text, str) and question_text.strip():
                return question_text.strip()
            label = next_field.get("label")
            if isinstance(label, str) and label.strip():
                return f"Please share your {label.lower()}."
        return _DISCOVERY_FALLBACK